import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
import os
import datetime
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import time
import json # Import json to parse the service account key
import base64 # Import base64 for decoding audio data from JS

# Firebase imports
import firebase_admin
from firebase_admin import credentials, storage

# --- Configuration Constants ---
MODEL_FILENAME = 'speaker_recognition_model.pkl'
LABELS_FILENAME = 'id_to_label_map.pkl'
TEMP_RECORDINGS_DIR = "temp_recordings" # For temporary local storage during processing

# Recording Specific (defaults, but actual recording is via browser API)
DEFAULT_NUM_SAMPLES = 5 # This is less relevant for a single custom recording
DEFAULT_DURATION = 4.0 # Duration for the custom recorder
DEFAULT_SAMPLE_RATE = 44100 # Standard sample rate

# Feature Extraction Specific
N_MFCC = 13 # Number of MFCCs to extract

# --- Firebase Initialization (using Streamlit Secrets) ---

@st.cache_resource
def initialize_firebase():
    """Initializes Firebase Admin SDK using Streamlit Secrets."""
    if not firebase_admin._apps: # Check if app is already initialized
        try:
            # Load credentials from Streamlit secrets
            cred_json_str = st.secrets["firebase_service_account_key"]
            bucket_name = st.secrets["firebase_storage_bucket"]

            # Parse the JSON string into a Python dictionary
            cred_dict = json.loads(cred_json_str)

            # Pass the dictionary directly to credentials.Certificate
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
            st.success("✅ Firebase initialized successfully.")
            return True
        except KeyError as e:
            st.error(f"❌ Firebase secret not found: {e}. Please ensure 'firebase_service_account_key' and 'firebase_storage_bucket' are set in your Streamlit secrets.toml.")
            return False
        except json.JSONDecodeError as e:
            st.error(f"❌ Error decoding Firebase service account key JSON: {e}. Please check the format in secrets.toml.")
            return False
        except Exception as e:
            st.error(f"❌ Error initializing Firebase: {e}")
            st.error("Please ensure your Firebase service account key JSON is correctly formatted in Streamlit secrets.")
            return False
    else:
        # If already initialized, return True
        return True


# --- Firebase Storage Functions ---

def upload_to_firebase(local_file_path, destination_blob_name):
    """Uploads a file to Firebase Cloud Storage."""
    try:
        bucket = storage.bucket()
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        st.info(f"⬆️ Uploaded {os.path.basename(local_file_path)} to Firebase as {destination_blob_name}")
        return True
    except Exception as e:
        st.error(f"❌ Error uploading {os.path.basename(local_file_path)} to Firebase: {e}")
        return False

def download_from_firebase(source_blob_name, local_file_path):
    """Downloads a file from Firebase Cloud Storage."""
    try:
        bucket = storage.bucket()
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(local_file_path)
        return True
    except Exception as e:
        st.error(f"❌ Error downloading {source_blob_name} from Firebase: {e}")
        return False

def list_firebase_files(prefix=""):
    """Lists files in the Firebase Storage bucket with a given prefix."""
    try:
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]
    except Exception as e:
        st.error(f"❌ Error listing files from Firebase: {e}")
        return []

# --- Utility Functions (modified for Streamlit & Firebase) ---

@st.cache_data(show_spinner=False)
def extract_features(file_path, n_mfcc=N_MFCC):
    """
    Extracts MFCCs from an audio file.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"⚠️ Error encountered while parsing file: {os.path.basename(file_path)} - {e}")
        return None
    return mfccs_processed

@st.cache_data(show_spinner="Loading data from Firebase...")
def load_data_from_firebase():
    """
    Loads all audio files from Firebase, downloads them temporarily,
    extracts features, and labels them.
    """
    X = [] # Features
    y = [] # Numeric labels
    labels_map = {} # Maps speaker name to a numeric ID
    id_to_label = [] # Maps numeric ID back to speaker name
    
    label_id_counter = 0

    st.write("--- Loading data from Firebase Cloud Storage ---")
    
    all_blobs = list_firebase_files()
    
    speaker_names = sorted(list(set([blob.split('/')[0] for blob in all_blobs if '/' in blob and blob.endswith('.wav')])))

    if not speaker_names:
        st.warning(f"❌ No speaker data found in Firebase Storage bucket '{st.secrets['firebase_storage_bucket']}'. Please record some voices first.")
        return np.array([]), np.array([]), {}, []

    # Ensure a local temp directory exists for downloads
    os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True)

    progress_text = st.empty()
    progress_bar = st.progress(0)
    total_files_to_process = sum([len([b for b in all_blobs if b.startswith(f"{s}/") and b.endswith('.wav')]) for s in speaker_names])
    files_processed_count = 0

    for speaker_name in speaker_names:
        if speaker_name not in labels_map:
            labels_map[speaker_name] = label_id_counter
            id_to_label.append(speaker_name)
            label_id_counter += 1

        current_label_id = labels_map[speaker_name]

        speaker_has_audio = False
        speaker_blobs = list_firebase_files(prefix=f"{speaker_name}/")

        for blob_name in speaker_blobs:
            if blob_name.endswith('.wav'):
                local_temp_file = os.path.join(TEMP_RECORDINGS_DIR, os.path.basename(blob_name))
                progress_text.text(f"Downloading and processing: {speaker_name}/{os.path.basename(blob_name)}")

                if download_from_firebase(blob_name, local_temp_file):
                    features = extract_features(local_temp_file)
                    if features is not None:
                        X.append(features)
                        y.append(current_label_id)
                        speaker_has_audio = True
                    os.remove(local_temp_file) # Clean up local temp file immediately
                else:
                    st.warning(f"Skipping {blob_name} due to download error.")
                
                files_processed_count += 1
                if total_files_to_process > 0:
                    progress_bar.progress(files_processed_count / total_files_to_process)

        if not speaker_has_audio:
            st.warning(f"   No valid .wav files found or downloaded for {speaker_name}. This speaker will be skipped for training.")
    
    progress_bar.empty()
    progress_text.empty()
    st.success("Data loading complete.")

    return np.array(X), np.array(y), labels_map, id_to_label

@st.cache_resource(show_spinner="Training model...")
def train_and_save_model():
    """
    Loads data from Firebase, trains a RandomForestClassifier model, and saves it locally.
    """
    st.write("\n--- Training Model ---")
    X, y, labels_map, id_to_label = load_data_from_firebase()

    if len(X) == 0:
        st.error("❌ No audio data found or features extracted from Firebase. Cannot train model.")
        return None, None
    
    unique_speakers = len(labels_map)
    if unique_speakers < 2:
        st.warning(f"❌ Need at least 2 distinct speakers ({unique_speakers} found) to train a meaningful model. Please add more data.")
        return None, None
    
    for speaker_id in range(unique_speakers):
        if np.sum(y == speaker_id) < 2:
            st.warning(f"❌ Speaker '{id_to_label[speaker_id]}' has fewer than 2 samples ({np.sum(y == speaker_id)} found). Each speaker needs at least 2 samples to train. Please add more data for this speaker.")
            return None, None
            
    st.write(f"\nTotal samples loaded: {len(X)}")
    st.write(f"Speakers found: {labels_map}")

    # Ensure enough samples for splitting
    if len(X) < 2:
        st.warning("Not enough total samples to split into training and testing sets. Need at least 2 samples.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    st.write(f"Training samples: {len(X_train)}")
    st.write(f"Testing samples: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.success("Model training complete. ✅")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"\nModel Accuracy on test set: **{accuracy * 100:.2f}%**")
    st.write("\nClassification Report:")
    st.code(classification_report(y_test, y_pred, target_names=id_to_label))

    # Save the trained model and the ID-to-label mapping locally (Streamlit Cloud ephemeral)
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(model, f)
    with open(LABELS_FILENAME, 'wb') as f:
        pickle.dump(id_to_label, f)
    st.info(f"Model saved locally to {MODEL_FILENAME} and {LABELS_FILENAME}")
    
    return model, id_to_label

@st.cache_resource(show_spinner="Loading model...")
def load_trained_model_and_labels():
    """
    Loads a pre-trained model and label mapping from local disk.
    If not found, it attempts to retrain or download from Firebase.
    """
    try:
        with open(MODEL_FILENAME, 'rb') as f:
            model = pickle.load(f)
        with open(LABELS_FILENAME, 'rb') as f:
            id_to_label = pickle.load(f)
        st.success("✅ Model and labels loaded successfully from local files.")
        return model, id_to_label
    except FileNotFoundError:
        st.warning("Model or labels file not found locally. Attempting to train a new model from Firebase data.")
        return train_and_save_model()


def recognize_speaker_from_audio_data(model, id_to_label, audio_bytes):
    """
    Recognizes a speaker from audio data (bytes), writes to temp file, processes, cleans up.
    """
    if model is None or id_to_label is None:
        st.error("Error: Model not loaded. Please add new data (Option 1) to train the model.")
        return "Not Available", 0.0

    os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True)
    temp_file = os.path.join(TEMP_RECORDINGS_DIR, f"temp_recognition_{int(time.time())}.wav")

    with open(temp_file, "wb") as f:
        f.write(audio_bytes)

    features = extract_features(temp_file)
    if features is None:
        os.remove(temp_file)
        return "Unknown Speaker (Feature Extraction Failed)", 0.0

    features = features.reshape(1, -1) # Reshape for prediction

    prediction_id = model.predict(features)[0]
    predicted_speaker = id_to_label[prediction_id]

    probabilities = model.predict_proba(features)[0]
    confidence = probabilities[prediction_id] * 100

    os.remove(temp_file) # Clean up the temporary file

    return predicted_speaker, confidence

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Speaker Recognition", layout="centered")
    st.title("🗣️ Speaker Recognition App")
    st.markdown("Powered by Firebase Cloud Storage for voice data.")

    # Ensure temporary recordings directory exists at startup
    os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True)

    # Initialize Firebase at the very beginning
    if not initialize_firebase():
        st.stop() # Stop the app if Firebase initialization fails

    # Load model and labels once and cache them
    trained_model, id_to_label_map = load_trained_model_and_labels()

    st.sidebar.header("Navigation")
    app_mode = st.sidebar.radio("Choose a mode:", ["Add New Speaker Data", "Recognize Speaker from File", "Live Speaker Recognition"])

    if app_mode == "Add New Speaker Data":
        st.header("1. Add New Speaker Voice Data (Custom Recorder)")
        st.info("Record a sample for a new or existing speaker using the custom recorder. This data will be uploaded to Firebase Cloud Storage and used to (re)train the model.")

        person_name = st.text_input("Enter the name of the person to record:", key="add_person_name_input")

        # --- Custom HTML/JS Recorder Component ---
        # The JavaScript here handles microphone access, recording, and sending Base64 data to Python.
        # It's embedded directly into the HTML component for simplicity.
        custom_recorder_html = f"""
        <style>
            .recorder-container {{
                border: 1px solid #ddd;
                padding: 15px;
                border-radius: 8px;
                background-color: #f9f9f9;
                text-align: center;
            }}
            .recorder-button {{
                background-color: #4CAF50; /* Green */
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 5px;
                transition: background-color 0.3s ease;
            }}
            .recorder-button:hover:not(:disabled) {{
                background-color: #45a049;
            }}
            .recorder-button:disabled {{
                background-color: #cccccc;
                cursor: not-allowed;
            }}
            #stopButton {{
                background-color: #f44336; /* Red */
            }}
            #stopButton:hover:not(:disabled) {{
                background-color: #da190b;
            }}
            #audioPlayback {{
                width: 90%;
                margin-top: 15px;
            }}
            #status {{
                margin-top: 10px;
                font-weight: bold;
                color: #333;
            }}
        </style>
        <div class="recorder-container">
            <button id="recordButton" class="recorder-button">Start Recording</button>
            <button id="stopButton" class="recorder-button" disabled>Stop Recording</button>
            <audio id="audioPlayback" controls style="display:block;margin: 15px auto;"></audio>
            <div id="status">Status: Ready</div>
        </div>

        <script>
            const recordButton = document.getElementById('recordButton');
            const stopButton = document.getElementById('stopButton');
            const audioPlayback = document.getElementById('audioPlayback');
            const statusDiv = document.getElementById('status');
            let mediaRecorder;
            let audioChunks = [];
            let stream; // To hold the media stream

            // Streamlit's way to communicate back to Python
            const streamlit = window.Streamlit;

            recordButton.onclick = async () => {{
                audioChunks = [];
                try {{
                    stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                    mediaRecorder = new MediaRecorder(stream);
                    
                    mediaRecorder.ondataavailable = event => {{
                        audioChunks.push(event.data);
                    }};
                    
                    mediaRecorder.onstop = () => {{
                        const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                        const reader = new FileReader();
                        reader.onloadend = () => {{
                            const base64data = reader.result; // Base64 encoded string
                            // Send data back to Streamlit
                            streamlit.setComponentValue(base64data);
                            audioPlayback.src = URL.createObjectURL(audioBlob);
                            statusDiv.innerText = "Status: Recording stopped. Audio sent to Streamlit.";
                            
                            // Stop all tracks in the stream to release microphone
                            if (stream) {{
                                stream.getTracks().forEach(track => track.stop());
                            }}
                        }};
                        reader.readAsDataURL(audioBlob);
                    }};
                    
                    mediaRecorder.start();
                    statusDiv.innerText = "Status: Recording... Speak now!";
                    recordButton.disabled = true;
                    stopButton.disabled = false;
                    audioPlayback.removeAttribute('src'); // Clear previous playback
                    
                    // Automatically stop after DEFAULT_DURATION seconds
                    setTimeout(() => {{
                        if (mediaRecorder.state === 'recording') {{
                            mediaRecorder.stop();
                            statusDiv.innerText = "Status: Recording stopped automatically.";
                            recordButton.disabled = false;
                            stopButton.disabled = true;
                        }}
                    }}, {int(DEFAULT_DURATION * 1000)}); // Convert seconds to milliseconds
                }} catch (err) {{
                    statusDiv.innerText = "Status: Microphone access denied or error: " + err.name + " - " + err.message;
                    console.error('Microphone access error:', err);
                }}
            }};

            stopButton.onclick = () => {{
                if (mediaRecorder && mediaRecorder.state === 'recording') {{
                    mediaRecorder.stop();
                    statusDiv.innerText = "Status: Recording stopped by user.";
                    recordButton.disabled = false;
                    stopButton.disabled = true;
                }}
            }};

            // Initialize component, tell Streamlit its ready
            // Use a small timeout to ensure the DOM is fully rendered before calculating height
            setTimeout(() => {{
                streamlit.setFrameHeight(document.documentElement.scrollHeight);
            }}, 100);
        </script>
        """
        # Render the custom component
        # The height is crucial for the component to be visible
        # The key is also important here for Streamlit to manage the component's state
        recorded_audio_base64 = st.components.v1.html(custom_recorder_html, height=250, scrolling=False)
            
            try:
                wav_audio_data = base64.b64decode(encoded)
                st.audio(wav_audio_data, format='audio/wav')

                if st.button("Save Recorded Sample to Firebase", key="save_add_speaker_button") and person_name:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    local_filename = os.path.join(TEMP_RECORDINGS_DIR, f"{person_name}_sample_{timestamp}.wav")
                    cloud_filename = f"{person_name}/{person_name}_sample_{timestamp}.wav"

                    with open(local_filename, "wb") as f:
                        f.write(wav_audio_data)

                    if upload_to_firebase(local_filename, cloud_filename):
                        st.success(f"Sample for {person_name} uploaded successfully!")
                        os.remove(local_filename)
                        
                        # Invalidate cache for training data so next training uses new data
                        st.cache_data.clear() 
                        st.cache_resource.clear() 
                        st.warning("New data added! The model needs to be retrained for these changes to take effect. The app will now reload to retrain.")
                        st.experimental_rerun() # Rerun the app to trigger retraining
                    else:
                        st.error("Failed to save sample. Check console for details.")
                        if os.path.exists(local_filename):
                            os.remove(local_filename)
                elif st.button("Save Recorded Sample to Firebase", key="save_add_speaker_button_no_name"):
                    st.warning("Please enter a person's name before saving the recording.")
            except base64.binascii.Error as e:
                st.error(f"❌ Error decoding audio data: {e}. The received data might be corrupted.")
            except Exception as e:
                st.error(f"An unexpected error occurred during audio processing: {e}")


    elif app_mode == "Recognize Speaker from File":
        st.header("2. Recognize Speaker from a Local File")
        st.info("Upload an audio file to recognize the speaker.")

        if trained_model is None:
            st.warning("Model not trained or loaded. Please add new data (Option 1) and train the model first.")
            return

        uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"], key="file_uploader_recognize")

        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            audio_bytes = uploaded_file.read()
            
            if st.button("Recognize Speaker", key="recognize_file_button"):
                with st.spinner("Processing file..."):
                    predicted_speaker, confidence = recognize_speaker_from_audio_data(trained_model, id_to_label_map, audio_bytes)
                    if predicted_speaker != "Not Available":
                        st.success(f"Predicted Speaker: **{predicted_speaker}** (Confidence: **{confidence:.2f}%**)")
                    else:
                        st.error("Could not recognize speaker.")


    elif app_mode == "Live Speaker Recognition":
        st.header("3. Recognize Speaker from Live Microphone Input")
        st.info(f"Record your voice for {DEFAULT_DURATION} seconds, and the app will try to recognize you.")

        if trained_model is None:
            st.warning("Model not trained or loaded. Please add new data (Option 1) and train the model first.")
            return

        # --- Custom HTML/JS Recorder Component for Live Recognition ---
        # This is a separate instance, so it needs its own key.
        # The functionality is similar to the "Add New Speaker Data" recorder.
        live_recorder_html = f"""
        <style>
            .recorder-container {{
                border: 1px solid #ddd;
                padding: 15px;
                border-radius: 8px;
                background-color: #f9f9f9;
                text-align: center;
            }}
            .recorder-button {{
                background-color: #4CAF50; /* Green */
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 5px;
                transition: background-color 0.3s ease;
            }}
            .recorder-button:hover:not(:disabled) {{
                background-color: #45a049;
            }}
            .recorder-button:disabled {{
                background-color: #cccccc;
                cursor: not-allowed;
            }}
            #stopButtonLive {{ /* Unique ID for live recorder stop button */
                background-color: #f44336; /* Red */
            }}
            #stopButtonLive:hover:not(:disabled) {{
                background-color: #da190b;
            }}
            #audioPlaybackLive {{ /* Unique ID for live recorder audio playback */
                width: 90%;
                margin-top: 15px;
            }}
            #statusLive {{ /* Unique ID for live recorder status */
                margin-top: 10px;
                font-weight: bold;
                color: #333;
            }}
        </style>
        <div class="recorder-container">
            <button id="recordButtonLive" class="recorder-button">Start Live Recording</button>
            <button id="stopButtonLive" class="recorder-button" disabled>Stop Live Recording</button>
            <audio id="audioPlaybackLive" controls style="display:block;margin: 15px auto;"></audio>
            <div id="statusLive">Status: Ready</div>
        </div>

        <script>
            const recordButtonLive = document.getElementById('recordButtonLive');
            const stopButtonLive = document.getElementById('stopButtonLive');
            const audioPlaybackLive = document.getElementById('audioPlaybackLive');
            const statusDivLive = document.getElementById('statusLive');
            let mediaRecorderLive;
            let audioChunksLive = [];
            let streamLive;

            const streamlit = window.Streamlit;

            recordButtonLive.onclick = async () => {{
                audioChunksLive = [];
                try {{
                    streamLive = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                    mediaRecorderLive = new MediaRecorder(streamLive);
                    
                    mediaRecorderLive.ondataavailable = event => {{
                        audioChunksLive.push(event.data);
                    }};
                    
                    mediaRecorderLive.onstop = () => {{
                        const audioBlobLive = new Blob(audioChunksLive, {{ type: 'audio/wav' }});
                        const readerLive = new FileReader();
                        readerLive.onloadend = () => {{
                            const base64dataLive = readerLive.result;
                            streamlit.setComponentValue(base64dataLive); // Send data back to Streamlit
                            audioPlaybackLive.src = URL.createObjectURL(audioBlobLive);
                            statusDivLive.innerText = "Status: Live recording stopped. Audio sent to Streamlit.";
                            
                            if (streamLive) {{
                                streamLive.getTracks().forEach(track => track.stop());
                            }}
                        }};
                        readerLive.readAsDataURL(audioBlobLive);
                    }};
                    
                    mediaRecorderLive.start();
                    statusDivLive.innerText = "Status: Recording... Speak now!";
                    recordButtonLive.disabled = true;
                    stopButtonLive.disabled = false;
                    audioPlaybackLive.removeAttribute('src');
                    
                    setTimeout(() => {{
                        if (mediaRecorderLive.state === 'recording') {{
                            mediaRecorderLive.stop();
                            statusDivLive.innerText = "Status: Live recording stopped automatically.";
                            recordButtonLive.disabled = false;
                            stopButtonLive.disabled = true;
                        }}
                    }}, {int(DEFAULT_DURATION * 1000)});
                }} catch (err) {{
                    statusDivLive.innerText = "Status: Microphone access denied or error: " + err.name + " - " + err.message;
                    console.error('Microphone access error:', err);
                }}
            }};

            stopButtonLive.onclick = () => {{
                if (mediaRecorderLive && mediaRecorderLive.state === 'recording') {{
                    mediaRecorderLive.stop();
                    statusDivLive.innerText = "Status: Live recording stopped by user.";
                    recordButtonLive.disabled = false;
                    stopButtonLive.disabled = true;
                }}
            }};

            setTimeout(() => {{
                streamlit.setFrameHeight(document.documentElement.scrollHeight);
            }}, 100);
        </script>
        """
        live_recorded_audio_base64 = st.components.v1.html(live_recorder_html, height=250, scrolling=False, key="custom_live_recorder")

        if live_recorded_audio_base64:
            st.info("Live audio received! Click 'Analyze Live Recording' to recognize.")
            if "," in live_recorded_audio_base64:
                header, encoded_live = live_recorded_audio_base64.split(",", 1)
            else:
                encoded_live = live_recorded_audio_base64

            try:
                live_wav_audio_data = base64.b64decode(encoded_live)
                st.audio(live_wav_audio_data, format='audio/wav')
                
                if st.button("Analyze Live Recording", key="analyze_live_button"):
                    with st.spinner("Analyzing live recording..."):
                        predicted_speaker, confidence = recognize_speaker_from_audio_data(trained_model, id_to_label_map, live_wav_audio_data)
                        
                        if predicted_speaker != "Not Available":
                            st.success(f"Predicted Speaker: **{predicted_speaker}** (Confidence: **{confidence:.2f}%**)")
                        else:
                            st.error("Could not recognize speaker.")
            except base64.binascii.Error as e:
                st.error(f"❌ Error decoding live audio data: {e}. The received data might be corrupted.")
            except Exception as e:
                st.error(f"An unexpected error occurred during live audio processing: {e}")


if __name__ == "__main__":
    main()
