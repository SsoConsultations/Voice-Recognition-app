import streamlit as st
import sounddevice as sd
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
import io # For handling in-memory audio data

# Firebase imports
import firebase_admin
from firebase_admin import credentials, storage

# --- Configuration Constants ---
# General
# DATA_BASE_DIR is now implicitly handled by Firebase Storage structure
MODEL_FILENAME = 'speaker_recognition_model.pkl'
LABELS_FILENAME = 'id_to_label_map.pkl'
TEMP_RECORDINGS_DIR = "temp_recordings" # For local temporary storage before/after Firebase interaction

# Recording Specific
DEFAULT_NUM_SAMPLES = 3     # Reduced for faster testing on Streamlit
DEFAULT_DURATION = 3.0      # Duration of each recording in seconds (can be float)
DEFAULT_SAMPLE_RATE = 44100 # Sample rate (samples per second). 44100 Hz is standard CD quality.

# Feature Extraction Specific
N_MFCC = 13 # Number of MFCCs to extract

# --- Firebase Configuration (for Streamlit Secrets) ---
# When deploying to Streamlit Cloud, you'll use st.secrets for these.
# For local testing, ensure firebase_service_account.json is in your project root.
try:
    FIREBASE_SERVICE_ACCOUNT_CONFIG = st.secrets["firebase"]["service_account"]
    FIREBASE_STORAGE_BUCKET = st.secrets["firebase"]["storage_bucket"]
    # Write the service account config to a temporary file
    with open("firebase_service_account.json", "w") as f:
        json.dump(FIREBASE_SERVICE_ACCOUNT_CONFIG, f)
    FIREBASE_SERVICE_ACCOUNT_KEY_PATH = "firebase_service_account.json"
except (KeyError, FileNotFoundError):
    st.warning("Firebase secrets not found. Attempting to load from local file. Make sure 'firebase_service_account.json' exists for local testing.")
    FIREBASE_SERVICE_ACCOUNT_KEY_PATH = 'firebase_service_account.json' # Path for local testing
    FIREBASE_STORAGE_BUCKET = 'your-project-id.appspot.com' # Replace with your actual bucket name for local testing


# --- Utility Functions (Adapted for Streamlit and Firebase) ---

# Initialize Firebase Admin SDK
@st.cache_resource
def initialize_firebase_app():
    if not firebase_admin._apps: # Check if app is already initialized
        try:
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY_PATH)
            firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_STORAGE_BUCKET})
            st.success("✅ Firebase initialized successfully.")
        except Exception as e:
            st.error(f"❌ Error initializing Firebase: {e}. Please check your Firebase credentials.")

initialize_firebase_app()


def upload_audio_to_firebase(local_file_path, destination_blob_name):
    """Uploads a file to Firebase Storage."""
    try:
        bucket = storage.bucket()
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        st.success(f"Uploaded {os.path.basename(local_file_path)} to Firebase Storage.")
        return True
    except Exception as e:
        st.error(f"❌ Error uploading {os.path.basename(local_file_path)} to Firebase: {e}")
        return False

def download_audio_from_firebase(source_blob_name, destination_file_path):
    """Downloads a blob from Firebase Storage."""
    try:
        bucket = storage.bucket()
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_path)
        return True
    except Exception as e:
        # st.error(f"❌ Error downloading {source_blob_name} from Firebase: {e}")
        return False

def list_files_in_firebase_storage(prefix=""):
    """Lists all blobs in the bucket that start with the given prefix."""
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]

def extract_features(file_path_or_buffer, n_mfcc=N_MFCC):
    """
    Extracts MFCCs from an audio file path or a file-like object (e.g., Streamlit UploadedFile).
    """
    try:
        y, sr = librosa.load(file_path_or_buffer, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"⚠️ Error encountered while parsing audio: {e}")
        return None
    return mfccs_processed

@st.cache_data
def load_data_from_firebase(data_prefix="data"):
    X = []
    y = []
    labels_map = {}
    id_to_label = []
    label_id_counter = 0

    all_blobs = list_files_in_firebase_storage(prefix=data_prefix + "/")
    speaker_names = sorted(list(set([blob.split('/')[1] for blob in all_blobs if len(blob.split('/')) > 1])))

    if not speaker_names:
        st.warning(f"No speaker data found in Firebase Storage under '{data_prefix}'.")
        return np.array([]), np.array([]), {}, []

    st.write(f"Processing speakers: {', '.join(speaker_names)}")
    progress_bar = st.progress(0)
    total_audio_files = sum(1 for blob in all_blobs if blob.endswith('.wav'))
    processed_count = 0

    for speaker_name in speaker_names:
        speaker_prefix = f"{data_prefix}/{speaker_name}/"
        
        if speaker_name not in labels_map:
            labels_map[speaker_name] = label_id_counter
            id_to_label.append(speaker_name)
            label_id_counter += 1

        current_label_id = labels_map[speaker_name]
        
        speaker_audio_blobs = [b for b in all_blobs if b.startswith(speaker_prefix) and b.endswith('.wav')]
        
        speaker_has_audio = False
        for firebase_audio_path in speaker_audio_blobs:
            temp_download_path = os.path.join(TEMP_RECORDINGS_DIR, os.path.basename(firebase_audio_path))
            os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True) # Ensure temp dir exists

            if download_audio_from_firebase(firebase_audio_path, temp_download_path):
                features = extract_features(temp_download_path)
                if features is not None:
                    X.append(features)
                    y.append(current_label_id)
                    speaker_has_audio = True
                os.remove(temp_download_path)
            
            processed_count += 1
            progress_bar.progress(processed_count / total_audio_files)

        if not speaker_has_audio:
            st.info(f"No valid .wav files found or downloaded for {speaker_name}.")
    
    return np.array(X), np.array(y), labels_map, id_to_label

@st.cache_resource # Use cache_resource for models as they are objects
def train_and_save_model():
    st.subheader("Training Model...")
    X, y, labels_map, id_to_label = load_data_from_firebase()

    if len(X) == 0:
        st.warning("No audio data found or features extracted from Firebase. Cannot train model.")
        return None, None
    
    unique_speakers = len(labels_map)
    if unique_speakers < 2:
        st.warning(f"Need at least 2 distinct speakers ({unique_speakers} found) to train a meaningful model. Please add more data.")
        return None, None
    
    # Check if each speaker has enough samples for stratified splitting
    for speaker_id in range(unique_speakers):
        if np.sum(y == speaker_id) < 2:
            st.warning(f"Speaker '{id_to_label[speaker_id]}' has fewer than 2 samples. Each speaker needs at least 2 samples to train. Please add more data for this speaker.")
            return None, None

    st.info(f"Total samples loaded: {len(X)}")
    st.info(f"Speakers found: {labels_map}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    st.info(f"Training samples: {len(X_train)}")
    st.info(f"Testing samples: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    with st.spinner("Training in progress..."):
        model.fit(X_train, y_train)
    st.success("Model training complete. ✅")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy on test set", value=f"{accuracy * 100:.2f}%")
    st.write("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=id_to_label))

    # Save and upload model/labels
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(model, f)
    with open(LABELS_FILENAME, 'wb') as f:
        pickle.dump(id_to_label, f)

    upload_audio_to_firebase(MODEL_FILENAME, MODEL_FILENAME)
    upload_audio_to_firebase(LABELS_FILENAME, LABELS_FILENAME)
    
    return model, id_to_label

@st.cache_resource
def load_trained_model():
    st.info("Attempting to load existing model from Firebase Storage...")
    try:
        temp_model_path = os.path.join(TEMP_RECORDINGS_DIR, MODEL_FILENAME)
        temp_labels_path = os.path.join(TEMP_RECORDINGS_DIR, LABELS_FILENAME)
        
        os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True)

        if not download_audio_from_firebase(MODEL_FILENAME, temp_model_path):
            st.warning(f"Model file '{MODEL_FILENAME}' not found in Firebase Storage.")
            return None, None
        if not download_audio_from_firebase(LABELS_FILENAME, temp_labels_path):
            st.warning(f"Labels file '{LABELS_FILENAME}' not found in Firebase Storage.")
            if os.path.exists(temp_model_path): os.remove(temp_model_path)
            return None, None

        with open(temp_model_path, 'rb') as f:
            model = pickle.load(f)
        with open(temp_labels_path, 'rb') as f:
            id_to_label = pickle.load(f)
        st.success("✅ Model and labels loaded successfully from Firebase.")
        
        if os.path.exists(temp_model_path): os.remove(temp_model_path)
        if os.path.exists(temp_labels_path): os.remove(temp_labels_path)
        return model, id_to_label
    except Exception as e:
        st.error(f"❌ Error loading model/labels from Firebase: {e}")
        return None, None

def recognize_speaker_from_audio(model, id_to_label, audio_data, sr):
    """
    Recognizes a speaker from audio data (numpy array) and sample rate.
    This is useful for live recordings or audio inputs from Streamlit.
    """
    if model is None or id_to_label is None:
        return "Not Available (Model not loaded)"

    # Save to a temporary WAV file for librosa to process
    temp_audio_path = os.path.join(TEMP_RECORDINGS_DIR, f"temp_rec_{int(time.time())}.wav")
    os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True)
    wav.write(temp_audio_path, sr, audio_data)

    features = extract_features(temp_audio_path)
    os.remove(temp_audio_path) # Clean up temp file

    if features is None:
        return "Unknown Speaker (Feature Extraction Failed)"

    features = features.reshape(1, -1) # Reshape for prediction

    prediction_id = model.predict(features)[0]
    predicted_speaker = id_to_label[prediction_id]

    probabilities = model.predict_proba(features)[0]
    confidence = probabilities[prediction_id] * 100

    st.write(f"Predicted Speaker: **{predicted_speaker}** (Confidence: {confidence:.2f}%)")
    return predicted_speaker

# --- Streamlit UI ---

st.title("🗣️ Speaker Recognition App")
st.markdown("---")

trained_model, id_to_label_map = load_trained_model()

# --- Main menu in sidebar for better navigation ---
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Add New Speaker Data", "Recognize Speaker from File", "Recognize Speaker Live"])

# --- Home Section ---
if app_mode == "Home":
    st.subheader("Welcome!")
    st.write("This application allows you to train a speaker recognition model and identify speakers from audio recordings.")
    st.write("To get started:")
    st.markdown("- Use **'Add New Speaker Data'** to record voice samples for different people. This will automatically train/retrain the model.")
    st.markdown("- Then, use **'Recognize Speaker from File'** or **'Recognize Speaker Live'** to test the model.")
    
    if trained_model:
        st.success("A model is currently loaded and ready for recognition!")
        st.write(f"Known speakers: {', '.join(id_to_label_map)}")
    else:
        st.warning("No model currently loaded. Please add new speaker data to train one.")

# --- Add New Speaker Data ---
elif app_mode == "Add New Speaker Data":
    st.header("➕ Add/Record New Speaker Voice Data")
    st.write("Record multiple voice samples for a person to train the recognition model.")

    person_name = st.text_input("Enter the name of the person:", key="person_name_input").strip()

    if person_name:
        st.info(f"Preparing to record {DEFAULT_NUM_SAMPLES} samples for **{person_name}**, each {DEFAULT_DURATION} seconds long.")
        st.write("Please speak clearly for each sample in a quiet environment.")

        if st.button(f"Start Recording Session for {person_name}"):
            recorded_successfully = 0
            for i in range(1, DEFAULT_NUM_SAMPLES + 1):
                st.write(f"--- Recording sample {i}/{DEFAULT_NUM_SAMPLES} for {person_name} ---")
                st.info("Click 'Start Recording' below and SPEAK NOW!")
                
                # Streamlit audio_recorder component (requires separate install for st_audiorec)
                # Or use st.audio_input for simple upload
                st.warning("Due to browser limitations, direct live recording via `sounddevice` in Streamlit Cloud is not straightforward.")
                st.info("For a better live recording experience, consider using a custom component like `streamlit-audio-recorder` (requires `pip install streamlit-audiorec`) or using the file upload option.")
                
                # Fallback to local recording prompt (will only work if app is run locally)
                st.write("**(If running locally, press Enter in your console to start recording for each sample.)**")
                # For web deployment, direct sd.rec will fail.
                # A more robust solution for live recording on Streamlit:
                # Use `streamlit-audio-recorder` custom component: https://github.com/stefanrmmr/streamlit-audio-recorder
                
                # Mock recording for demonstration purposes or for local testing with `sd.rec`
                audio_data = None
                with st.spinner(f"Recording sample {i}..."):
                    try:
                        # This part will likely not work directly on Streamlit Cloud
                        recording = sd.rec(int(DEFAULT_DURATION * DEFAULT_SAMPLE_RATE), 
                                           samplerate=DEFAULT_SAMPLE_RATE, channels=1, dtype=np.int16)
                        sd.wait()
                        audio_data = recording
                        st.success(f"Sample {i} recorded locally.")
                    except Exception as e:
                        st.error(f"Error during recording: {e}. Live recording might not be supported in your environment (e.g., Streamlit Cloud).")
                        st.info("Please use the file upload option instead if live recording fails.")
                        continue # Skip to next sample if recording failed

                if audio_data is not None:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    local_filename = os.path.join(TEMP_RECORDINGS_DIR, f"{person_name}_sample_{i}_{timestamp}.wav")
                    os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True)
                    wav.write(local_filename, DEFAULT_SAMPLE_RATE, audio_data)

                    firebase_path = f"data/{person_name}/{os.path.basename(local_filename)}"
                    if upload_audio_to_firebase(local_filename, firebase_path):
                        recorded_successfully += 1
                    os.remove(local_filename)
                    st.write(f"Sample {i} processing complete.")
                else:
                    st.warning(f"Sample {i} skipped due to recording issue.")
                
                time.sleep(1) # Small pause between recordings

            if recorded_successfully > 0:
                st.success(f"Recording session for {person_name} complete! {recorded_successfully} samples recorded and uploaded.")
                st.info("Automatically retraining model with updated data...")
                trained_model, id_to_label_map = train_and_save_model()
            else:
                st.warning("No new samples were successfully recorded or uploaded. Model not retrained.")
    else:
        st.info("Please enter a person's name to start recording.")

# --- Recognize Speaker from File ---
elif app_mode == "Recognize Speaker from File":
    st.header("🔍 Recognize Speaker from a File")
    
    if trained_model is None:
        st.warning("Cannot recognize. Model not trained or loaded. Please add new data (Option 1) first.")
    else:
        uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            # Save the uploaded file temporarily to process it with librosa
            temp_upload_path = os.path.join(TEMP_RECORDINGS_DIR, uploaded_file.name)
            os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True)
            with open(temp_upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write("Analyzing uploaded file...")
            recognized_speaker = recognize_speaker_from_file(trained_model, id_to_label_map, temp_upload_path)
            st.success(f"File analysis complete. Predicted Speaker: **{recognized_speaker}**")
            os.remove(temp_upload_path) # Clean up temp file

# --- Recognize Speaker Live ---
elif app_mode == "Recognize Speaker Live":
    st.header("🎤 Recognize Speaker from Live Microphone Input")

    if trained_model is None:
        st.warning("Cannot recognize. Model not trained or loaded. Please add new data (Option 1) first.")
    else:
        st.write(f"Speak for {DEFAULT_DURATION} seconds to get a live prediction.")
        
        # Use streamlit-audio-recorder for proper live recording in Streamlit
        # pip install streamlit-audiorec
        try:
            from st_audiorec import st_audiorec
            wav_audio_data = st_audiorec() # Returns raw audio bytes
            
            if wav_audio_data is not None:
                st.audio(wav_audio_data, format='audio/wav')
                st.write("Analyzing live recording...")
                
                # Save the recorded audio bytes to a BytesIO object for librosa
                audio_buffer = io.BytesIO(wav_audio_data)
                
                recognized_speaker = recognize_speaker_from_file(trained_model, id_to_label_map, audio_buffer)
                st.success(f"Live analysis complete. Predicted Speaker: **{recognized_speaker}**")

        except ImportError:
            st.error("The `streamlit-audiorec` component is not installed. Please install it (`pip install streamlit-audiorec`) for live recording functionality.")
            st.info("Alternatively, you can manually record and upload a file using the 'Recognize Speaker from File' option.")
        except Exception as e:
            st.error(f"An error occurred during live recording recognition: {e}")

# --- Clean up temporary directory on exit (optional, can be problematic on cloud deployments) ---
# It's better to let Streamlit's ephemeral containers handle cleanup or use a dedicated cleanup process.
# if os.path.exists(TEMP_RECORDINGS_DIR):
#     import shutil
#     shutil.rmtree(TEMP_RECORDINGS_DIR)
#     print(f"Cleaned up {TEMP_RECORDINGS_DIR}")

# Create temp recordings dir on startup
os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True)
