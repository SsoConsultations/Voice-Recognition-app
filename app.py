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
import io # For handling in-memory audio data (BytesIO)
import json # For handling Firebase service account JSON

# Firebase imports
import firebase_admin
from firebase_admin import credentials, storage

# Attempt to import the custom Streamlit audio recorder component
try:
    from st_audiorec import st_audiorec
except ImportError:
    st.error("The `streamlit-audiorec` component is not installed. Please add `streamlit-audiorec` to your requirements.txt and redeploy.")
    st.stop() # Stop the app if this crucial component is missing

# --- Configuration Constants ---
MODEL_FILENAME = 'speaker_recognition_model.pkl'
LABELS_FILENAME = 'id_to_label_map.pkl'
TEMP_RECORDINGS_DIR = "temp_recordings" # For local temporary storage before/after Firebase interaction

# Recording Specific
DEFAULT_NUM_SAMPLES = 5     # Number of audio samples to record for each person
DEFAULT_DURATION = 5.0      # Duration of each recording in seconds
DEFAULT_SAMPLE_RATE = 44100 # Sample rate (samples per second). 44100 Hz is standard CD quality.

# Feature Extraction Specific
N_MFCC = 13 # Number of MFCCs to extract

# --- Firebase Configuration & Initialization ---
# This block handles loading Firebase credentials from Streamlit secrets.
# For local testing, ensure 'firebase_service_account.json' is in your project root.
@st.cache_resource(show_spinner=False) # Cache the Firebase app initialization
def initialize_firebase_app():
    """Initializes the Firebase Admin SDK, trying Streamlit secrets first, then a local file."""
    if not firebase_admin._apps: # Check if Firebase app is already initialized
        try:
            # Access the JSON string from secrets
            firebase_service_account_json_str = st.secrets["firebase"]["service_account_json"]
            firebase_storage_bucket = st.secrets["firebase"]["storage_bucket"]

            # Parse the JSON string into a dictionary
            firebase_config_dict = json.loads(firebase_service_account_json_str)
            
            # Use from_service_account_info to initialize with a dictionary
            cred = credentials.Certificate(firebase_config_dict)
            firebase_admin.initialize_app(cred, {'storageBucket': firebase_storage_bucket})
            st.success("✅ Firebase initialized successfully from secrets.")
            return True
        except (KeyError, json.JSONDecodeError, Exception) as e:
            # Fallback for local development if secrets.toml isn't set up or file is missing
            st.warning(f"Firebase secrets not found or error during initialization: {e}. Attempting to load from local 'firebase_service_account.json'.")
            local_service_account_path = 'firebase_service_account.json'
            local_storage_bucket = 'face-recogniser-app.appspot.com' # REMEMBER TO REPLACE THIS FOR LOCAL TESTING

            if os.path.exists(local_service_account_path):
                try:
                    cred = credentials.Certificate(local_service_account_path)
                    firebase_admin.initialize_app(cred, {'storageBucket': local_storage_bucket})
                    st.success("✅ Firebase initialized successfully from local file.")
                    return True
                except Exception as e_local:
                    st.error(f"❌ Error initializing Firebase from local file: {e_local}. Please ensure your 'firebase_service_account.json' is correct.")
                    return False
            else:
                st.error("❌ Firebase service account file not found locally. Please ensure 'firebase_service_account.json' is in your project root or configure Streamlit secrets.")
                return False
    return True # Already initialized

# Ensure temporary directory exists on startup
os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True)

# Initialize Firebase (this will run once due to @st.cache_resource)
if not initialize_firebase_app():
    st.stop() # Stop the app if Firebase cannot be initialized

# --- Firebase Storage Utility Functions ---

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
        # st.error(f"❌ Error downloading {source_blob_name} from Firebase: {e}") # Suppress for "not found" cases
        return False

def list_files_in_firebase_storage(prefix=""):
    """Lists all blobs in the bucket that start with the given prefix."""
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]

# --- Feature Extraction Function ---

def extract_features(file_path_or_buffer, n_mfcc=N_MFCC):
    """
    Extracts MFCCs from an audio file path or a file-like object (e.g., Streamlit UploadedFile, BytesIO).
    """
    try:
        # librosa.load can directly take a file-like object or a path
        y, sr = librosa.load(file_path_or_buffer, sr=None) # sr=None to preserve original sample rate
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        st.error(f"⚠️ Error encountered while parsing audio for feature extraction: {e}")
        return None
    return mfccs_processed

# --- Data Loading Function (from Firebase) ---

@st.cache_data(show_spinner="Loading audio data from Firebase...")
def load_data_from_firebase(data_prefix="data"):
    """
    Loads all audio files from Firebase Storage, extracts features, and labels them.
    """
    X = [] # Features
    y = [] # Numeric labels
    labels_map = {} # Maps speaker name to a numeric ID
    id_to_label = [] # Maps numeric ID back to speaker name
    
    label_id_counter = 0

    # Get all blobs from the 'data/' prefix
    all_blobs = list_files_in_firebase_storage(prefix=data_prefix + "/")
    
    # Extract unique speaker names from blob paths (e.g., 'data/JohnDoe/sample.wav' -> 'JohnDoe')
    speaker_names = sorted(list(set([blob.split('/')[1] for blob in all_blobs if len(blob.split('/')) > 1 and blob.endswith('.wav')])))

    if not speaker_names:
        st.warning(f"No speaker audio data found in Firebase Storage under '{data_prefix}'.")
        return np.array([]), np.array([]), {}, []

    st.info(f"Processing speakers found in Firebase: {', '.join(speaker_names)}")
    
    total_audio_files = sum(1 for blob in all_blobs if blob.endswith('.wav'))
    if total_audio_files == 0:
        st.warning("No WAV files found in Firebase Storage for processing.")
        return np.array([]), np.array([]), {}, []

    progress_bar = st.progress(0, text="Downloading and processing audio files...")
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
            local_download_path = os.path.join(TEMP_RECORDINGS_DIR, os.path.basename(firebase_audio_path))
            
            if download_audio_from_firebase(firebase_audio_path, local_download_path):
                features = extract_features(local_download_path)
                if features is not None:
                    X.append(features)
                    y.append(current_label_id)
                    speaker_has_audio = True
                os.remove(local_download_path) # Clean up downloaded file immediately
            else:
                st.warning(f"Skipping {firebase_audio_path} due to download error or file not found.")
            
            processed_count += 1
            progress_bar.progress(processed_count / total_audio_files, text=f"Processed {processed_count}/{total_audio_files} files...")

    progress_bar.empty() # Hide progress bar after completion
    return np.array(X), np.array(y), labels_map, id_to_label

# --- Model Training and Saving/Loading Functions ---

@st.cache_resource(show_spinner="Training model...") # Use cache_resource for models as they are objects
def train_and_save_model():
    """
    Loads data (from Firebase), trains a RandomForestClassifier model, and saves it
    to local files, then uploads to Firebase.
    """
    st.subheader("⚙️ Training Model...")
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
            st.warning(f"Speaker '{id_to_label[speaker_id]}' has fewer than 2 samples ({np.sum(y == speaker_id)} found). Each speaker needs at least 2 samples to train. Please add more data for this speaker.")
            return None, None

    st.info(f"Total samples loaded: {len(X)}")
    st.info(f"Speakers found: {labels_map}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    st.info(f"Training samples: {len(X_train)}")
    st.info(f"Testing samples: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    with st.spinner("Model training in progress... This might take a moment."):
        model.fit(X_train, y_train)
    st.success("Model training complete. ✅")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy on test set", value=f"{accuracy * 100:.2f}%")
    st.write("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=id_to_label))

    # Save the trained model and the ID-to-label mapping locally first
    # Then upload to Firebase Storage
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(model, f)
    with open(LABELS_FILENAME, 'wb') as f:
        pickle.dump(id_to_label, f)

    upload_audio_to_firebase(MODEL_FILENAME, MODEL_FILENAME)
    upload_audio_to_firebase(LABELS_FILENAME, LABELS_FILENAME)
    
    # Clean up local model files after upload
    os.remove(MODEL_FILENAME)
    os.remove(LABELS_FILENAME)

    return model, id_to_label

@st.cache_resource(show_spinner="Loading existing model...")
def load_trained_model():
    """
    Loads a pre-trained model and label mapping from Firebase Storage.
    """
    try:
        temp_model_path = os.path.join(TEMP_RECORDINGS_DIR, MODEL_FILENAME)
        temp_labels_path = os.path.join(TEMP_RECORDINGS_DIR, LABELS_FILENAME)
        
        # Try downloading the model and labels from Firebase
        model_downloaded = download_audio_from_firebase(MODEL_FILENAME, temp_model_path)
        labels_downloaded = download_audio_from_firebase(LABELS_FILENAME, temp_labels_path)

        if not model_downloaded or not labels_downloaded:
            st.warning("No existing model or labels found in Firebase Storage. Please add new data to train the model.")
            # Ensure cleanup if only one part downloaded
            if os.path.exists(temp_model_path): os.remove(temp_model_path)
            if os.path.exists(temp_labels_path): os.remove(temp_labels_path)
            return None, None

        with open(temp_model_path, 'rb') as f:
            model = pickle.load(f)
        with open(temp_labels_path, 'rb') as f:
            id_to_label = pickle.load(f)
        st.success("✅ Model and labels loaded successfully from Firebase.")
        
        # Clean up temporary downloaded files
        os.remove(temp_model_path)
        os.remove(temp_labels_path)
        return model, id_to_label
    except Exception as e:
        st.error(f"❌ Error loading model/labels from Firebase: {e}")
        return None, None

# --- Speaker Recognition Functions ---

def recognize_speaker_from_audio_source(model, id_to_label, audio_source_buffer, sample_rate):
    """
    Recognizes a speaker from an audio source (BytesIO buffer) and sample rate.
    This is a unified function for both file uploads and live recordings.
    """
    if model is None or id_to_label is None:
        return "Not Available (Model not loaded)"

    with st.spinner("Extracting features and predicting..."):
        # Reset buffer position to the beginning before passing to librosa
        audio_source_buffer.seek(0) 
        features = extract_features(audio_source_buffer)

    if features is None:
        return "Unknown Speaker (Feature Extraction Failed)"

    features = features.reshape(1, -1) # Reshape for prediction

    prediction_id = model.predict(features)[0]
    predicted_speaker = id_to_label[prediction_id]

    probabilities = model.predict_proba(features)[0]
    confidence = probabilities[prediction_id] * 100

    st.write(f"Predicted Speaker: **{predicted_speaker}** (Confidence: {confidence:.2f}%)")
    return predicted_speaker

# --- Streamlit UI Layout ---

st.set_page_config(page_title="Speaker Recognition", layout="centered", initial_sidebar_state="auto")

# Initialize session state for login
if 'logged_in_as' not in st.session_state:
    st.session_state.logged_in_as = None

# Load model at the start (cached) - this will happen only once unless caches are cleared
trained_model, id_to_label_map = load_trained_model()

# --- Logout Function ---
def logout():
    """Resets the login state and clears relevant session variables."""
    st.session_state.logged_in_as = None
    # Optionally clear relevant session states for recording/recognition if needed
    if 'recorded_samples_count' in st.session_state: del st.session_state.recorded_samples_count
    if 'temp_audio_files' in st.session_state: del st.session_state.temp_audio_files
    if 'current_sample_processed' in st.session_state: del st.session_state.current_sample_processed
    st.rerun()

# Display Logout button if logged in
if st.session_state.logged_in_as:
    st.sidebar.button("Logout", on_click=logout)

# --- Login Page ---
if st.session_state.logged_in_as is None:
    # Custom CSS for centering and button styling
    st.markdown(
        """
        <style>
        .centered-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 80vh; /* Adjust height as needed */
            text-align: center;
        }
        .login-buttons {
            display: flex;
            gap: 20px; /* Space between buttons */
            margin-top: 30px;
        }
        .stButton button {
            background-color: #4CAF50; /* Green */
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            border: none;
            transition-duration: 0.4s;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    
    # --- HERE IS THE CHANGE FOR THE LOGO ---
    st.image("sso_logo.png", width=150) 
    
    st.markdown("## SSO Consultants Face Recogniser") # Title based on image
    st.write("Please choose your login type.")

    col1, col2 = st.columns([1, 1]) # Create two columns for buttons

    with col1:
        if st.button("Login as User", key="login_user"):
            st.session_state.logged_in_as = 'user'
            st.rerun()
    with col2:
        if st.button("Login as Admin", key="login_admin"):
            st.session_state.logged_in_as = 'admin'
            st.rerun()
            
    st.markdown('<p style="margin-top: 50px; font-size: 0.9em; color: grey;">SSO Consultants Face Recognition Tool © 2025 | All Rights Reserved.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- User Section ---
elif st.session_state.logged_in_as == 'user':
    st.title("🗣️ User Portal: Recognize Speaker")
    st.markdown("---")
    
    st.sidebar.header("User Options")
    user_mode = st.sidebar.radio("Choose Recognition Method", ["Recognize Speaker from File", "Recognize Speaker Live"])

    if user_mode == "Recognize Speaker from File":
        st.header("🔍 Recognize Speaker from a File")
        if trained_model is None:
            st.warning("Cannot recognize. Model not trained or loaded. Please ask an Admin to add new data and train the model.")
        else:
            uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

            if uploaded_file is not None:
                st.audio(uploaded_file, format='audio/wav')
                
                audio_buffer = io.BytesIO(uploaded_file.getvalue())
                
                st.write("Analyzing uploaded file...")
                recognized_speaker = recognize_speaker_from_audio_source(trained_model, id_to_label_map, audio_buffer, DEFAULT_SAMPLE_RATE)
                st.success(f"File analysis complete. Predicted Speaker: **{recognized_speaker}**")

    elif user_mode == "Recognize Speaker Live":
        st.header("🎤 Recognize Speaker from Live Microphone Input")

        if trained_model is None:
            st.warning("Cannot recognize. Model not trained or loaded. Please ask an Admin to add new data and train the model.")
        else:
            st.write(f"Click 'Start Recording' and speak for a few seconds to get a live prediction.")
            
            wav_audio_data = st_audiorec()
            
            if wav_audio_data is not None:
                st.audio(wav_audio_data, format='audio/wav')
                
                audio_buffer = io.BytesIO(wav_audio_data)
                
                st.write("Analyzing live recording...")
                recognized_speaker = recognize_speaker_from_audio_source(trained_model, id_to_label_map, audio_buffer, DEFAULT_SAMPLE_RATE)
                st.success(f"Live analysis complete. Predicted Speaker: **{recognized_speaker}**")

# --- Admin Section ---
elif st.session_state.logged_in_as == 'admin':
    st.title("⚙️ Admin Portal: Manage Speaker Data")
    st.markdown("---")

    st.sidebar.header("Admin Options")
    admin_mode = st.sidebar.radio("Choose Admin Action", ["Add New Speaker Data", "Retrain Model (Manual)"])

    if admin_mode == "Add New Speaker Data":
        st.header("➕ Add/Record New Speaker Voice Data")
        st.write("Record multiple voice samples for a person to train the recognition model. Each sample will be uploaded to Firebase Storage.")

        person_name = st.text_input("Enter the name of the person to record:", key="admin_person_name_input").strip()

        if person_name:
            st.info(f"You will record {DEFAULT_NUM_SAMPLES} samples for **{person_name}**, each {DEFAULT_DURATION} seconds long.")
            st.markdown(f"**Instructions:** For each sample, click 'Start Recording', speak for approximately **{DEFAULT_DURATION} seconds**, then **click 'Stop'** to finalize the sample. After processing, click 'Next Sample' to continue.")

            if 'recorded_samples_count' not in st.session_state:
                st.session_state.recorded_samples_count = 0
                st.session_state.temp_audio_files = [] # Store paths of locally saved temp files
                st.session_state.current_sample_processed = False # New state for managing flow

            if st.session_state.recorded_samples_count < DEFAULT_NUM_SAMPLES:
                st.subheader(f"Recording Sample {st.session_state.recorded_samples_count + 1}/{DEFAULT_NUM_SAMPLES}")
                
                if not st.session_state.current_sample_processed:
                    wav_audio_data = st_audiorec()

                    if wav_audio_data is not None:
                        st.audio(wav_audio_data, format='audio/wav')
                        
                        # Process the recorded audio
                        with st.spinner("Processing recorded sample..."):
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            local_filename = os.path.join(TEMP_RECORDINGS_DIR, f"{person_name}_sample_{st.session_state.recorded_samples_count + 1}_{timestamp}.wav")
                            
                            with open(local_filename, "wb") as f:
                                f.write(wav_audio_data)
                            
                            st.session_state.temp_audio_files.append(local_filename)
                            st.session_state.recorded_samples_count += 1
                            st.success(f"Sample {st.session_state.recorded_samples_count} recorded and saved locally.")
                            st.session_state.current_sample_processed = True # Mark as processed
                            st.rerun() # Rerun to show the 'Next Sample' button
                else:
                    if st.button(f"Next Sample ({st.session_state.recorded_samples_count}/{DEFAULT_NUM_SAMPLES} collected)"):
                        st.session_state.current_sample_processed = False # Reset for next recording
                        st.rerun() # Rerun to display the recorder for the next sample
                    else:
                        st.info(f"Sample {st.session_state.recorded_samples_count} collected. Click 'Next Sample' to continue.")

            else: # All samples collected
                st.success(f"All {DEFAULT_NUM_SAMPLES} samples recorded for {person_name}!")
                
                if st.button("Upload Samples and Train Model"):
                    with st.spinner("Uploading samples to Firebase and retraining model..."):
                        uploaded_count = 0
                        for local_file_path in st.session_state.temp_audio_files:
                            firebase_path = f"data/{person_name}/{os.path.basename(local_file_path)}"
                            if upload_audio_to_firebase(local_file_path, firebase_path):
                                uploaded_count += 1
                            os.remove(local_file_path) # Clean up local temp file
                        
                        st.info(f"{uploaded_count} samples uploaded for {person_name}.")
                        
                        # Clear caches to ensure new data is loaded
                        load_data_from_firebase.clear()
                        train_and_save_model.clear()
                        load_trained_model.clear()

                        # Declare global *before* reassignment within this scope
                        global trained_model, id_to_label_map 

                        # Retrain the model with the new data
                        trained_model, id_to_label_map = train_and_save_model()
                        
                        st.session_state.recorded_samples_count = 0 # Reset for next session
                        st.session_state.temp_audio_files = []
                        st.session_state.current_sample_processed = False # Reset for next session
                        st.rerun()
                else:
                    st.info("Click 'Upload Samples and Train Model' to finalize and update the model.")
        else:
            st.info("Please enter a person's name to start recording samples.")
            # Reset session state if name is cleared
            if 'recorded_samples_count' in st.session_state:
                del st.session_state.recorded_samples_count
            if 'temp_audio_files' in st.session_state:
                del st.session_state.temp_audio_files
            if 'current_sample_processed' in st.session_state:
                del st.session_state.current_sample_processed

    elif admin_mode == "Retrain Model (Manual)":
        st.header("🔄 Manually Retrain Model")
        st.write("This option allows you to force a model retraining with all available data in Firebase Storage. This is useful if you manually added files to Firebase or want to ensure the latest data is used.")
        
        if st.button("Retrain Model Now"):
            load_data_from_firebase.clear() # Clear data cache to ensure fresh load
            train_and_save_model.clear() # Clear model cache to force retraining
            load_trained_model.clear() # Clear loaded model cache to pick up new model
            
            # Declare global *before* reassignment within this scope
            global trained_model, id_to_label_map
            
            trained_model, id_to_label_map = train_and_save_model()
            
            if trained_model:
                st.success("Model retraining initiated and completed successfully!")
            else:
                st.error("Model retraining failed. Check logs for details.")
