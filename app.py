import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
import os
import datetime
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Added confusion_matrix
import pickle
import time
import io # For handling in-memory audio data (BytesIO)
import json # For handling Firebase service account JSON
import pandas as pd # NEW: For handling data with st.data_editor
import seaborn as sns # For confusion matrix visualization
import matplotlib.pyplot as plt # For confusion matrix visualization

# Firebase imports
import firebase_admin
from firebase_admin import credentials, storage, firestore # Import firestore

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
METADATA_COLLECTION = 'actors_actresses_metadata' # Firestore collection name for metadata

# Recording Specific
DEFAULT_NUM_SAMPLES = 5     # Number of audio samples to record for each person (increased to 5)
DEFAULT_DURATION = 5.0      # Duration of each recording in seconds (increased to 5.0)
DEFAULT_SAMPLE_RATE = 44100 # Sample rate (samples per second). 44100 Hz is standard CD quality.

# Feature Extraction Specific
N_MFCC = 13 # Number of MFCCs to extract
# Consider adding more features for better accuracy:
# N_CHROMA = 12
# N_MEL = 128
# N_TONNETZ = 6

# --- Firebase Configuration & Initialization ---
# This block handles loading Firebase credentials from Streamlit secrets.
# For local testing, ensure 'firebase_service_account.json' is in your project root.
@st.cache_resource(show_spinner=False) # Cache the Firebase app initialization
def initialize_firebase_app():
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
            return True
        except (KeyError, json.JSONDecodeError, Exception) as e:
            # Fallback for local development if secrets.toml isn't set up or file is missing
            st.warning(f"Firebase secrets not found or error during initialization: {e}. Attempting to load from local 'firebase_service_account.json'.")
            local_service_account_path = 'firebase_service_account.json'
            local_storage_bucket = 'face-recogniser-app.appspot.com' # REMEMBER TO REPLACE THIS FOR LOCAL TESTING
            # !! IMPORTANT: Replace 'face-recogniser-app.appspot.com' with your actual Firebase Storage bucket for local testing !!

            if os.path.exists(local_service_account_path):
                try:
                    cred = credentials.Certificate(local_service_account_path)
                    firebase_admin.initialize_app(cred, {'storageBucket': local_storage_bucket})
                    st.success("✅ Firebase initialized successfully from local file.")
                    return True
                except Exception as e_local:
                    st.error(f"❌ Error initializing Firebase from local file: {e_local}. Please ensure your 'firebase_service_account.json' is correct and readable.")
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

# Get Firestore client
@st.cache_resource(show_spinner=False)
def get_firestore_client():
    return firestore.client()

db = get_firestore_client()

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

# --- Firestore Metadata Functions ---
def save_actor_metadata(actor_name, age, height, total_films, hit_films, industry): # Added industry
    """Saves/updates an actor's metadata in Firestore."""
    try:
        doc_ref = db.collection(METADATA_COLLECTION).document(actor_name)
        doc_ref.set({
            'age': age,
            'height': height,
            'total_films': total_films,
            'hit_films': hit_films,
            'industry': industry, # Added industry field
            'last_updated': firestore.SERVER_TIMESTAMP
        }, merge=True) # merge=True allows updating specific fields without overwriting the whole document
        st.success(f"Metadata for {actor_name} saved/updated in Firestore.")
        return True
    except Exception as e:
        st.error(f"❌ Error saving metadata for {actor_name} to Firestore: {e}")
        return False

@st.cache_data(ttl=3600) # Cache metadata for an hour to reduce reads
def get_actor_metadata(actor_name):
    """Retrieves an actor's metadata from Firestore."""
    try:
        doc_ref = db.collection(METADATA_COLLECTION).document(actor_name)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error retrieving metadata for {actor_name} from Firestore: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner="Loading all actor/actress metadata...") # NEW FUNCTION
def get_all_actor_metadata():
    """Retrieves all actor/actress metadata from Firestore."""
    try:
        docs = db.collection(METADATA_COLLECTION).stream()
        all_metadata = []
        for doc in docs:
            data = doc.to_dict()
            data['Name'] = doc.id # The document ID is the actor's name
            all_metadata.append(data)
        return pd.DataFrame(all_metadata)
    except Exception as e:
        st.error(f"❌ Error retrieving all metadata from Firestore: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Feature Extraction Function ---

def extract_features(file_path_or_buffer, n_mfcc=N_MFCC):
    """
    Extracts MFCCs (and potentially other features) from an audio file path or a file-like object.
    Robust to different sample rates by resampling internally if necessary.
    """
    try:
        # librosa.load can directly take a file-like object or a path
        # It automatically resamples to `sr` (default 22050 Hz) if not None.
        # Explicitly setting sr=DEFAULT_SAMPLE_RATE ensures consistency if source audio varies.
        y, sr = librosa.load(file_path_or_buffer, sr=DEFAULT_SAMPLE_RATE)

        # Ensure audio is not silent or too short
        if len(y) < sr * 0.1: # At least 0.1 seconds of audio
            st.warning("Audio too short for meaningful feature extraction.")
            return None
        if np.max(np.abs(y)) < 0.01: # Check for near-silent audio
            st.warning("Audio too quiet for meaningful feature extraction. Please speak louder.")
            return None

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0) # Mean across time for a single feature vector

        # --- Potential Accuracy Improvement: Add more features ---
        # You can concatenate different types of features for a richer representation.
        # This often improves model performance.
        # For example:
        # chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
        # chroma_processed = np.mean(chroma.T, axis=0)
        # mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
        # mel_processed = np.mean(librosa.power_to_db(mel, ref=np.max).T, axis=0)
        # tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr, chroma=chroma)
        # tonnetz_processed = np.mean(tonnetz.T, axis=0)

        # features_combined = np.hstack([mfccs_processed, chroma_processed, mel_processed, tonnetz_processed])
        # return features_combined
        # For now, sticking to MFCCs to keep it simple, but this is a key area for improvement.

    except Exception as e:
        st.error(f"⚠️ Error encountered while parsing audio for feature extraction: {e}")
        st.exception(e) # Display full traceback for debugging
        return None
    return mfccs_processed

# --- Data Loading Function (from Firebase) ---

@st.cache_data(show_spinner="Loading audio data from Firebase...")
def load_data_from_firebase(data_prefix="data"):
    """
    Loads all audio files from Firebase Storage, extracts features, and labels them.
    Includes robust error handling for individual file processing.
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
        st.warning(f"No speaker audio data found in Firebase Storage under '{data_prefix}'. Please add data in the Admin Panel.")
        return np.array([]), np.array([]), {}, []
        st.info(f"Processing speakers found in Firebase: {', '.join(speaker_names)}")

    total_audio_files = sum(1 for blob in all_blobs if blob.endswith('.wav'))
    if total_audio_files == 0:
        st.warning("No WAV files found in Firebase Storage for processing.")
        return np.array([]), np.array([]), {}, []

    progress_bar = st.progress(0, text="Downloading and processing audio files...")
    processed_count = 0
    successful_features_count = 0

    for speaker_name in speaker_names:
        speaker_prefix = f"{data_prefix}/{speaker_name}/"

        if speaker_name not in labels_map:
            labels_map[speaker_name] = label_id_counter
            id_to_label.append(speaker_name)
            label_id_counter += 1

        current_label_id = labels_map[speaker_name]

        speaker_audio_blobs = [b for b in all_blobs if b.startswith(speaker_prefix) and b.endswith('.wav')]

        for firebase_audio_path in speaker_audio_blobs:
            local_download_path = os.path.join(TEMP_RECORDINGS_DIR, os.path.basename(firebase_audio_path))

            download_success = False
            try:
                if download_audio_from_firebase(firebase_audio_path, local_download_path):
                    download_success = True
            except Exception as e:
                st.error(f"Error downloading {firebase_audio_path}: {e}")

            if download_success:
                features = extract_features(local_download_path)
                if features is not None:
                    X.append(features)
                    y.append(current_label_id)
                    successful_features_count += 1
                else:
                    st.warning(f"Failed to extract features from {firebase_audio_path}. Skipping.")
                os.remove(local_download_path) # Clean up downloaded file immediately
            else:
                st.warning(f"Skipping {firebase_audio_path} due to download error or file not found in Firebase.")

            processed_count += 1
            progress_bar.progress(processed_count / total_audio_files, text=f"Processed {processed_count}/{total_audio_files} files...")

    progress_bar.empty() # Hide progress bar after completion

    if successful_features_count == 0:
        st.error("No features could be extracted from any audio files. Please check audio quality and file formats.")
        return np.array([]), np.array([]), {}, []

    return np.array(X), np.array(y), labels_map, id_to_label

# --- Model Training and Saving/Loading Functions ---

@st.cache_resource(show_spinner="Training model...") # Use cache_resource for models as they are objects
def train_and_save_model():
    """
    Loads data (from Firebase), trains a RandomForestClassifier model, and saves it
    to local files, then uploads to Firebase. Includes model evaluation and visualization.
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
    # Stratified split requires at least 2 samples per class in both train and test sets for non-zero test_size.
    # For a 0.2 test_size, it implicitly means you need at least 1 / 0.2 = 5 samples per class.
    # To be safe and avoid errors, we'll check for 2, and warn if less than 5.
    insufficient_samples_speakers = []
    for speaker_id in range(unique_speakers):
        count = np.sum(y == speaker_id)
        if count < 2: # Absolute minimum for split
            st.error(f"Speaker '{id_to_label[speaker_id]}' has only {count} sample(s). Cannot train model. Each speaker needs at least 2 samples.")
            return None, None
        elif count < 5: # Recommended minimum for a reasonable stratified split
            insufficient_samples_speakers.append(f"'{id_to_label[speaker_id]}' ({count} samples)")

    if insufficient_samples_speakers:
        st.warning(f"Some speakers have few samples: {', '.join(insufficient_samples_speakers)}. This might affect model accuracy. Consider adding more samples for them.")


    st.info(f"Total samples loaded: {len(X)}")
    st.info(f"Speakers found ({unique_speakers}): {labels_map}")

    # Split data into training and testing sets
    # Ensure stratify is used to maintain class distribution in splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.info(f"Training samples: {len(X_train)}")
    st.info(f"Testing samples: {len(X_test)}")

    # Model parameters can be tuned for better accuracy.
    # RandomForestClassifier is a good baseline. Consider GridSearchCV for hyperparameter tuning.
    model = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42, class_weight='balanced')
    # Increased n_estimators, added max_depth, min_samples_leaf, and class_weight='balanced' for potentially better performance on imbalanced datasets

    with st.spinner("Model training in progress... This might take a moment."):
        model.fit(X_train, y_train)
    st.success("Model training complete. ✅")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy on test set", value=f"{accuracy * 100:.2f}%")

    st.write("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=id_to_label))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=id_to_label, yticklabels=id_to_label, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix for Speaker Recognition')
    st.pyplot(fig) # Display the plot in Streamlit

    # Save the trained model and the ID-to-label mapping locally first
    # Then upload to Firebase Storage
    try:
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(model, f)
        with open(LABELS_FILENAME, 'wb') as f:
            pickle.dump(id_to_label, f)

        upload_audio_to_firebase(MODEL_FILENAME, MODEL_FILENAME)
        upload_audio_to_firebase(LABELS_FILENAME, LABELS_FILENAME)

        # Clean up local model files after upload
        os.remove(MODEL_FILENAME)
        os.remove(LABELS_FILENAME)
        st.success("Trained model and labels saved to Firebase Storage.")
    except Exception as e:
        st.error(f"❌ Error saving/uploading model or labels: {e}")
        st.exception(e) # Show traceback

    return model, id_to_label

@st.cache_resource(show_spinner="Loading existing model...")
def load_trained_model():
    """
    Loads a pre-trained model and label mapping from Firebase Storage.
    """
    temp_model_path = os.path.join(TEMP_RECORDINGS_DIR, MODEL_FILENAME)
    temp_labels_path = os.path.join(TEMP_RECORDINGS_DIR, LABELS_FILENAME)
    model = None
    id_to_label = None

    try:
        # Try downloading the model and labels from Firebase
        model_downloaded = download_audio_from_firebase(MODEL_FILENAME, temp_model_path)
        labels_downloaded = download_audio_from_firebase(LABELS_FILENAME, temp_labels_path)

        if not model_downloaded or not labels_downloaded:
            st.warning("No existing model or labels found in Firebase Storage. Please add new data in the Admin Panel to train the model.")
            return None, None

        with open(temp_model_path, 'rb') as f:
            model = pickle.load(f)
        with open(temp_labels_path, 'rb') as f:
            id_to_label = pickle.load(f)

        return model, id_to_label
    except FileNotFoundError:
        st.warning(f"Model or label file not found locally after download attempt.")
        return None, None
    except pickle.UnpicklingError as e:
        st.error(f"❌ Error unpickling model or labels (corrupted file?): {e}")
        st.warning("Consider retraining the model to generate new files.")
        return None, None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading model/labels: {e}")
        st.exception(e) # Show traceback for other errors
        return None, None
    finally:
        # Ensure cleanup regardless of success or failure
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        if os.path.exists(temp_labels_path):
            os.remove(temp_labels_path)

# --- Speaker Recognition Functions ---

def recognize_speaker_from_audio_source(model, id_to_label, audio_source_buffer, sample_rate):
    """
    Recognizes a speaker from an audio source (BytesIO buffer) and sample rate.
    This is a unified function for both file uploads and live recordings.
    """
    if model is None or id_to_label is None:
        st.error("Model or label mapping not loaded. Cannot perform recognition.")
        return "Model Not Loaded", None

    with st.spinner("Extracting features and predicting..."):
        features = extract_features(audio_source_buffer)

    if features is None:
        st.error("Feature extraction failed. Cannot recognize speaker.")
        return "Unknown Speaker (Feature Extraction Failed)", None

    # Check if features dimension matches model's expected input
    if features.shape[0] != model.n_features_in_:
        st.error(f"Feature dimension mismatch. Expected {model.n_features_in_}, got {features.shape[0]}. "
                 "This can happen if the model was trained with a different number of features (e.g., N_MFCC changed). "
                 "Please retrain the model in the Admin Panel.")
        return "Feature Mismatch Error", None

    features = features.reshape(1, -1) # Reshape for prediction

    try:
        prediction_id = model.predict(features)[0]
        predicted_speaker = id_to_label[prediction_id]

        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction_id] * 100

        st.write(f"Predicted Speaker: **{predicted_speaker}**")

        # Fetch and display metadata
        metadata = get_actor_metadata(predicted_speaker)
        if metadata:
            st.subheader(f"Details for {predicted_speaker}:")
            st.write(f"**Age:** {metadata.get('age', 'N/A')} years")
            st.write(f"**Height:** {metadata.get('height', 'N/A')}")
            st.write(f"**Total Films:** {metadata.get('total_films', 'N/A')}")
            st.write(f"**Hit Films:** {metadata.get('hit_films', 'N/A')}")
            st.write(f"**Industry:** {metadata.get('industry', 'N/A')}") # Display industry
        else:
            st.info(f"No additional metadata found for {predicted_speaker}. Admin can add this in the 'View/Update Actor Details' panel.")

        return predicted_speaker, metadata # Return both for potential future use
    except Exception as e:
        st.error(f"❌ Error during speaker prediction: {e}")
        st.exception(e) # Show traceback
        return "Prediction Error", None

# --- Streamlit UI Layout ---

st.set_page_config(page_title="Speaker Recognition", layout="centered", initial_sidebar_state="auto")

# --- TOP HEADER WITH LOGO AND CUSTOM TITLE ---
col1, col2 = st.columns([0.1, 0.9]) # Adjust column width for logo and text
with col1:
    st.image("sso_logo.png", width=70) # Adjust width as needed
with col2:
    st.markdown("<h1 style='display: inline-block; vertical-align: middle;'>SSO Consultants Voice Recogniser</h1>", unsafe_allow_html=True)

st.markdown("---") # Separator line


# Initialize session state for login status and role
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None # 'admin' or 'user'

# --- Login Functions ---
def admin_login(username, password):
    # Fetch credentials from Streamlit secrets
    try:
        ADMIN_USERNAME = st.secrets["login"]["admin_username"]
        ADMIN_PASSWORD = st.secrets["login"]["admin_password"]
    except KeyError:
        st.error("Admin login credentials not found in Streamlit secrets.toml. Please configure them.")
        return False

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        st.session_state.logged_in = True
        st.session_state.user_role = 'admin'
        st.success("Admin login successful!")
        return True
    else:
        st.error("Invalid admin credentials.")
        return False

def user_login(username, password):
    # Fetch credentials from Streamlit secrets
    try:
        USER1_USERNAME = st.secrets["login"]["user1_username"]
        USER1_PASSWORD = st.secrets["login"]["user1_password"]
        USER2_USERNAME = st.secrets["login"]["user2_username"]
        USER2_PASSWORD = st.secrets["login"]["user2_password"]
    except KeyError:
        st.error("User login credentials not found in Streamlit secrets.toml. Please configure them.")
        return False

    if username == USER_USERNAME and password == USER_PASSWORD:
        st.session_state.logged_in = True
        st.session_state.user_role = 'user'
        st.success("User login successful!")
        return True
    else:
        st.error("Invalid user credentials.")
        return False

def logout():
    st.session_state.logged_in = False
    st.session_state.user_role = None
    # Clear caches that might hold sensitive data or models
    load_trained_model.clear()
    load_data_from_firebase.clear()
    train_and_save_model.clear()
    get_actor_metadata.clear() # Clear metadata cache too!
    get_all_actor_metadata.clear() # NEW: Clear all metadata cache on logout

# --- Home Page / Login Screen ---
if not st.session_state.logged_in:
    st.subheader("Welcome! Please Login to Continue.")

    login_type = st.radio("Select Login Type:", ("User Login", "Admin Login"), key="login_type_radio")

    with st.form("login_form"):
        username = st.text_input("Username", key="login_username_input")
        password = st.text_input("Password", type="password", key="login_password_input")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if login_type == "Admin Login":
                if admin_login(username, password):
                    st.rerun()
            elif login_type == "User Login":
                if user_login(username, password):
                    st.rerun()

else:
    # --- Sidebar content ---
    # Add your logo to the sidebar here
    st.sidebar.image("sso_logo.png", width=150) # Adjust width as needed for sidebar logo
    st.sidebar.markdown("---") # Add a separator below the logo
    st.sidebar.write(f"Logged in as: **{st.session_state.user_role.capitalize()}**")


    if st.session_state.user_role == 'admin':
        app_mode = st.sidebar.radio("Go to", ["Admin Panel: Add Speaker Data", "Admin Panel: Retrain Model", "Admin Panel: View/Update Actor Details"], key="admin_app_mode") # MODIFIED: Added new option
    elif st.session_state.user_role == 'user':
        app_mode = st.sidebar.radio("Go to", ["User Panel: Recognize Speaker from File", "User Panel: Recognize Speaker Live"], key="user_app_mode")

    st.sidebar.button("Logout", on_click=logout, key="sidebar_logout_btn")

    # Load model at the start (cached) after login
    # This will attempt to load the model and labels once per session state change if needed.
    trained_model, id_to_label_map = load_trained_model()


    # --- Admin Panel ---
    if st.session_state.user_role == 'admin':
        if app_mode == "Admin Panel: Add Speaker Data":
            st.header("Add/Record New Actor/Actress Voice Data & Metadata")

            person_name = st.text_input("Enter the name of the Actor/Actress and click ENTER to record samples:", key="admin_person_name_input").strip()

            # New input fields for metadata
            st.subheader("Actor/Actress Details")
            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                age = st.number_input("Age (Years):", min_value=1, max_value=120, value=30, key="actor_age_input")
                total_films = st.number_input("Total Films:", min_value=0, value=10, key="actor_total_films_input")
            with col_meta2:
                height = st.text_input("Height (e.g., 5'8\" or 175cm):", value="N/A", key="actor_height_input")
                hit_films = st.number_input("Hit Films:", min_value=0, value=2, key="actor_hit_films_input")

            industry = st.text_input("Industry (e.g., Bollywood, Hollywood, Tollywood):", value="N/A", key="actor_industry_input") # Added industry input

            if person_name:
                st.markdown(f"**Instructions:** For each sample, click 'Start Recording', speak clearly for approximately **{DEFAULT_DURATION} seconds**, then **click 'Stop'** to finalize the sample. After processing, click 'Next Sample' to continue.")

                if 'admin_recorded_samples_count' not in st.session_state:
                    st.session_state.admin_recorded_samples_count = 0
                    st.session_state.admin_temp_audio_files = [] # Store paths of locally saved temp files
                    st.session_state.admin_current_sample_processed = False # New state for managing flow

                if st.session_state.admin_recorded_samples_count < DEFAULT_NUM_SAMPLES:
                    st.subheader(f"Recording Sample {st.session_state.admin_recorded_samples_count + 1}/{DEFAULT_NUM_SAMPLES}")

                    # Only show the recorder if the current sample hasn't been processed yet
                    if not st.session_state.admin_current_sample_processed:
                        # Streamlit audiorec provides the raw WAV bytes
                        wav_audio_data = st_audiorec()

                        if wav_audio_data is not None:
                            st.audio(wav_audio_data, format='audio/wav')

                            # Process the recorded audio
                            with st.spinner("Processing recorded sample..."):
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                # Sanitize person_name for filename (replace spaces with underscores, remove special chars)
                                local_filename = os.path.join(TEMP_RECORDINGS_DIR, f"{person_name}_sample_{st.session_state.admin_recorded_samples_count + 1}_{timestamp}.wav")

                                try:
                                    with open(local_filename, "wb") as f:
                                        f.write(wav_audio_data)
                                    st.session_state.admin_temp_audio_files.append(local_filename)
                                    st.session_state.admin_recorded_samples_count += 1
                                    st.success(f"Sample {st.session_state.admin_recorded_samples_count} recorded and saved locally.")
                                    st.session_state.admin_current_sample_processed = True # Mark as processed
                                    st.rerun() # Rerun to show the 'Next Sample' button
                                except Exception as e:
                                    st.error(f"Error saving recorded audio locally: {e}")
                                    st.exception(e)
                    else:
                        # If sample processed, show "Next Sample" button
                        if st.button(f"Next Sample ({st.session_state.admin_recorded_samples_count}/{DEFAULT_NUM_SAMPLES} collected)", key="admin_next_sample_btn"):
                            st.session_state.admin_current_sample_processed = False # Reset for next recording
                            st.rerun() # Rerun to display the recorder for the next sample
                        else:
                            st.info(f"Sample {st.session_state.admin_recorded_samples_count} collected. Click 'Next Sample' to continue.")

                else: # All samples collected
                    st.success(f"All {DEFAULT_NUM_SAMPLES} samples recorded for {person_name}!")

                    if st.button("Upload Samples & Save Metadata and Train Model", key="admin_upload_train_btn"):
                        with st.spinner("Uploading samples to Firebase, saving metadata, and retraining model..."):
                            uploaded_audio_count = 0
                            for local_file_path in st.session_state.admin_temp_audio_files:
                                # Ensure consistent naming with metadata by using the sanitized name
                                firebase_path = f"data/{person_name}/{os.path.basename(local_file_path)}"
                                if upload_audio_to_firebase(local_file_path, firebase_path):
                                    uploaded_audio_count += 1
                                try:
                                    os.remove(local_file_path) # Clean up local temp file
                                except OSError as e:
                                    st.warning(f"Could not remove local file {local_file_path}: {e}")

                            st.info(f"{uploaded_audio_count} audio samples uploaded for {person_name}.")

                            # Save metadata to Firestore (Updated call with 'industry')
                            if save_actor_metadata(person_name, age, height, total_films, hit_films, industry):
                                st.info(f"Metadata for {person_name} successfully saved.")
                            else:
                                st.error(f"Failed to save metadata for {person_name}.")


                            # Clear caches to ensure new data is loaded for model training/loading
                            load_data_from_firebase.clear()
                            train_and_save_model.clear()
                            load_trained_model.clear()
                            get_actor_metadata.clear() # Clear metadata cache to ensure fresh data for users
                            get_all_actor_metadata.clear() # NEW: Clear all metadata cache

                            # Retrain the model with the new data
                            trained_model, id_to_label_map = train_and_save_model()
                            st.session_state.admin_recorded_samples_count = 0 # Reset for next session
                            st.session_state.admin_temp_audio_files = []
                            st.session_state.admin_current_sample_processed = False # Reset for next session
                            st.success("Data uploaded, metadata saved, and model retraining completed. You can now recognize this speaker!")
                            st.rerun()
                    else:
                        st.info("Click 'Upload Samples & Save Metadata and Train Model' to finalize and update the model and metadata. (Don't navigate away until done!)")
            else:
                # Reset session state if name is cleared
                if 'admin_recorded_samples_count' in st.session_state:
                    del st.session_state.admin_recorded_samples_count
                if 'admin_temp_audio_files' in st.session_state:
                    del st.session_state.admin_temp_audio_files
                if 'admin_current_sample_processed' in st.session_state:
                    del st.session_state.admin_current_sample_processed

        elif app_mode == "Admin Panel: Retrain Model":
            st.header("Retrain Speaker Recognition Model")
            st.write("This will retrain the model using all available audio data in Firebase Storage. This is useful if you've manually added/removed data or want to refresh the model.")
            
            if st.button("Trigger Model Retraining", key="trigger_retrain_btn"):
                # Clear all relevant caches before retraining
                load_data_from_firebase.clear()
                train_and_save_model.clear()
                load_trained_model.clear()
                # No need to clear get_actor_metadata here, as retraining doesn't change metadata
                get_all_actor_metadata.clear() # Clear all metadata cache to ensure consistency

                trained_model, id_to_label_map = train_and_save_model()
                if trained_model:
                    st.success("Model retraining initiated and completed successfully! The updated model is now active.")
                else:
                    st.error("Model retraining failed. Check previous messages for details or ensure enough diverse audio data exists.")
                st.rerun()

        elif app_mode == "Admin Panel: View/Update Actor Details": # NEW ADMIN PANEL SECTION
            st.header("View and Update Actor/Actress Details")
            st.write("Edit actor/actress metadata directly in the table below. Changes will be saved to Firestore.")

            # Load all existing metadata
            current_df = get_all_actor_metadata()

            if not current_df.empty:
                # Reorder columns for better display, put 'Name' first
                # Ensure all expected columns are present, add if missing (e.g., if old data doesn't have 'industry')
                display_cols_order = ['Name', 'age', 'height', 'total_films', 'hit_films', 'industry', 'last_updated']
                for col in display_cols_order:
                    if col not in current_df.columns:
                        current_df[col] = None # Add missing columns as None/NaN

                # Ensure 'last_updated' is datetime type for proper formatting
                if 'last_updated' in current_df.columns:
                    current_df['last_updated'] = pd.to_datetime(current_df['last_updated'], unit='s', errors='coerce') # Convert Unix timestamp to datetime

                current_df = current_df.reindex(columns=display_cols_order)

                st.subheader("Current Actor/Actress Data")
                # Make 'Name' and 'last_updated' columns non-editable
                column_config = {
                    "Name": st.column_config.TextColumn(
                        "Actor/Actress Name",
                        disabled=True, # Make name not directly editable as it's the document ID
                    ),
                    "last_updated": st.column_config.DatetimeColumn(
                        "Last Updated",
                        format="YYYY-MM-DD HH:mm:ss",
                        disabled=True,
                    ),
                    "age": st.column_config.NumberColumn(
                        "Age (Years)",
                        min_value=1, max_value=120, step=1,
                        # Default to None for empty values
                        default=None
                    ),
                    "total_films": st.column_config.NumberColumn(
                        "Total Films",
                        min_value=0, step=1,
                        default=None
                    ),
                    "hit_films": st.column_config.NumberColumn(
                        "Hit Films",
                        min_value=0, step=1,
                        default=None
                    ),
                    "height": st.column_config.TextColumn(
                        "Height (e.g., 5'8\" or 175cm)"
                    ),
                    "industry": st.column_config.TextColumn(
                        "Industry (e.g., Bollywood)"
                    )
                }

                edited_df = st.data_editor(
                    current_df,
                    key="actor_details_editor",
                    num_rows="fixed", # Prevent adding/deleting rows from here directly
                    use_container_width=True,
                    column_config=column_config
                )

                if st.button("Save Changes to Firestore", key="save_actor_details_btn"):
                    # Check for changes and update Firestore
                    has_changes = False
                    for index, row in edited_df.iterrows():
                        original_row = current_df.loc[index]
                        actor_name = row['Name']

                        # Compare edited row with original row
                        changed_fields = {}
                        for col in original_row.index:
                            # Skip 'last_updated' as it's updated by Firestore server timestamp
                            if col == 'last_updated' or col == 'Name': # Skip Name as well, it's the ID
                                continue

                            # Compare values, handling potential type differences or NaN/None
                            original_value = original_row[col]
                            edited_value = row[col]

                            # Convert numpy types to native Python types for comparison/Firestore
                            # This helps with st.data_editor which can return numpy types
                            if isinstance(original_value, (np.integer, int)): # Include native int
                                original_value = int(original_value)
                            if isinstance(edited_value, (np.integer, int)):
                                edited_value = int(edited_value)
                            if isinstance(original_value, (np.floating, float)): # Include native float
                                original_value = float(original_value)
                            if isinstance(edited_value, (np.floating, float)):
                                edited_value = float(edited_value)

                            # Handle potential NaNs from number inputs if they were cleared
                            if pd.isna(edited_value):
                                edited_value = None
                            if pd.isna(original_value):
                                original_value = None

                            # Robust comparison for equality, especially for None vs "" or None vs 0
                            if original_value != edited_value:
                                # Special handling for empty strings from text inputs
                                if col in ['height', 'industry'] and edited_value == "":
                                    edited_value = None # Store empty strings as None in Firestore
                                changed_fields[col] = edited_value

                        if changed_fields:
                            st.info(f"Detected changes for **{actor_name}**: {changed_fields}")
                            # Update the document in Firestore
                            try:
                                doc_ref = db.collection(METADATA_COLLECTION).document(actor_name)
                                doc_ref.set(changed_fields, merge=True) # Use merge=True
                                st.success(f"Updated {actor_name}'s details in Firestore.")
                                has_changes = True
                            except Exception as e:
                                st.error(f"❌ Error updating {actor_name} in Firestore: {e}")

                    if has_changes:
                        st.success("All detected changes saved successfully!")
                        # Clear cache to force reload of updated metadata
                        get_all_actor_metadata.clear()
                        get_actor_metadata.clear() # Also clear single actor cache
                        st.rerun() # Rerun to display updated table
                    else:
                        st.info("No changes detected to save.")
            else:
                st.info("No actor/actress metadata found in Firestore to display.")
                st.warning("Please add new actor/actress data first using the 'Add Speaker Data' panel.")

    # --- User Panel ---
    elif st.session_state.user_role == 'user':

        if app_mode == "User Panel: Recognize Speaker from File":
            st.header("Recognize Actor/Actress from a File")

            if trained_model is None:
                st.warning("Cannot recognize. Model not trained or loaded. Please inform the admin to train one.")
            else:
                uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

                if uploaded_file is not None:
                    st.audio(uploaded_file, format='audio/wav')

                    audio_buffer = io.BytesIO(uploaded_file.getvalue())

                    st.write("Analyzing uploaded file...")
                    recognized_speaker, metadata = recognize_speaker_from_audio_source(trained_model, id_to_label_map, audio_buffer, DEFAULT_SAMPLE_RATE)
                    # The display of metadata is now handled directly within recognize_speaker_from_audio_source

        elif app_mode == "User Panel: Recognize Speaker Live":
            st.header("Recognize Actor/Actress Live")
            st.write(f"**Instructions:** To recognise a voice, click 'Start Recording', speak clearly , then **click 'Stop'** to finalize the recording.")


            if trained_model is None:
                st.warning("Cannot recognize. Model not trained or loaded. Please inform the admin to train one.")
            else:

                live_audio_data = st_audiorec()

                if live_audio_data is not None:
                    st.audio(live_audio_data, format='audio/wav')
                    audio_buffer = io.BytesIO(live_audio_data) # st_audiorec returns bytes directly

                    st.write("Analyzing live recording...")
                    recognized_speaker, metadata = recognize_speaker_from_audio_source(trained_model, id_to_label_map, audio_buffer, DEFAULT_SAMPLE_RATE)
                    # The display of metadata is now handled directly within recognize_speaker_from_audio_source
