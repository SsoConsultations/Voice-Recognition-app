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
import tempfile # For robust temporary file handling

# --- Firebase Integration ---
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase Admin SDK
firebase_enabled = False
try:
    cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY)
    firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_STORAGE_BUCKET})
    print("✅ Firebase initialized successfully.")
    firebase_enabled = True
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    print("Ensure 'FIREBASE_SERVICE_ACCOUNT_KEY' and 'FIREBASE_STORAGE_BUCKET' are correct.")
    print("Proceeding without Firebase Cloud Storage integration for recording/training.")


# --- Configuration Constants ---
# General
# DATA_BASE_DIR is no longer used for permanent local storage, but TEMP_RECORDINGS_DIR is for transient files
TEMP_RECORDINGS_DIR = "temp_recordings" # For temporary local files during record/download
MODEL_FILENAME = 'speaker_recognition_model.pkl'
LABELS_FILENAME = 'id_to_label_map.pkl'

# Recording Specific
DEFAULT_NUM_SAMPLES = 5     # Number of audio samples to record for each person
DEFAULT_DURATION = 4.0      # Duration of each recording in seconds (can be float)
DEFAULT_SAMPLE_RATE = 44100 # Sample rate (samples per second). 44100 Hz is standard CD quality.

# Feature Extraction Specific
N_MFCC = 13 # Number of MFCCs to extract

# --- Utility Functions ---

def upload_to_firebase_storage(local_file_path, destination_blob_name):
    """Uploads a file to Firebase Cloud Storage."""
    if not firebase_enabled:
        print("Firebase is not enabled. Skipping upload.")
        return False
    try:
        bucket = storage.bucket()
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        print(f"⬆️ Uploaded {local_file_path} to gs://{FIREBASE_STORAGE_BUCKET}/{destination_blob_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload {local_file_path} to Firebase Storage: {e}")
        return False

def record_voice(temp_output_dir, person_name, num_samples, duration, fs):
    """
    Records multiple voice samples for a given person, uploads to Firebase, and deletes local copy.
    """
    os.makedirs(temp_output_dir, exist_ok=True) # Ensure temp directory exists
    
    print(f"\n--- Recording for {person_name} ({num_samples} samples, {duration} seconds each) ---")
    print("Please speak clearly for each sample. Ensure you have a quiet environment.")

    for i in range(1, num_samples + 1):
        print(f"\nPress Enter to start recording sample {i}/{num_samples} for {person_name}...")
        input()

        print(f"Recording sample {i}... SPEAK NOW!")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
        sd.wait()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        local_filename = f"{person_name.lower().replace(' ', '_')}_sample_{i}_{timestamp}.wav"
        local_file_path = os.path.join(temp_output_dir, local_filename) # Save temporarily

        wav.write(local_file_path, fs, recording)
        print(f"Sample {i} saved temporarily to {local_file_path}")

        # Upload to Firebase Storage
        storage_path = f"{FIREBASE_VOICES_PREFIX}{person_name.lower().replace(' ', '_')}/{local_filename}"
        if upload_to_firebase_storage(local_file_path, storage_path):
            os.remove(local_file_path) # Delete local copy after successful upload
            print(f"Local temporary file {local_file_path} deleted.")
        else:
            print(f"Failed to upload to Firebase. Local file {local_file_path} retained.")


def extract_features(file_path, n_mfcc=N_MFCC):
    """
    Extracts MFCCs from an audio file.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"⚠️ Error encountered while parsing file: {file_path} - {e}")
        return None
    return mfccs_processed

def load_data_from_firebase_storage(firebase_voices_prefix, temp_local_dir):
    """
    Loads all audio files from Firebase Storage, extracts features, and labels them.
    Downloads files temporarily and deletes them after processing.
    """
    X = [] # Features
    y = [] # Numeric labels
    labels_map = {} # Maps speaker name to a numeric ID
    id_to_label = [] # Maps numeric ID back to speaker name
    
    label_id_counter = 0

    if not firebase_enabled:
        print("❌ Firebase is not enabled. Cannot load data from cloud for training.")
        return np.array([]), np.array([]), {}, []

    os.makedirs(temp_local_dir, exist_ok=True) # Ensure temp directory exists for downloads

    bucket = storage.bucket()
    print(f"\n--- Loading Data from Firebase Storage (gs://{FIREBASE_STORAGE_BUCKET}/{firebase_voices_prefix}) ---")
    blobs = bucket.list_blobs(prefix=firebase_voices_prefix)

    # Group blobs by speaker for organized processing
    speaker_files = {} # { "speaker_slug": [blob_name_1, blob_name_2, ...]}
    for blob in blobs:
        if blob.name.endswith('.wav'):
            # Example blob.name: voices/alice_smith/alice_smith_sample_1_timestamp.wav
            parts = blob.name.split('/')
            if len(parts) >= 2 and parts[0] == firebase_voices_prefix.strip('/'): # Ensure it's under our voices prefix
                speaker_slug = parts[-2] # e.g., 'alice_smith' from 'voices/alice_smith/...'
                if speaker_slug not in speaker_files:
                    speaker_files[speaker_slug] = []
                speaker_files[speaker_slug].append(blob)
    
    if not speaker_files:
        print("❌ No voice data found in Firebase Storage under the specified prefix. Please record some voices.")
        return np.array([]), np.array([]), {}, []

    # Process files speaker by speaker
    for speaker_slug in sorted(speaker_files.keys()):
        if speaker_slug not in labels_map:
            labels_map[speaker_slug] = label_id_counter
            id_to_label.append(speaker_slug)
            label_id_counter += 1
        
        current_label_id = labels_map[speaker_slug]
        print(f"Downloading and processing speaker: {speaker_slug} (ID: {current_label_id}) from Firebase...")

        speaker_has_audio_processed = False
        for blob in speaker_files[speaker_slug]:
            # Create a temporary file path
            # Using tempfile.NamedTemporaryFile is safer for temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=temp_local_dir) as temp_wav_file:
                temp_file_path = temp_wav_file.name

            try:
                blob.download_to_filename(temp_file_path)
                features = extract_features(temp_file_path)
                if features is not None:
                    X.append(features)
                    y.append(current_label_id)
                    speaker_has_audio_processed = True
            except Exception as e:
                print(f"⚠️ Error processing blob {blob.name}: {e}")
            finally:
                # Ensure the temporary file is deleted
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        if not speaker_has_audio_processed:
            print(f"  No valid audio files processed for {speaker_slug} from Firebase. This speaker will be skipped for training.")

    return np.array(X), np.array(y), labels_map, id_to_label


def train_and_save_model(model_file=MODEL_FILENAME, labels_file=LABELS_FILENAME):
    """
    Loads data from Firebase Storage, trains a RandomForestClassifier model, and saves it locally.
    """
    print("\n--- Training Model ---")
    X, y, labels_map, id_to_label = load_data_from_firebase_storage(FIREBASE_VOICES_PREFIX, TEMP_RECORDINGS_DIR)

    if len(X) == 0:
        print("❌ No audio data found or features extracted from Firebase. Cannot train model.")
        return None, None
    
    unique_speakers = len(labels_map)
    if unique_speakers < 2:
        print(f"❌ Need at least 2 distinct speakers ({unique_speakers} found) to train a meaningful model. Please add more data.")
        return None, None
    
    # Check if each speaker has enough samples for stratified splitting
    for speaker_id in range(unique_speakers):
        if np.sum(y == speaker_id) < 2:
            print(f"❌ Speaker '{id_to_label[speaker_id]}' has fewer than 2 samples. Each speaker needs at least 2 samples to train. Please add more data for this speaker.")
            return None, None
        
    print(f"\nTotal samples loaded: {len(X)}")
    print(f"Speakers found: {labels_map}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Model training complete. ✅")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on test set: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=id_to_label))

    # Save the trained model and the ID-to-label mapping locally
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved locally to {model_file}")

    with open(labels_file, 'wb') as f:
        pickle.dump(id_to_label, f)
    print(f"ID-to-label map saved locally to {labels_file}")
    
    return model, id_to_label

def load_trained_model(model_file=MODEL_FILENAME, labels_file=LABELS_FILENAME):
    """
    Loads a pre-trained model and label mapping from disk.
    """
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(labels_file, 'rb') as f:
            id_to_label = pickle.load(f)
        print("✅ Model and labels loaded successfully.")
        return model, id_to_label
    except FileNotFoundError:
        return None, None

def recognize_speaker_from_file(model, id_to_label, audio_sample_path):
    """
    Recognizes a speaker from a given audio file.
    """
    if model is None or id_to_label is None:
        print("Error: Model not loaded. Please train or load a model first by adding new data (Option 1).")
        return "Not Available"

    features = extract_features(audio_sample_path)
    if features is None:
        return "Unknown Speaker (Feature Extraction Failed)"

    features = features.reshape(1, -1) # Reshape for prediction

    prediction_id = model.predict(features)[0]
    predicted_speaker = id_to_label[prediction_id]

    probabilities = model.predict_proba(features)[0]
    confidence = probabilities[prediction_id] * 100

    print(f"Predicted Speaker: {predicted_speaker} (Confidence: {confidence:.2f}%)")
    return predicted_speaker

def live_recognition(model, id_to_label, duration=DEFAULT_DURATION, fs=DEFAULT_SAMPLE_RATE, temp_dir=TEMP_RECORDINGS_DIR):
    """
    Records audio from the microphone, saves it temporarily, and recognizes the speaker.
    """
    if model is None or id_to_label is None:
        print("Error: Model not loaded. Please train or load a model first by adding new data (Option 1).")
        return "Not Available"

    os.makedirs(temp_dir, exist_ok=True)
    
    # Use tempfile for robust temporary file creation
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=temp_dir) as temp_wav_file:
        temp_file_path = temp_wav_file.name

    print(f"\n🎙️ Please speak for {duration} seconds... STARTING RECORDING NOW!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    
    wav.write(temp_file_path, fs, recording)
    print(f"Recording saved temporarily to {temp_file_path}")

    recognized_speaker = recognize_speaker_from_file(model, id_to_label, temp_file_path)
    
    os.remove(temp_file_path) # Clean up the temporary file
    print(f"Temporary file {temp_file_path} removed.")
    
    return recognized_speaker

# --- Main Application Logic ---

def main():
    trained_model = None
    id_to_label_map = None

    # Try to load existing model at startup
    print("Attempting to load existing model...")
    trained_model, id_to_label_map = load_trained_model()
    if trained_model is None:
        print("No existing model found locally. Please add new data (Option 1) to train and save a model.")


    while True:
        print("\n--- Speaker Recognition App ---")
        print("1. Add/Record New Speaker Voice Data (Stores on Firebase Cloud)")
        print("2. Recognize Speaker from a File")
        print("3. Recognize Speaker from Live Microphone Input")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            added_new_data_in_session = False # Flag to check if any data was added in this session
            while True:
                person_name = input("\nEnter the name of the person to record (or 'done' to finish recording this session): ").strip()
                if person_name.lower() == 'done':
                    break
                if not person_name:
                    print("Person name cannot be empty. Please try again.")
                    continue
                
                # record_voice now handles temporary local saving and direct Firebase upload
                record_voice(TEMP_RECORDINGS_DIR, person_name, 
                             num_samples=DEFAULT_NUM_SAMPLES, 
                             duration=DEFAULT_DURATION, 
                             fs=DEFAULT_SAMPLE_RATE)
                added_new_data_in_session = True
            
            if added_new_data_in_session:
                print("\nRecording session complete. Automatically training model with updated data from Firebase...")
                # train_and_save_model now reads from Firebase Storage
                trained_model, id_to_label_map = train_and_save_model()
            else:
                print("\nNo new data added in this session. Model not retrained.")
            
        elif choice == '2':
            if trained_model is None:
                print("Cannot recognize. Model not trained or loaded. Please add new data (Option 1) first.")
                continue
            
            file_path = input("Enter the path to the WAV file for recognition: ").strip()
            if os.path.exists(file_path) and file_path.lower().endswith('.wav'):
                recognize_speaker_from_file(trained_model, id_to_label_map, file_path)
            else:
                print(f"❌ Error: File not found or not a WAV file at '{file_path}'")

        elif choice == '3':
            if trained_model is None:
                print("Cannot recognize. Model not trained or loaded. Please add new data (Option 1) first.")
                continue
            
            live_recognition(trained_model, id_to_label_map, 
                             duration=DEFAULT_DURATION, fs=DEFAULT_SAMPLE_RATE)

        elif choice == '4':
            print("Exiting Speaker Recognition App. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    # Ensure temporary recordings directory exists
    os.makedirs(TEMP_RECORDINGS_DIR, exist_ok=True)
    main()
