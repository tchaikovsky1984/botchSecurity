import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import glob
import cv2  # Import OpenCV for SIFT
from PIL import Image
import torchaudio
import torchaudio.transforms as T

# FastAPI imports
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import tempfile

# --- Configuration ---
# These MUST match the values used during your model training!
DATA_ROOT = './dataset/'  # Adjust this path as necessary for your deployment
MODEL_PATH = '../../weights/multimodal_person_model.pth' # Path to your saved .pth model
AUDIO_MAX_LEN = 500
AUDIO_N_MFCC = 40
AUDIO_SAMPLE_RATE_FOR_PREPROCESS = 16000
MAX_SIFT_DESCRIPTORS = 100 # From your MultiModalPersonDataset
SIFT_DESCRIPTOR_DIM = 128 # Default for SIFT

# Prediction thresholds
THRESHOLD_FUSED = 0.7

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"FastAPI app will use device: {DEVICE}")

# --- SIFT Feature Extraction Function (from your ipynb) ---
def get_sift_features(image_path):
    """
    Extracts SIFT feature vectors from an image.
    Args:
        image_path (str): The path to the image file.
    Returns:
        tuple: A tuple containing:
            - keypoints (list): A list of cv2.KeyPoint objects.
            - descriptors (numpy.ndarray): A NumPy array of SIFT descriptors,
                                         or None if no features are found.
    """
    sift = cv2.SIFT_create()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return [], None

    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

# --- 1. MultiModalPersonDataset Class (Minimal for mappings) ---
# This class needs to be identical in terms of person-to-index mapping
# and `not_identified_label` to the one used during training.
# Data loading methods are stubbed out as they're not needed for inference mapping.
class MultiModalPersonDataset:
    def __init__(self, root_dir, audio_sample_rate=16000, audio_n_mfcc=40, audio_max_len=500, is_train=True, use_sift=False, max_sift_descriptors=100):
        self.root_dir = root_dir
        self.audio_sample_rate = audio_sample_rate
        self.audio_n_mfcc = audio_n_mfcc
        self.audio_max_len = audio_max_len
        self.is_train = is_train
        self.use_sift = use_sift
        self.max_sift_descriptors = max_sift_descriptors
        self.sift_descriptor_dim = 128 # Default for SIFT

        self.persons = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.person_to_idx = {person: idx for idx, person in enumerate(self.persons)}
        self.idx_to_person = {idx: person for person, idx in self.person_to_idx.items()}

        self.not_identified_label = len(self.persons)
        self.person_to_idx['not identified'] = self.not_identified_label
        self.idx_to_person[self.not_identified_label] = 'not identified'
        self.num_classes = len(self.person_to_idx)

    # These methods are for training data loading, not needed for inference mappings
    def _collect_paths(self): pass
    def _create_data_pairs_with_mismatched(self): return []
    def __len__(self): return 0
    def __getitem__(self, idx): raise NotImplementedError("Not a dataset for data loading.")

# --- 2. ANNModel Architecture (Identical to your ipynb) ---
class ANNModel(nn.Module):
    def __init__(self, num_classes, sift_input_dim, mfcc_input_dim):
        super(ANNModel, self).__init__()

        # Define the SIFT branch (simple ANNs)
        self.sift_fc1 = nn.Linear(sift_input_dim, 512)
        self.sift_fc2 = nn.Linear(512, 256)

        # Define the MFCC branch (simple ANNs)
        self.mfcc_fc1 = nn.Linear(mfcc_input_dim, 512)
        self.mfcc_fc2 = nn.Linear(512, 256)

        # Define the combined branch
        combined_input_dim = 256 + 256
        self.combined_fc1 = nn.Linear(combined_input_dim, 128)
        self.combined_fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self, sift_features, mfcc_features):
        # Process SIFT features
        sift_out = self.sift_fc1(sift_features)
        sift_out = self.relu(sift_out)
        sift_out = self.dropout(sift_out)
        sift_out = self.sift_fc2(sift_out)
        sift_out = self.relu(sift_out)
        sift_out = self.dropout(sift_out)

        # Process MFCC features (flatten before passing through ANN)
        mfcc_features_flat = mfcc_features.view(mfcc_features.size(0), -1) # Flatten time and n_mfcc dims
        mfcc_out = self.mfcc_fc1(mfcc_features_flat)
        mfcc_out = self.relu(mfcc_out)
        mfcc_out = self.dropout(mfcc_out)
        mfcc_out = self.mfcc_fc2(mfcc_out)
        mfcc_out = self.relu(mfcc_out)
        mfcc_out = self.dropout(mfcc_out)

        # Concatenate outputs from both branches
        combined_out = torch.cat((sift_out, mfcc_out), dim=1)

        # Process combined features
        combined_out = self.combined_fc1(combined_out)
        combined_out = self.relu(combined_out)
        combined_out = self.dropout(combined_out)
        logits = self.combined_fc2(combined_out)

        return logits

# --- 3. Inference Function (Adapted for ANNModel's single output) ---
def identify_person_multimodal_inference(model, sift_tensor, audio_tensor, person_idx_to_name,
                                         not_identified_idx, threshold_fused=0.7):
    model.eval()
    device = next(model.parameters()).device
    sift_tensor = sift_tensor.to(device)
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        logits = model(sift_tensor, audio_tensor)
        probs = F.softmax(logits, dim=1)

        conf, pred_idx = torch.max(probs, dim=1)
        conf = conf.item()
        pred_idx = pred_idx.item()

        pred_name = person_idx_to_name.get(pred_idx, "Unknown_ID")

        details = {
            "confidence": f"{conf:.4f}",
            "predicted_label": pred_name
        }

        if pred_idx == not_identified_idx:
            if conf >= threshold_fused:
                return "Not Identified", "Model confidently predicted 'Not Identified'", details
            else:
                return "Not Identified", "Model predicted 'Not Identified' but with low confidence", details
        else:
            if conf >= threshold_fused:
                return pred_name, f"Model confidently identified {pred_name}", details
            else:
                return "Not Identified", f"Model predicted '{pred_name}' but with low confidence", details

# --- Helper functions for single input preprocessing ---
def preprocess_single_image_sift(img_file_path, max_descriptors=MAX_SIFT_DESCRIPTORS, descriptor_dim=SIFT_DESCRIPTOR_DIM):
    """
    Loads an image, extracts SIFT features, and formats them for the ANN model.
    """
    _, descriptors = get_sift_features(img_file_path)

    if descriptors is not None:
        num_descriptors = descriptors.shape[0]
        fixed_size_descriptor_vector = torch.zeros(max_descriptors, descriptor_dim)

        if num_descriptors > 0:
            descriptors_tensor = torch.tensor(descriptors, dtype=torch.float32)
            if num_descriptors > max_descriptors:
                fixed_size_descriptor_vector = descriptors_tensor[:max_descriptors, :]
            else:
                fixed_size_descriptor_vector[:num_descriptors, :] = descriptors_tensor

        image_input = fixed_size_descriptor_vector.flatten()
    else:
        # Handle case where no features are found (return zeros)
        image_input = torch.zeros(max_descriptors * descriptor_dim)

    return image_input.unsqueeze(0) # Add batch dimension


def preprocess_single_audio(aud_file_path, sample_rate=AUDIO_SAMPLE_RATE_FOR_PREPROCESS, n_mfcc=AUDIO_N_MFCC, max_len=AUDIO_MAX_LEN):
    """Loads and preprocesses a single audio file for model input."""
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={
                                "n_fft": 400, "hop_length": 160, "n_mels": 64})

    try:
        waveform, sr = torchaudio.load(aud_file_path)
        if sr != sample_rate:
            resampler = T.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mfcc_features = mfcc_transform(waveform).squeeze(0).transpose(0, 1)

        if mfcc_features.shape[0] > max_len:
            mfcc_features = mfcc_features[:max_len, :]
        elif mfcc_features.shape[0] < max_len:
            padding = torch.zeros(
                max_len - mfcc_features.shape[0], n_mfcc, dtype=mfcc_features.dtype)
            mfcc_features = torch.cat((mfcc_features, padding), dim=0)

        mean = mfcc_features.mean()
        std = mfcc_features.std()
        if std == 0:
            std = 1e-6
        mfcc_features = (mfcc_features - mean) / std
        return mfcc_features.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing audio {aud_file_path}: {e}")
        return torch.zeros(1, max_len, n_mfcc)  # Return dummy tensor on error


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multi-Modal Person Identification API (SIFT + MFCC ANN)",
    description="API for identifying persons using SIFT features from images and MFCCs from audio.",
    version="1.0.0"
)

# Global variables to store model and dataset info
model = None
person_idx_to_name = None
not_identified_idx = None
sift_input_dim = None
mfcc_input_dim = None

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_model_and_mappings():
    """
    Loads the trained model and dataset mappings when the FastAPI application starts.
    """
    global model, person_idx_to_name, not_identified_idx, sift_input_dim, mfcc_input_dim

    print("FastAPI startup: Loading dataset info for class mapping and model dimensions...")
    try:
        # Instantiate MultiModalPersonDataset to get mappings and dimensions
        # Use the same parameters as during training to get correct dimensions
        dataset_info = MultiModalPersonDataset(DATA_ROOT, use_sift=True, max_sift_descriptors=MAX_SIFT_DESCRIPTORS)
        num_classes = dataset_info.num_classes
        person_idx_to_name = dataset_info.idx_to_person
        not_identified_idx = dataset_info.not_identified_label
        sift_input_dim = dataset_info.max_sift_descriptors * dataset_info.sift_descriptor_dim
        mfcc_input_dim = dataset_info.audio_max_len * dataset_info.audio_n_mfcc

        print(f"Loaded class mappings for {num_classes} persons (including 'not identified').")
        print(f"Determined SIFT input dimension: {sift_input_dim}")
        print(f"Determined MFCC input dimension: {mfcc_input_dim}")
    except Exception as e:
        print(f"ERROR: Could not load dataset info. Ensure DATA_ROOT ('{DATA_ROOT}') is correct and contains person directories.")
        raise HTTPException(status_code=500, detail=f"Server startup error: {e}")

    print(f"FastAPI startup: Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'. Please ensure the file exists.")
        raise HTTPException(status_code=500, detail="Model file not found.")

    try:
        # Initialize the ANNModel with the determined input dimensions and number of classes
        model = ANNModel(num_classes=num_classes, sift_input_dim=sift_input_dim, mfcc_input_dim=mfcc_input_dim).to(DEVICE)

        # Load the state_dict
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("FastAPI startup: Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model from '{MODEL_PATH}': {e}")
        raise HTTPException(status_code=500, detail=f"Server startup error: Could not load model: {e}")

@app.get("/")
async def read_root():
    """
    Root endpoint for a simple health check.
    """
    return {"message": "Multi-Modal Person Identification API (SIFT + MFCC ANN) is running. Go to /docs for API documentation."}

@app.post("/predict/")
async def predict_person(image_file: UploadFile = File(...), audio_file: UploadFile = File(...)):
    """
    Predicts the identity of a person from an image (using SIFT) and an audio file (using MFCC).
    Args:
        image_file (UploadFile): The uploaded image file (JPEG, PNG).
        audio_file (UploadFile): The uploaded audio file (MP3, M4A, AAC).
    Returns:
        JSONResponse: A JSON object containing the identified person, reason, details, and prediction time.
    """
    if model is None or person_idx_to_name is None or not_identified_idx is None or sift_input_dim is None or mfcc_input_dim is None:
        raise HTTPException(
            status_code=503, detail="Model or mappings not loaded. Server is still starting up or encountered an error.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1]) as temp_img:
        shutil.copyfileobj(image_file.file, temp_img)
        temp_img_path = temp_img.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_aud:
        shutil.copyfileobj(audio_file.file, temp_aud)
        temp_aud_path = temp_aud.name

    try:
        start_time = time.time()
        # Use the SIFT preprocessing function
        image_sift_tensor = preprocess_single_image_sift(temp_img_path, MAX_SIFT_DESCRIPTORS, SIFT_DESCRIPTOR_DIM)
        audio_mfcc_tensor = preprocess_single_audio(temp_aud_path, sample_rate=AUDIO_SAMPLE_RATE_FOR_PREPROCESS)

        identified_person, reason, details = identify_person_multimodal_inference(
            model, image_sift_tensor, audio_mfcc_tensor, person_idx_to_name,
            not_identified_idx, THRESHOLD_FUSED
        )
        end_time = time.time()
        prediction_time = end_time - start_time

        return JSONResponse(content={
            "identified_person": identified_person,
            "reason": reason,
            "details": details,
            "prediction_time_seconds": f"{prediction_time:.4f}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    finally:
        # Clean up temporary files
        os.unlink(temp_img_path)
        os.unlink(temp_aud_path)