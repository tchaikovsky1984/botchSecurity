import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import torchaudio
import torchaudio.transforms as T
import numpy as np
import time
import glob
from collections import defaultdict
import random # Not directly used in FastAPI inference, but kept for consistency if parts are reused

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil # For saving uploaded files temporarily
import tempfile # For creating temporary files/directories

# --- Configuration ---
# These MUST match the values used during your model training!
DATA_ROOT = '../../dataset/'
MODEL_SAVE_DIR = './' # Directory where models are saved
AUDIO_MAX_LEN = 500
AUDIO_N_MFCC = 40
AUDIO_SAMPLE_RATE_FOR_PREPROCESS = 16000

# Prediction thresholds (can be adjusted for desired sensitivity)
# For AttentionFusionModel, these apply to the single fused output's confidence.
THRESHOLD_FUSED = 0.7 # Confidence threshold for classification (e.g., if confidence < 0.7, label as "not identified")

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"FastAPI app will use device: {DEVICE}")

# --- 1. MultiModalPersonDataset Class (Minimal for mappings) ---
# This class needs to be identical in terms of person-to-index mapping
# and `not_identified_label` to the one used during training.
# Data loading methods are stubbed out as they're not needed for inference mapping.
class MultiModalPersonDataset: # Changed from Dataset to a regular class as it's not iterated
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # Original persons collected from directory names
        self.persons = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.person_to_idx = {person: idx for idx, person in enumerate(self.persons)}
        self.idx_to_person = {idx: person for idx, person in enumerate(self.persons)}

        # Add the 'not identified' class exactly as in your training code
        self.not_identified_label = len(self.persons)
        self.person_to_idx['not identified'] = self.not_identified_label
        self.idx_to_person[self.not_identified_label] = 'not identified'
        self.num_classes = len(self.person_to_idx)

    # These methods are for training data loading, not needed for inference mappings
    def _collect_paths(self): pass
    def _create_data_pairs_with_mismatched(self): return []
    def __len__(self): return 0
    def __getitem__(self, idx): raise NotImplementedError("Not a dataset for data loading.")

# --- 2. Model Architecture Classes (Identical to your ipynb) ---
class ImageCNN_Complex(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ImageCNN_Complex, self).__init__()
        self.backbone = models.resnet50(weights=None) # Ensure this matches your trained model
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        return self.backbone(x)

class AudioRNN_Complex(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, embedding_dim=512): # Match training params
        super(AudioRNN_Complex, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x):
        gru_out, hidden = self.gru(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        embedding = self.fc(hidden)
        return embedding

class AttentionFusionModel(nn.Module):
    def __init__(self, num_classes, image_embedding_dim=512, audio_embedding_dim=512, attention_dim=128):
        super(AttentionFusionModel, self).__init__()
        self.image_model = ImageCNN_Complex(embedding_dim=image_embedding_dim)
        
        # Initialize audio model directly with fixed audio_n_mfcc from config
        self.audio_model = AudioRNN_Complex(input_dim=AUDIO_N_MFCC, embedding_dim=audio_embedding_dim)

        self.image_embedding_dim = image_embedding_dim
        self.audio_embedding_dim = audio_embedding_dim
        self.attention_dim = attention_dim

        self.attention_layer1 = nn.Linear(self.image_embedding_dim + self.audio_embedding_dim, self.attention_dim)
        self.attention_layer2 = nn.Linear(self.attention_dim, 2) # Output 2 scores: one for image, one for audio

        self.fusion_fc = nn.Linear(self.image_embedding_dim + self.audio_embedding_dim, num_classes)

    def forward(self, image_input, audio_input):
        # Get embeddings from individual models
        image_embedding = self.image_model(image_input)
        audio_embedding = self.audio_model(audio_input)

        # Concatenate original embeddings for attention
        combined_original = torch.cat((image_embedding, audio_embedding), dim=1)

        # Calculate attention weights
        attention_scores = self.attention_layer2(torch.tanh(self.attention_layer1(combined_original)))
        attention_weights = F.softmax(attention_scores, dim=1)

        # Split attention weights and apply
        image_attn_weight = attention_weights[:, 0].unsqueeze(1)
        audio_attn_weight = attention_weights[:, 1].unsqueeze(1)

        image_attended = image_embedding * image_attn_weight.expand_as(image_embedding)
        audio_attended = audio_embedding * audio_attn_weight.expand_as(audio_embedding)

        # Concatenate attended embeddings
        combined_attended_embedding = torch.cat((image_attended, audio_attended), dim=1)

        # Pass through the final classification layer
        output = self.fusion_fc(combined_attended_embedding)

        return output

# --- 3. Inference Function (Adapted for AttentionFusionModel's single output) ---
def identify_person_multimodal_inference(model, image_tensor, audio_tensor, person_idx_to_name,
                                         not_identified_idx, threshold_fused=0.7):
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        fused_logits = model(image_tensor, audio_tensor)
        fused_probs = F.softmax(fused_logits, dim=1) # Get probabilities

        # Get the highest probability and its corresponding predicted index/name
        conf_fused, pred_fused_idx = torch.max(fused_probs, dim=1)
        conf_fused = conf_fused.item() # Convert to scalar
        pred_fused_idx = pred_fused_idx.item() # Convert to scalar

        pred_fused_name = person_idx_to_name.get(pred_fused_idx, "Unknown_ID")

        details = {
            "fused_confidence": f"{conf_fused:.4f}",
            "predicted_label": pred_fused_name
        }

        # Decision logic based on the fused confidence and predicted label
        if pred_fused_idx == not_identified_idx:
            # If the model explicitly predicts "not identified"
            if conf_fused >= threshold_fused:
                return "Not Identified", "Model confidently predicted 'Not Identified'", details
            else:
                return "Not Identified", "Model predicted 'Not Identified' but with low confidence", details
        else:
            # If the model predicts a specific person
            if conf_fused >= threshold_fused:
                return pred_fused_name, f"Model confidently identified {pred_fused_name}", details
            else:
                # If confidence for a specific person is low, fallback to "Not Identified"
                return "Not Identified", f"Model predicted '{pred_fused_name}' but with low confidence", details

# --- Helper functions for single input preprocessing (identical to previous) ---
def preprocess_single_image(img_file_path):
    """Loads and preprocesses a single image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_file_path).convert('RGB')
    return transform(image).unsqueeze(0) # Add batch dimension

def preprocess_single_audio(aud_file_path, sample_rate=AUDIO_SAMPLE_RATE_FOR_PREPROCESS, n_mfcc=AUDIO_N_MFCC, max_len=AUDIO_MAX_LEN):
    """Loads and preprocesses a single audio file for model input."""
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64})
    
    try:
        waveform, sr = torchaudio.load(aud_file_path)
        if sr != sample_rate:
            resampler = T.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1: # Convert to mono if stereo
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        mfcc_features = mfcc_transform(waveform).squeeze(0).transpose(0, 1) # (n_mfcc, time) -> (time, n_mfcc)

        if mfcc_features.shape[0] > max_len:
            mfcc_features = mfcc_features[:max_len, :]
        elif mfcc_features.shape[0] < max_len:
            padding = torch.zeros(max_len - mfcc_features.shape[0], n_mfcc, dtype=mfcc_features.dtype)
            mfcc_features = torch.cat((mfcc_features, padding), dim=0)

        mean = mfcc_features.mean()
        std = mfcc_features.std()
        if std == 0: std = 1e-6
        mfcc_features = (mfcc_features - mean) / std
        return mfcc_features.unsqueeze(0) # Add batch dimension
    except Exception as e:
        print(f"Error processing audio {aud_file_path}: {e}")
        return torch.zeros(1, max_len, n_mfcc) # Return dummy tensor on error

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multi-Modal Person Identification API",
    description="API for identifying persons using image and audio inputs.",
    version="1.0.0"
)

# Global variables to store model and dataset info
model = None
person_idx_to_name = None
not_identified_idx = None

@app.on_event("startup")
async def load_model_and_mappings():
    """
    Loads the trained model and dataset mappings when the FastAPI application starts.
    """
    global model, person_idx_to_name, not_identified_idx

    print("FastAPI startup: Loading dataset info for class mapping...")
    try:
        # Instantiate MultiModalPersonDataset to get mappings
        dataset_info = MultiModalPersonDataset(DATA_ROOT) # No is_train, audio_sample_rate etc. needed here
        num_classes = dataset_info.num_classes
        person_idx_to_name = dataset_info.idx_to_person
        not_identified_idx = dataset_info.not_identified_label
        print(f"Loaded class mappings for {num_classes} persons (including 'not identified').")
    except Exception as e:
        print(f"ERROR: Could not load dataset info. Ensure DATA_ROOT ('{DATA_ROOT}') is correct and contains person directories.")
        raise HTTPException(status_code=500, detail=f"Server startup error: {e}")

    print("FastAPI startup: Searching for the latest model...")
    list_of_files = glob.glob(os.path.join(MODEL_SAVE_DIR, '*.pth'))
    if not list_of_files:
        print(f"ERROR: No model files found in '{MODEL_SAVE_DIR}'. Please train the model and save it there.")
        raise HTTPException(status_code=500, detail="No trained model found. Please train the model.")

    latest_model_path = max(list_of_files, key=os.path.getctime)
    print(f"FastAPI startup: Loading latest model from: {latest_model_path}")

    try:
        # Initialize the model with the correct number of classes
        model = AttentionFusionModel(num_classes=num_classes).to(DEVICE)
        
        # Load the state_dict
        model.load_state_dict(torch.load(latest_model_path, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        print("FastAPI startup: Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load model from '{latest_model_path}': {e}")
        raise HTTPException(status_code=500, detail=f"Server startup error: Could not load model: {e}")


@app.get("/")
async def read_root():
    """
    Root endpoint for a simple health check.
    """
    return {"message": "Multi-Modal Person Identification API is running. Go to /docs for API documentation."}

@app.post("/predict/")
async def predict_person(image_file: UploadFile = File(...), audio_file: UploadFile = File(...)):
    """
    Predicts the identity of a person from an image and an audio file.

    Args:
        image_file (UploadFile): The uploaded image file (JPEG, PNG).
        audio_file (UploadFile): The uploaded audio file (MP3, M4A, AAC).

    Returns:
        JSONResponse: A JSON object containing the identified person, reason, details, and prediction time.
    """
    if model is None or person_idx_to_name is None or not_identified_idx is None:
        raise HTTPException(status_code=503, detail="Model or mappings not loaded. Server is still starting up or encountered an error.")

    # Create temporary files to save uploads
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1]) as temp_img:
        shutil.copyfileobj(image_file.file, temp_img)
        temp_img_path = temp_img.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_aud:
        shutil.copyfileobj(audio_file.file, temp_aud)
        temp_aud_path = temp_aud.name

    try:
        # Preprocess inputs
        start_time = time.time()
        image_tensor = preprocess_single_image(temp_img_path)
        audio_tensor = preprocess_single_audio(temp_aud_path, sample_rate=AUDIO_SAMPLE_RATE_FOR_PREPROCESS)
        
        # Perform inference
        identified, reason, details = identify_person_multimodal_inference(
            model,
            image_tensor,
            audio_tensor,
            person_idx_to_name,
            not_identified_idx,
            threshold_fused=THRESHOLD_FUSED
        )
        end_time = time.time()
        prediction_time = end_time - start_time

        return JSONResponse(content={
            "identified_person": identified,
            "reason": reason,
            "prediction_details": details,
            "prediction_time_seconds": f"{prediction_time:.4f}"
        })

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        # Clean up temporary files
        os.unlink(temp_img_path)
        os.unlink(temp_aud_path)
