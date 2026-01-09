"""
Sentiment Analysis Router with FHE support (TenSEAL)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import tenseal as ts
import joblib
import os

from sentiment_analysis.model import SentimentNet

router = APIRouter()

# Global variables for model and vectorizer
sentiment_model = None
vectorizer = None
labels = None
ts_context = None

class SentimentInput(BaseModel):
    text: str

class EncryptedSentimentInput(BaseModel):
    encrypted_features: list  # List of encrypted TF-IDF features

def load_sentiment_model():
    """Load the trained sentiment model and vectorizer"""
    global sentiment_model, vectorizer, labels
    
    try:
        # Get backend directory path
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load vectorizer
        vectorizer_path = os.path.join(backend_dir, "sentiment_vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            print(f"Warning: {vectorizer_path} not found. Run train.py first.")
            return False
        
        vectorizer = joblib.load(vectorizer_path)
        
        # Load labels
        labels_path = os.path.join(backend_dir, "sentiment_labels.pkl")
        if os.path.exists(labels_path):
            labels = joblib.load(labels_path)
        else:
            labels = ['Negative', 'Neutral', 'Positive']
        
        # Load model
        model_path = os.path.join(backend_dir, "sentiment_model.pth")
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Run train.py first.")
            return False
        
        sentiment_model = SentimentNet(input_dim=256, hidden_dim=64, output_dim=3)
        sentiment_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        sentiment_model.eval()
        
        print("✅ Sentiment model loaded successfully")
        return True
        
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_tenseal_context():
    """Create TenSEAL context for FHE operations"""
    global ts_context
    
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    ts_context = context
    return context

@router.get("/sentiment/status")
async def sentiment_status():
    """Check if sentiment model is loaded"""
    return {
        "model_loaded": sentiment_model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "labels": labels if labels else []
    }

@router.post("/sentiment/get-vectorizer-params")
async def get_vectorizer_params():
    """
    Return vectorizer parameters for client-side TF-IDF
    Client needs these to transform text before encryption
    """
    if vectorizer is None:
        raise HTTPException(status_code=500, detail="Vectorizer not loaded")
    
    # Convert vocabulary_ dict to have int values instead of numpy.int64
    vocab_dict = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    
    return {
        "vocabulary": vocab_dict,
        "idf": vectorizer.idf_.tolist(),
        "max_features": len(vectorizer.vocabulary_)
    }

@router.post("/sentiment/predict")
async def predict_sentiment(payload: SentimentInput):
    """
    Plaintext prediction (for testing)
    """
    if sentiment_model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Transform text to TF-IDF features
        features = vectorizer.transform([payload.text]).toarray()[0]
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = sentiment_model(features_tensor)
            _, predicted = torch.max(output, 1)
            prediction_idx = predicted.item()
        
        return {
            "prediction": labels[prediction_idx] if labels else f"Class {prediction_idx}",
            "prediction_index": prediction_idx
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/sentiment/predict-encrypted")
async def predict_sentiment_encrypted(payload: EncryptedSentimentInput):
    """
    FHE-encrypted prediction
    Client sends encrypted TF-IDF features
    Server performs inference on encrypted data
    Returns encrypted prediction
    """
    if sentiment_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # For now, simulating FHE by accepting array and encrypting/decrypting
        # In production, client would encrypt features
        features = np.array(payload.encrypted_features)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Predict (in FHE mode, this would operate on encrypted tensors)
        with torch.no_grad():
            output = sentiment_model(features_tensor)
            _, predicted = torch.max(output, 1)
            prediction_idx = predicted.item()
        
        # In real FHE, we'd return encrypted result
        # For this demo, returning plaintext similar to digit recognition
        return {
            "prediction": prediction_idx,
            "label": labels[prediction_idx] if labels else f"Class {prediction_idx}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encrypted prediction error: {str(e)}")
