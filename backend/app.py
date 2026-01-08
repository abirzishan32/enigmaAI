from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import tenseal as ts
import json
from pathlib import Path
from typing import List, Dict, Any

app = FastAPI(title="FHE Digit Recognition API (TenSEAL)")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
context = None
model_metadata = None


class SimpleMNISTNet(nn.Module):
    """Simple neural network for MNIST digit classification"""
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class InferenceRequest(BaseModel):
    pixels: List[float]


class EncryptedInferenceRequest(BaseModel):
    encrypted_input: str  # Base64 encoded encrypted data


@app.on_event("startup")
async def load_model():
    """Load model and TenSEAL context on startup"""
    global model, context, model_metadata
    
    try:
        model_path = Path("model.pth")
        context_path = Path("full_context.bin")
        metadata_path = Path("metadata.json")
        
        if model_path.exists() and context_path.exists():
            print("Loading PyTorch model...")
            model = SimpleMNISTNet()
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()
            print("✓ Model loaded successfully!")
            
            print("Loading TenSEAL context...")
            with open(context_path, "rb") as f:
                context = ts.context_from(f.read())
            print("✓ TenSEAL context loaded!")
            
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    model_metadata = json.load(f)
                print(f"✓ Model metadata loaded: Test accuracy = {model_metadata.get('test_accuracy', 'N/A')}%")
        else:
            print("⚠ Model not found. Please run train_model_tenseal.py first.")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "FHE Digit Recognition API (TenSEAL)",
        "status": "success",
        "fhe_enabled": model is not None and context is not None,
        "framework": "TenSEAL + PyTorch"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "context_loaded": context is not None,
        "model_metadata": model_metadata
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
