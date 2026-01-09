from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import tenseal as ts
import numpy as np
from digit_recognition.model import ConvNet
from routers import chat, sentiment

app = FastAPI(title="FHE Digit Recognition API (TenSEAL)")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat.router)
app.include_router(sentiment.router)

# Global variables
model = None
context = None

class ImageInput(BaseModel):
    image: list  # Expecting a flat list or 2D list of pixel values

def setup_tenseal_context():
    """
    Setup TenSEAL context for CKKS scheme.
    CKKS is good for floating point operations (like neural networks).
    """
    # bit_scale: scales the message to preserve precision
    # poly_modulus_degree: degree of the polynomial modulus (security parameter)
    # coeff_module_bit_sizes: bit sizes of the coefficient modulus primes
    # Increasing to 16384 and adding more primes to support depth (Linear -> Square -> Linear)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

@app.on_event("startup")
async def startup_event():
    global model, context
    
    # Load Model
    print("Loading model...")
    device = torch.device("cpu") # TenSEAL works on CPU
    model = ConvNet().to(device)
    
    model_path = "mnist_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    else:
        print("Warning: mnist_model.pth not found. Please run train.py first.")

    # Setup TenSEAL
    print("Setting up TenSEAL context...")
    context = setup_tenseal_context()
    print("TenSEAL context ready.")
    
    # Load Sentiment Analysis Model
    print("Loading sentiment analysis model...")
    sentiment.load_sentiment_model()


@app.post("/classify")
async def classify_digit(payload: ImageInput):
    global model, context
    
    if model is None or context is None:
        raise HTTPException(status_code=503, detail="Model or Context not initialized")

    try:
        # 1. Preprocess Input
        # Frontend should send 28x28 normalized values (0-1) or we normalize here.
        # Assuming frontend sends flattened array of 784 float values (0-1).
        input_data = np.array(payload.image, dtype=np.float32).flatten()
        
        if len(input_data) != 784:
            raise HTTPException(status_code=400, detail=f"Expected 784 pixels, got {len(input_data)}")

        # 2. Encrypt
        print("Encrypting input...")
        # ckks_vector encrypts a vector of float numbers
        enc_input = ts.ckks_vector(context, input_data)

        # 3. Model Prediction (Homomorphic)
        print("Running FHE prediction...")
        # Extract weights and biases from the PyTorch model
        fc1_weight = model.fc1.weight.data.numpy().T
        fc1_bias = model.fc1.bias.data.numpy()
        
        fc2_weight = model.fc2.weight.data.numpy().T
        fc2_bias = model.fc2.bias.data.numpy()

        # Layer 1: Linear
        enc_hidden = enc_input.matmul(fc1_weight) + fc1_bias
        
        # Layer 1: Activation (Square)
        enc_hidden.square_() # In-place square
        
        # Layer 2: Linear
        enc_output = enc_hidden.matmul(fc2_weight) + fc2_bias
        
        # 4. Decrypt
        print("Decrypting result...")
        output_vec = enc_output.decrypt()
        
        # 5. Helper: Argmax
        prediction = int(np.argmax(output_vec))
        confidence = float(np.max(output_vec)) # Not a real probability without Softmax, but relative score

        return {
            "prediction": prediction,
            "confidence": confidence, # Raw score
            "encrypted": True
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
