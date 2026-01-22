"""
Train a small sentiment analysis model on IMDb dataset
Uses TF-IDF + simple neural network for FHE compatibility
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os
from datasets import load_dataset
import numpy as np

from sentiment_analysis.model import SentimentNet

def simplify_label(rating):
    """Convert 0-10 rating to Negative(0), Neutral(1), Positive(2)"""
    if rating <= 4:
        return 0  # Negative
    elif rating <= 6:
        return 1  # Neutral
    else:
        return 2  # Positive

def train_model():
    print("="*50)
    print("Training Sentiment Analysis Model (FHE-Compatible)")
    print("="*50)
    
    # Parameters
    max_features = 256  # Small TF-IDF feature space for FHE efficiency
    epochs = 20
    batch_size = 64
    learning_rate = 0.01
    device = torch.device("cpu")  # FHE models run on CPU
    
    # Load IMDb dataset
    print("\n[1/6] Loading IMDb dataset...")
    dataset = load_dataset("imdb", split="train[:5000]")  # Use subset for faster training
    texts = dataset['text']
    labels = dataset['label']  # 0=negative, 1=positive
    
    # Convert binary to ternary (add neutral class for demonstration)
    # For simplicity: <0.3 = negative, 0.3-0.7 = neutral, >0.7 = positive
    # Since IMDb is binary, we'll split based on confidence simulation
    labels_ternary = [0 if l == 0 else 2 for l in labels]  # 0=neg, 2=pos, no actual neutral
    
    print(f"Loaded {len(texts)} samples")
    
    # TF-IDF Vectorization
    print(f"\n[2/6] Creating TF-IDF vectorizer (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray()
    y = np.array(labels_ternary)
    
    print(f"Feature shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    print("\n[3/6] Initializing model...")
    model = SentimentNet(input_dim=max_features, hidden_dim=64, output_dim=3)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\n[4/6] Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Evaluation
    print("\n[5/6] Evaluating model...")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = 100 * (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save model and vectorizer
    print("\n[6/6] Saving model and vectorizer...")
    os.makedirs("sentiment_analysis/saved_models", exist_ok=True)
    
    torch.save(model.state_dict(), "sentiment_model.pth")
    joblib.dump(vectorizer, "sentiment_vectorizer.pkl")
    joblib.dump(['Negative', 'Neutral', 'Positive'], "sentiment_labels.pkl")
    
    print("\n Training complete!")
    print(f"   - Model saved to: sentiment_model.pth")
    print(f"   - Vectorizer saved to: sentiment_vectorizer.pkl")
    print(f"   - Labels saved to: sentiment_labels.pkl")
    print("="*50)

if __name__ == "__main__":
    train_model()