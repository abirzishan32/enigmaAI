"""
Simple Logistic Regression model for Sentiment Analysis
Compatible with TenSEAL FHE encryption
"""

import numpy as np
import torch
import torch.nn as nn

class SentimentNet(nn.Module):
    """
    Simple feedforward network for sentiment classification
    Architecture: Linear(256 -> 64) -> ReLU -> Linear(64 -> 3)
    """
    def __init__(self, input_dim=256, hidden_dim=64, output_dim=3):
        super(SentimentNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
