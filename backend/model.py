import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, hidden=64):
        super(ConvNet, self).__init__()
        # We use a simple MLP because CNNs are harder to implement purely in TenSEAL (though possible)
        # without client-side modifications. 
        # A simple linear model (28*28 -> 10) can actually get 90%+ acc on MNIST.
        # But to show "Square" activation, we can do 1 hidden layer.
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        # Flatten
        x = x.view(-1, 28 * 28)
        # Activation: Square function (approximate to ReLU/Sigmoid for FHE)
        # FHE friendly activation: x^2
        x = self.fc1(x)
        x = x * x 
        x = self.fc2(x)
        return x
