import torch
import torch.nn as nn

class OneLayerNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, digit):
        out = self.fc1(digit)
        out = self.fc2(nn.functional.relu(out))
        return out
