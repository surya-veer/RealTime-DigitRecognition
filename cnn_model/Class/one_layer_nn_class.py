import torch
import torch.nn as nn
import numpy as np

class OneLayerNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, digit):
        out = self.fc1(digit)
        out = self.fc2(nn.functional.relu(out))
        return out
class OneLayerNN_2:
    def __init__(self):
        self.weight_1 = np.random.uniform(-0.5, 0.5, (28, 784))
        self.weight_2 = np.random.uniform(-0.5, 0.5, (10, 28))
        self.bias_1 = np.zeros((28, 1))
        self.bias_2 = np.zeros((10, 1))

    def forward(self, x):
        x.shape += (1,)
        h_pre = self.bias_1 + np.dot(self.weight_1, x)
        h = 1 / (1 + np.exp(-h_pre))
        o_pre = self.bias_2 + np.dot(self.weight_2, h)
        o = 1 / (1 + np.exp(-o_pre))
        return o

    def load_weights(self, file_path):
        loaded_weights = np.load(file_path)
        self.weight_1 = loaded_weights['weight_1']
        self.bias_1 = loaded_weights['bias_1']
        self.weight_2 = loaded_weights['weight_2']
        self.bias_2 = loaded_weights['bias_2']