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
        self.w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
        self.w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
        self.b_i_h = np.zeros((20, 1))
        self.b_h_o = np.zeros((10, 1))

    def forward(self, x):
        x.shape += (1,)
        h_pre = self.b_i_h + self.w_i_h @ x
        h = 1 / (1 + np.exp(-h_pre))
        o_pre = self.b_h_o + self.w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        o = 1 / (1 + np.exp(-o_pre))
        return o

    def load_weights(self, file_path):
        loaded_weights = np.load(file_path)
        self.w_i_h = loaded_weights['w_i_h']
        self.b_i_h = loaded_weights['b_i_h']
        self.w_h_o = loaded_weights['w_h_o']
        self.b_h_o = loaded_weights['b_h_o']