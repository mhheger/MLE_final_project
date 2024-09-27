import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, field_dim, n_actions):
        super(DQN, self).__init__()
        self.input_dim = field_dim  # This is assumed to be the width/height of the 2D input
        self.conv_layer1 = nn.Conv2d(5, 16, (3, 3))  # In channels: 5, out channels: 16
        self.pool_layer1 = nn.MaxPool2d(3, stride=2)
        self.conv_layer2 = nn.Conv2d(16, 32, (2, 2))
        self.pool_layer2 = nn.MaxPool2d(2, stride=2)

        # Dynamically calculate the size of the flattened conv layer output
        # Assume square input
        conv_output_size = self._calculate_conv_output_size(field_dim)

        self.layer1 = nn.Linear(conv_output_size + 2, 512)  # +5 for the concatenated part
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def _calculate_conv_output_size(self, input_size):
        # Conv Layer 1 (3x3 kernel)
        size = input_size - 2  # (input_size - 3 + 1 for stride 1)
        
        # Pool Layer 1 (3x3 kernel, stride 2)
        size = (size - 3) // 2 + 1
        
        # Conv Layer 2 (2x2 kernel)
        size = size - 1  # (size - 2 + 1 for stride 1)
        
        # Pool Layer 2 (2x2 kernel, stride 2)
        size = (size - 2) // 2 + 1
        
        # The final output will be 32 channels * size * size (for the flattened output)
        return 32 * size * size

    def forward(self, x):
        batch_size = 1 # Get batch size (even if it's 1 for single input)
        d = self.input_dim
        
        if len(x.size()) == 1:  # Single input, reshape accordingly
            y = torch.reshape(x[:-2], (1, 5, d, d))
            z = torch.reshape(x[-2:], (1, 2))  # This will be concatenated later
        else:  # Batch input
            batch_size = x.size(0)
            y = torch.reshape(x[:, :-2], (batch_size, 5, d, d))
            z = x[:, -2:]  # This will be concatenated directly
        
        y = self.conv_layer1(y)
        y = self.pool_layer1(y)
        y = F.relu(y)
        y = self.conv_layer2(y)
        y = self.pool_layer2(y)
        y = F.relu(y)
        y = y.view(batch_size, -1)  # Flatten the conv output to [batch_size, features]
        w = torch.cat((y, z), dim=1)  # Concatenate conv features with the last 5 elements

        w = F.relu(self.layer1(w))
        w = F.relu(self.layer2(w))
        return self.layer3(w)

    

    
