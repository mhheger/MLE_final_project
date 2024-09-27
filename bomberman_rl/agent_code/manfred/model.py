import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        #self.conv_layer1 = nn.Conv3d(1,3,(3,3,3)) 
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

    
