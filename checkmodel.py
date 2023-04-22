# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:37:16 2022

@author: Kin Chan
"""

	
from torchvision import models
import torch

if __name__ == "__main__":
    
    model = torch.load("Weight/default-epoch-099.pth")
    alex = torch.load_state_dict("Weight/alexnet-owt-7be5be79.pth")
    alex.eval()
    