import torch
import torch.nn as nn
import os
import torchprofile
from .unet import UNet
from .deeplabv3_plus import DeepLab_V3_plus
import time
import torch.nn.functional as F
import functools

class DE_framework(nn.Module):
    def __init__(self, models=[UNet(n_channels=1, num_classes=4), DeepLab_V3_plus(4)], weight_list = None):  
        super(DE_framework, self).__init__()
        self.models = nn.ModuleList(models)  
        self.weight_list = weight_list
        self.process = self.weighted_avg

    def forward(self, x):
        return self.process(x)
    
    def weighted_avg(self, x):
        final_output = torch.zeros(x.size(0), 4, x.size(2), x.size(3)).to(x.device)  # Create new tensor with the required size
        for weight, model in zip(self.weight_list, self.models):
            model_output = model(x)
            model_output = F.softmax(model_output, dim=1)
            model_output = model_output * weight
            final_output += model_output.clone()  # Add the cloned model output to avoid in-place operations on shared memory
        final_output = F.softmax(final_output, dim=1)
        return final_output
    
    def save_model(self, save_dir="models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for idx, model in enumerate(self.models):
            model_name = f"{model.__class__.__name__}_{idx + 1}"
            
            save_path = os.path.join(save_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)

    def load_model(self, load_dir="models"):
        for idx, model in enumerate(self.models):
            model_name = f"{model.__class__.__name__}_{idx + 1}"
            
            load_path = os.path.join(load_dir, f"{model_name}.pth")
            
            if os.path.exists(load_path):
                model.load_state_dict(torch.load(load_path))
            else:
                print(f"Model file {load_path} does not exist!")


