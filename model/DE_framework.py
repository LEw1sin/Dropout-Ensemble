import torch
import torch.nn as nn
import os
from .unet import UNet_linear
from .deeplabv3_plus import DeepLabV3P_linear
import torch.nn.functional as F

class DE_framework_linear(nn.Module):
    def __init__(self, args, models=[UNet_linear(input_channel=1, num_classes=4), DeepLabV3P_linear(4)], weight_list = None):  
        super(DE_framework_linear, self).__init__()
        self.models = nn.ModuleList(models)  
        self.weight_list = weight_list
        self.process = self.weighted_avg
        self.num_classes = args.num_classes

    def forward(self, x, valid_mask):
        return self.process(x, valid_mask)
    
    def weighted_avg(self, img, valid_mask):
        x = img[valid_mask].unsqueeze(1)
        final_output = torch.zeros(x.size(0), self.num_classes, x.size(2), x.size(3)).to(img.device)  # Create new tensor with the required size
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

class DE_framework_mem(nn.Module):
    def __init__(self, args, models, weight_list = None):  
        super(DE_framework_mem, self).__init__()
        self.models = nn.ModuleList(models)  
        self.weight_list = weight_list
        self.num_classes = args.num_classes

    def forward(self, img, valid_mask):
        B, D, H, W= img.size()
        img_ = img[valid_mask].unsqueeze(1)
        true_counts = valid_mask.sum(dim=1).tolist() 
        valid_depth = img_.size(0)

        base_learner_outputs = []
        base_learner_preweights = []
        for model in self.models:
            base_learner_output = model(img_)
            base_learner_output = F.softmax(base_learner_output, dim=1)
            base_learner_outputs.append(base_learner_output)
            base_learner_output_splits = torch.split(base_learner_output, true_counts)
            model_prewight = []
            for base_learner_output_split in base_learner_output_splits:
                depth, num_class = base_learner_output_split.size(0), base_learner_output_split.size(1)
                sample_mean = base_learner_output_split.mean(dim=0)
                sample_var = sample_mean.var(dim=0)
                sample_preweight = torch.exp(sample_var).unsqueeze(0).unsqueeze(0).expand(depth, num_class, -1, -1)
                model_prewight.append(sample_preweight)
            base_learner_preweights.append(torch.cat(model_prewight, dim=0))

        sum_weight = torch.sum(torch.stack(base_learner_preweights), dim=0)
        model_wights = [preweight/sum_weight for preweight in base_learner_preweights]
        final_output = torch.zeros(valid_depth, self.num_classes, H, W).to(img.device)  # Create new tensor with the required size
        for weight, model_output in zip(model_wights, base_learner_outputs):
            final_output += weight*model_output.clone()
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
                model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True))
            else:
                print(f"Model file {load_path} does not exist!")

import numpy as np
from scipy import ndimage

def random_rot_flip(image):
    image = np.transpose(image, (1, 2, 0))
    
    k = np.random.randint(0, 4)  
    image = np.rot90(image, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()


    image = np.transpose(image, (2, 0, 1))

    return image, k, axis  

def random_rotate(image):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    
    return image, angle  

def inverse_random_rot_flip(image, k, axis):
    image = np.rot90(image, 4 - k, axes=(2,3))

    return image


def inverse_random_rotate(image, angle):
    image = ndimage.rotate(image, -angle, axes=(2,3), order=0, reshape=False)

    return image

class DE_framework_Augmenting(nn.Module):
    def __init__(self, args, models, weight_list = None):  
        super(DE_framework_Augmenting, self).__init__()
        self.models = nn.ModuleList(models)  
        self.model = self.models[0]
        self.weight_list = 1
        self.process = self.weighted_avg
        self.num_classes = args.num_classes

    def forward(self, x, valid_mask):
        return self.process(x, valid_mask)
    
    def weighted_avg(self, img, valid_mask):
        x = img[valid_mask].cpu().detach().numpy()
        x_rot_flip, k, axis = random_rot_flip(x)
        x_rotate, angle = random_rotate(x)

        final_output = np.zeros((x.shape[0], self.num_classes, x.shape[1], x.shape[2]))  # Create new tensor with the required size

        x_rot_flip = torch.from_numpy(x_rot_flip).to(img.device).unsqueeze(1)
        x_rotate = torch.from_numpy(x_rotate).to(img.device).unsqueeze(1)
        rot_flip_output = self.model(x_rot_flip)
        # rot_flip_output = F.softmax(rot_flip_output, dim=1)
        rot_flip_output = inverse_random_rot_flip(rot_flip_output.cpu().detach().numpy(), k, axis)
        final_output += rot_flip_output

        rotate_output = self.model(x_rotate)
        # rotate_output = F.softmax(rotate_output, dim=1)
        rotate_output = inverse_random_rotate(rotate_output.cpu().detach().numpy(), angle)
        final_output += rotate_output

        final_output = F.softmax(torch.from_numpy(final_output), dim=1)
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

def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params > 1e3 and num_params < 1e6:
        num_params = f'{num_params/1e3:.2f}K'
    elif num_params > 1e6:
        num_params = f'{num_params/1e6:.2f}M'
    return num_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
    args = parser.parse_args()
    t = torch.rand(1, 30, 224, 224).to(device='cuda:1')
    valid_mask = torch.cat([torch.ones(11, dtype=torch.bool), torch.zeros(30 - 11, dtype=torch.bool)]).unsqueeze(0)
    net = DE_framework_mem(args, models=[UNet_linear(num_classes = 4, max_channels=256)]).to(device=t.device)
    pred = net(t,valid_mask)
    print(pred.size())
    print(count_parameters(net))
    # from fvcore.nn import FlopCountAnalysis
    # flops = FlopCountAnalysis(net, (t, valid_mask))
    # print(f"FLOPs: {flops.total()}")

