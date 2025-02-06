import torch
import torch.nn as nn
import os
import torchprofile
from .unet import UNet_linear, UNet_logvar
from .deeplabv3_plus import DeepLabV3P_linear, DeepLabV3P_logvar
import time
import torch.nn.functional as F
import functools

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

class DE_framework_logvar(nn.Module):
    def __init__(self, args, models=[UNet_linear(input_channel=1, num_classes=4), DeepLabV3P_linear(4)], weight_list = None):  
        super(DE_framework_logvar, self).__init__()
        self.models = nn.ModuleList(models)  
        self.weight_list = weight_list
        self.process = self.weighted_avg
        self.num_classes = args.num_classes

    def forward(self, x):
        return self.process(x)
    
    def weighted_avg(self, x):
        final_output = torch.zeros(x.size(0), self.num_classes, x.size(2), x.size(3)).to(x.device)  # Create new tensor with the required size
        for model in self.models:
            uncertainty_weights, model_output = model(x)
            model_output = F.softmax(model_output, dim=1)
            model_output = model_output * uncertainty_weights
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
                model.load_state_dict(torch.load(load_path), map_location='cpu', weights_only=True)
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

# def forward(self, img, valid_mask):
#     B, D, H, W = img.size()
#     img_ = img[valid_mask].unsqueeze(1)  # [valid_depth, 1, H, W]
#     true_counts = valid_mask.sum(dim=1)  # [B]
#     counts = true_counts.to(img.device)  # [B]
#     valid_depth = img_.size(0)

#     # 创建一个映射，每个有效样本对应其所属的批次索引
#     batch_idx = torch.repeat_interleave(torch.arange(B, device=img.device), counts)  # [valid_depth]

#     base_learner_outputs = []
#     base_learner_preweights = []
    
#     for model in self.models:
#         # 获取模型输出并应用 softmax
#         base_learner_output = F.softmax(model(img_), dim=1)  # [valid_depth, num_classes, H, W]
#         base_learner_outputs.append(base_learner_output)
        
#         # 计算每个批次的均值
#         sample_mean = torch.zeros(B, base_learner_output.size(1), H, W, device=img.device)
#         sample_mean = sample_mean.index_add(0, batch_idx, base_learner_output) / counts.view(B, 1, 1, 1)
        
#         # 计算每个批次的方差并应用指数函数
#         sample_var = sample_mean.var(dim=1)  # [B, H, W]
#         preweight = torch.exp(sample_var).unsqueeze(1).expand(-1, base_learner_output.size(1), -1, -1)  # [B, num_classes, H, W]
        
#         # 将权重映射回每个有效样本
#         preweight_samples = preweight[batch_idx]  # [valid_depth, num_classes, H, W]
#         base_learner_preweights.append(preweight_samples)

#     # 堆叠所有模型的输出和权重
#     base_learner_outputs = torch.stack(base_learner_outputs, dim=0)  # [num_models, valid_depth, num_classes, H, W]
#     base_learner_preweights = torch.stack(base_learner_preweights, dim=0)  # [num_models, valid_depth, num_classes, H, W]
    
#     # 计算所有模型权重的总和
#     sum_weight = torch.sum(base_learner_preweights, dim=0)  # [valid_depth, num_classes, H, W]
    
#     # 规范化每个模型的权重
#     model_weights = base_learner_preweights / sum_weight  # [num_models, valid_depth, num_classes, H, W]
    
#     # 计算加权最终输出
#     final_output = torch.sum(model_weights * base_learner_outputs, dim=0)  # [valid_depth, num_classes, H, W]
#     final_output = F.softmax(final_output, dim=1)
    
#     return final_output
    
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
    args = parser.parse_args()
    t = torch.rand(4, 10, 1, 224, 224).to(device='cuda:1')
    net = DE_framework_mem(args, models=[UNet_logvar(num_classes = 4, max_channels=256),
                                            DeepLabV3P_logvar(num_classes = 4, max_channels=256)]).to(device=t.device)
    pred = net(t)
    print(pred.size())

