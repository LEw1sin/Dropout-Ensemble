import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.modules.loss import CrossEntropyLoss

def make_one_hot(input, num_classes, device):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).to(device=device)
    result = result.scatter_(1, input, 1)
    return result

class MulticlassCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MulticlassCrossEntropyLoss, self).__init__()

    def forward(self, pred, target):
        """Compute multiclass cross entropy loss

        Args:
            pred: A tensor of shape [N, num_classes, *], raw probabilities from the model.
            target: A tensor of shape [N, 1, *], class indices.
            num_classes: An int, number of classes.
        
        Returns:
            A scalar tensor, the average cross entropy loss.
        """
        # Compute cross entropy loss
        loss = -torch.sum(target * torch.log(pred + 1e-9)) / pred.shape[0]
        return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-6, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Clamping values to avoid extreme cases
        predict = torch.clamp(predict, min=self.smooth, max=1 - self.smooth)
        target = torch.clamp(target, min=self.smooth, max=1 - self.smooth)

        num = torch.sum(predict * target, dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
            
        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    dice_loss = dice_loss * self.weight[i].item()
                total_loss = total_loss + dice_loss

        return total_loss/target.shape[1]

def multi_label_focal_loss(probs, targets, gamma=2.0, reduction='mean'):
    probs = torch.clamp(probs, min=1e-9)
    num_classes = probs.size(1)
    alpha = [0.2 for _ in range(num_classes)]
    log_probs = torch.log(probs)
    ce_loss = -targets * log_probs  
    pt = probs * targets + (1 - probs) * (1 - targets)  
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * ce_loss
    alpha = torch.tensor(alpha, dtype=probs.dtype, device=probs.device)
    alpha = alpha.view(1, -1, 1, 1)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    return loss

class Evaluator(object):
    def __init__(self, args, net, pred, label, label_one_hot, valid_mask, weights=None, train_eval=False):
        self.num_classes = args.num_classes
        self.train_eval=train_eval
        self.pred = pred
        self.label = label
        self.label_one_hot = label_one_hot
        self.valid_mask = valid_mask
        self.weights = weights
        # self.weights = None
        self.confuse_matrix = self.compute_batch_confuse_matrix()
        self.dice_coff = self.compute_dice_coff(self.confuse_matrix)
        self.net = net
        self.l1_lambda = 1e-5
        self.dice_loss_func = DiceLoss(weight=self.weights)

        
    def compute_batch_confuse_matrix(self):
        if self.train_eval:
            pred = self.pred.cpu().detach().numpy()
            label = self.label.cpu().detach().numpy()
            self.batch_size = label.shape[0]
        else:
            pred = self.pred
            label = self.label
            self.batch_size = self.label.shape[0]

        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        for i in range(self.batch_size):
            pred_image = np.argmax(pred[i], axis=0)  # (height, width)
            label_image = label[i]  # (height, width)
            confusion_matrix += self.compute_iter_confuse_matrix(pred_image, label_image)
        
        return confusion_matrix
    
    def compute_iter_confuse_matrix(self, pred_image, label_image):
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        for true_class in range(self.num_classes):
            for pred_class in range(self.num_classes):
                mask = (label_image == true_class) & (pred_image == pred_class)
                confusion_matrix[true_class, pred_class] += np.sum(mask)

        return confusion_matrix
    
    def compute_pa(self, confuse_matrix):
        pa = confuse_matrix.trace() / confuse_matrix.sum()
        return pa

    def compute_mpa(self, confuse_matrix):
        sums = confuse_matrix.sum(axis=1)
        mpa = np.divide(np.diag(confuse_matrix), sums, where=sums!=0)  
        mpa = np.nanmean(mpa)  
        return mpa
    
    def compute_miou(self, confuse_matrix):
        intersection = np.diag(confuse_matrix)
        union = (confuse_matrix.sum(1) + confuse_matrix.sum(0) - intersection)
        mIoU = np.divide(intersection, union, where=union!=0)  
        return mIoU
    
    def compute_end_coff(self, threshold):

        end_idx_list = [0, -1, 1, -2]
        true_counts = self.valid_mask.sum(dim=1).tolist() 
        split_indices = np.cumsum(true_counts)[:-1]  
        pred_splits = np.split(self.pred, split_indices, axis=0)
        label_splits = np.split(self.label, split_indices, axis=0)

        end_coff_e = 0

        for i, (pred_split, label_split) in enumerate(zip(pred_splits, label_splits)):
            # miou_case = 0
            dice_case = 0
            for idx in end_idx_list:
                # miou_case += self.compute_miou(self.compute_iter_confuse_matrix(np.argmax(pred_split[idx], axis=0), label_split[idx]))
                dice_case_idx = self.compute_dice_coff(self.compute_iter_confuse_matrix(np.argmax(pred_split[idx], axis=0), label_split[idx]))
                # dice_case_idx = [np.float64(1.0) if x == 0 else x for x in dice_case_idx]
                dice_case += self.dice_score(dice_case_idx)

            # if np.sum(miou_case) / len(miou_case) > threshold:
            if dice_case / len(end_idx_list) > threshold:
                end_coff_e += 1

        return end_coff_e , len(true_counts)
    
    def compute_dice_coff(self, confusion_matrix):
        dice_scores = []
        for i in range(self.num_classes):
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i, :]) - TP
            TN = np.sum(confusion_matrix) - (TP + FP + FN)

            dice_i = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else np.float64(1.0)
            dice_scores.append(dice_i)

        return dice_scores
    
    def dice_score(self, dice_case):
        return sum(dice_case) / len(dice_case)

    def compute_metrics(self):
        pa = self.compute_pa(self.confuse_matrix)
        mpa = self.compute_mpa(self.confuse_matrix)
        miou = self.compute_miou(self.confuse_matrix)
        dice = self.dice_score(self.dice_coff)
        return pa, mpa, miou, dice
    
    def dice_loss(self):
        dice_loss_val_weighted = self.dice_loss_func(self.pred, self.label_one_hot)
        return dice_loss_val_weighted
    
    def ce_loss(self):
        return F.binary_cross_entropy(self.pred, self.label_one_hot, weight=self.weights)
    
    def l1_loss(self):
        return sum(p.abs().sum() for p in self.net.parameters())
    
    def focal_loss(self):
        return multi_label_focal_loss(self.pred, self.label_one_hot)
    
    def mse_loss(self):
        return F.mse_loss(self.pred, self.label_one_hot)
    
    def loss(self):
        return self.dice_loss() + 10*self.focal_loss()