from utils.MedicalImageDataset import MedicalImageDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from utils.image_trans import ImageTransform
from utils.label_trans import LabelTransform
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.metrics import *
from model.DE_framework import DE_framework
from model.unet import UNet
from model.deeplabv3_plus import DeepLab_V3_plus
from utils.initialize import *
import torch.distributed as dist
from torch.optim import lr_scheduler
import argparse
import os

def main(args,net):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    net_weights = args.pretrained_weights
    args.lr *= args.world_size

    if rank == 0: 
        logging.info(args)

    # if there is a pretrained model, load it
    if os.path.exists(net_weights) and args.pretrained:
        net.load_model(net_weights)
        logging.info(f"load weights from {net_weights}")

    net = net.to(device)
    if args.world_size > 1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

    try:
        train_dataset, val_dataset = load_dataset(args)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    except Exception as e:
        logging.error(f"Error during DataLoader iteration: {e}", exc_info=True)

    pg = [p for p in net.parameters() if p.requires_grad]

    optimizer = torch.optim.RMSprop(pg, lr=lr, weight_decay=args.l2_norm, momentum=0.9)
    def lf(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return ((1 + math.cos((epoch - args.warmup_epochs) * math.pi / (args.epochs - args.warmup_epochs))) / 2) * (1 - args.lr) + args.lr
        
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    model_train_acc = []
    model_train_loss = []

    model_valid_acc = []
    model_valid_loss = []

    best_loss = float('inf')
    best_acc = 0
    patience = 0 # early stop patience
    for epoch in range(epochs):
        train_acc, train_losses = train_epoch(args, net, optimizer, train_loader, device, epoch, scheduler)
        model_train_acc.append(train_acc)
        model_train_loss.append(train_losses)

        valid_losses, valid_acc = valid_epoch(args, net, valid_loader, device, epoch)
        model_valid_loss.append(valid_losses)
        model_valid_acc.append(valid_acc)

        # select the best model
        if best_acc < valid_acc:
            best_acc = valid_acc
            logging.info(f'save the model in epoch{epoch+1}')
            net.save_model(args.net_weights)
            patience = 0
        else:
            patience += 1
            if patience > args.patience:
                logging.info(f'early stop in epoch{epoch+1}')
                break

def process_epoch(args, net, loader, device, mode, optimizer=None, scaler=None, scheduler=None):
    net.train() if mode == 'train' else net.eval()
    total_losses = 0
    total_acc = 0
    miou = 0
    iter_num = 0
    weights = torch.tensor(args.channel_weights).to(device=device)

    for step, data_batch in enumerate(loader):
        image, label = data_batch
        image = image.unsqueeze(1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
        label_one_hot = make_one_hot(label.unsqueeze(1), args.num_classes, device)

        with torch.set_grad_enabled(mode == 'train'):
            with torch.cuda.amp.autocast():
                pred = net(image)
                evaluator = Evaluator(args, net, pred, label, label_one_hot, weights=weights)
                loss = evaluator.loss()
                total_losses += loss.item()
                total_acc += evaluator.dice_score()
                miou += evaluator.compute_miou()

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

        del image, label, pred, label_one_hot, loss, evaluator
        torch.cuda.empty_cache()
        iter_num += 1

    if scheduler and mode == 'train':
        scheduler.step()

    total_losses /= iter_num
    total_acc /= iter_num
    miou /= iter_num

    return total_losses, total_acc, miou

def train_epoch(args, net, optimizer, train_loader, device, epoch, scheduler):
    scaler = torch.cuda.amp.GradScaler()

    if is_main_process():
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]", unit="batch")

    train_losses, train_acc, _= process_epoch(args, net, train_loader, device, 'train', optimizer, scaler, scheduler)

    if is_main_process():
        logging.info('train_loss:' + str(train_losses) + f' in epoch{epoch+1}')
        logging.info('train_acc:' + str(train_acc) + f' in epoch{epoch+1}')
        logging.info(f"Epoch {epoch+1}: Learning Rate: {scheduler.get_last_lr()}")

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return train_acc, train_losses

@torch.no_grad()
def valid_epoch(args, net, valid_loader, device, epoch):

    if is_main_process():
        valid_loader = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", unit="batch")

    valid_losses, valid_acc, miou = process_epoch(args, net, valid_loader, device, 'valid')

    if is_main_process():
        logging.info('valid_loss:' + str(valid_losses) + f' in epoch{epoch+1}')
        logging.info('valid_acc:' + str(valid_acc) + f' in epoch{epoch+1}')
        logging.info("Mean iou = %.2f%%" % (np.sum(miou) * 100 / len(miou)) + f' in epoch{epoch+1}')

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return valid_losses, valid_acc

def load_dataset(args):
    target_depth, target_height, target_width = args.target_depth, args.target_height, args.target_width
    image_transform = transforms.Compose([
        ImageTransform(target_depth, target_height, target_width),
        lambda x: torch.from_numpy(x)
    ])

    label_transform = transforms.Compose([
        LabelTransform(target_depth, target_height, target_width),
        lambda x: torch.from_numpy(x)
    ])

    try:
        if args.dataset_mode == 'ACDC':
            train_dataset = MedicalImageDataset(
                acdc_folder=args.acdc_path,
                dataset_mode = 'ACDC', 
                train_valid = 'train',
                target_depth=target_depth,
                target_height=target_height,
                target_width=target_width,
                image_transform=image_transform, 
                label_transform=label_transform,
                aug=args.aug
            )
            val_dataset = MedicalImageDataset(
                acdc_folder=args.acdc_path,
                dataset_mode = 'ACDC', 
                train_valid = 'valid',
                target_depth=target_depth,
                target_height=target_height,
                target_width=target_width,
                image_transform=image_transform, 
                label_transform=label_transform,
                aug=args.aug
            )
        elif args.dataset_mode == 'MM':
            train_dataset = MedicalImageDataset(
                mm_folder=args.mm_path,
                dataset_mode = 'MM', 
                train_valid = 'train',
                target_depth=target_depth,
                target_height=target_height,
                target_width=target_width,
                image_transform=image_transform, 
                label_transform=label_transform,
                aug=args.aug
            )
            val_dataset = MedicalImageDataset(
                mm_folder=args.mm_path,
                dataset_mode = 'MM', 
                train_valid = 'valid',
                target_depth=target_depth,
                target_height=target_height,
                target_width=target_width,
                image_transform=image_transform, 
                label_transform=label_transform,
                aug=args.aug
            )
        else:
            raise ValueError(f"dataset_mode {args.dataset_mode} is not supported")
        return train_dataset, val_dataset
    except Exception as e:
        logging.error(f"Error during dataset creation: {e}", exc_info=True)
            

if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--dist_url', type=str, help='dist url')
    parser.add_argument('--device', default='cuda', type=str, help='device to use for training / testing')
    parser.add_argument('--pretrained', default=False, type=bool, help='load pretrained model weights')
    parser.add_argument('--channel_weights', nargs='+', type=float, default=[1.0, 2.0, 1.0, 1.0], help='Channel weights')
    parser.add_argument('--batch_size', default=8, type=int, help='input batch size for training')
    parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
    parser.add_argument('--pretrained_weights', default='./MM_weights_3unet', type=str, help='pretrained weights path')
    parser.add_argument('--net_weights', default='./MM_weights_3unet', type=str, help='net weights path')
    parser.add_argument('--log_file_path', default='./training.log', type=str, help='training log path')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='training and testing epochs')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='training warmup epochs')
    parser.add_argument('--target_depth', default=21, type=int, help='the size of unified image depth')
    parser.add_argument('--target_height', default=256, type=int, help='the size of unified image height')
    parser.add_argument('--target_width', default=256, type=int, help='the size of unified image width')
    parser.add_argument('--acdc_path', default='./ACDC', type=str, help='the directory of acdc')
    parser.add_argument('--mm_path', default='./MM', type=str, help='the directory of mm')
    parser.add_argument('--dataset_mode', default='MM', type=str, help='choose which dataset to use')
    parser.add_argument('--max_channel', default=512, type=int, help='the maximum number of channels')
    parser.add_argument('--l2_norm', default=1e-7, type=float, help='the coefficient of l2 norm')
    parser.add_argument('--patience', default=15, type=float, help='the patience of early stop')
    parser.add_argument('--aug', default=True, type=bool, help='whether to use data augmentation')
    args = parser.parse_args()
    setup_logging(args.log_file_path)
    weight_list = [0.33,0.33,0.33]
    net = DE_framework(models=[UNet(num_classes = args.num_classes, max_channels=args.max_channel,),
                               UNet(num_classes = args.num_classes, max_channels=args.max_channel,),
                                UNet(num_classes = args.num_classes, max_channels=args.max_channel,)],
                                         weight_list=weight_list)
    # Deeplab_V3_plus(num_classes = args.num_classes)]
    num_cuda_devices = torch.cuda.device_count()
    logging.info(f"Let's use {num_cuda_devices} GPUs!")
    main(args, net)