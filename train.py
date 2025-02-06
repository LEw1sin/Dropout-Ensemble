from utils.cmr_dataset import cmr_dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.metrics import *
from model.DE_framework import DE_framework_linear, DE_framework_logvar, DE_framework_mem
from model.unet import UNet_linear, UNet_logvar
from model.deeplabv3_plus import DeepLabV3P_linear, DeepLabV3P_logvar
from utils.initialize import *
from torch.optim import lr_scheduler
import argparse
import os
import wandb
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main(args,net):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    device = torch.device(args.device)
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    net_weights = args.pretrained_weights
    logging.info(args)

    # if there is a pretrained model, load it
    if os.path.exists(net_weights) and args.pretrained:
        net.load_model(net_weights)
        logging.info(f"load weights from {net_weights}")

    net = net.to(device)

    try:
        train_dataset = cmr_dataset(data_dir=f'../../de_data/preprocessed_{args.dataset_mode}/train', cache=args.cache)
        val_dataset = cmr_dataset(data_dir=f'../../de_data/preprocessed_{args.dataset_mode}/eval', cache=args.cache)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
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
            net.save_model(args.log_dir)
            patience = 0
        else:
            patience += 1
            # if patience > args.patience:
            #     logging.info(f'early stop in epoch{epoch+1}')
            #     break

def process_epoch(args, net, loader, device, mode, optimizer=None, scaler=None, scheduler=None):
    net.train() if mode == 'train' else net.eval()
    total_losses = 0
    total_acc = 0
    miou = 0
    iter_num = 0
    weights = torch.tensor(args.channel_weights).to(device=device)

    for step, data_batch in enumerate(loader):
        image, label, mask = data_batch
        valid_mask = mask > 0
        image = image.to(device=device, dtype=torch.float32)
        label = label[valid_mask]
        label = label.to(device=device, dtype=torch.long)
        label_one_hot = make_one_hot(label.unsqueeze(1), args.num_classes, device)

        with torch.set_grad_enabled(mode == 'train'):
            with torch.amp.autocast(device_type='cuda'):
                pred = net(image, valid_mask)
                evaluator = Evaluator(args, net, pred, label, label_one_hot, valid_mask, weights=weights, train_eval=True)
                loss = evaluator.loss()
                total_losses += loss.item()
                total_acc += evaluator.dice_score(evaluator.dice_coff)
                miou += evaluator.compute_miou(evaluator.confuse_matrix)

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

        # del image, label, pred, label_one_hot, loss, evaluator
        # torch.cuda.empty_cache()
        iter_num += 1

    if scheduler and mode == 'train':
        scheduler.step()

    total_losses /= iter_num
    total_acc /= iter_num
    miou /= iter_num

    return total_losses, total_acc, miou

def train_epoch(args, net, optimizer, train_loader, device, epoch, scheduler):
    scaler = torch.amp.GradScaler()

    train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]", unit="batch")

    train_losses, train_acc, _= process_epoch(args, net, train_loader, device, 'train', optimizer, scaler, scheduler)

    logging.info('train_loss:' + str(train_losses) + f' in epoch{epoch+1}')
    logging.info('train_acc:' + str(train_acc) + f' in epoch{epoch+1}')
    logging.info(f"Epoch {epoch+1}: Learning Rate: {scheduler.get_last_lr()}")
    if args.wandb:
        wandb.log({'train_loss': train_losses, 'train_acc': train_acc}, step=epoch)
        wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch)

    return train_acc, train_losses

@torch.no_grad()
def valid_epoch(args, net, valid_loader, device, epoch):

    valid_loader = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", unit="batch")

    valid_losses, valid_acc, miou = process_epoch(args, net, valid_loader, device, 'valid')

    logging.info('valid_loss:' + str(valid_losses) + f' in epoch{epoch+1}')
    logging.info('valid_acc:' + str(valid_acc) + f' in epoch{epoch+1}')
    logging.info("Mean iou = %.2f%%" % (np.sum(miou) * 100 / len(miou)) + f' in epoch{epoch+1}')

    if args.wandb:
        wandb.log({'valid_loss': valid_losses, 'valid_acc': valid_acc}, step=epoch)
        wandb.log({'miou': np.sum(miou) * 100 / len(miou)}, step=epoch)

    return valid_losses, valid_acc

            

if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--device', default='cuda:1', type=str, help='device to use for training / testing')
    parser.add_argument('--channel_weights', nargs='+', type=float, default=[1.0, 2.0, 1.0, 1.0], help='Channel weights')
    parser.add_argument('--batch_size', default=2, type=int, help='input batch size for training')
    parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
    parser.add_argument('--pretrained', default=False, type=bool, help='load pretrained model weights')
    parser.add_argument('--pretrained_weights', default='./Synapse_weights_1u1d', type=str, help='pretrained weights path')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epochs', default=500, type=int, help='training and testing epochs')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='training warmup epochs')
    parser.add_argument('--dataset_mode', default='ACDC', type=str, help='choose which dataset to use')
    parser.add_argument('--max_channel', default=256, type=int, help='the maximum number of channels')
    parser.add_argument('--l2_norm', default=1e-7, type=float, help='the coefficient of l2 norm')
    parser.add_argument('--patience', default=15, type=float, help='the patience of early stop')
    parser.add_argument('--wandb', default=True, type=bool, help='whether to use wandb')
    parser.add_argument('--cache', default=True, type=bool, help='whether to load the dataset into memory')
    parser.add_argument('--ed_es_only', default='', type=str, help='test on ED or ES slices only')
    args = parser.parse_args()

    # weight_list = [0.33,0.33,0.33]
    # net = DE_framework(models=[UNet(num_classes = args.num_classes, max_channels=args.max_channel,),
    #                            UNet(num_classes = args.num_classes, max_channels=args.max_channel,),
    #                             UNet(num_classes = args.num_classes, max_channels=args.max_channel,)],
    #                                      weight_list=weight_list)
    # weight_list = [0.6, 0.4]
    # net = DE_framework(args, models=[UNet(num_classes = args.num_classes, max_channels=args.max_channel,)
    #                            ,DeepLab_V3_plus(num_classes = args.num_classes)],
    #                                      weight_list=weight_list)
    # weight_list = [0.5, 0.5]
    # net = DE_framework_linear(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel),
    #                                         UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel),],
    #                                      weight_list=weight_list)
    # net = DE_framework_linear(args, models=[DeepLab_V3_plus_linear(num_classes = args.num_classes, max_channels=args.max_channel,)],
    #                                      weight_list=weight_list)
    # net = DE_framework_mem(args, models=[UNet_linear(num_classes = 4, max_channels=256),
    #                                         DeepLabV3P_linear(num_classes = 4, max_channels=256)])
    net = DE_framework_linear(args, models=[UNet_linear(num_classes = 4, max_channels=256),
                                            DeepLabV3P_linear(num_classes = 4, max_channels=256)], weight_list=[0.7,0.3])
    # net = DE_framework_mem(args, models=[UNet_linear(num_classes = 4, max_channels=256),
    #                                         UNet_linear(num_classes = 4, max_channels=256),
    #                                         UNet_linear(num_classes = 4, max_channels=256),])
    
    # net = DE_framework_mem(args, models=[UNet_linear(num_classes = 4, max_channels=256),
    #                                         UNet_linear(num_classes = 4, max_channels=256),
    #                                         UNet_linear(num_classes = 4, max_channels=256),
    #                                         UNet_linear(num_classes = 4, max_channels=256),])
    
    args.log_dir = get_log_dir(net, args.dataset_mode)
    args.log_file_path = os.path.join(args.log_dir, "training.log")
    setup_logging(args.log_file_path)
    num_cuda_devices = torch.cuda.device_count()
    logging.info(f"Let's use {num_cuda_devices} GPUs!")
    if args.wandb:
        wandb.init(project='DEUNET', config=args, name=args.log_dir.split('/')[-2])
    main(args, net)
