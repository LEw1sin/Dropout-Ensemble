import torch
import numpy as np
import torch.nn.functional as F
import logging
from tqdm import tqdm
from utils.metrics import *
from model.unet import UNet_linear, UNet_logvar
from model.deeplabv3_plus import DeepLabV3P_linear, DeepLabV3P_logvar
from model.DE_framework import DE_framework_linear, DE_framework_logvar, DE_framework_mem, DE_framework_Augmenting
from model.tools import dense_crf
import argparse
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from utils.initialize import setup_logging
from medpy.metric.binary import hd
from utils.initialize import get_log_dir
from utils.cmr_dataset import cmr_dataset
from torch.utils.data import DataLoader

def compute_hausdorff_distance(pred, label_slice):
    hausdorff_distances = []
    dummy_iter = np.array([0, 0, 0])
    for i in range(1, 4):
        class_slice = np.zeros_like(pred)
        class_slice[pred == i] = pred[pred == i] 
        class_label = np.zeros_like(label_slice)
        class_label[label_slice == i] = label_slice[label_slice == i]
        if class_label.max() != 0 and class_slice.max() != 0:
            hausdorff_distance_class = hd(class_slice, class_label)
            hausdorff_distances.append(hausdorff_distance_class)
        else:
            dummy_iter[i-1] += 1
            hausdorff_distance_class = 0
            hausdorff_distances.append(hausdorff_distance_class)

    hausdorff_distances = np.array(hausdorff_distances).astype(np.float32)
    
    return hausdorff_distances, dummy_iter

def visualize(args, pred, label, valid_mask, file_list, viz_idx_list=[0, 'mid', -1, 1, -2]):
    true_counts = valid_mask.sum(dim=1).tolist() 
    split_indices = np.cumsum(true_counts)[:-1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    pred_splits = np.split(pred, split_indices, axis=0)
    label_splits = np.split(label, split_indices, axis=0)
    for i, (pred_split, label_split) in enumerate(zip(pred_splits, label_splits)):
        middle_idx = pred_split.shape[0] // 2
        viz_idx_list[1] = middle_idx
        for idx in viz_idx_list:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(pred_split[idx], cmap='gray')
            ax = plt.subplot(1, 2, 2)
            ax.imshow(label_split[idx], cmap='gray')

            save_path = os.path.join(args.log_dir, 'img/',file_list[0].replace('.h5', f'_slice_{idx}.png'))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        file_list.pop(0)
    return file_list

def main(args,net):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for predicting.")

    net_weights = args.net_weights
    device = torch.device(args.device)
    
    if os.path.exists(net_weights):
        net.load_model(net_weights)
        logging.info(f"load weights from {net_weights}")
    else:
        raise EnvironmentError("not find model weights.")

    net = net.to(device)
    net.eval()

    hist = np.zeros((4, 4))
    iter_num = 0
    dummy_iter = np.array([0, 0, 0])
    pa = 0
    mpa = 0
    miou = 0
    dice = 0
    end_coff = 0
    batch_num = 0

    hausdorff_distance = 0
    val_dataset = cmr_dataset(data_dir=f'../../de_data/preprocessed_{args.dataset_mode}/test', cache=args.cache, ed_es_only=args.ed_es_only)
    file_list = val_dataset.files
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    weights = torch.tensor(args.channel_weights).to(device=device)

    with torch.no_grad():
        for step, data_batch in enumerate(tqdm(valid_loader, desc="Validation Progress")):
            image, label, mask = data_batch
            valid_mask = mask > 0
            image = image.to(device=device, dtype=torch.float32)
            label = label[valid_mask]
            label = label.to(device=device, dtype=torch.long)
            label_one_hot = make_one_hot(label.unsqueeze(1), args.num_classes, device)
            label = label.cpu().detach().numpy()
 
            pred = net(image, valid_mask)
            pred_argmax = torch.argmax(pred, dim=1).cpu().detach().numpy()
            if args.visualize:
                file_list = visualize(args, pred_argmax, label, valid_mask, file_list)
            pred = pred.cpu().detach().numpy()
            # pred = dense_crf(probs=pred, n_classes=4)
            iter_num += 1

            evaluator = Evaluator(args, net, pred, label, label_one_hot, valid_mask, weights=weights)
            hist += evaluator.confuse_matrix
            pa_e, mpa_e, miou_e, dice_e = evaluator.compute_metrics()
            end_coff_e, batch_num_e = evaluator.compute_end_coff(args.end_coff_threshold)

            hausdorff_distance_e , dummy_iter_e= compute_hausdorff_distance(pred_argmax, label)
            pa += pa_e
            mpa += mpa_e
            miou += miou_e
            dice += dice_e
            end_coff += end_coff_e
            batch_num += batch_num_e

            hausdorff_distance += hausdorff_distance_e
            dummy_iter += dummy_iter_e

        pa, mpa, miou, dice, end_coff = pa / iter_num, mpa / iter_num, miou / iter_num, dice / iter_num, end_coff / batch_num
        iter_num_expanded = np.full((1, 3), iter_num)
        iter_num_expanded -= dummy_iter
        hausdorff_distance = hausdorff_distance / iter_num_expanded
        logging.info("Pixel Accuracy = %.2f%%" % (pa * 100))
        logging.info('Mean Pixel Accuracy = %.2f%%' % (mpa * 100))
        logging.info("Mean iou = %.2f%%" % (np.sum(miou) * 100 / len(miou)))
        logging.info("End Coff = %.2f%%" % (end_coff * 100))

        precision = []
        recall = []
        accuracy = []
        dice = []  #
        for i in range(args.num_classes):
            TP = hist[i, i]
            FP = np.sum(hist[:, i]) - TP
            FN = np.sum(hist[i, :]) - TP
            TN = np.sum(hist) - (TP + FP + FN)
            
            precision_i = TP / (TP + FP) if (TP + FP) != 0 else 0
            precision.append(precision_i)
            
            recall_i = TP / (TP + FN) if (TP + FN) != 0 else 0
            recall.append(recall_i)
            
            accuracy_i = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
            accuracy.append(accuracy_i)
            
            dice_i = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
            dice.append(dice_i)

        logging.info(f"BG: precision= {precision[0] * 100:.2f}, recall: {recall[0] * 100:.2f}, accuracy: {accuracy[0] * 100:.2f}, dice: {dice[0] * 100:.2f}")
        logging.info(f"RV: precision= {precision[1] * 100:.2f}, recall: {recall[1] * 100:.2f}, accuracy: {accuracy[1] * 100:.2f}, dice: {dice[1] * 100:.2f}, HD: {hausdorff_distance[0][0].item():.2f}")
        logging.info(f"MYO: precision= {precision[2] * 100:.2f}, recall: {recall[2] * 100:.2f}, accuracy: {accuracy[2] * 100:.2f}, dice: {dice[2] * 100:.2f}, HD: {hausdorff_distance[0][1].item():.2f}")
        logging.info(f"LV: precision= {precision[3] * 100:.2f}, recall: {recall[3] * 100:.2f}, accuracy: {accuracy[3] * 100:.2f}, dice: {dice[3] * 100:.2f}, HD: {hausdorff_distance[0][2].item():.2f}")
        logging.info("Dice Coeff = %.2f%%" % (((dice[1]+dice[2]+dice[3]) / 3)* 100))


if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    parser = argparse.ArgumentParser(description="Predicting")
    parser.add_argument('--device', default='cuda:1', type=str, help='device to use for testing')
    parser.add_argument('--net_weights', default='../../de_logistics/ACDC_2UNetlinear_02-05-01-00-04', type=str, help='net weights path')
    parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
    parser.add_argument('--dataset_mode', default='ACDC', type=str, help='choose which dataset to use')
    parser.add_argument('--max_channel', default=256, type=int, help='the maximum number of channels')
    parser.add_argument('--cache', default=True, type=bool, help='whether to load the dataset into memory')
    parser.add_argument('--batch_size', default=4, type=int, help='input batch size for testing')
    parser.add_argument('--channel_weights', nargs='+', type=float, default=[1.0, 2.0, 1.0, 1.0], help='Channel weights')
    parser.add_argument('--visualize', default=True, type=bool, help='whether to visualize the prediction')
    parser.add_argument('--end_coff_threshold', default=0.8, type=float, help='threshold for end slices confidence')
    parser.add_argument('--ed_es_only', default='', type=str, help='test on ED or ES slices only')

    args = parser.parse_args()
    # net = DE_framework_mem(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel),
    #                                         DeepLabV3P_linear(num_classes = args.num_classes, max_channels=args.max_channel)])
    # net = DE_framework_linear(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel),
    #                                         DeepLabV3P_linear(num_classes = args.num_classes, max_channels=args.max_channel)], weight_list=[0.7,0.3])
    # net = DE_framework_Augmenting(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel)], weight_list=[1])
    net = DE_framework_mem(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channe),
                                            UNet_linear(num_classes = args.num_classes, max_channels=args.max_channe),
                                            UNet_linear(num_classes = args.num_classes, max_channels=args.max_channe)])
    args.log_dir = get_log_dir(net, args.dataset_mode, train_eval='eval', ed_es_only=args.ed_es_only)
    args.log_file_path = os.path.join(args.log_dir, "predict.log")
    setup_logging(args.log_file_path)
    main(args, net)