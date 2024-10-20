import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from utils.image_trans import ImageTransform
from utils.label_trans import LabelTransform
import logging
from tqdm import tqdm
from utils.metrics import *
from model.unet import UNet
from model.deeplabv3_plus import DeepLab_V3_plus
from model.DE_framework import DE_framework
from model.tools import dense_crf
import argparse
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from utils.initialize import setup_logging
from medpy.metric.binary import hd

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
    image_transform = transforms.Compose([
        ImageTransform(),
        lambda x: torch.tensor(x, dtype=torch.float32)
    ])
    label_transform = transforms.Compose([
        LabelTransform(),
        lambda x: torch.tensor(x, dtype=torch.float32)
    ])

    folder_path = args.acdc_path if args.dataset_mode == 'ACDC' else args.mm_path
    image_folder = os.path.join(folder_path, 'images')
    label_folder = os.path.join(folder_path, 'labels')
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.nii.gz')]
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    hist = np.zeros((4, 4))
    iter_num = 0
    dummy_iter = np.array([0, 0, 0])
    pa = 0
    mpa = 0
    miou = 0
    dice = 0

    hausdorff_distance = 0
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.nii.gz')]

    with torch.no_grad():
        for filename in tqdm(image_files,desc=f"Predicting slices from {image_folder}"):
            image_path = os.path.join(image_folder, filename)

            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)
            image_array = image_array.astype(np.float32)

            label_path = os.path.join(label_folder, filename)
            label = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label)
            label_array = label_array.astype(np.float32)

            image_array = image_transform(image_array)
            label_array = label_transform(label_array)

            mid_slice = image_array.shape[0] // 2
            for z_index in range(image_array.shape[0]):
                image_slice = image_array[z_index,:,:].unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
                label_slice = label_array[z_index,:,:].unsqueeze(0).to(device, dtype=torch.long)
                label_one_hot = make_one_hot(label_slice.unsqueeze(0), args.num_classes, device)
                if label_slice.max() == 0:
                    continue
 
                pred = net(image_slice)
                pred = pred.cpu().detach().numpy()
                pred = dense_crf(probs=pred, n_classes=4)
                pred = torch.from_numpy(pred).unsqueeze(0).to(device, dtype=torch.float32)
                iter_num += 1

                evaluator = Evaluator(args, net, pred, label_slice, label_one_hot)
                hist += evaluator.compute_confuse_matrix()
                label_slice = label_slice.squeeze(0).squeeze(0).cpu().detach().numpy()
                pa_e, mpa_e, miou_e, dice_e = evaluator.compute_metrics()

                pred = torch.argmax(pred, dim=1)
                pred = pred.squeeze(0).cpu().detach().numpy()

                hausdorff_distance_e , dummy_iter_e= compute_hausdorff_distance(pred, label_slice)
                pa += pa_e
                mpa += mpa_e
                miou += miou_e
                dice += dice_e

                hausdorff_distance += hausdorff_distance_e
                dummy_iter += dummy_iter_e

                if z_index == mid_slice:
                    plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(pred, cmap='gray')
                    ax = plt.subplot(1, 2, 2)
                    ax.imshow(label_slice, cmap='gray')

                    save_path = os.path.join(args.predicted_directory, filename.replace('.nii.gz', '.png'))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                    plt.close()

        pa, mpa, miou, dice = pa / iter_num, mpa / iter_num, miou / iter_num, dice / iter_num
        iter_num_expanded = np.full((1, 3), iter_num)
        iter_num_expanded -= dummy_iter
        hausdorff_distance = hausdorff_distance / iter_num_expanded
        logging.info("Pixel Accuracy = %.2f%%" % (pa * 100))
        logging.info('Mean Pixel Accuracy = %.2f%%' % (mpa * 100))
        logging.info("Mean iou = %.2f%%" % (np.sum(miou) * 100 / len(miou)))

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
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--device', default='cuda', type=str, help='device to use for training / testing')
    parser.add_argument('--net_weights', default='./ACDC_weights_3unet', type=str, help='net weights path')
    parser.add_argument('--log_file_path', default='./predict.log', type=str, help='training log path')
    parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
    parser.add_argument('--acdc_path', default='./ACDC/test', type=str, help='the directory of acdc')
    parser.add_argument('--mm_path', default='./MM/ED/test', type=str, help='the directory of mm') #change ED or ES
    parser.add_argument('--dataset_mode', default='ACDC', type=str, help='choose which dataset to use')
    parser.add_argument('--predicted_directory', default='./ACDC_predict', type=str, help='the directory of predicted directory')
    parser.add_argument('--max_channel', default=512, type=int, help='the maximum number of channels')
    args = parser.parse_args()
    setup_logging(args.log_file_path)
    weight_list = [0.33,0.33,0.33]
    net = DE_framework(models=[UNet(num_classes = args.num_classes, max_channels=args.max_channel,),
                               UNet(num_classes = args.num_classes, max_channels=args.max_channel,),
                                UNet(num_classes = args.num_classes, max_channels=args.max_channel,)],
                                         weight_list=weight_list)
    num_cuda_devices = torch.cuda.device_count()
    logging.info(f"Let's use {num_cuda_devices} GPUs!")
    main(args, net)