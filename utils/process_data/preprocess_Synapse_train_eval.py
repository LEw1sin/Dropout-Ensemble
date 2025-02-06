from tqdm import tqdm
import SimpleITK as sitk
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import h5py
from scipy import ndimage
from scipy.ndimage import zoom
import random
import torch
import torch.nn.functional as F

original_labels = np.array([0, 1, 2, 3, 4, 6, 7, 8, 11])
new_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
max_label = 13
mapping_array = np.full(max_label + 1, fill_value=0, dtype=int)
mapping_array[original_labels] = new_labels

def random_rot_flip(image, label):
    image, label = np.transpose(image, (1, 2, 0)), np.transpose(label, (1, 2, 0))
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    image, label = np.transpose(image, (2, 0, 1)), np.transpose(label, (2, 0, 1))

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def normalize(self, image):
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return normalized_image

    def __call__(self, image, label):
        image_rot_flip, label_rot_flip = random_rot_flip(image, label)
        image_rotate, label_rotate = random_rotate(image, label)
        image, image_rot_flip, image_rotate = self.normalize(image), self.normalize(image_rot_flip), self.normalize(image_rotate)
        _, x_origin, y_origin = image.shape
        _, x_rot_flip, y_rot_flip = image_rot_flip.shape
        _, x_rotate, y_rotate = image_rotate.shape

        image = zoom(image, (1, self.output_size[0] / x_origin, self.output_size[1] / y_origin), order=3) 
        label = zoom(label, (1, self.output_size[0] / x_origin, self.output_size[1] / y_origin), order=0)
        image_rot_flip = zoom(image_rot_flip, (1, self.output_size[0] / x_rot_flip, self.output_size[1] / y_rot_flip), order=3) 
        label_rot_flip = zoom(label_rot_flip, (1, self.output_size[0] / x_rot_flip, self.output_size[1] / y_rot_flip), order=0)
        image_rotate = zoom(image_rotate, (1, self.output_size[0] / x_rotate, self.output_size[1] / y_rotate), order=3)
        label_rotate = zoom(label_rotate, (1, self.output_size[0] / x_rotate, self.output_size[1] / y_rotate), order=0)

        image = np.concatenate((image, image_rot_flip, image_rotate), axis=0)
        label = np.concatenate((label, label_rot_flip, label_rotate), axis=0).astype(int)

        filtered_label = mapping_array[label]

        return image, filtered_label

Abdomen_img_path = '/data2/lyw/Abdomen/RawData/Training/img'
train_split_txt_path = '../lists_Synapse/train.txt'
eval_vol_split_txt_path = '../lists_Synapse/eval_vol.txt'

train_path = '/data2/lyw/data/preprocessed_Synapse/train'
eval_path = '/data2/lyw/data/preprocessed_Synapse/eval'

Abdomen_label_path = Abdomen_img_path.replace('img', 'label')
train_eval_img_files = os.listdir(Abdomen_img_path)

# Split the list into two parts
train_img_files = []
eval_img_files = []

with open(train_split_txt_path, 'r') as file:
    train_name = [line.strip().split('_')[0] for line in file]
    for train_img_file in train_eval_img_files:
        t1 = train_img_file.split('.')
        t2 = t1[0][-4:]
        if 'case'+t2 in train_name: 
            train_img_files.append(train_img_file)

with open(eval_vol_split_txt_path, 'r') as file:
    eval_name = [line.strip().split('_')[0] for line in file]
    for eval_img_file in train_eval_img_files:
        t1 = eval_img_file.split('.')
        t2 = t1[0][-4:]
        if 'case'+t2 in eval_name:
            eval_img_files.append(eval_img_file)

# Train and evaluation files with '_seg'
train_files_seg = [file.replace('img', 'label') for file in train_img_files]
eval_files_seg = [file.replace('img', 'label') for file in eval_img_files]

train_h5_dir_path = f'{train_path}_h5'
eval_h5_dir_path = f'{eval_path}_h5'

os.makedirs(train_h5_dir_path, exist_ok=True)
os.makedirs(eval_h5_dir_path, exist_ok=True)

transfrom = RandomGenerator([224, 224])
train_sample_num = 0
eval_sample_num = 0
     
for origin_path, seg_path in tqdm(zip(train_img_files, train_files_seg), total=len(train_img_files)):
    origin = sitk.ReadImage(os.path.join(Abdomen_img_path, origin_path))
    origin_array = sitk.GetArrayFromImage(origin)
    origin_array = origin_array.astype(np.float32)

    seg = sitk.ReadImage(os.path.join(Abdomen_label_path, seg_path))
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array = seg_array.astype(np.float32)

    origin_array, seg_array = transfrom(origin_array, seg_array)

    z_len = origin_array.shape[0]
    train_sample_num += z_len
    for slice_idx in range(z_len):
        file_h5_name = origin_path.split('.')[0].replace('img','case') + '_slice' + str(slice_idx) +'.h5'
        with h5py.File(os.path.join(train_h5_dir_path, file_h5_name), 'w') as f:
            f.create_dataset('image', data=origin_array[slice_idx])
            f.create_dataset('label', data=seg_array[slice_idx])    


for origin_path, seg_path in tqdm(zip(eval_img_files, eval_files_seg), total=len(eval_img_files)):
    origin = sitk.ReadImage(os.path.join(Abdomen_img_path, origin_path))
    origin_array = sitk.GetArrayFromImage(origin)
    origin_array = origin_array.astype(np.float32)

    seg = sitk.ReadImage(os.path.join(Abdomen_label_path, seg_path))
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array = seg_array.astype(np.float32)

    origin_array, seg_array = transfrom(origin_array, seg_array)

    z_len = origin_array.shape[0]
    eval_sample_num += z_len
    for slice_idx in range(z_len):
        file_h5_name = origin_path.split('.')[0].replace('img','case') + '_slice' + str(slice_idx) +'.h5'
        with h5py.File(os.path.join(eval_h5_dir_path, file_h5_name), 'w') as f:
            f.create_dataset('image', data=origin_array[slice_idx])
            f.create_dataset('label', data=seg_array[slice_idx])    

print(f'Train sample number: {train_sample_num}')
print(f'Evaluation sample number: {eval_sample_num}')