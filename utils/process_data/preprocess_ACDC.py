from tqdm import tqdm
import SimpleITK as sitk
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import h5py
from aug_utils import *

sitk.ProcessObject_SetGlobalWarningDisplay(False)
original_ACDC_image_path = '../../../de_data/ACDC/images'
original_ACDC_label_path = '../../../de_data/ACDC/labels'

train_image_path = original_ACDC_image_path + '/train'
train_label_path = original_ACDC_label_path + '/train'
eval_image_path = original_ACDC_image_path + '/eval'
eval_label_path = original_ACDC_label_path + '/eval'

target_train_path = '../../../de_data/preprocessed_ACDC/train'
target_eval_path = '../../../de_data/preprocessed_ACDC/eval'

os.makedirs(target_train_path, exist_ok=True)
os.makedirs(target_eval_path, exist_ok=True)
random_gen = RandomGenerator((224, 224))

for train_image_name, train_label_name in tqdm(zip(os.listdir(train_image_path), os.listdir(train_label_path)), total=len(os.listdir(train_image_path))):
    train_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(train_image_path, train_image_name)))
    train_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(train_label_path, train_label_name)))

    train_image = train_image.astype(np.float32)
    train_label = train_label.astype(np.int32)

    image, label, image_rot_flip, label_rot_flip, image_rotate, label_rotate = random_gen(train_image, train_label)
    image, mask = padding(image, 30)
    label, _ = padding(label, 30)
    image_rot_flip, _ = padding(image_rot_flip, 30)
    label_rot_flip, _ = padding(label_rot_flip, 30)
    image_rotate, _ = padding(image_rotate, 30)
    label_rotate, _ = padding(label_rotate, 30)  

    with h5py.File(os.path.join(target_train_path, train_image_name.replace('.nii.gz', '.h5')), 'w') as f:
        f.create_dataset('image', data=image)
        f.create_dataset('label', data=label)
        f.create_dataset('mask', data=mask)

    with h5py.File(os.path.join(target_train_path, train_image_name.replace('.nii.gz', '_rot_flip.h5')), 'w') as f:
        f.create_dataset('image', data=image_rot_flip)
        f.create_dataset('label', data=label_rot_flip)
        f.create_dataset('mask', data=mask)
    
    with h5py.File(os.path.join(target_train_path, train_image_name.replace('.nii.gz', '_rotate.h5')), 'w') as f:
        f.create_dataset('image', data=image_rotate)
        f.create_dataset('label', data=label_rotate)
        f.create_dataset('mask', data=mask)

for eval_image_name, eval_label_name in tqdm(zip(os.listdir(eval_image_path), os.listdir(eval_label_path)), total=len(os.listdir(eval_image_path))):
    eval_image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(eval_image_path, eval_image_name)))
    eval_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(eval_label_path, eval_label_name)))

    eval_image = eval_image.astype(np.float32)
    eval_label = eval_label.astype(np.int32)
    image, label, image_rot_flip, label_rot_flip, image_rotate, label_rotate = random_gen(eval_image, eval_label)
    image, mask = padding(image, 30)
    label, _ = padding(label, 30)

    with h5py.File(os.path.join(target_eval_path, eval_image_name.replace('.nii.gz', '.h5')), 'w') as f:
        f.create_dataset('image', data=image)
        f.create_dataset('label', data=label)
        f.create_dataset('mask', data=mask)


def copy_folder_contents(src_folder, dst_folder):
    import shutil
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)  

    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(dst_folder, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)  
        else:
            shutil.copy2(src_path, dst_path)  # 

copy_folder_contents(target_eval_path, target_eval_path.replace('eval', 'test'))


