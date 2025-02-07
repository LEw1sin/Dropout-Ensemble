import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from aug_utils import *
import numpy as np
import h5py
import SimpleITK as sitk
from tqdm import tqdm
import nibabel as nib

def is_orthonormal(matrix):
    identity_matrix = np.eye(3)
    matrix = np.array(matrix).reshape(3, 3)
    return np.allclose(np.dot(matrix.T, matrix), identity_matrix)

def read_image(image_path):
    try:
        image = sitk.ReadImage(image_path)
        direction = image.GetDirection()

        if is_orthonormal(direction):
            return sitk.GetArrayFromImage(image)
        else:
            img = nib.load(image_path)  
            return img.get_fdata()

    except Exception as e:
        img = nib.load(image_path) 
        img = img.get_fdata()
        img = img.transpose(3, 2, 1, 0)
        return img

def list_leaf_directories(directory):
    image_dirs = []
    label_dirs = []
    for root, dirs, files in os.walk(directory):
        if not dirs:  
            for file in files:
                if 'gt' not in file:
                    image_dirs.append(os.path.join(root, file))
                else:
                    label_dirs.append(os.path.join(root, file))
    return image_dirs, label_dirs

def process_train_eval_test(original_path, target_path, ref_dict, aug=False):
    image_dirs, label_dirs = list_leaf_directories(original_path)
    os.makedirs(target_path, exist_ok=True)
    random_gen = RandomGenerator((224, 224))
    for image_name, label_name in tqdm(zip(image_dirs, label_dirs), total=len(image_dirs)):
        case_name = os.path.basename(os.path.dirname(image_name))
        image = read_image(image_name)
        label = read_image(label_name)

        image = image.astype(np.float32)
        label = label.astype(np.int32)

        for i in range(image.shape[0]):
            label_i = label[i]
            if np.max(label_i) == 0 and i != ref_dict[case_name]['ED'] and i != ref_dict[case_name]['ES']:
                continue

            image_i = image[i]
            label_i[label_i == 3] = -1  
            label_i[label_i == 1] = 3
            label_i[label_i == -1] = 1  

            image_i, label_i, image_i_rot_flip, label_i_rot_flip, image_i_rotate, label_i_rotate = random_gen(image_i, label_i)
            image_i, mask = padding(image_i, 30)
            label_i, _ = padding(label_i, 30)
            image_i_rot_flip, _ = padding(image_i_rot_flip, 30)
            label_i_rot_flip, _ = padding(label_i_rot_flip, 30)
            image_i_rotate, _ = padding(image_i_rotate, 30)
            label_i_rotate, _ = padding(label_i_rotate, 30)

            output_name = os.path.join(target_path, case_name + '_slice' + str(i))
            if i == ref_dict[case_name]['ED']:
                output_name = output_name + '_ED'
            elif i == ref_dict[case_name]['ES']:
                output_name = output_name + '_ES'

            with h5py.File(output_name + '.h5', 'w') as f:
                f.create_dataset('image', data=image_i)
                f.create_dataset('label', data=label_i)
                f.create_dataset('mask', data=mask)

            if aug:
                with h5py.File(output_name  + '_rot_flip.h5', 'w') as f:
                    f.create_dataset('image', data=image_i_rot_flip)
                    f.create_dataset('label', data=label_i_rot_flip)
                    f.create_dataset('mask', data=mask)
                
                with h5py.File(output_name +  '_rotate.h5', 'w') as f:
                    f.create_dataset('image', data=image_i_rotate)
                    f.create_dataset('label', data=label_i_rotate)
                    f.create_dataset('mask', data=mask)

ref = pd.read_csv('../../../de_data/MnMs/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
ref_dict = {}
for index, row in ref.iterrows():
    ref_dict[row['External code']] = {'ED': row['ED'], 'ES': row['ES']}

original_train_path = '../../../de_data/MnMs/Training' 
target_train_path = '../../../de_data/preprocessed_MnMs/train'   
process_train_eval_test(original_train_path, target_train_path, ref_dict, aug=True)

original_eval_path = '../../../de_data/MnMs/Validation'
target_eval_path = '../../../de_data/preprocessed_MnMs/eval'
process_train_eval_test(original_eval_path, target_eval_path, ref_dict)

original_test_path = '../../../de_data/MnMs/Testing'
target_test_path = '../../../de_data/preprocessed_MnMs/test'
process_train_eval_test(original_test_path, target_test_path, ref_dict)


