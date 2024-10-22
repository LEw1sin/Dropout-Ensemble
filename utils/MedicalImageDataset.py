import torch
import os
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import gc  
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MedicalImageDataset(Dataset):
    def __init__(self, acdc_folder=None, mm_folder=None, dataset_mode = 'ACDC', train_valid = 'train',
                 target_depth=20, target_height=256, target_width=256,
                 image_transform=None, label_transform=None, aug=True):

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.aug = aug
        self.target_depth, self.target_height, self.target_width = target_depth, target_height, target_width
        image_shape = (target_height, target_width)
        self.image_slices_index = 0
        self.label_slices_index = 0
        self.argumentation_type = []
        self.dataset_mode = dataset_mode

        try:
            if self.dataset_mode == 'ACDC':
                acdc_folder = os.path.join(acdc_folder, train_valid)
                self.acdc_image_folder = os.path.join(acdc_folder, 'images')
                self.acdc_label_folder = os.path.join(acdc_folder, 'labels')
                total_slices = self.cal_file_nums(self.acdc_image_folder)
                total_slices *= 3 if aug else 1
                self.image_slices = torch.empty((total_slices,) + image_shape, dtype=torch.float32)
                self.label_slices = torch.empty((total_slices,) + image_shape, dtype=torch.float32)
                self._load_silices(self.acdc_image_folder, self.acdc_label_folder)
            elif self.dataset_mode == 'MM':
                self.mm_ED_folder = os.path.join(mm_folder, 'ED')
                self.mm_ED_folder = os.path.join(self.mm_ED_folder, train_valid)
                self.mm_ED_image_folder = os.path.join(self.mm_ED_folder, 'images')
                self.mm_ED_label_folder = os.path.join(self.mm_ED_folder, 'labels')

                self.mm_ES_folder = os.path.join(mm_folder, 'ES')
                self.mm_ES_folder = os.path.join(self.mm_ES_folder, train_valid)
                self.mm_ES_image_folder = os.path.join(self.mm_ES_folder, 'images')
                self.mm_ES_label_folder = os.path.join(self.mm_ES_folder, 'labels')
                total_slices = self.cal_file_nums(self.mm_ED_image_folder)
                total_slices += self.cal_file_nums(self.mm_ES_image_folder)
                total_slices *= 3 if aug else 1
                self.image_slices = torch.empty((total_slices,) + image_shape, dtype=torch.float32)
                self.label_slices = torch.empty((total_slices,) + image_shape, dtype=torch.float32)
                self._load_silices(self.mm_ED_image_folder, self.mm_ED_label_folder)
                self._load_silices(self.mm_ES_image_folder, self.mm_ES_label_folder)
            else:
                raise ValueError('Invalid mode')
        except Exception as e:
            logging.error(f"Error during file loading: {e}")
            raise
        
    def _load_silices(self, image_folder=None, label_folder=None):
        try:
            image_files = [f for f in os.listdir(image_folder) if f.endswith('.nii.gz')]
            num_files = len(image_files)
            sitk.ProcessObject_SetGlobalWarningDisplay(False)

            with tqdm(total=num_files, desc=f"Loading slices from {image_folder}", unit="file") as pbar:
                for filename in image_files:
                    image_path = os.path.join(image_folder, filename)
                    image = sitk.ReadImage(image_path)
                    image_array = sitk.GetArrayFromImage(image)
                    image_array = image_array.astype(np.float32)
 
                    if self.dataset_mode == 'MM':
                        filename = filename.replace('.nii.gz', '_gt.nii.gz')
                    label_path = os.path.join(label_folder, filename)
                    label = sitk.ReadImage(label_path)
                    label_array = sitk.GetArrayFromImage(label)
                    label_array = label_array.astype(np.float32)


                    image_array = self.size_correction(image_array)
                    label_array = self.size_correction(label_array)

                    if self.image_transform is not None:
                        image_array = self.image_transform(image_array)
                    if self.label_transform is not None:
                        label_array = self.label_transform(label_array)
                    for slice in image_array:
                        self.image_slices[self.image_slices_index] = slice
                        self.image_slices_index = self.image_slices_index + 1
                    for slice in label_array:
                        self.label_slices[self.label_slices_index] = slice
                        self.label_slices_index = self.label_slices_index + 1
                        self.argumentation_type.append('original')

                    if self.aug:
                        # Apply affine and shear transformations
                        affine_image, affine_label = self.augment(image_array, label_array, apply_affine=True, apply_shear=False)
                        for slice in affine_image:
                            self.image_slices[self.image_slices_index] = slice
                            self.image_slices_index = self.image_slices_index + 1
                        for slice in affine_label:
                            self.label_slices[self.label_slices_index] = slice
                            self.label_slices_index = self.label_slices_index + 1
                            self.argumentation_type.append('affine')

                        shear_image, shear_label = self.augment(image_array, label_array, apply_affine=False, apply_shear=True)
                        for slice in shear_image:
                            self.image_slices[self.image_slices_index] = slice
                            self.image_slices_index = self.image_slices_index + 1
                        for slice in shear_label:
                            self.label_slices[self.label_slices_index] = slice
                            self.label_slices_index = self.label_slices_index + 1
                            self.argumentation_type.append('shear')
                    
                    pbar.update(1)

                    gc.collect()
            
        except Exception as e:
            logging.error(f"Error during slice loading: {e}")
            raise
    
    def cal_file_nums(self, folder):
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        num_files = 0

        for file in os.listdir(folder):
            image_path = os.path.join(folder, file)
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)
            z, _, _ = image_array.shape
            num_files += z

        return num_files

    def __getitem__(self, index):
        try:
            image_slice = self.image_slices[index]
            label_slice = self.label_slices[index]

            return image_slice, label_slice
        except Exception as e:
            logging.error(f"Error during __getitem__: {e}")
            raise
    
    def __len__(self):
        return len(self.image_slices)
    
    def size_correction(self, img):
        num_channels, img_height, img_width = img.shape

        # Handle height
        if img_height < self.target_height:
            pad_height = self.target_height - img_height
            img = np.pad(img, ((0, 0), (0, pad_height), (0, 0)), mode='constant', constant_values=0)

        # Handle width
        if img_width < self.target_width:
            pad_width = self.target_width - img_width
            img = np.pad(img, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)

        # Crop or pad the height and width to target dimensions
        img = img[:, :self.target_height, :self.target_width]

        return img
    
    def augment(self, image, label, apply_affine=False, apply_shear=False,
                angle_range=(-5, 5), tx_range=(-3, 3), ty_range=(-3, 3),
                tz_range=(-3, 3), scale_range=(0.5, 1.5), shear_range=(-10, 10)):
        try:
            device = image.device
            
            # Convert to 5D tensor (B=1, C=1, D, H, W)
            image_tensor = image.unsqueeze(0).unsqueeze(0).to(device)
            label_tensor = label.unsqueeze(0).unsqueeze(0).to(device)
            
            if apply_affine:
                while(True):
                    affine_matrix = self.get_affine_matrix(angle_range, tx_range, ty_range, tz_range, scale_range)
                    affine_matrix = affine_matrix.to(device)
                    grid = F.affine_grid(affine_matrix.unsqueeze(0), image_tensor.size(), align_corners=False)
                    transformed_image_tensor = F.grid_sample(image_tensor, grid, mode='bilinear', padding_mode='border', align_corners=False)
                    transformed_label_tensor = F.grid_sample(label_tensor, grid, mode='nearest', padding_mode='border', align_corners=False)
                    if torch.count_nonzero(transformed_label_tensor) / torch.count_nonzero(label_tensor) > 0.9 :
                        break

            if apply_shear:
                while(True):
                    shear_matrix = self.get_shear_matrix(shear_range)
                    shear_matrix = shear_matrix.to(device)
                    grid = F.affine_grid(shear_matrix.unsqueeze(0), image_tensor.size(), align_corners=False)
                    transformed_image_tensor = F.grid_sample(image_tensor, grid, mode='bilinear', padding_mode='border', align_corners=False)
                    transformed_label_tensor = F.grid_sample(label_tensor, grid, mode='nearest', padding_mode='border', align_corners=False)
                    if torch.count_nonzero(transformed_label_tensor) / torch.count_nonzero(label_tensor) > 0.9 :
                        break  
            image = transformed_image_tensor.squeeze(0).squeeze(0)
            label = transformed_label_tensor.squeeze(0).squeeze(0)

            return image, label

        except Exception as e:
            print(f"Error during augmentation: {e}")
            return image, label
        
    def is_all_zero(self, tensor):
        return torch.all(tensor.eq(0))
    
    def get_affine_matrix(self, angle_range, tx_range, ty_range, tz_range, scale_range):
        angle_z = np.random.uniform(angle_range[0], angle_range[1])
        tx = np.random.uniform(tx_range[0], tx_range[1])
        ty = np.random.uniform(ty_range[0], ty_range[1])
        tz = np.random.uniform(tz_range[0], tz_range[1])
        scale = np.random.uniform(scale_range[0], scale_range[1])
        
        affine_matrix = torch.eye(3, 4, dtype=torch.float32)
        affine_matrix[0, 0] = scale * np.cos(np.radians(angle_z))
        affine_matrix[0, 1] = -scale * np.sin(np.radians(angle_z))
        affine_matrix[1, 0] = scale * np.sin(np.radians(angle_z))
        affine_matrix[1, 1] = scale * np.cos(np.radians(angle_z))
        affine_matrix[0, 3] = tx
        affine_matrix[1, 3] = ty
        affine_matrix[2, 3] = tz
        return affine_matrix
    
    def get_shear_matrix(self, shear_range):
        shear_x = np.random.uniform(shear_range[0], shear_range[1])
        shear_y = np.random.uniform(shear_range[0], shear_range[1])
        shear_z = np.random.uniform(shear_range[0], shear_range[1])
        
        shear_matrix = torch.eye(3, 4, dtype=torch.float32)
        shear_matrix[0, 1] = np.tan(np.radians(shear_x))
        shear_matrix[1, 2] = np.tan(np.radians(shear_y))
        shear_matrix[2, 0] = np.tan(np.radians(shear_z))

        return shear_matrix