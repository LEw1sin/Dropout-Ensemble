import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

original_labels = np.array([0, 1, 2, 3, 4, 6, 7, 8, 11])
new_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
max_label = 13
mapping_array = np.full(max_label + 1, fill_value=0, dtype=int)
mapping_array[original_labels] = new_labels

class Synapse_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        with h5py.File(file_path, 'r') as f:
            image, label = f['image'][:], f['label'][:]

        # label = label.astype(int)
        # filtered_label = mapping_array[label]

        return image, label
    
if __name__ == "__main__":
    import os
    from tqdm import tqdm
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # db_eval = Synapse_dataset(data_dir='../../data/preprocessed_Synapse/eval_h5')
    # unique_set = set()
    # label_idx_max=0
    # for i in tqdm(range(len(db_eval))):
    #     image, label = db_eval[i]
    #     label_idx_max = max(label.max(), label_idx_max)
    #     unique_set.update(np.unique(label).tolist())
    # print(f'max label id: {label_idx_max}')
    # print(f'unique labels: {unique_set}')

    db_train = Synapse_dataset(data_dir='../../data/preprocessed_Synapse/train_h5')
    unique_set = set()
    label_idx_max=0
    for i in tqdm(range(len(db_train))):
        image, label = db_train[i]
        label_idx_max = max(label.max(), label_idx_max)
        unique_set.update(np.unique(label).tolist())
    print(f'max label id: {label_idx_max}')
    print(f'unique labels: {unique_set}')
