import os
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import h5py

class MM_dataset(Dataset):
    def __init__(self, data_dir, cache=True, ed_es_only=''):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.h5') and ed_es_only in f]
        if cache:
            self.get_all_data()
            self.getitem = self.getitem_w_cache
        else:
            self.getitem = self.getitem_wo_cache

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_sequence, label_sequence, mask_sequence = self.getitem(idx)
        return image_sequence, label_sequence, mask_sequence
    
    def get_all_data(self):
        self.images_sequence = []
        self.labels_sequence = []
        self.masks_sequence = []
        for idx in range(len(self.files)):
            file_name = self.files[idx]
            file_path = os.path.join(self.data_dir, file_name)

            with h5py.File(file_path, 'r') as f:
                image_sequence, label_sequence, mask_sequence = f['image'][:], f['label'][:], f['mask'][:]
            self.images_sequence.append(image_sequence)
            self.labels_sequence.append(label_sequence)
            self.masks_sequence.append(mask_sequence)
    
    def getitem_wo_cache(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        with h5py.File(file_path, 'r') as f:
            image_sequence, label_sequence, mask_sequence = f['image'][:], f['label'][:], f['mask'][:]

        return image_sequence, label_sequence, mask_sequence
    
    def getitem_w_cache(self, idx):
        return self.images_sequence[idx], self.labels_sequence[idx], self.masks_sequence[idx]
    
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

    db_train = MM_dataset(data_dir='../../de_data/preprocessed_MM/test')
    unique_set = set()
    label_idx_max=0
    for i in tqdm(range(len(db_train))):
        image, label = db_train[i]
        label_idx_max = max(label.max(), label_idx_max)
        unique_set.update(np.unique(label).tolist())
    print(f'max label id: {label_idx_max}')
    print(f'unique labels: {unique_set}')

