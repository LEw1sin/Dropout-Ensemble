from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import os
import h5py

test_data_path = "/data2/lyw/data/preprocessed_Synapse/test"
test_vol_h5_path = "/data2/lyw/data/preprocessed_Synapse/test_vol_h5"

for file in tqdm(os.listdir(test_data_path)):
    image = sitk.ReadImage(os.path.join(test_data_path, file))
    image_array = sitk.GetArrayFromImage(image)
    image_array = image_array.astype(np.float32)
    file_name =  file.split('.')[0]
    file_h5_name = file_name + ".h5"
    with h5py.File(os.path.join(test_vol_h5_path, file_h5_name), 'w') as f:
        f.create_dataset('image', data=image_array)
