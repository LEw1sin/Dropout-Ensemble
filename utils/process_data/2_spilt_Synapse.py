import os
import shutil

def move_files(src_folder, target_folder, file_names):
    os.makedirs(target_folder, exist_ok=True)
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file in file_names:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(target_folder, file)
                shutil.move(src_path, dest_path)
                print(f"Moved: {file}")

# Train and evaluation files
train_eval_files = [
    "DET0000101_avg.nii.gz", "DET0000201_avg.nii.gz", "DET0000801_avg.nii.gz", "DET0001101_avg.nii.gz", 
    "DET0001201_avg.nii.gz", "DET0001301_avg.nii.gz", "DET0001401_avg.nii.gz", "DET0001501_avg.nii.gz", 
    "DET0001601_avg.nii.gz", "DET0001701_avg.nii.gz", "DET0001801_avg.nii.gz", "DET0002001_avg.nii.gz", 
    "DET0002401_avg.nii.gz", "DET0002501_avg.nii.gz", "DET0002601_avg.nii.gz", "DET0002701_avg.nii.gz", 
    "DET0002801_avg.nii.gz", "DET0003101_avg.nii.gz", "DET0003201_avg.nii.gz", "DET0003301_avg.nii.gz", 
    "DET0003401_avg.nii.gz", "DET0003501_avg.nii.gz", "DET0003601_avg.nii.gz", "DET0003801_avg.nii.gz", 
    "DET0003901_avg.nii.gz", "DET0004001_avg.nii.gz", "DET0004101_avg.nii.gz", "DET0004201_avg.nii.gz", 
    "DET0004301_avg.nii.gz", "DET0004401_avg.nii.gz", "DET0004601_avg.nii.gz", "DET0004701_avg.nii.gz", 
    "DET0004801_avg.nii.gz", "DET0004901_avg.nii.gz", "DET0005001_avg.nii.gz", "DET0005101_avg.nii.gz", 
    "DET0005201_avg.nii.gz", "DET0005401_avg.nii.gz", "DET0005601_avg.nii.gz", "DET0005701_avg.nii.gz", 
    "DET0005801_avg.nii.gz", "DET0005901_avg.nii.gz", "DET0006001_avg.nii.gz", "DET0006101_avg.nii.gz", 
    "DET0006201_avg.nii.gz", "DET0006301_avg.nii.gz", "DET0006401_avg.nii.gz", "DET0006501_avg.nii.gz", 
    "DET0007101_avg.nii.gz", "DET0008801_avg.nii.gz", "DET0008901_avg.nii.gz", "DET0009001_avg.nii.gz", 
    "DET0009301_avg.nii.gz", "DET0010601_avg.nii.gz", "DET0014101_avg.nii.gz", "DET0014201_avg.nii.gz", 
    "DET0015201_avg.nii.gz", "DET0015401_avg.nii.gz", "DET0015601_avg.nii.gz", "DET0016101_avg.nii.gz", 
    "DET0021501_avg.nii.gz", "DET0021701_avg.nii.gz", "DET0024401_avg.nii.gz", "DET0026901_avg.nii.gz", 
    "DET0028301_avg.nii.gz", "DET0028601_avg.nii.gz", "DET0028801_avg.nii.gz", "DET0029001_avg.nii.gz", 
    "DET0030901_avg.nii.gz", "DET0035501_avg.nii.gz", "DET0037101_avg.nii.gz", "DET0039401_avg.nii.gz", 
    "DET0039501_avg.nii.gz", "DET0040001_avg.nii.gz", "DET0040101_avg.nii.gz", "DET0040201_avg.nii.gz", 
    "DET0042001_avg.nii.gz", "DET0042401_avg.nii.gz", "DET0042501_avg.nii.gz", "DET0042601_avg.nii.gz", 
    "DET0043101_avg.nii.gz", "DET0043201_avg.nii.gz", "DET0044601_avg.nii.gz"
]

# Train and evaluation files with '_seg'
train_eval_files_seg = [file.replace('.nii.gz', '_seg.nii.gz') for file in train_eval_files]

# Test files
test_files = [
    "DET0000301_avg.nii.gz", "DET0000601_avg.nii.gz", "DET0000701_avg.nii.gz", "DET0007301_avg.nii.gz", 
    "DET0007401_avg.nii.gz", "DET0007501_avg.nii.gz", "DET0007601_avg.nii.gz", "DET0007801_avg.nii.gz", 
    "DET0008301_avg.nii.gz", "DET0008501_avg.nii.gz", "DET0008601_avg.nii.gz", "DET0008701_avg.nii.gz", 
    "DET0009101_avg.nii.gz", "DET0009401_avg.nii.gz", "DET0009501_avg.nii.gz", "DET0009701_avg.nii.gz", 
    "DET0010001_avg.nii.gz", "DET0010201_avg.nii.gz", "DET0010401_avg.nii.gz", "DET0010501_avg.nii.gz", 
    "DET0010701_avg.nii.gz", "DET0010801_avg.nii.gz", "DET0010901_avg.nii.gz", "DET0011201_avg.nii.gz", 
    "DET0011301_avg.nii.gz", "DET0011401_avg.nii.gz", "DET0011501_avg.nii.gz", "DET0011601_avg.nii.gz", 
    "DET0011701_avg.nii.gz", "DET0012001_avg.nii.gz", "DET0012101_avg.nii.gz", "DET0012201_avg.nii.gz", 
    "DET0012401_avg.nii.gz", "DET0012501_avg.nii.gz", "DET0012901_avg.nii.gz", "DET0013001_avg.nii.gz", 
    "DET0013101_avg.nii.gz", "DET0013301_avg.nii.gz", "DET0013401_avg.nii.gz", "DET0013801_avg.nii.gz", 
    "DET0014001_avg.nii.gz", "DET0014401_avg.nii.gz", "DET0014501_avg.nii.gz", "DET0014601_avg.nii.gz", 
    "DET0014701_avg.nii.gz", "DET0014801_avg.nii.gz", "DET0014901_avg.nii.gz", "DET0015001_avg.nii.gz", 
    "DET0015101_avg.nii.gz", "DET0015501_avg.nii.gz", "DET0019401_avg.nii.gz", "DET0020201_avg.nii.gz", 
    "DET0021401_avg.nii.gz", "DET0024301_avg.nii.gz", "DET0025001_avg.nii.gz", "DET0026201_avg.nii.gz", 
    "DET0026601_avg.nii.gz", "DET0026701_avg.nii.gz", "DET0027601_avg.nii.gz", "DET0029201_avg.nii.gz", 
    "DET0029301_avg.nii.gz", "DET0029601_avg.nii.gz", "DET0029901_avg.nii.gz", "DET0030301_avg.nii.gz", 
    "DET0035301_avg.nii.gz", "DET0042201_avg.nii.gz", "DET0042301_avg.nii.gz", "DET0042701_avg.nii.gz", 
    "DET0043001_avg.nii.gz", "DET0043301_avg.nii.gz", "DET0044001_avg.nii.gz", "DET0044901_avg.nii.gz"
]

#'/data2/lyw/data/Synapse' is the file path of the original Synapse dataset you downloaded using 1_synapse_download.py

# Move files to appropriate target folders
move_files('/data2/lyw/data/Synapse', '/data2/lyw/data/preprocessed_Synapse/train_eval', train_eval_files)
move_files('/data2/lyw/data/Synapse', '/data2/lyw/data/preprocessed_Synapse/train_eval', train_eval_files_seg)
move_files('/data2/lyw/data/Synapse', '/data2/lyw/data/preprocessed_Synapse/test', test_files)
