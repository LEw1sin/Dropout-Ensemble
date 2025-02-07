# I. Clarity

It is the code repository of our paper "*Memory-based Ensemble Learning in CMR Semantic Segmentation*".

# II. Usage

The recommended rough sketch of the file directory is as follows:

```
.
├── de_data
    ├── ACDC
    ├── MnMs
    ├── preprocessed_ACDC             
    └── preprocessed_MnMs           
        ├── train
        ├── valid
        └── test
            ├── images
            └── labels
            └── *.h5
├── de_logistics
    ├── ACDC_2UNetlinear_02-02-16-35-56
    ├── eval_ACDC_2UNetlinear_02-05-11-19-24
    ├── MnMs_3UNetlinear_02-03-14-00-38
    └── ···
└── de_mem_based
    ├── model
    ├── utils
    ├── *.py
    └── README.md 
```

``preprocessed_* folder`` got by ``../de_mem_based/utils/process_data/preprocess_*.py``, or you can click on [here](https://drive.google.com/drive/folders/1-l-w_3aVs6I0obVT0o517Ikjd5C_-L5F?usp=drive_link) to download the dataset we have preprocessed directly.

For training, you can directly execute the following code in the terminal:
```python train.py```

In ``train.py``:
```
···
parser.add_argument('--channel_weights', nargs='+', type=float, default=[1.0, 2.0, 1.0, 1.0], help='Channel weights')
parser.add_argument('--dataset_mode', default='ACDC', type=str, help='choose which dataset to use')
parser.add_argument('--wandb', default=True, type=bool, help='whether to use wandb')
parser.add_argument('--cache', default=True, type=bool, help='whether to load the dataset into memory')
···
```
``[1.0, 2.0, 1.0, 1.0]`` is the weight of BG, RV, MYO, LV respectively. ``'--dataset_mode'`` controls the dataset for training, it takes MnMs as the dataset for training when it is ``'MnMs'``. ``'--wandb'`` controls whether to demonstrate the training process in ``wandb``. `'--cache'` controls whether to load the dataset into memory in advance. 

For predicting, you can directly execute the following code in the terminal:
```python predict.py```

In ``predict.py``:
```
···
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_2UNetlinear_02-05-01-00-04', type=str, help='net weights path')
parser.add_argument('--visualize', default=True, type=bool, help='whether to visualize the prediction')
parser.add_argument('--end_coff_threshold', default=0.8, type=float, help='threshold for end slices confidence')
···
```
``'--net_weights'`` is the checkpoint path used for predicting.``'--visualize'`` controls whether to visualize the output. ``'--end_coff_threshold'`` controls the threshold of End Cofficient (EC).

Both in ``train.py`` and ``predict.py``, you can construct the combination of classifiers yourself. The following are some examples:

If you want to implement *Fixed* in our paper, you can set:
```
net = DE_framework_linear(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel),DeepLabV3P_linear(num_classes = args.num_classes, max_channels=args.max_channel)], weight_list=[0.7,0.3])
```
The above is the implementation of 1 UNet + 1 DeeplabV3+ weighted by 0.7 and 0.3 respectively. Linear means linear combination of classifiers weighted by fixed value.

```
net = DE_framework_mem(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channe),
                                        UNet_linear(num_classes = args.num_classes, max_channels=args.max_channe),
                                        UNet_linear(num_classes = args.num_classes, max_channels=args.max_channe)])
```
The above is the implementation of 3 * UNet weighted by memory-based Uncertainty. Here ``weight_list`` is not required because their weights are calculated by Uncertainty of a specific 3D CMR frame.

# III. Reproducibility
The model checkpoints with the numeric performance claimed in our paper can be downloaded on [here](https://drive.google.com/drive/folders/1-l-w_3aVs6I0obVT0o517Ikjd5C_-L5F?usp=drive_link) (included *Augmenting*, *Bagging*, *Stacking*), and unzip them under ``de_logistics``. It contains:
```
├──  ACDC_1DeepLabV3Plinear
├──  ACDC_1UNetlinear
├──  ACDC_1UNetlinear_1DeepLabV3Plinear_fixed(0.7_0.3)
├──  ACDC_1UNetlinear_1DeeplabV3Plinear_Stacking
├──  ACDC_1UNetlinear_1DeepLabV3Plinear_Uncertainty
├──  ACDC_2UNetlinear_Bagging
├──  ACDC_2UNetlinear_fixed(0.5_0.5)
├──  ACDC_2UNetlinear_Uncertainty
├──  ACDC_3UNetlinear_Uncertainty
├──  ACDC_4UNetlinear_Uncertainty
├──  MnMs_2UNetlinear_Uncertainty
└──  MnMs_3UNetlinear_Uncertainty
    ├── training.log
    └── *.pth

```

The following are the settings in ``predict.py`` to reproduce the results of Table.3 in our paper:

UNet:
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_1UNetlinear', type=str, help='net weights path')
net = DE_framework_linear(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel)], weight_list=[1])
```

DeeplabV3+:
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_1DeepLabV3Plinear', type=str, help='net weights path')
net = DE_framework_linear(args, models=[DeepLabV3P_linear(num_classes = args.num_classes, max_channels=args.max_channel)], weight_list=[1])
```

1UNet + 1DeeplabV3+ (*Fixed*):
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_1UNetlinear_1DeepLabV3Plinear_fixed(0.7_0.3)', type=str, help='net weights path')
net = DE_framework_linear(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), DeepLabV3P_linear(num_classes = args.num_classes, max_channels=args.max_channel)], weight_list=[0.7, 0.3])
```

1UNet + 1DeeplabV3+ (*Stacking*)
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_1UNetlinear_1DeeplabV3Plinear_Stacking', type=str, help='net weights path')
net = DE_framework_linear(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), DeepLabV3P_linear(num_classes = args.num_classes, max_channels=args.max_channel)], weight_list=[0.7, 0.3])
```

1UNet + 1DeeplabV3+ (Uncertainty)
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_1UNetlinear_1DeepLabV3Plinear_Uncertainty', type=str, help='net weights path')
net = DE_framework_mem(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), DeepLabV3P_linear(num_classes = args.num_classes, max_channels=args.max_channel)])
```

2 * UNet (*Fixed*)
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_2UNetlinear_fixed(0.5_0.5)', type=str, help='net weights path')
net = DE_framework_linear(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel)], weight_list=[0.5, 0.5])
```

2 * UNet (*Bagging*) 
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_2UNetlinear_Bagging', type=str, help='net weights path')
net = DE_framework_linear(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel)], weight_list=[0.5, 0.5])
```

2 * UNet (*Augmenting*)
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_1UNetlinear', type=str, help='net weights path')
net = DE_framework_Augmenting(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel)], weight_list=[1])
```

2 * UNet (Uncertainty, Ours)
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_2UNetlinear_Uncertainty', type=str, help='net weights path')
net = DE_framework_mem(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel)])
```

3 * UNet (Uncertainty, Ours)
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_3UNetlinear_Uncertainty', type=str, help='net weights path')
net = DE_framework_mem(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel)])
```

4 * UNet (Uncertainty, Ours)
```
parser.add_argument('--net_weights', default='../../de_logistics/ACDC_4UNetlinear_Uncertainty', type=str, help='net weights path')
net = DE_framework_mem(args, models=[UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel), UNet_linear(num_classes = args.num_classes, max_channels=args.max_channel)])
```
