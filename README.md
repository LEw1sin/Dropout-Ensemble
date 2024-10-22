# I. Clarity

It is the code repository of our paper "*A Simple Ensemble Learning Implementation in Ventricle Semantic Segmentation*".

# II. Usage

Place the ACDC or MM dataset into the corresponding folder in the project. The directory structure should be:

```
MM/
  ├── ED/
  │   ├── train/
  │   │   ├── images/
  │   │   │   ├── 001_SA_ED.nii.gz
  │   │   │   └── ...
  │   │   └── labels/
  │   │       ├── 001_SA_ED_gt.nii.gz
  │   │       └── ...
  │   ├── valid/
  │   │   ├── images/
  │   │   │   ├── 001_SA_ES.nii.gz
  │   │   │   └── ...
  │   │   └── labels/
  │   │       ├── 001_SA_ES_gt.nii.gz
  │   │       └── ...
  │   └── test/
  │       ├── images/
  │       │   ├── ...
  │       └── labels/
  │           ├── ...
  └── ES/
      ├── train/
      │   ├── images/
      │   │   ├── ...
      │   └── labels/
      │       ├── ...
      ├── valid/
      │   ├── images/
      │   │   ├── ...
      │   └── labels/
      │       ├── ...
      └── test/
          ├── images/
          │   ├── ...
          └── labels/
              ├── ...
```



```
ACDC/
  ├── train/
  │   ├── images/
  │   │   ├── patient001_frame01.nii.gz
  │   │   └── ...
  │   └── labels/
  │       ├── patient001_frame01.nii.gz
  │       └── ...
  ├── valid/
  │   ├── images/
  │   │   ├── ...
  │   └── labels/
  │       ├── ...
  └── test/
      ├── images/
      │   ├── ...
      └── labels/
          ├── ...
```

In train.py and predict.py, custom parameters are provided, which users can change according to their needs. In fact, the parameters are pre-configured by default, so train.py and predict.py can be run directly.
