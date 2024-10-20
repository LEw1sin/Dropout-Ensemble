# I. Clarity

It is the code repository of our paper "*A Simple Ensemble Learning Implementation in Ventricle Semantic Segmentation*".

# II. Usage

Place the ACDC or MM dataset into the corresponding folder in the project. The directory structure should be:

```
MM
  ├── ED
  │   ├── train
  │   │   ├── images
  │   │   └── labels
  │   ├── valid
  │   │   ├── images
  │   │   └── labels
  │   └── test
  │       ├── images
  │       └── labels
  └── ES
      ├── train
      │   ├── images
      │   └── labels
      ├── valid
      │   ├── images
      │   └── labels
      └── test
          ├── images
          └── labels
```



```
ACDC
  ├── train
  │   ├── images
  │   └── labels
  ├── valid
  │   ├── images
  │   └── labels
  └── test
      ├── images
      └── labels
```

In train.py and predict.py, custom parameters are provided, which users can change according to their needs. In fact, the parameters are pre-configured by default, so *train.py* and *predict.py* can be run directly.

Data augmentation is used in our code, thus it may take some time to load the dataset, if you do not want to use it, you could modify the parser setting in *train.py*, setting 'aug' to False.

The training, validation, and testing procedure are conducted on the two datasets respectively, so it is sufficient to have files in just one directory. You could modify the parser setting in *train.py*, setting 'dataset_mode' to 'ACDC' if you want to use ACDC and to 'MM' if you want to use M&Ms.
