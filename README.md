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



In train.py and predict.py, custom parameters are provided, which users can change according to their needs. In fact, the parameters are pre-configured by default, so train.py and predict.py can be run directly.