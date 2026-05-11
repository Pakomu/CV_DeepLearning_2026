Markdown
# NYCU Computer Vision 2026 HW3
* Student ID: 111550028
* Name: 黃柏翔
## Introduction
For this assignment, inspired by the lessons mentioned and I tested the training speed, my core methodology used the Mask R-CNN I finally get 0.43 and pass the strongbaseline. 
## Environment Setup
Use anaconda and build an environment.
```
conda env create -f environment.yml
conda activate cv_hw3_env
```


The path is show below:
```bash
HW3/
├── train/     # store my pred.json             
├── main4_c.py                # Main training script       # Inference and ensemble voting script
└── README.md
```
## Usage
### 1. Model Training
Open `main4_c.py` and modify the `CONFIG` dictionary to customize your training hyperparameters. Below is a recommended setup:

```python
NUM_CLASSES        = 5      
ACCUMULATION_STEPS = 4       
NUM_EPOCHS         = 80

LEARNING_RATE = 1e-4         
MAX_LR        = 1e-3         
WEIGHT_DECAY  = 1e-4
USE_ONECYCLE  = True   

BACKBONE         = 'resnet101'
TRAINABLE_LAYERS = 3
USE_CASCADE      = True
DICE_WEIGHT      = 0.5
FOCAL_GAMMA      = 2.0

SCORE_THRESHOLD = 0.05       
NMS_THRESHOLD   = 0.60       
USE_TTA         = True
```

Once configured, start the training process by running:
```bash
python main4_c.py
``` 


## Performance Snapshot
![picture](./snapshot.png)
