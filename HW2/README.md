Markdown
# NYCU Computer Vision 2026 HW1
* Student ID: 111550028
* Name: 黃柏翔
## Introduction
For this assignment, inspired by the lessons mentioned and I tested the training speed, my core methodology used the deformable DETR according to its convergence speed. Also, I implemented the augmentation method by applying the albumentation package. However, I finally cannot reach the pass score of this homework. 
## Environment Setup
Use anaconda and build an environment.
```
conda create -n detr_hw2 python=3.10
conda activate detr_hw2
pip install -r requirements.txt
```


The path is show below:
```bash
HW2/
├── nycu-hw2-data/      # store my pred.json             
│   ├── test/
│   ├── train/
│   └── valid/
│   └── train.json/
│   └── valid.json/
├── checkpoints/          
├── train.py                # Main training script
├── test_add_NMS.py        # Inference and ensemble voting script
└── README.md
```
## Usage
### 1. Model Training
Open `train.py` and modify the `CONFIG` dictionary to customize your training hyperparameters. Below is a recommended setup:

```python
CONFIG = {
    "experiment_name": "DeformableDETR_ResNet50",
    "batch_size": 2,    # hyperparameter
    "accumulation_steps": 8,    # hyperparameter
    "epochs": 50,   # hyperparameter
    "resume_epoch": 0, # if you want to resume the training(the resume_checkpoint cannot be empty.)
    "learning_rate": 1e-4, # hyperparameter
    "backbone_lr": 1e-5,   # hyperparameter       
    "weight_decay": 1e-4,   # hyperparameter
    "checkpoint_dir": "checkpoints", # save the model_direction
    "best_model_name": "best_model", # save model name.
    "resume_checkpoint": "",
    "score_threshold": 0.3,
    "num_workers": 4,
}
```

Once configured, start the training process by running:
```bash
python train.py
``` 

### 2. Model Selection & Ensemble Setup
After the training is complete, the model weights will be saved in the `./checkpoints` directory. To prepare for the ensemble inference:

1. Select your top-performing models from the checkpoints.
2. When running the test_add_NMS.py, add --checkpoint [your checkpoint path], set your threshold


```bash
python test_add_NMS.py --checkpoint [your_path] --threshold 
``` 
## Performance Snapshot
![picture](./Snapshot.png)
