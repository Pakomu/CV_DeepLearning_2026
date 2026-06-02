import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

from utils.dataset_utils import PromptTrainDataset
from utils.dataset_utils import HW4TrainDataset # <--- 修改這裡
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class CompositeLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha # 權重分配

    def forward(self, x, y):
        l1_loss = self.l1(x, y)
        # SSIM 越高越好(最大為1)，所以 Loss 要用 1 減去 SSIM
        ssim_loss = 1 - ssim(x, y, data_range=1.0, size_average=True) 
        
        # 結合兩者
        return (1 - self.alpha) * l1_loss + self.alpha * ssim_loss
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # Charbonnier Formula
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss
    
class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = CompositeLoss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = HW4TrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(
    dirpath=opt.ckpt_dir,
    filename='promptir-{epoch:03d}-{train_loss:.4f}', # 檔名會自動標記 Epoch 與 Loss
    monitor='train_loss', # 監控訓練誤差
    mode='min',           # 尋找誤差「最小」的
    save_top_k=3,         # 只保留分數最好的前 3 名，其他的自動刪除防爆硬碟
    save_last=True        # 強制保留最後一個 Epoch 作為備用
    )
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = PromptIRModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=1,strategy="auto",precision="16-mixed",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



