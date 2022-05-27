import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchio 
from torchio import DATA
import pytorch_lightning as pl
from surface_distance import metrics
from sklearn.model_selection import train_test_split

from unet import UNet
from utils import Criterion, softmax_helper, SoftDiceLoss, BCE


MRI = 'MRI'
LABEL = 'LABEL'
SUB = 'SUB'

def create_model(params):  
    model = UNet(
          in_channels=params.in_channels,
          out_classes=4,
          dimensions=3,
          num_encoding_blocks=params.num_encoding_blocks,
          out_channels_first_layer=params.out_channels_first_layer,
          preactivation=params.preactivation,
          normalization='batch',
          upsampling_type=params.upsampling,
          padding=True,
          activation=params.activation,
      )
    return model

class LightUnet(pl.LightningModule):
    def __init__(self,params, validation_set):
        super().__init__()
        self.params = params
        self.model = create_model(params)
        # self.loss = SoftDiceLoss(softmax_helper, batch_dice=True, smooth=1e-5, do_bg=False)
        self.loss = BCE()
        self.save_hyperparameters(params)
        self.validation_set = validation_set

    def forward(self, x):
        out = self.model(x)
        return out
    
    def prepare_batch(self, batch):
        return batch[MRI][DATA], batch[LABEL][DATA]
    
    def training_step(self, batch, batch_idx):
        inputs, targets = self.prepare_batch(batch)
        logits = self.model(inputs)
        loss = self.loss(logits, targets)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        del inputs, targets, logits
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = self.prepare_batch(batch)
        logits = self.model(inputs)
        loss = self.loss(logits, targets)
        del targets, logits
        self.log('valid_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return {"x": loss}
    
    def test_step(self, batch, batch_idx):
        img,_ = self.prepare_batch(batch)
        df=pd.DataFrame()
        for i in tqdm(range(len(self.validation_set)), leave=False):
            sample = self.validation_set[i]
            targets = sample[LABEL][DATA]
            grid_sampler = torchio.inference.GridSampler(
                sample,
                self.params.patch_size,
                self.params.patch_overlap,
            )
            patch_loader = torch.utils.data.DataLoader(
                grid_sampler, batch_size=self.params.validation_batch_size, num_workers=0)
            aggregator = torchio.inference.GridAggregator(grid_sampler)
    
            for patches_batch in patch_loader:
                    inputs = patches_batch[MRI][DATA].type_as(img)
                    locations = patches_batch['location']
                    logits = self.model(inputs)
                    labels = logits.argmax(dim=1, keepdim=True).int()
                    aggregator.add_batch(labels, locations)
            predictions = aggregator.get_output_tensor()
            df=df.append(calculate_metrics_brats(targets, predictions))
        print(f"Dice_ET {df['Dice_ET'].mean()} Dice_TC {df['Dice_TC'].mean()}  Dice_WT {df['Dice_WT'].mean()} Hausdorff95_ET {df['Hausdorff95_ET'].mean()}  Hausdorff95_TC {df['Hausdorff95_TC'].mean()}  Hausdorff95_WT {df['Hausdorff95_WT'].mean()} Sensitivity_ET {df['Sensitivity_ET'].mean()} Sensitivity_TC {df['Sensitivity_TC'].mean()} Sensitivity_WT {df['Sensitivity_WT'].mean()} Specificity_ET {df['Specificity_ET'].mean()} Specificity_TC {df['Specificity_TC'].mean()} Specificity_WT {df['Specificity_WT'].mean()} Surface_dice_ET {df['Surface_dice_ET'].mean()} Surface_dice_TC {df['Surface_dice_TC'].mean()}  Surface_dice_WT {df['Surface_dice_WT'].mean()}")

    
    def validation_epoch_end(self, validation_step_outputs):
        l = torch.stack([x['x'] for x in validation_step_outputs]).mean()
        self.log("val_epoch_loss", l, on_epoch=True, prog_bar=True)
        self.evaluate(self.validation_set, validation_step_outputs[0]['x'])
        
    def evaluate(self, evaluation_set, img):
        dice, dice_et, dice_tc, dice_wt = [], [], [], []
        for i in tqdm(range(len(evaluation_set)), leave=False):
            sample = evaluation_set[i]
            targets = sample[LABEL][DATA]
            grid_sampler = torchio.inference.GridSampler(
                sample,
                self.params.patch_size,
                self.params.patch_overlap,
            )
            patch_loader = torch.utils.data.DataLoader(
                grid_sampler, batch_size=self.params.validation_batch_size, num_workers=0)
            aggregator = torchio.inference.GridAggregator(grid_sampler)
    
            for patches_batch in patch_loader:
                    inputs = patches_batch[MRI][DATA].type_as(img)
                    locations = patches_batch['location']
                    logits = self.model(inputs)
                    labels = logits.argmax(dim=1, keepdim=True).int()
                    aggregator.add_batch(labels, locations)
            predictions = aggregator.get_output_tensor()
            dice.append(metrics.compute_dice_coefficient(targets, predictions))
            if (targets==1).sum() + (predictions==1).sum() != 0:
                dice_et.append(metrics.compute_dice_coefficient(targets==1, predictions==1))
            if (targets==2).sum() + (predictions==2).sum() != 0:
                dice_tc.append(metrics.compute_dice_coefficient(targets==2, predictions==2))
            if (targets==3).sum() + (predictions==3).sum() != 0:
                dice_wt.append(metrics.compute_dice_coefficient(targets==3, predictions==3))

        del inputs, locations, logits, labels, predictions
        self.log(f"val_dice", torch.stack(dice).mean(), prog_bar=True)
        self.log(f"val_dice_et", torch.stack(dice_et).mean(), prog_bar=True)
        self.log(f"val_dice_tc", torch.stack(dice_tc).mean(), prog_bar=True)
        self.log(f"val_dice_wt", torch.stack(dice_wt).mean(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.params.learning_rate, momentum=self.params.momentum, weight_decay=self.params.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
        return ([optimizer], [scheduler])
