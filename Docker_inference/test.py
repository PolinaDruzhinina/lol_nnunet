import os
import re
import sys
import torch
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchio
import torchio as tio
from torchio import AFFINE, DATA, PATH, TYPE, STEM
import nilearn
from nilearn import image
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from unet import UNet

seed_everything(42, workers=True)

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='path to input dir', type=str, default='/input')
parser.add_argument('--output', help='path to output dir', type=str, default='/output')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='/app/epoch=25-val_dice=0.78.ckpt')
parser.add_argument('--patch_size', default=128, type=int, help='patch_size')
parser.add_argument('--patch_overlap', default=0, type=int, help='patch_overlap')
parser.add_argument('--validation_batch_size', default=2, type=int, help='Batch size for validation')

parser.add_argument('--in_channels', default=2, type=float, help='Size of input channels')
parser.add_argument('--num_encoding_blocks', default=5, type=float, help='num_encoding_blocks')
parser.add_argument('--out_channels_first_layer', default=16, type=float, help='out_channels_first_layer')
parser.add_argument('--preactivation', default=True, type=bool, help='preactivation')
parser.add_argument('--upsampling', default='trilinear', type=str, help='upsampling_type')
parser.add_argument('--activation', default='PReLU', type=str, help='activation')

parser.add_argument('--gpus', default=-1, type=int, help="Id of gpu device")
args = parser.parse_args()

sys.stdout.flush()


MRI = 'MRI'
LABEL = 'LABEL'
SUB = 'SUB'

LI_LANDMARKS = "0 8.06305571158 15.5085721044 18.7007018006 21.5032879029 26.1413278906 29.9862059045 33.8384058795 38.1891334787 40.7217966068 44.0109152758 58.3906435207 100.0"
LI_LANDMARKS = np.array([float(n) for n in LI_LANDMARKS.split()])
landmarks_dict = {'MRI': LI_LANDMARKS}

class MedicalDecathlonDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.subjects = None
        self.validation_set = None

    def prepare_data(self):
        """
        The function creates dataset from the list of files from cunstumised dataloader.
        """
        self.subjects = []
        path = self.params.input
        sub_list=sorted(os.listdir(path))
        for s in tqdm(sub_list):    
            image_path_t1ce=os.path.join(path,s,s+ '_'+ 't1ce' +'.nii.gz')
            image_path_flair=os.path.join(path,s,s+ '_'+ 'flair' +'.nii.gz')
            label_path=os.path.join(path,s,s+ '_seg.nii.gz')
            subject_dict = {
                        MRI : torchio.Image(path = [image_path_t1ce,image_path_flair] , type= torchio.INTENSITY),
                        LABEL: tio.LabelMap(label_path),
                        SUB: image_path_t1ce
                    }

            subject = torchio.Subject(subject_dict)
            self.subjects.append(subject)

    def setup(self):
        validation_transform = tio.Compose([tio.HistogramStandardization(
                                      landmarks_dict, masking_method=tio.ZNormalization.mean),
                                            tio.RemapLabels({0:0, 1:1, 2:2, 4:3})])
        self.validation_set = torchio.SubjectsDataset(
                self.subjects, transform=validation_transform)
        self.patches_validation_set = torchio.Queue(
                subjects_dataset=self.validation_set,
                max_length=240,
                samples_per_volume=8,
                sampler=torchio.sampler.UniformSampler(self.params.patch_size),
                num_workers=0,
                shuffle_subjects=False,
                shuffle_patches=False,
            )
        
    def test_dataloader(self):
        return DataLoader(self.patches_validation_set, batch_size=self.params.validation_batch_size, pin_memory=True)
    
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
        self.validation_set = validation_set
    
    def prepare_batch(self, batch):
        return batch[MRI][DATA], batch[LABEL][DATA]
    
    def test_step(self, batch, batch_idx):
        img,_ = self.prepare_batch(batch)
        for i in tqdm(range(len(self.validation_set)), leave=False):
            sample = self.validation_set[i]
            targets = sample[LABEL][DATA]
            name = re.search(r"_(.*)_", sample[SUB].split('/')[-1]).group(1)        
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
            output_np = predictions.squeeze(0).cpu().detach().numpy()
            output_nii = nib.Nifti1Image(output_np.astype('uint8'), np.eye(4))
            nib.save(output_nii, os.path.join(args.output, '{}.nii.gz'.format(name)))
   
        
        
def get_model_and_optimizer(num_encoding_blocks=5, out_channels_first_layer=16, patience=3):

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = UNet(
          in_channels=4,
          out_classes=4,
          dimensions=3,
          num_encoding_blocks=num_encoding_blocks,
          out_channels_first_layer=out_channels_first_layer,
          preactivation=True,
          normalization='batch',
          upsampling_type='trilinear',
          padding=True,
          activation='PReLU',
      )
    return model


if __name__ == '__main__':
    args = parser.parse_args()

    data = MedicalDecathlonDataModule(args)
    data.prepare_data()
    data.setup()
    print('Validation: ', len(data.validation_set))
    
    model = LightUnet.load_from_checkpoint(
    checkpoint_path=args.resume_from_checkpoint,
    map_location=None,params = args, validation_set=data.validation_set
    )
    
    trainer = Trainer.from_argparse_args(args, precision=16, deterministic=True, limit_test_batches=1)
    
    trainer.test(model, datamodule= data)