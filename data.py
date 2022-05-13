import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchio 
import torchio as tio
from torchio import transforms, AFFINE, DATA, PATH, TYPE, STEM
import pytorch_lightning as pl
import os
import enum
import numpy as np
from tqdm import tqdm 

import enum
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchio

MRI = 'MRI'
LABEL = 'LABEL'
SUB = 'SUB'

LI_LANDMARKS = "0 8.06305571158 15.5085721044 18.7007018006 21.5032879029 26.1413278906 29.9862059045 33.8384058795 38.1891334787 40.7217966068 44.0109152758 58.3906435207 100.0"
LI_LANDMARKS = np.array([float(n) for n in LI_LANDMARKS.split()])
landmarks_dict = {'MRI': LI_LANDMARKS}
                      
class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

class MedicalDecathlonDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.subjects = None
        self.training_set = None
        self.validation_set = None
        self.patches_training_set = None
        self.patches_validation_set = None

    def prepare_data(self):
        """
        The function creates dataset from the list of files from cunstumised dataloader.
        """
        self.subjects = []
        path = self.params.data_path
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
    
    def get_transforms(self, data_augmentation_train=None, data_augmentation_val=None):
        """
        Outputs the transformations that will be applied to the dataset
        :param data_augmentation_tain/val: (list[str]) list of data augmentation performed on the training/val set.
        :return:
        - container torchio.Compose including transforms to apply in train mode.
        - container torchio.Compose including transforms to apply in evaluation mode.
        """
        augmentation_dict_train = {  "Anisotropic":  tio.RandomAnisotropy(p=0.25),              # make images look anisotropic 25% of times
                                     "HistStandart": tio.HistogramStandardization(
                                              landmarks_dict, masking_method=tio.ZNormalization.mean),        # standardize histogram of foreground
                                     "Noise":  tio.RandomNoise(p=0.25), # Gaussian noise 25% of times
                                     "BiasField": tio.RandomBiasField(p=0.3), # magnetic field inhomogeneity 30% of times
                                     "Blur": tio.RandomBlur(p=0.25),
                                     "RemapLab": tio.RemapLabels({0:0, 1:1, 2:2, 4:3}),
                                     "None": None}
        augmentation_dict_val = {"HistStandart": tio.HistogramStandardization(
                                      landmarks_dict, masking_method=tio.ZNormalization.mean),        # standardize histogram of foreground
                                 "RemapLab": tio.RemapLabels({0:0, 1:1, 2:2, 4:3}),
                                 "None": None}

        if data_augmentation_train and data_augmentation_val:
            train_augmentation_list = [augmentation_dict_train[augmentation] for augmentation in data_augmentation_train]
            val_augmentation_list = [augmentation_dict_val[augmentation] for augmentation in data_augmentation_val]
        else:
            train_augmentation_list = []
            val_augmentation_list = []

        train_transform = tio.Compose(train_augmentation_list)

        validation_transform = tio.Compose(val_augmentation_list)
        return train_transform, validation_transform

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            training_subjects, validation_subjects = train_test_split(
                self.subjects, train_size=0.7, shuffle=False, random_state=42
            )

            train_transform, validation_transform = self.get_transforms(self.params.data_augmentation_train, self.params.data_augmentation_val)
            self.training_set = torchio.SubjectsDataset(
                training_subjects, transform=train_transform)

            self.validation_set = torchio.SubjectsDataset(
                validation_subjects, transform=validation_transform)

            self.patches_training_set = torchio.Queue(
                subjects_dataset=self.training_set,
                max_length=self.params.max_queue_length,
                samples_per_volume=self.params.samples_per_volume,
                sampler=torchio.sampler.UniformSampler(self.params.patch_size),
                num_workers=self.params.num_workers,
                shuffle_subjects=True,
                shuffle_patches=True,
            )
            
            self.patches_validation_set = torchio.Queue(
                subjects_dataset=self.validation_set,
                max_length=self.params.max_queue_length,
                samples_per_volume=self.params.samples_per_volume,
                sampler=torchio.sampler.UniformSampler(self.params.patch_size),
                num_workers=self.params.num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            print('You should download the dataset for the test!!!!!!!!!!')

    def train_dataloader(self):
        return DataLoader(self.patches_training_set, batch_size=self.params.training_batch_size, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.patches_validation_set, batch_size=self.params.validation_batch_size, pin_memory=True)
