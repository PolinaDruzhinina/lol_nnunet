import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
import wandb
from model import LightUnet
from utils import MyPrintingCallback
from data import MedicalDecathlonDataModule
seed_everything(42, workers=True)

AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)


parser = argparse.ArgumentParser()
parser.add_argument('--weights_stem', type=str, help='weights_stem')
parser.add_argument('--project_name', type=str, default='brats', help='project name for W&B')
parser.add_argument('--group_name', type=str, default='t1c_flair', help='group name for W&B')
parser.add_argument('--weights_save_path', type=str, default='/home/gliomas/brats/models', help='path to output dir')
parser.add_argument('--data_path', type=str, default='/data', help='path to input dir')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='load weights')
parser.add_argument('--training_batch_size', default=2, type=int, help='Batch size for training')
parser.add_argument('--validation_batch_size', default=2, type=int, help='Batch size for validation')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--gpus', default=1, type=int, help="Id of gpu device")
parser.add_argument('--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--patch_size', default=128, type=float, help='patch_size')
parser.add_argument('--patch_overlap', default=0, type=float, help='patch_overlap')
parser.add_argument('--samples_per_volume', default=8, type=float, help='samples_per_volume')
parser.add_argument('--max_queue_length', default=240, type=float, help='max_queue_length')

parser.add_argument('--in_channels', default=2, type=float, help='Size of input channels')
parser.add_argument('--num_encoding_blocks', default=5, type=float, help='num_encoding_blocks')
parser.add_argument('--out_channels_first_layer', default=16, type=float, help='out_channels_first_layer')
parser.add_argument('--preactivation', default=True, type=bool, help='preactivation')
parser.add_argument('--upsampling', default='trilinear', type=str, help='upsampling_type')
parser.add_argument('--activation', default='PReLU', type=str, help='activation')

parser.add_argument('--max_epochs', default=300, type=int, help='max epoch for training')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
parser.add_argument('--momentum', default=0.0001, type=float, help='momentum')
parser.add_argument('--tolerance', default=0, type=float, help='tolerance')
parser.add_argument('--patience', default=3, type=float, help='patience')
parser.add_argument('--data_augmentation_train', default=['Anisotropic', 'HistStandart', 'Noise', 'BiasField', 'Blur', 'RemapLab'], 
                    help='Augmentation fot training')
parser.add_argument('--data_augmentation_val', default=['HistStandart', 'RemapLab'], help='Augmentation fot validation')
parser.add_argument('--verbose', '-v', action='count', default=0)
parser.add_argument('--fast_dev_run', type=bool, default=False, help='Runs 1 train, val, test batch and program ends')

sys.stdout.flush()



def main(args):
    data = MedicalDecathlonDataModule(args)
    data.prepare_data()
    data.setup()
    print('Training:  ', len(data.training_set))
    print('Validation: ', len(data.validation_set))
    
    model = LightUnet(args, data.validation_set)
    model.load_from_checkpoint(
    checkpoint_path=args.resume_from_checkpoint
)
    early_stopping = EarlyStopping('val_loss', patience=params.patience, verbose = True)
    checkpoint_callback_loss = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3, filename=os.path.join(params.weights_stem, '{epoch:02d}-{val_loss:.2f}')) 
    checkpoint_callback_dice = ModelCheckpoint(monitor='val_dice', mode='max', save_top_k=3, filename=os.path.join(params.weights_stem, '{epoch:02d}-{val_dice:.2f}')) 
    trainer = Trainer.from_argparse_args(args, precision=16, deterministic=True,logger=wandb_logger, callbacks=[MyPrintingCallback(), checkpoint_callback_loss,checkpoint_callback_dice, early_stopping])
    
    trainer.test(model, dataloaders=data.val_dataloader, verbose=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)