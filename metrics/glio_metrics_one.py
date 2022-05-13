import os
import sys
import argparse
import numpy as np
import pandas as pd
import nibabel as nib    
from pathlib import Path
from surface_distance import metrics
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_pred', type=str,  default='',  help='path to prediction file')
parser.add_argument('--path_to_lab', type=str, default='', help='path to mask file ')
parser.add_argument('--out_folder', type=str, default='', help='path to save prediction .csv')
      

def sensitivity_and_specificity(mask_gt, mask_pred):
    """ Computes sensitivity and specificity
     sensitivity  = TP/(TP+FN)
     specificity  = TN/(TN+FP) """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    tp = (mask_gt & mask_pred).sum()
    tn = (~mask_gt & ~mask_pred).sum()
    fp = (~mask_gt & mask_pred).sum()
    fn = (mask_gt & ~mask_pred).sum()
    
    return tp/(tp+fn), tn/(tn+fp)


def calculate_metrics_brats(true_mask, pred_mask):
    """ Takes two file locations as input and validates surface distances.
    Be careful with dimensions of saved `pred` it should be 3D.
    
    """
    
    _columns = ['Dice_all', 'Dice_0', 'Dice_gtv', 'Dice_2', 
               'Hausdorff95_all', 'Hausdorff95_0', 'Hausdorff95_1', 'Hausdorff95_2', 
               'Sensitivity_all', 'Sensitivity_0', 'Sensitivity_1', 'Sensitivity_2', 
               'Specificity_all', 'Specificity_0', 'Specificity_1', 'Specificity_2', 
               'Surface_dice_all', 'Surface_dice_0', 'Surface_dice_1', 'Surface_dice_2',]
    
    df = pd.DataFrame(columns = _columns)
    #sum of all classes
    distances = metrics.compute_surface_distances((true_mask > 0), (pred_mask > 0), [1,1,1])
    df.at[0,'Dice_all'] = metrics.compute_dice_coefficient((true_mask > 0), (pred_mask > 0))
    df.at[0,'Surface_dice_all'] = metrics.compute_surface_dice_at_tolerance(distances, 1)
    df.at[0,'Hausdorff95_all'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, spec = sensitivity_and_specificity((true_mask > 0), (pred_mask > 0))
    df.at[0,'Sensitivity_all'] = sens 
    df.at[0,'Specificity_all'] = spec
    # class 0
    distances = metrics.compute_surface_distances((true_mask == 0), (pred_mask == 0), [1,1,1])
    df.at[0,'Dice_0'] = metrics.compute_dice_coefficient((true_mask == 0), (pred_mask == 0))
    df.at[0,'Surface_dice_0'] = metrics.compute_surface_dice_at_tolerance(distances, 1)
    df.at[0,'Hausdorff95_0'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, spec = sensitivity_and_specificity((true_mask == 0), (pred_mask == 0))
    df.at[0,'Sensitivity_0'] = sens 
    df.at[0,'Specificity_0'] = spec
    #class 1
    distances = metrics.compute_surface_distances((true_mask == 1), (pred_mask == 1), [1,1,1])
    df.at[0,'Dice_gtv'] = metrics.compute_dice_coefficient((true_mask == 1), (pred_mask == 1))
    df.at[0,'Surface_dice_1'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_1'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, spec = sensitivity_and_specificity((true_mask == 1), (pred_mask == 1))
    df.at[0,'Sensitivity_1'] = sens
    df.at[0,'Specificity_1'] = spec
    #class 2
    distances = metrics.compute_surface_distances((true_mask == 2), (pred_mask == 2), [1,1,1])
    df.at[0,'Dice_2'] = metrics.compute_dice_coefficient((true_mask == 2), (pred_mask == 2))
    df.at[0,'Surface_dice_2'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_2'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, spec= sensitivity_and_specificity((true_mask == 2), (pred_mask == 2))
    df.at[0,'Sensitivity_2'] = sens
    df.at[0,'Specificity_2'] = spec
    
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.DataFrame()
    pred_file = Path(args.path_to_pred)
    target_file = Path(args.path_to_lab)
    out_folder = Path(args.out_folder)
    
    targets = nib.load(pred_file).get_fdata()
    predictions = nib.load(target_file).get_fdata()
    # dimensions should be preserved
    assert(targets.shape == predictions.shape)
    pred = np.round(predictions, 0)
    df = df.append(calculate_metrics_brats(targets.astype('int'), pred.astype('int')))
    
    df.to_csv(out_folder / 'dice_hausdorff_metrics.csv')
    