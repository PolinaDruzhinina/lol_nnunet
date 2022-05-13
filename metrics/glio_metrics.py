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
parser.add_argument('--path_to_pred', type=str,  default='/anvar/private_datasets/glioma_burdenko/hd_glio_auto',  help='path to prediction')
parser.add_argument('--path_to_lab', type=str, default='/anvar/private_datasets/glioma_burdenko/glioma_burdenko_nii_gz', help='path to masks')
      

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


def calculate_metrics_brats(true_mask, pred_mask, ids):
    """ Takes two file locations as input and validates surface distances.
    Be careful with dimensions of saved `pred` it should be 3D.
    
    """
    
    _columns = ['Ids', 'Dice_all', 'Dice_0', 'Dice_gtv', 'Dice_2',
               'Hausdorff95_all', 'Hausdorff95_0', 'Hausdorff95_gtv', 'Hausdorff95_2',
               'Sensitivity_all', 'Sensitivity_0', 'Sensitivity_gtv', 'Sensitivity_2',
               'Specificity_all', 'Specificity_0', 'Specificity_gtv', 'Specificity_2',
               'Surface_dice_all', 'Surface_dice_0', 'Surface_dice_gtv', 'Surface_dice_2']
    
    df = pd.DataFrame(columns = _columns)
    df.at[0,'Ids'] = ids
    #all classes
#     distances = metrics.compute_surface_distances(true_mask, pred_mask, [1,1,1])
#     df.at[0,'Dice_all'] = metrics.compute_dice_coefficient(true_mask, pred_mask)
#     df.at[0,'Surface_dice_all'] = metrics.compute_surface_dice_at_tolerance(distances, 1)
#     df.at[0,'Hausdorff95_all'] = metrics.compute_robust_hausdorff(distances, 95)
#     sens, spec = sensitivity_and_specificity(true_mask, pred_mask)
#     df.at[0,'Sensitivity_all'] = sens 
#     df.at[0,'Specificity_all'] = spec
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
    df.at[0,'Surface_dice_gtv'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_gtv'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, spec = sensitivity_and_specificity((true_mask == 1), (pred_mask == 1))
    df.at[0,'Sensitivity_gtv'] = sens
    df.at[0,'Specificity_gtv'] = spec
    #class 2
    distances = metrics.compute_surface_distances((true_mask == 1), (pred_mask == 2), [1,1,1])
    df.at[0,'Dice_2'] = metrics.compute_dice_coefficient((true_mask == 1), (pred_mask == 2))
    df.at[0,'Surface_dice_2'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_2'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, spec= sensitivity_and_specificity((true_mask == 1), (pred_mask == 2))
    df.at[0,'Sensitivity_2'] = sens
    df.at[0,'Specificity_2'] = spec
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    df=pd.DataFrame()
    pred_folder = Path(args.path_to_pred)
    target_folder = Path(args.path_to_lab)
    for ids in tqdm(os.listdir(pred_folder)):
        targets = nib.load(target_folder / ids / 'mask_GTV_FLAIR.nii.gz').get_fdata()
        predictions = nib.load(pred_folder / ids / 'seg_to_ref.nii.gz').get_fdata()
        assert(targets.shape == predictions.shape)
#         out, c = np.unique(predictions, return_counts=True)
#         print(np.unique(targets.astype('int'),return_counts=True))
#         print(out), print(c), print(len(c))
        pred = np.round(predictions, 0)
#         out, c = np.unique(pred.astype('int'), return_counts=True)
#         print(out), print(c), print(len(c))
        df=df.append(calculate_metrics_brats(targets.astype('int'), pred.astype('int'), ids))
    
    df.to_csv('glio_out.csv')
    print(f"Dice_0 {df['Dice_0'].mean()} Dice_gtv {df['Dice_gtv'].mean()}  Dice_2 {df['Dice_2'].mean()} Hausdorff95_0 {df['Hausdorff95_0'].mean()}  Hausdorff95_gtv {df['Hausdorff95_gtv'].mean()}  Hausdorff95_2 {df['Hausdorff95_2'].mean()} Sensitivity_0 {df['Sensitivity_0'].mean()} Sensitivity_2 {df['Sensitivity_2'].mean()} Sensitivity_gtv {df['Sensitivity_gtv'].mean()} Specificity_0 {df['Specificity_0'].mean()} Specificity_gtv {df['Specificity_gtv'].mean()} Specificity_2 {df['Specificity_2'].mean()} Surface_dice_0 {df['Surface_dice_0'].mean()} Surface_dice_gtv {df['Surface_dice_gtv'].mean()}  Surface_dice_2 {df['Surface_dice_2'].mean()}")
