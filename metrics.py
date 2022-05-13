from surface_distance import metrics

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
    
    _columns = ['Dice_ET', 'Dice_TC', 'Dice_WT',
               'Hausdorff95_ET', 'Hausdorff95_TC', 'Hausdorff95_WT',
               'Sensitivity_ET', 'Sensitivity_TC', 'Sensitivity_WT',
               'Specificity_ET', 'Specificity_TC', 'Specificity_WT',
               'Surface_dice_ET', 'Surface_dice_TC', 'Surface_dice_WT']
    
    df = pd.DataFrame(columns = _columns)
    #ET
    distances = metrics.compute_surface_distances((true_mask == 1), (pred_mask == 1), [1,1,1])
    df.at[0,'Dice_ET'] = metrics.compute_dice_coefficient((true_mask == 1), (pred_mask == 1))
    df.at[0,'Surface_dice_ET'] = metrics.compute_surface_dice_at_tolerance(distances, 1)
    df.at[0,'Hausdorff95_ET'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, spec = sensitivity_and_specificity((true_mask == 1), (pred_mask == 1))
    df.at[0,'Sensitivity_ET'] = sens 
    df.at[0,'Specificity_ET'] = spec
    #TC
    distances = metrics.compute_surface_distances((true_mask == 2), (pred_mask == 2), [1,1,1])
    df.at[0,'Dice_TC'] = metrics.compute_dice_coefficient((true_mask == 2), (pred_mask == 2))
    df.at[0,'Surface_dice_TC'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_TC'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, spec = sensitivity_and_specificity((true_mask == 2), (pred_mask == 2))
    df.at[0,'Sensitivity_TC'] = sens
    df.at[0,'Specificity_TC'] = spec
    #WT
    distances = metrics.compute_surface_distances((true_mask == 3), (pred_mask == 3), [1,1,1])
    df.at[0,'Dice_WT'] = metrics.compute_dice_coefficient((true_mask == 3), (pred_mask == 3))
    df.at[0,'Surface_dice_WT'] = metrics.compute_surface_dice_at_tolerance(distances,1)
    df.at[0,'Hausdorff95_WT'] = metrics.compute_robust_hausdorff(distances, 95)
    sens, spec= sensitivity_and_specificity((true_mask == 3), (pred_mask == 3))
    df.at[0,'Sensitivity_WT'] = sens
    df.at[0,'Specificity_WT'] = spec
    return df