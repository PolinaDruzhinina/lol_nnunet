## Brain tumor segmentation based on deep learning mechanisms using MRI multi-modalities brain images
In this repository we evaluat the state-of-the-art methods for the accurate segmentation of intrinsically heterogeneous brain glioma sub-regions based on Brain Tumor Segmentation dataset with mpMRI scans from RSNA-ASNR-MICCAI BraTS 2021 challenge.

## DATASET
  All BraTS mpMRI scans are available as NIfTI files (.nii.gz). These mpMRI scans describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple institutions. The sub-regions considered for evaluation are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT). The ground truth data were created after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm^3) and skull-stripped.
 
## Content

-nvidia_unet - pipline with optimized Unet from NVIDIA
     --nvidia_unet/nnunet - models unet,unetr,seg_res_net
     --nvidia_unet/scripts_training - examples of scripts with experiments
     --nvidia_unet/notebooks - jupyter notebooks with smth (preprocessing, metrics, inferense
-notebooks - examples of vanilla 3d unet
-vanila_nnunet_torchio - attempt of nn-unet with torchio
 

