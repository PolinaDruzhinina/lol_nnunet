# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

task = {
    "01": "Task01_BrainTumour",
    "02": "Task02_Heart",
    "03": "Task03_Liver",
    "04": "Task04_Hippocampus",
    "05": "Task05_Prostate",
    "06": "Task06_Lung",
    "07": "Task07_Pancreas",
    "08": "Task08_HepaticVessel",
    "09": "Task09_Spleen",
    "10": "Task10_Colon",
    "11": "BraTS2021_train",
    "12": "BraTS2021_val",
    "13": "brats_pipeline_out_train_GTV",
    "14": "EGD_train",
    "15": "brats_pipeline_out_train_GTV_brain",
    "16": "brats_pipeline_out_train_GTV_brain_1cl",
    "18": "brats_pipeline_out_train_GTV_2mod",
    "20": "gbm_3a_atlas_train",
    "21": "gbm_1_reg_train",
    "22": "gbm_2a_interp",
    "23": "gbm_3b_n4",
    "24": "gbm_3c_n4_susan",
    "25": "3d_susan",
    "26": "TCGA_GBM",
    "27": "3a_susan",
    "28": "5_ss_shared",
    "29": "2b_n4",
    "30": "2c_n4_susan",
    "31.0": "6_histogram_fold_0",
    "31.1": "6_histogram_fold_1",
    "31.2": "6_histogram_fold_2",
    "32": "7a_resample",
    "33": "schw_1_reg",
    "34": "bgpd_1_reg",
    "35": "bgpd_2a_interp",
    "36": "schw_2a_interp",
    "37": "schw_3a_atlas",
    "38": "gbm_4a_resamp",
    "39": "schw_4a_resamp",
    "40": "bgpd_4a_resamp",
    "41": "schw_4b_n4",
    "42": "schw_4d_susan",
    "39": "gbm_4a_resamp",# wo z-scoring
    
}

patch_size = {
    "01_3d": [128, 128, 128],
    "02_3d": [80, 192, 160],
    "03_3d": [128, 128, 128],
    "04_3d": [40, 56, 40],
    "05_3d": [20, 320, 256],
    "06_3d": [80, 192, 160],
    "07_3d": [40, 224, 224],
    "08_3d": [64, 192, 192],
    "09_3d": [64, 192, 160],
    "10_3d": [56, 192, 160],
    "11_3d": [128, 128, 128],
    "12_3d": [128, 128, 128],
    "13_3d": [128, 128, 128],
    "14_3d": [128, 128, 128],
    "15_3d": [128, 128, 128],
    "16_3d": [128, 128, 128],
    "17_3d": [128, 128, 128],
    "18_3d": [128, 128, 128],
    "19_3d": [128, 128, 128],
    "20_3d": [128, 128, 128],
    "21_3d": [128, 128, 128],
    "22_3d": [128, 128, 128],
    "23_3d": [128, 128, 128],
    "24_3d": [128, 128, 128],
    "25_3d": [128, 128, 128],
    "26_3d": [128, 128, 128],
    "27_3d": [128, 128, 128],
    "28_3d": [128, 128, 128],
    "29_3d": [128, 128, 128],
    "30_3d": [128, 128, 128],
    "31.0_3d": [128, 128, 128],
    "31.1_3d": [128, 128, 128],
    "31.2_3d": [128, 128, 128],
    "32_3d": [128, 128, 128],
    "33_3d": [128, 128, 128],
    "34_3d": [128, 128, 128],
    "35_3d": [128, 128, 128],
    "36_3d": [128, 128, 128],
    "37_3d": [128, 128, 128],
    "38_3d": [128, 128, 128],
    "39_3d": [128, 128, 128],
    "40_3d": [128, 128, 128],
    "41_3d": [128, 128, 128],
    "42_3d": [128, 128, 128],
    "01_2d": [192, 160],
    "02_2d": [320, 256],
    "03_2d": [512, 512],
    "04_2d": [56, 40],
    "05_2d": [320, 320],
    "06_2d": [512, 512],
    "07_2d": [512, 512],
    "08_2d": [512, 512],
    "09_2d": [512, 512],
    "10_2d": [512, 512],
}


spacings = {
    "01_3d": [1.0, 1.0, 1.0],
    "02_3d": [1.37, 1.25, 1.25],
    "03_3d": [1, 0.7676, 0.7676],
    "04_3d": [1.0, 1.0, 1.0],
    "05_3d": [3.6, 0.62, 0.62],
    "06_3d": [1.24, 0.79, 0.79],
    "07_3d": [2.5, 0.8, 0.8],
    "08_3d": [1.5, 0.8, 0.8],
    "09_3d": [1.6, 0.79, 0.79],
    "10_3d": [3, 0.78, 0.78],
    "11_3d": [1.0, 1.0, 1.0],
    "12_3d": [1.0, 1.0, 1.0],
    "13_3d": [1.0, 1.0, 1.0],
    "14_3d": [1.0, 1.0, 1.0],
    "15_3d": [1.0, 1.0, 1.0],
    "16_3d": [1.0, 1.0, 1.0],
    "20_3d": [1.0, 1.0, 1.0],
    "21_3d": [1.0, 1.0, 1.0],
    "23_3d": [1.0, 1.0, 1.0],
    "24_3d": [1.0, 1.0, 1.0],
    "25_3d": [1.0, 1.0, 1.0],
    "26_3d": [1.0, 1.0, 1.0],
    "27_3d": [1.0, 1.0, 1.0],
    "28_3d": [1.0, 1.0, 1.0],
    "30_3d": [1.0, 1.0, 1.0],
    "29_3d": [1.0, 1.0, 1.0],
    "30_3d": [1.0, 1.0, 1.0],
    "31.0_3d": [1.0, 1.0, 1.0],
    "31.1_3d": [1.0, 1.0, 1.0],
    "31.2_3d": [1.0, 1.0, 1.0],
    "32_3d": [1.0, 1.0, 1.0],
    "33_3d": [1.0, 1.0, 1.0],
    "34_3d": [1.0, 1.0, 1.0],
    "35_3d": [1.0, 1.0, 1.0],
    "36_3d": [1.0, 1.0, 1.0],
    "37_3d": [1.0, 1.0, 1.0],
    "38_3d": [1.0, 1.0, 1.0],
    "39_3d": [1.0, 1.0, 1.0],
    "40_3d": [1.0, 1.0, 1.0],
    "41_3d": [1.0, 1.0, 1.0],
    "42_3d": [1.0, 1.0, 1.0],
    "18_3d": [1.0, 1.0, 1.0],
    "01_2d": [1.0, 1.0],
    "02_2d": [1.25, 1.25],
    "03_2d": [0.7676, 0.7676],
    "04_2d": [1.0, 1.0],
    "05_2d": [0.62, 0.62],
    "06_2d": [0.79, 0.79],
    "07_2d": [0.8, 0.8],
    "08_2d": [0.8, 0.8],
    "09_2d": [0.79, 0.79],
    "10_2d": [0.78, 0.78],
}

ct_min = {
    "03": -17,
    "06": -1024,
    "07": -96,
    "08": -3,
    "09": -41,
    "10": -30,
}

ct_max = {
    "03": 201,
    "06": 325,
    "07": 215,
    "08": 243,
    "09": 176,
    "10": 165.82,
}

ct_mean = {"03": 99.4, "06": -158.58, "07": 77.9, "08": 104.37, "09": 99.29, "10": 62.18}

ct_std = {"03": 39.36, "06": 324.7, "07": 75.4, "08": 52.62, "09": 39.47, "10": 32.65}
