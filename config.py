import copy
import os
import warnings
import logging
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WARMUP = True
N_SPLITS = 5
RANDOM_STATE = 1335
EPOCHS = 8
LR = [1e-3, 3e-3]
DATA_PATH = "/kaggle/input/child-mind-institute-problematic-internet-use"
DROP_NAN = True
CYCLES = 1
QWK_WEIGHT = 0.25
CE_WEIGHT = 0.75
WARMUP_RATIO = [0.0, 1.0]
BATCH_SIZE = 32
DROPOUT = 0.3

COLS = [
    "Basic_Demos-Enroll_Season", "CGAS-Season", "Physical-Season", "Fitness_Endurance-Season",
    "FGC-Season", "BIA-Season", "PAQ_C-Season", "SDS-Season", "PreInt_EduHx-Season",
    "FGC-FGC_PU", "BIA-BIA_SMM", "BIA-BIA_BMR", "BIA-BIA_FFMI", "BIA-BIA_TBW", "Basic_Demos-Sex",
    "BIA-BIA_LDM", "Fitness_Endurance-Time_Mins", "FGC-FGC_GSND", "Basic_Demos-Age", "Physical-HeartRate",
    "FGC-FGC_SRL", "Physical-Waist_Circumference", "Physical-Systolic_BP", "CGAS-CGAS_Score",
    "BIA-BIA_ECW", "PAQ_A-PAQ_A_Total", "FGC-FGC_SRR", "PreInt_EduHx-computerinternet_hoursday",
    "SDS-SDS_Total_Raw", "FGC-FGC_GSD", "PAQ_C-PAQ_C_Total", "BIA-BIA_BMI", "Fitness_Endurance-Time_Sec",
    "Physical-Height", "SDS-SDS_Total_T", "FGC-FGC_CU", "Physical-Weight", "FGC-FGC_TL",
    "Physical-Diastolic_BP", "Physical-BMI", "Fitness_Endurance-Max_Stage", "BIA-BIA_FMI", "BIA-BIA_BMC",
    "BIA-BIA_DEE", "BIA-BIA_ICW", "BIA-BIA_Fat", "BIA-BIA_LST", "BIA-BIA_Activity_Level_num",
]
