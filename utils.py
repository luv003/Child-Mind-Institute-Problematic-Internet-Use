import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from typing import List

from config import COLS, DATA_PATH

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def quadratic_weighted_kappa(predictions, targets):
    return cohen_kappa_score(predictions, targets, weights="quadratic")

def threshold_rounder(oof_non_rounded: np.ndarray, thresholds: List[float]) -> np.ndarray:
    return np.where(
        oof_non_rounded < thresholds[0],
        0,
        np.where(
            oof_non_rounded < thresholds[1],
            1,
            np.where(oof_non_rounded < thresholds[2], 2, 3),
        ),
    )

def evaluate_predictions(thresholds: List[float], y_true: np.ndarray, oof_non_rounded: np.ndarray) -> float:
    rounded_p = threshold_rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)

def create_preprocessor(categorical_features: List[int], numerical_features: List[int]):
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", QuantileTransformer()),
                    ]
                ),
                numerical_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
