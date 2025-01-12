import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA_PATH, COLS, BATCH_SIZE, RANDOM_STATE, DROP_NAN
from utils import threshold_rounder, quadratic_weighted_kappa, logger
from dataset import HybridDataset, collate_fn_test
from model import HybridModel
import lightgbm as lgb

def train_lightgbm(train_df: pd.DataFrame, preprocessors, targets: np.ndarray):
    preprocessor = preprocessors[0]
    tab_data = preprocessor.transform(train_df[COLS])
    lgb_params = {
        "objective": "multiclass",
        "num_class": 4,
        "verbosity": -1,
        "seed": RANDOM_STATE
    }
    lgb_train = lgb.Dataset(tab_data, label=targets)
    model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
    return model

def inference(preprocessors, optimal_thresholds):
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    tabular_data = test_df[COLS]
    ids = test_df["id"].values
    def ts_exists(x, mode='test'):
        path = os.path.join(DATA_PATH, f"series_{mode}.parquet/id={x}/part-0.parquet")
        return 1 if os.path.exists(path) else 0
    ts_indicator = np.array([ts_exists(x, 'test') for x in ids]).reshape(-1, 1)
    test_predictions_nn = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting inference with neural network models.")
    for fold in range(N_SPLITS):
        preprocessor = preprocessors[fold]
        tabular_data_processed = preprocessor.transform(tabular_data)
        tabular_data_processed = np.column_stack([tabular_data_processed, ts_indicator])
        categorical_features = list(range(9))
        categorical_dims = [
            len(
                preprocessors[fold].named_transformers_["cat"]
                .named_steps["encoder"]
                .categories_[i]
            )
            for i in range(len(categorical_features))
        ]
        test_dataset = HybridDataset(tabular_data_processed, ids, is_test=True, cat_len=9)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_test,
            num_workers=4,
        )
        model = HybridModel(
            categorical_dims=categorical_dims,
            numerical_dim=tabular_data_processed.shape[1] - len(categorical_dims) - 1,
            time_series_dim=7,
            embedding_dim=8,
            hidden_dim=128,
            num_classes=4,
        ).to(device)
        fold_preds = []
        for it in range(2):
            model_path = f"best_model_fold_{fold+1}_{it}.pth"
            if not os.path.exists(model_path):
                logger.warning(f"Model checkpoint missing: {model_path}")
                continue
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            preds_tmp = []
            with torch.no_grad():
                for categorical, numerical, time_series in test_loader:
                    categorical, numerical, time_series = (
                        categorical.to(device),
                        numerical.to(device),
                        time_series.to(device),
                    )
                    outputs = model(categorical, numerical, time_series)
                    preds_tmp.extend(
                        (outputs.softmax(dim=1) * torch.tensor([0, 1, 2, 3], device=device))
                        .sum(dim=1)
                        .cpu()
                        .numpy()
                    )
            fold_preds.append(preds_tmp)
        if len(fold_preds) > 0:
            fold_preds = np.mean(fold_preds, axis=0)
            test_predictions_nn.append(fold_preds)
    if len(test_predictions_nn) == 0:
        logger.error("No neural network predictions were generated.")
        test_predictions_nn_final = np.zeros(len(test_df))
    else:
        test_predictions_nn_final = np.mean(test_predictions_nn, axis=0)
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    if DROP_NAN:
        train_df = train_df.dropna(subset=["sii"])
    else:
        train_df["sii"] = train_df["sii"].fillna(0)
    targets = train_df["sii"].values
    lgb_model = train_lightgbm(train_df, preprocessors, targets)
    tabular_test_processed = preprocessors[0].transform(tabular_data)
    lgb_preds = lgb_model.predict(tabular_test_processed)
    lgb_preds_cont = (lgb_preds * np.array([0,1,2,3])).sum(axis=1)
    final_predictions_cont = (test_predictions_nn_final + lgb_preds_cont) / 2
    final_predictions_class = threshold_rounder(final_predictions_cont, optimal_thresholds)
    test_df["sii"] = final_predictions_class
    test_df[["id", "sii"]].to_csv("submission.csv", index=False)
    logger.info("Inference complete. Submission saved to submission.csv.")
