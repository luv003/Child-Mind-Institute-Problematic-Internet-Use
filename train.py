import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

from config import (
    WARMUP,
    N_SPLITS,
    RANDOM_STATE,
    EPOCHS,
    LR,
    DATA_PATH,
    DROP_NAN,
    CYCLES,
    QWK_WEIGHT,
    CE_WEIGHT,
    WARMUP_RATIO,
    BATCH_SIZE,
    DROPOUT,
)
from utils import (
    set_seed,
    quadratic_weighted_kappa,
    threshold_rounder,
    evaluate_predictions,
    create_preprocessor,
    logger,
)
from dataset import HybridDataset, collate_fn
from model import HybridModel, QuadraticWeightedKappaLoss
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    patience=10,
    fold=1,
    wur=0.0,
    it=0,
):
    ce_loss = nn.CrossEntropyLoss()
    qwk_loss = QuadraticWeightedKappaLoss(num_classes=4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(wur * total_steps) if WARMUP else 0,
        num_training_steps=total_steps,
        num_cycles=1,
    )
    best_val_qwk = 0
    epochs_no_improve = 0
    pbar = tqdm(range(epochs), desc=f"Fold {fold}, LR={lr}")
    for _ in pbar:
        model.train()
        train_loss = 0.0
        for categorical, numerical, time_series, targets in train_loader:
            categorical, numerical, time_series, targets = (
                categorical.to(device),
                numerical.to(device),
                time_series.to(device),
                targets.to(device),
            )
            optimizer.zero_grad()
            outputs = model(categorical, numerical, time_series)
            loss = CE_WEIGHT * ce_loss(outputs, targets) + QWK_WEIGHT * qwk_loss(
                outputs, targets
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for (categorical, numerical, time_series, targets) in val_loader:
                categorical, numerical, time_series, targets = (
                    categorical.to(device),
                    numerical.to(device),
                    time_series.to(device),
                    targets.to(device),
                )
                outputs = model(categorical, numerical, time_series)
                loss = CE_WEIGHT * ce_loss(outputs, targets) + QWK_WEIGHT * qwk_loss(
                    outputs, targets
                )
                val_loss += loss.item()
                temp_val_preds = (
                    (outputs.softmax(dim=1) * torch.tensor([0, 1, 2, 3], device=device))
                    .sum(dim=1)
                    .cpu()
                    .numpy()
                )
                temp_targets = targets.cpu().numpy()
                val_preds.extend(temp_val_preds)
                val_targets.extend(temp_targets)
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_preds_rounded = np.round(val_preds).astype(int)
        val_qwk = quadratic_weighted_kappa(val_targets, val_preds_rounded)
        pbar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "Val QWK": f"{val_qwk:.4f}",
            }
        )
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save(model.state_dict(), f"best_model_fold_{fold}_{it}.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            break
    oof_predictions = np.zeros(len(val_loader.dataset))
    oof_targets = np.zeros(len(val_loader.dataset))
    model.load_state_dict(torch.load(f"best_model_fold_{fold}_{it}.pth"))
    model.eval()
    with torch.no_grad():
        start_idx = 0
        for i, (categorical, numerical, time_series, targets) in enumerate(val_loader):
            categorical, numerical, time_series, targets = (
                categorical.to(device),
                numerical.to(device),
                time_series.to(device),
                targets.to(device),
            )
            outputs = model(categorical, numerical, time_series)
            temp_val_preds = (
                (outputs.softmax(dim=1) * torch.tensor([0, 1, 2, 3], device=device))
                .sum(dim=1)
                .cpu()
                .numpy()
            )
            temp_targets = targets.cpu().numpy()
            batch_size = targets.size(0)
            end_idx = start_idx + batch_size
            oof_predictions[start_idx:end_idx] = temp_val_preds
            oof_targets[start_idx:end_idx] = temp_targets
            start_idx = end_idx
    fold_qwk = cohen_kappa_score(
        oof_predictions.round(), oof_targets, weights="quadratic"
    )
    logger.info(f"Fold {fold} QWK: {fold_qwk:.4f}")
    return fold_qwk, oof_predictions, oof_targets

def train_main():
    import pandas as pd
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    if DROP_NAN:
        train_df = train_df.dropna(subset=["sii"])
    else:
        train_df["sii"] = train_df["sii"].fillna(0)
    if train_df["sii"].isnull().any():
        raise ValueError("Some target values are still missing!")
    tabular_data = train_df[COLS]
    targets = train_df["sii"].values
    ids = train_df["id"].values
    categorical_features = list(range(9))
    numerical_features = list(range(9, len(COLS)))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_qwks = []
    all_oof_predictions = np.zeros(len(targets))
    all_oof_targets = np.zeros(len(targets))
    preprocessors = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    for fold, (train_idx, val_idx) in enumerate(skf.split(tabular_data, targets), 1):
        logger.info(f"Starting fold {fold}")
        preprocessor = create_preprocessor(categorical_features, numerical_features)
        fold_train_data = tabular_data.iloc[train_idx]
        fold_val_data = tabular_data.iloc[val_idx]
        tabular_train = preprocessor.fit_transform(fold_train_data)
        tabular_val = preprocessor.transform(fold_val_data)
        preprocessors.append(preprocessor)
        ids_train, ids_val = ids[train_idx], ids[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]
        def ts_exists(x, mode='train'):
            path = os.path.join(DATA_PATH, f"series_{mode}.parquet/id={x}/part-0.parquet")
            return 1 if os.path.exists(path) else 0
        train_ts_indicator = np.array([ts_exists(x, 'train') for x in ids_train]).reshape(-1, 1)
        val_ts_indicator = np.array([ts_exists(x, 'train') for x in ids_val]).reshape(-1, 1)
        tabular_train = np.column_stack([tabular_train, train_ts_indicator])
        tabular_val = np.column_stack([tabular_val, val_ts_indicator])
        categorical_dims = [
            len(
                preprocessor.named_transformers_["cat"]
                .named_steps["encoder"]
                .categories_[i]
            )
            for i in range(len(categorical_features))
        ]
        train_dataset = HybridDataset(tabular_train, ids_train, y_train)
        val_dataset = HybridDataset(tabular_val, ids_val, y_val)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )
        model = HybridModel(
            categorical_dims=categorical_dims,
            numerical_dim=tabular_train.shape[1] - len(categorical_dims) - 1,
            time_series_dim=7,
            embedding_dim=8,
            hidden_dim=128,
            num_classes=4,
        ).to(device)
        fold_qwk_1, oof_predictions_1, oof_targets_1 = train_and_evaluate(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=LR[0],
            device=device,
            patience=15,
            fold=fold,
            wur=WARMUP_RATIO[0],
            it=0,
        )
        fold_qwk_2, oof_predictions_2, oof_targets_2 = train_and_evaluate(
            copy.deepcopy(model),
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=LR[1],
            device=device,
            patience=15,
            fold=fold,
            wur=WARMUP_RATIO[1],
            it=1,
        )
        combined_predictions = (oof_predictions_1 + oof_predictions_2) / 2
        combined_targets = (oof_targets_1 + oof_targets_2) / 2
        fold_qwks.extend([fold_qwk_1, fold_qwk_2])
        all_oof_predictions[val_idx] = combined_predictions
        all_oof_targets[val_idx] = combined_targets
    Kappa_optimizer = minimize(
        evaluate_predictions,
        x0=[0.5, 1.5, 2.5],
        args=(all_oof_targets, all_oof_predictions),
        method="Nelder-Mead",
    )
    if not Kappa_optimizer.success:
        logger.warning("Threshold optimization did not converge. Using default thresholds.")
        optimal_thresholds = [0.5, 1.5, 2.5]
    else:
        optimal_thresholds = Kappa_optimizer.x
    oof_tuned = threshold_rounder(all_oof_predictions, optimal_thresholds)
    kappa_optimized = quadratic_weighted_kappa(all_oof_targets, oof_tuned)
    default_thresholds = [0.5, 1.5, 2.5]
    oof_not_tuned = threshold_rounder(all_oof_predictions, default_thresholds)
    oof_qwk = quadratic_weighted_kappa(all_oof_targets, oof_not_tuned)
    logger.info(f"Mean of fold QWKs: {np.mean(fold_qwks):.4f}")
    logger.info(f"OOF QWK (not optimized): {oof_qwk:.4f}")
    logger.info(f"OOF QWK (optimized): {kappa_optimized:.4f}")
    logger.info(f"Optimal thresholds: {optimal_thresholds}")
    return preprocessors, optimal_thresholds
