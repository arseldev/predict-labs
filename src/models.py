"""
models.py — Model Training (Logistic Regression, LightGBM, XGBoost)

Fungsi utama:
- train_logistic_regression(): Baseline model dengan StandardScaler
- train_lightgbm(): Model utama menggunakan LightGBM Classifier
- train_xgboost(): Model alternatif menggunakan XGBoost Classifier
- save_model(): Menyimpan model terlatih ke file pickle
- load_model(): Memuat model dari file pickle
"""

import os
import pickle
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Baseline model — StandardScaler -> LogisticRegression.
    """
    logger.info(f"Training Logistic Regression baseline with {len(X_train)} samples...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            solver="lbfgs",
            class_weight="balanced", 
            random_state=42
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None
) -> lgb.LGBMClassifier:
    """
    Model utama menggunakan LightGBM Classifier dengan early stopping opsional.
    """
    logger.info(f"Training LightGBM model with {len(X_train)} samples...")
    
    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    }
    
    if params:
        default_params.update(params)
        
    model = lgb.LGBMClassifier(**default_params)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
    else:
        model.fit(X_train, y_train)
        
    return model

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None
) -> xgb.XGBClassifier:
    """
    Model alternatif menggunakan XGBoost Classifier.
    """
    logger.info(f"Training XGBoost model with {len(X_train)} samples...")
    
    # Hitung ratio scale_pos_weight untuk imbalance class
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    scale_pw = float(num_neg) / float(max(1, num_pos))
    
    default_params = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pw,
        "tree_method": "hist",
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": -1
    }
    
    if params:
        default_params.update(params)
        
    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path: str, feature_cols: list = None):
    """
    Menyimpan model terlatih ke file pickle.
    Format: dict {"model": estimator, "feature_cols": [...]}
    """
    Path_dir = os.path.dirname(model_path)
    if Path_dir:
        os.makedirs(Path_dir, exist_ok=True)

    # Selalu simpan sebagai dict agar feature_cols ikut tersimpan
    payload = {"model": model, "feature_cols": feature_cols or []}
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Successfully saved model to {model_path} (feature_cols={len(feature_cols or [])} features)")

def load_model(model_path: str):
    """
    Memuat model dari file pickle.
    Mengembalikan tuple (model, feature_cols).
    Kompatibel dengan format lama (pickle langsung LGBMClassifier).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    # Format baru: dict {"model": ..., "feature_cols": [...]}
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_cols = payload.get("feature_cols", [])
        logger.info(f"Successfully loaded model from {model_path} (format=dict, feature_cols={len(feature_cols)})")
        return model, feature_cols

    # Format lama: langsung estimator
    model = payload
    feature_cols = []
    if hasattr(model, "feature_name_"):
        feature_cols = list(model.feature_name_)
        logger.info(f"Loaded legacy model format, feature_cols extracted from model ({len(feature_cols)})")
    else:
        logger.warning("Legacy model has no feature_name_ attribute. feature_cols will be empty!")
    logger.info(f"Successfully loaded model from {model_path} (format=legacy)")
    return model, feature_cols
