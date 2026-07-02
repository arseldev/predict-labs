# Phase 4 — Model & Validasi Walk-Forward

> **Tujuan:** Train model prediksi arah harga dengan validasi yang benar (walk-forward), evaluasi per fold, dan hindari semua bentuk data leakage. Model dengan akurasi "bagus" di sini baru layak lanjut ke Phase 5.

---

## 4.1 Prinsip Validasi yang Benar

### ❌ Cara Salah (JANGAN LAKUKAN)
```python
# SALAH: Random split — data finansial berurutan waktu!
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SALAH: K-Fold biasa — melanggar urutan waktu!
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
```

### ✅ Cara Benar: Walk-Forward Validation
```
Timeline:  ─────────────────────────────────────────────────────────────▶

Fold 1:   [====TRAIN====][TEST]
Fold 2:   [========TRAIN========][TEST]
Fold 3:   [===========TRAIN===========][TEST]
Fold 4:   [==============TRAIN==============][TEST]
...

Setiap fold: train pada data SEBELUMNYA, test pada data BERIKUTNYA.
Tidak pernah "mengintip" masa depan.
```

---

## 4.2 `src/validation.py` — Walk-Forward Implementation

```python
"""
validation.py — Time-Series Cross-Validation

Implements:
1. Walk-Forward Validation (utama)
2. Purged K-Fold CV (alternatif yang lebih statistik)
"""

from dataclasses import dataclass
from typing import Iterator, Tuple
import pandas as pd
import numpy as np

@dataclass
class WalkForwardConfig:
    n_splits: int = 10
    train_period_days: int = 45
    test_period_days: int = 7
    embargo_periods: int = 12  # candle, bukan hari
    min_train_samples: int = 1000

def walk_forward_splits(
    df: pd.DataFrame,
    config: WalkForwardConfig
) -> Iterator[Tuple[pd.Index, pd.Index]]:
    """
    Generator: menghasilkan (train_idx, test_idx) untuk setiap fold.
    
    Strategi: expanding window (train window makin besar setiap fold).
    Alternatif: rolling window (ukuran train fixed) — implementasikan keduanya.
    
    Timeline untuk n_splits=5, train=45 hari, test=7 hari:
        Fold 1: Train [day 0 - 44], Test [day 45 - 51]
        Fold 2: Train [day 0 - 51], Test [day 52 - 58]
        Fold 3: Train [day 0 - 58], Test [day 59 - 65]
        ...
    
    embargo_periods: candle yang dibuang antara train akhir dan test awal
    untuk menghindari label overlap (triple-barrier).
    
    Args:
        df: DataFrame dengan DatetimeIndex
        config: WalkForwardConfig
    
    Yields:
        (train_indices, test_indices) sebagai pd.Index
    """
    freq = pd.infer_freq(df.index[:100])  # e.g., '5T' untuk 5 menit
    candles_per_day = 24 * 60 // 5  # = 288 untuk 5m
    
    train_candles = config.train_period_days * candles_per_day
    test_candles = config.test_period_days * candles_per_day
    embargo = config.embargo_periods
    
    n = len(df)
    
    for i in range(config.n_splits):
        # Expanding window: setiap fold train sampai test_end fold sebelumnya
        train_end_idx = train_candles + i * test_candles
        if train_end_idx >= n:
            break
        
        # Embargo: lewati N candle setelah train end
        test_start_idx = train_end_idx + embargo
        test_end_idx = min(test_start_idx + test_candles, n)
        
        if test_start_idx >= n or len(df.index[test_start_idx:test_end_idx]) == 0:
            break
        
        train_idx = df.index[:train_end_idx]
        test_idx = df.index[test_start_idx:test_end_idx]
        
        if len(train_idx) < config.min_train_samples:
            continue
        
        yield train_idx, test_idx


def purged_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    embargo_pct: float = 0.01
) -> Iterator[Tuple[pd.Index, pd.Index]]:
    """
    Purged K-Fold CV: buang sampel training yang labelnya overlap dengan test set.
    
    Digunakan sebagai alternatif/pelengkap walk-forward untuk hyperparameter tuning.
    BUKAN pengganti walk-forward untuk evaluasi final performa.
    
    Args:
        df: DataFrame dengan DatetimeIndex
        n_folds: jumlah fold
        embargo_pct: persentase data yang di-embargo setelah test set
    
    Yields:
        (train_indices, test_indices) sebagai pd.Index
    """
    n = len(df)
    fold_size = n // n_folds
    embargo_size = int(n * embargo_pct)
    
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = test_start + fold_size
        
        # Embargo sebelum dan setelah test set
        purge_start = max(0, test_start - embargo_size)
        purge_end = min(n, test_end + embargo_size)
        
        train_mask = np.ones(n, dtype=bool)
        train_mask[purge_start:purge_end] = False
        
        train_idx = df.index[train_mask]
        test_idx = df.index[test_start:test_end]
        
        yield train_idx, test_idx
```

---

## 4.3 `src/models.py` — Model Training

### Model 1: Logistic Regression (Baseline Wajib)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Baseline model — WAJIB dijalankan pertama.
    Kalau model canggih tidak mengalahkan ini, ada masalah.
    
    Pipeline: StandardScaler → LogisticRegression
    
    Kenapa perlu StandardScaler: LR sensitif terhadap skala fitur.
    Kenapa Pipeline: supaya scaling juga di-apply saat predict, bukan cuma saat train.
    
    Hyperparameter:
        C=1.0 (regularization default)
        max_iter=1000
        solver='lbfgs'
        class_weight='balanced'  # handle class imbalance
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe
```

### Model 2: LightGBM (Model Utama)

```python
import lightgbm as lgb

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None
) -> lgb.LGBMClassifier:
    """
    Model utama berdasarkan rekomendasi literatur untuk 5-menit prediction.
    
    Default params (dari config.yaml):
        n_estimators: 500
        max_depth: 6
        learning_rate: 0.05
        num_leaves: 63
        min_child_samples: 50
        subsample: 0.8
        colsample_bytree: 0.8
        class_weight: 'balanced'
        random_state: 42
    
    Early stopping:
        Jika X_val, y_val tersedia → gunakan sebagai validation set untuk early stopping.
        Ini mencegah overfitting pada data training.
        early_stopping_rounds=50
    
    Feature importance:
        Setelah training, log feature importance (gain-based) ke file.
        Ini kritis untuk debugging dan feature selection.
    
    Args:
        X_train, y_train: training data
        X_val, y_val: opsional validation data untuk early stopping
        params: override default params
    
    Returns:
        Trained LGBMClassifier
    """
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
        "verbose": -1
    }
    if params:
        default_params.update(params)
    
    model = lgb.LGBMClassifier(**default_params)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
    else:
        model.fit(X_train, y_train)
    
    return model
```

### Model 3: XGBoost (Alternatif)

```python
import xgboost as xgb

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None
) -> xgb.XGBClassifier:
    """
    Alternatif LightGBM. Gunakan untuk perbandingan.
    
    Default params:
        n_estimators: 500
        max_depth: 5
        learning_rate: 0.05
        subsample: 0.8
        colsample_bytree: 0.8
        scale_pos_weight: (n_neg / n_pos)  # handle class imbalance
        tree_method: 'hist'  # lebih cepat
        random_state: 42
    
    XGBoost vs LightGBM:
        - LightGBM biasanya lebih cepat untuk data besar
        - XGBoost lebih stable untuk dataset kecil
        - Keduanya perlu dicoba untuk perbandingan
    """
    scale_pw = (y_train == 0).sum() / (y_train == 1).sum()
    
    default_params = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pw,
        "tree_method": "hist",
        "random_state": 42,
        "eval_metric": "logloss"
    }
    if params:
        default_params.update(params)
    
    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model
```

### Model 4: LSTM (Opsional — Setelah Baseline Solid)

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM untuk prediksi arah harga. Arsitektur standar.
    
    Architecture:
        Input: (batch_size, seq_len, n_features)
        LSTM layers: 2 layers, hidden_size=128, dropout=0.2
        Output: Linear → Sigmoid → probabilitas [0, 1]
    
    Kapan dipakai:
        - Setelah LightGBM baseline sudah solid dan validated
        - Butuh data lebih banyak dari tree-based model
        - Sekuensial pattern mungkin lebih baik ditangkap LSTM
    
    Catatan:
        Performa LSTM di 5-menit biasanya setara dengan LightGBM (dari literatur),
        bukan signifikan lebih baik. Tapi bisa lebih baik untuk ensemble.
    
    Args:
        input_size: jumlah fitur
        hidden_size: ukuran hidden state LSTM (default 128)
        num_layers: jumlah layer LSTM (default 2)
        dropout: dropout rate (default 0.2)
        seq_len: panjang sequens historis (default 24 = 2 jam pada 5m)
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # ambil output timestep terakhir
        return self.sigmoid(self.fc(out)).squeeze()
```

---

## 4.4 Walk-Forward Evaluation Loop

```python
from src.evaluate import compute_fold_metrics, FoldResult

def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str,
    model_type: str = "lightgbm",
    wf_config: WalkForwardConfig = None,
    model_params: dict = None
) -> list[FoldResult]:
    """
    Jalankan walk-forward validation end-to-end.
    
    Args:
        df: DataFrame dengan fitur + label, DatetimeIndex
        feature_cols: list nama kolom fitur
        label_col: nama kolom label ('label_fh' atau 'label_tb_bin')
        model_type: 'lightgbm', 'xgboost', 'logistic_regression', 'lstm'
        wf_config: konfigurasi walk-forward
        model_params: override parameter model
    
    Returns:
        List of FoldResult (satu per fold) dengan semua metrik
    """
    if wf_config is None:
        wf_config = WalkForwardConfig()
    
    X = df[feature_cols]
    y = df[label_col]
    
    fold_results = []
    
    for fold_num, (train_idx, test_idx) in enumerate(walk_forward_splits(df, wf_config)):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        
        logger.info(f"Fold {fold_num+1}: Train {len(X_train)} samples "
                    f"({train_idx[0]} → {train_idx[-1]}), "
                    f"Test {len(X_test)} samples ({test_idx[0]} → {test_idx[-1]})")
        
        # Train model
        if model_type == "lightgbm":
            model = train_lightgbm(X_train, y_train, params=model_params)
        elif model_type == "xgboost":
            model = train_xgboost(X_train, y_train, params=model_params)
        elif model_type == "logistic_regression":
            model = train_logistic_regression(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        result = compute_fold_metrics(
            fold_num=fold_num,
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            test_idx=test_idx,
            model=model,
            feature_cols=feature_cols
        )
        fold_results.append(result)
        
        # Log per fold
        logger.info(f"Fold {fold_num+1} results: acc={result.accuracy:.4f}, "
                    f"precision={result.precision:.4f}, recall={result.recall:.4f}")
    
    return fold_results
```

---

## 4.5 `src/evaluate.py` — Metrik Evaluasi

```python
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report
)

@dataclass
class FoldResult:
    fold_num: int
    accuracy: float
    precision: float           # per kelas
    recall: float              # per kelas
    f1: float
    roc_auc: float
    log_loss: float
    n_train: int
    n_test: int
    train_period: tuple        # (start, end) datetime
    test_period: tuple         # (start, end) datetime
    feature_importance: dict   # {feature_name: importance_score}
    market_regime: str         # 'bullish', 'bearish', 'sideways'

def compute_fold_metrics(
    fold_num: int,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    test_idx: pd.Index,
    model,
    feature_cols: list
) -> FoldResult:
    """
    Hitung semua metrik untuk satu fold.
    
    Metrik yang dihitung:
    1. Accuracy (keseluruhan)
    2. Precision per kelas (naik vs turun) — JANGAN hanya global
    3. Recall per kelas
    4. F1 score
    5. ROC-AUC
    6. Log-loss (kalibrasi probabilitas)
    7. Feature importance (LightGBM/XGBoost: gain-based)
    8. Market regime (berdasarkan trend harga di periode test)
    """
    return FoldResult(
        fold_num=fold_num,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba),
        log_loss=log_loss(y_true, y_proba),
        n_train=0,  # diisi caller
        n_test=len(y_true),
        train_period=(None, None),
        test_period=(test_idx[0], test_idx[-1]),
        feature_importance=_get_feature_importance(model, feature_cols),
        market_regime=_classify_regime(y_true)
    )

def summarize_walk_forward(fold_results: list[FoldResult]) -> dict:
    """
    Summarize semua fold menjadi satu laporan.
    
    Output:
        {
          'mean_accuracy': float,
          'std_accuracy': float,
          'mean_roc_auc': float,
          'mean_log_loss': float,
          'per_regime': {
            'bullish': {mean_accuracy, std_accuracy},
            'bearish': {mean_accuracy, std_accuracy},
            'sideways': {mean_accuracy, std_accuracy}
          },
          'top_features': [(feature_name, mean_importance), ...],
          'all_folds': [FoldResult, ...]
        }
    
    PENTING: Laporkan per fold, bukan hanya rata-rata.
    Rata-rata tinggi tapi std tinggi = model tidak konsisten = risiko tinggi.
    """
    accs = [r.accuracy for r in fold_results]
    aucs = [r.roc_auc for r in fold_results]
    
    return {
        "mean_accuracy": np.mean(accs),
        "std_accuracy": np.std(accs),
        "min_accuracy": np.min(accs),
        "max_accuracy": np.max(accs),
        "mean_roc_auc": np.mean(aucs),
        "fold_details": [
            {
                "fold": r.fold_num,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "roc_auc": r.roc_auc,
                "log_loss": r.log_loss,
                "regime": r.market_regime,
                "test_period": f"{r.test_period[0]} → {r.test_period[1]}"
            }
            for r in fold_results
        ]
    }

def _classify_regime(price_series: pd.Series) -> str:
    """Klasifikasi regime pasar berdasarkan pergerakan harga di periode test."""
    # Sederhana: cek apakah net movement > threshold
    # Bullish: naik >5% dalam periode; Bearish: turun >5%; Sideways: di antaranya
    net_change = (price_series.iloc[-1] - price_series.iloc[0]) / price_series.iloc[0]
    if net_change > 0.05:
        return "bullish"
    elif net_change < -0.05:
        return "bearish"
    else:
        return "sideways"
```

---

## 4.6 Laporan Walk-Forward yang Harus Dihasilkan

```
============================================================
WALK-FORWARD VALIDATION REPORT
Model: LightGBM | Label: Triple-Barrier | Features: 38
============================================================

Fold | Period                     | Acc    | AUC    | LogLoss | Regime
-----|----------------------------|--------|--------|---------|--------
  1  | 2024-01-01 → 2024-01-07   | 0.5723 | 0.5891 | 0.6821  | bullish
  2  | 2024-01-08 → 2024-01-14   | 0.5512 | 0.5634 | 0.6934  | sideways
  3  | 2024-01-15 → 2024-01-21   | 0.5891 | 0.6012 | 0.6712  | bearish
  4  | 2024-01-22 → 2024-01-28   | 0.5634 | 0.5756 | 0.6845  | bullish
  5  | 2024-01-29 → 2024-02-04   | 0.5478 | 0.5598 | 0.6967  | sideways
  ...
-----|----------------------------|--------|--------|---------|--------
MEAN |                            | 0.5647 | 0.5778 | 0.6856  |
STD  |                            | ±0.014 | ±0.015 | ±0.008  |
MIN  |                            | 0.5312 |        |         |
MAX  |                            | 0.5923 |        |         |

Per-Regime Performance:
  bullish:   mean_acc=0.5712 ± 0.012
  bearish:   mean_acc=0.5634 ± 0.018
  sideways:  mean_acc=0.5523 ± 0.021  ← biasanya lebih lemah

Top 10 Features by SHAP:
  1. obi_5              0.0823
  2. tfi_5m             0.0712
  3. ret_1              0.0634
  4. micro_price_dev    0.0598
  5. rvol_12            0.0512
  ...

KESIMPULAN: Lanjut ke Phase 5 jika mean_accuracy >= 0.52 dan min_accuracy >= 0.50
============================================================
```

---

## 4.7 Hyperparameter Tuning (Opsional)

Setelah walk-forward baseline selesai, optionally tuning dengan Purged K-Fold:

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    
    # Gunakan purged k-fold untuk tuning (lebih cepat dari full walk-forward)
    fold_accs = []
    for train_idx, test_idx in purged_kfold_splits(df, n_folds=5):
        model = train_lightgbm(X.loc[train_idx], y.loc[train_idx], params=params)
        preds = model.predict(X.loc[test_idx])
        fold_accs.append(accuracy_score(y.loc[test_idx], preds))
    
    return np.mean(fold_accs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
best_params = study.best_params
```

---

## 4.8 Unit Tests (`tests/test_validation.py`)

```python
class TestWalkForward:
    def test_no_temporal_leakage(self):
        """Test bahwa test data tidak overlap dengan train data."""
        df = make_dummy_dataset(1000)  # helper function
        config = WalkForwardConfig(n_splits=5, train_period_days=30, test_period_days=7)
        
        for train_idx, test_idx in walk_forward_splits(df, config):
            # Test tidak boleh ada overlap
            overlap = train_idx.intersection(test_idx)
            assert len(overlap) == 0, "Train dan test index overlap — temporal leakage!"
            
            # Test harus selalu setelah train
            assert train_idx[-1] < test_idx[0], "Test period ada sebelum train period!"
    
    def test_expanding_window(self):
        """Train window harus makin besar setiap fold."""
        df = make_dummy_dataset(2000)
        config = WalkForwardConfig(n_splits=5, train_period_days=30, test_period_days=7)
        
        train_sizes = []
        for train_idx, test_idx in walk_forward_splits(df, config):
            train_sizes.append(len(train_idx))
        
        # Setiap fold train size harus lebih besar atau sama dengan fold sebelumnya
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1], "Train window tidak expanding!"
    
    def test_embargo_respected(self):
        """Pastikan embargo period ada antara train dan test."""
        df = make_dummy_dataset(1000)
        config = WalkForwardConfig(
            n_splits=3, train_period_days=20, test_period_days=5,
            embargo_periods=12
        )
        
        for train_idx, test_idx in walk_forward_splits(df, config):
            train_end_pos = df.index.get_loc(train_idx[-1])
            test_start_pos = df.index.get_loc(test_idx[0])
            gap = test_start_pos - train_end_pos
            assert gap >= config.embargo_periods, \
                f"Embargo tidak cukup: gap={gap}, required={config.embargo_periods}"
```

---

## 4.9 Kriteria Selesai Phase 4

- [ ] `validation.py` dengan `walk_forward_splits()` diimplementasikan dan ditest
- [ ] `models.py` dengan Logistic Regression (baseline) dan LightGBM (utama) diimplementasikan
- [ ] Walk-forward loop berjalan tanpa error di atas 60-90 hari data kline
- [ ] Laporan walk-forward dihasilkan per fold (bukan cuma rata-rata)
- [ ] Precision & recall per kelas dilaporkan
- [ ] Performa model diuji per market regime (bullish/bearish/sideways)
- [ ] LightGBM beat Logistic Regression baseline (kalau tidak, cek fitur)
- [ ] Feature importance dianalisis (SHAP) — top 10 fitur diidentifikasi
- [ ] Semua unit test di `test_validation.py` PASS
- [ ] **Kriteria minimum:** mean_accuracy walk-forward >= 0.52, min_accuracy >= 0.50

**→ Lanjut ke [Phase 5 — Backtest](./phase-5-backtest.md)**
