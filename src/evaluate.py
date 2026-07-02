"""
evaluate.py — Metrik Evaluasi (ML Validation & Backtest)

Fungsi utama:
- FoldResult: Kelas data penyimpan metrik per fold
- compute_fold_metrics(): Hitung metrik performa ML untuk satu fold
- summarize_walk_forward(): Rekap hasil walk-forward splits
- compute_backtest_metrics(): Hitung metrik trading (EV, Sharpe, Max Drawdown)
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)

@dataclass
class FoldResult:
    fold_num: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss: float
    n_train: int
    n_test: int
    train_period: tuple        # (start, end) datetime
    test_period: tuple         # (start, end) datetime
    feature_importance: dict   # {feature_name: importance}
    market_regime: str         # 'bullish', 'bearish', 'sideways'

def _get_feature_importance(model, feature_cols: list) -> dict:
    """Ekstrak feature importance dari LightGBM/XGBoost/Logistic Regression."""
    importance_dict = {}
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_dict = {name: float(imp) for name, imp in zip(feature_cols, importances)}
        elif hasattr(model, "named_steps") and "model" in model.named_steps:
            # Logistic Regression pipeline
            inner_model = model.named_steps["model"]
            if hasattr(inner_model, "coef_"):
                coefs = np.abs(inner_model.coef_[0])
                importance_dict = {name: float(c) for name, c in zip(feature_cols, coefs)}
    except Exception:
        pass
    return importance_dict

def _classify_regime(y_true: pd.Series) -> str:
    """Klasifikasi sederhana kondisi pasar berdasarkan distribusi kelas label."""
    # Bullish jika net ratio label 1 > 0.55, Bearish jika < 0.45, else Sideways
    ratio_up = y_true.mean()
    if ratio_up > 0.55:
        return "bullish"
    elif ratio_up < 0.45:
        return "bearish"
    else:
        return "sideways"

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
    Hitung metrik evaluasi klasifikasi ML untuk satu fold.
    """
    # Pastikan data non-kosong untuk menghindari pembagian dengan nol
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    rec = recall_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    
    try:
        auc = roc_auc_score(y_true, y_proba) if len(y_true) > 0 and len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5
        
    try:
        loss = log_loss(y_true, y_proba) if len(y_true) > 0 else 1.0
    except Exception:
        loss = 1.0
        
    regime = _classify_regime(y_true)
    importance = _get_feature_importance(model, feature_cols)
    
    return FoldResult(
        fold_num=fold_num,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc,
        log_loss=loss,
        n_train=0,  # Akan diisi oleh orchestrator run_walk_forward
        n_test=len(y_true),
        train_period=(None, None),
        test_period=(test_idx[0], test_idx[-1]) if len(test_idx) > 0 else (None, None),
        feature_importance=importance,
        market_regime=regime
    )

def summarize_walk_forward(fold_results: list[FoldResult]) -> dict:
    """
    Kompilasi semua FoldResult menjadi statistik rekapitulasi.
    """
    if not fold_results:
        return {}
        
    accs = [r.accuracy for r in fold_results]
    aucs = [r.roc_auc for r in fold_results]
    losses = [r.log_loss for r in fold_results]
    
    # Rata-rata feature importance dari semua fold
    all_importances = {}
    for r in fold_results:
        for name, val in r.feature_importance.items():
            all_importances[name] = all_importances.get(name, []) + [val]
            
    mean_importances = {name: float(np.mean(vals)) for name, vals in all_importances.items()}
    sorted_importances = sorted(mean_importances.items(), key=lambda x: x[1], reverse=True)
    
    # Kinerja per regime
    regime_groups = {}
    for r in fold_results:
        regime_groups[r.market_regime] = regime_groups.get(r.market_regime, []) + [r.accuracy]
        
    per_regime = {
        regime: {
            "mean_accuracy": float(np.mean(acc_list)),
            "std_accuracy": float(np.std(acc_list)) if len(acc_list) > 1 else 0.0,
            "n_folds": len(acc_list)
        }
        for regime, acc_list in regime_groups.items()
    }
    
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "min_accuracy": float(np.min(accs)),
        "max_accuracy": float(np.max(accs)),
        "mean_roc_auc": float(np.mean(aucs)),
        "mean_log_loss": float(np.mean(losses)),
        "per_regime": per_regime,
        "top_features": sorted_importances[:10],
        "fold_details": [
            {
                "fold": r.fold_num,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "roc_auc": r.roc_auc,
                "log_loss": r.log_loss,
                "regime": r.market_regime,
                "test_period": f"{r.test_period[0]} -> {r.test_period[1]}"
            }
            for r in fold_results
        ]
    }

def compute_backtest_metrics(
    trades: list,
    equity_curve: pd.DataFrame,
    config
) -> dict:
    """
    Hitung metrik trading pasca backtest (Expected Value, Sharpe, Max Drawdown).
    """
    if not trades:
        return {"error": "No trades generated"}
        
    net_pnls = [t.net_pnl_pct for t in trades]
    winning_trades = [t for t in trades if t.net_pnl > 0]
    losing_trades = [t for t in trades if t.net_pnl <= 0]
    
    win_rate = len(winning_trades) / len(trades)
    avg_win = float(np.mean([t.net_pnl_pct for t in winning_trades])) if winning_trades else 0.0
    avg_loss = float(abs(np.mean([t.net_pnl_pct for t in losing_trades]))) if losing_trades else 0.0
    
    # Expected Value (EV) per trade
    ev = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
    
    # Biaya transaksi
    total_fee = sum(t.fee_paid for t in trades)
    total_slippage = sum(t.slippage_cost for t in trades)
    total_gross_pnl = sum(t.gross_pnl for t in trades)
    
    # Sharpe Ratio harian di-annualisasikan (288 candle 5m = 1 hari)
    equity_returns = equity_curve["equity"].pct_change().dropna()
    if len(equity_returns) > 1 and equity_returns.std() > 0:
        sharpe = float((equity_returns.mean() / equity_returns.std()) * np.sqrt(288.0 * 365.0))
    else:
        sharpe = 0.0
        
    # Maximum Drawdown
    rolling_max = equity_curve["equity"].cummax()
    drawdowns = (equity_curve["equity"] - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())
    
    total_return = float((equity_curve["equity"].iloc[-1] - equity_curve["equity"].iloc[0]) / equity_curve["equity"].iloc[0])
    rr_ratio = avg_win / max(avg_loss, 1e-10)
    
    # Exit Reason distribution
    exit_counts = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
    exit_breakdown = {reason: float(count / len(trades)) for reason, count in exit_counts.items()}
    
    # Probability Calibration
    buckets = {"0.60-0.65": [], "0.65-0.70": [], "0.70-0.75": [], "0.75+": []}
    for t in trades:
        p = t.predicted_proba
        if p < 0.65:
            bucket = "0.60-0.65"
        elif p < 0.70:
            bucket = "0.65-0.70"
        elif p < 0.75:
            bucket = "0.70-0.75"
        else:
            bucket = "0.75+"
        buckets[bucket].append(t.net_pnl > 0)
        
    prob_buckets = {
        bucket: {
            "n_trades": len(results),
            "win_rate": float(np.mean(results)) if results else None
        }
        for bucket, results in buckets.items()
    }
    
    return {
        "ev_per_trade": ev,
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "risk_reward_ratio": rr_ratio,
        "total_return_pct": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown,
        "n_trades": len(trades),
        "n_winning": len(winning_trades),
        "n_losing": len(losing_trades),
        "total_fee_paid": total_fee,
        "total_slippage_cost": total_slippage,
        "fee_as_pct_gross_pnl": float(total_fee / max(abs(total_gross_pnl), 1e-10)),
        "exit_breakdown": exit_breakdown,
        "prob_buckets": prob_buckets
    }
