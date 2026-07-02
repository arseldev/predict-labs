"""
main.py — Entry Point Utama BTC 5-Minute Prediction System

Usage:
    python main.py --mode backtest
    python main.py --mode paper
    python main.py --mode retrain
"""

import argparse
import sys
import os
import time
from loguru import logger
import pandas as pd

from src.data_fetch import load_config, get_binance_client, fetch_all_historical_klines
from src.features import build_all_features
from src.labeling import build_labels
from src.validation import WalkForwardConfig, walk_forward_splits
from src.models import train_lightgbm, save_model, load_model
from src.backtest import run_backtest, BacktestConfig
from src.evaluate import compute_backtest_metrics, summarize_walk_forward, compute_fold_metrics
from src.live_predict import LivePredictor

# Setup loguru logging format
logger.remove()
logger.add(
    sys.stdout, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

def run_backtest_flow(config: dict, model_path: str):
    logger.info("Starting Backtest and Training Pipeline...")
    
    testnet = config["binance"].get("testnet", True)
    client = get_binance_client(testnet=testnet)
    
    # 1. Fetch data kline historis
    symbol = config["binance"]["symbol"]
    days_back = config["data"]["history"].get("days_back", 90)
    
    logger.info("Step 1: Fetching historical kline data...")
    df_5m = fetch_all_historical_klines(client, symbol, "5m", days_back=days_back)
    df_1h = fetch_all_historical_klines(client, symbol, "1h", days_back=days_back)
    
    if df_5m.empty or df_1h.empty:
        logger.error("Failed to fetch historical kline data.")
        return
        
    # 2. Build features
    logger.info("Step 2: Building features...")
    df_features = build_all_features(
        df_5m=df_5m,
        df_1h=df_1h,
        config=config.get("features", {})
    )
    
    # 3. Build labels
    logger.info("Step 3: Creating labels...")
    df_dataset = build_labels(df_features, config)
    
    # Ambil list kolom fitur yang valid
    feature_cols = [
        col for col in df_dataset.columns 
        if col not in ["label", "label_fh", "future_ret", "label_tb", "label_tb_bin", "barrier_top", "barrier_bot", "hit_barrier", "sample_weight", "close_time"]
    ]
    
    # Simpan list feature columns ke config agar runtime prediction tau fiturnya
    config["features"]["feature_columns"] = feature_cols
    
    # 4. Train Model & Walk-Forward Validation
    logger.info("Step 4: Running Walk-Forward validation training...")
    wf_cfg = WalkForwardConfig(
        n_splits=config["validation"]["walk_forward"].get("n_splits", 10),
        train_period_days=config["validation"]["walk_forward"].get("train_period_days", 45),
        test_period_days=config["validation"]["walk_forward"].get("test_period_days", 7),
        embargo_periods=config["validation"]["walk_forward"].get("embargo_periods", 12)
    )
    
    fold_results = []
    trained_model = None
    
    for fold_num, (train_idx, test_idx) in enumerate(walk_forward_splits(df_dataset, wf_cfg)):
        X_train = df_dataset.loc[train_idx, feature_cols]
        y_train = df_dataset.loc[train_idx, "label"]
        X_test = df_dataset.loc[test_idx, feature_cols]
        y_test = df_dataset.loc[test_idx, "label"]
        
        logger.info(f"Fold {fold_num+1}: Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train model utama (LightGBM)
        model = train_lightgbm(X_train, y_train)
        
        # Prediksi test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Hitung metrik
        metrics = compute_fold_metrics(
            fold_num=fold_num + 1,
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            test_idx=test_idx,
            model=model,
            feature_cols=feature_cols
        )
        # Isi parameter n_train
        metrics.n_train = len(X_train)
        fold_results.append(metrics)
        
        # Simpan instance model terakhir untuk dipakai backtest/live
        trained_model = model
        
    if not fold_results:
        logger.error("No splits were generated. Check dataset length vs train/test periods.")
        return
        
    # Tampilkan rekap walk forward
    summary = summarize_walk_forward(fold_results)
    logger.info(f"Walk-Forward Validation Mean Accuracy: {summary['mean_accuracy']:.4f} (±{summary['std_accuracy']:.4f})")
    logger.info(f"Walk-Forward Validation Mean ROC-AUC: {summary['mean_roc_auc']:.4f}")
    
    # Simpan model terakhir beserta daftar fiturnya ke disk
    save_model(trained_model, model_path, feature_cols=feature_cols)
    
    # 5. Jalankan Backtest Realistis di Seluruh Dataset
    logger.info("Step 5: Simulating backtest on full dataset...")
    bt_cfg = BacktestConfig(
        fee_taker=config["trading"].get("fee_taker", 0.001),
        fee_maker=config["trading"].get("fee_maker", 0.001),
        slippage_pct=config["trading"].get("slippage_pct", 0.0002),
        probability_threshold=config["trading"].get("probability_threshold", 0.60),
        position_size_pct=config["trading"].get("position_size_pct", 0.02),
        profit_target_pct=config["trading"].get("profit_target_pct", 0.0015),
        stop_loss_pct=config["trading"].get("stop_loss_pct", 0.0015),
        max_hold_candles=config["trading"].get("max_hold_candles", 6),
        max_daily_loss_pct=config["trading"].get("max_daily_loss_pct", 0.03)
    )
    
    trades, equity = run_backtest(df_dataset, trained_model, feature_cols, bt_cfg)
    
    if not trades:
        logger.warning("No trades executed during backtest simulation.")
        return
        
    # Hitung metrik backtest
    bt_metrics = compute_backtest_metrics(trades, equity, bt_cfg)
    
    logger.info("============================================================")
    logger.info("BACKTEST RESULTS REPORT")
    logger.info(f"Total Trades: {bt_metrics.get('n_trades')}")
    logger.info(f"Win Rate:     {bt_metrics.get('win_rate'):.2%}")
    logger.info(f"Avg Win:      {bt_metrics.get('avg_win_pct'):.4%}")
    logger.info(f"Avg Loss:     {bt_metrics.get('avg_loss_pct'):.4%}")
    logger.info(f"Risk-Reward:  {bt_metrics.get('risk_reward_ratio'):.2f}x")
    logger.info(f"Sharpe Ratio: {bt_metrics.get('sharpe_ratio'):.2f}")
    logger.info(f"Max Drawdown: {bt_metrics.get('max_drawdown_pct'):.2%}")
    logger.info(f"Total Return: {bt_metrics.get('total_return_pct'):.2%}")
    logger.info(f"Expected Value (EV) per trade: {bt_metrics.get('ev_per_trade'):.4%}")
    logger.info("============================================================")
    
    if bt_metrics.get('ev_per_trade', -1) > 0.0:
        logger.info("✅ KESIMPULAN: EV > 0, Lolos kriteria untuk Paper Trading!")
    else:
        logger.warning("❌ KESIMPULAN: EV <= 0, Sistem rugi setelah fee/slippage. Tune fitur atau model.")

def run_paper_trading(config: dict, model_path: str):
    logger.info("Starting Paper Trading on Binance Testnet...")
    
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}. Please run backtest first to train a model.")
        return
        
    predictor = LivePredictor(config, model_path)
    
    try:
        predictor.start()
        # Keep main thread alive
        while True:
            time.sleep(60)
            # Tampilkan report performa singkat
            perf = predictor._pred_logger.get_live_performance(days_back=1)
            logger.info(f"Live Stats (Past 24h) | Trades: {perf['total_trades']} | Win Rate: {perf['win_rate']:.1%} | Net PnL: ${perf['total_net_pnl']:.2f}")
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
        predictor.stop()
    except Exception as e:
        logger.critical(f"Critical error in predictor: {e}", exc_info=True)
        predictor.stop()

def main():
    parser = argparse.ArgumentParser(description="BTC 5m Prediction Trading Bot")
    parser.add_argument("--mode", choices=["backtest", "paper", "live", "retrain"], required=True,
                        help="Mode untuk menjalankan sistem")
    parser.add_argument("--config", default="config/config.yaml", help="Path ke config.yaml")
    parser.add_argument("--model", default="models/latest.pkl", help="Path ke file model pickle")
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
        
    if args.mode == "live":
        confirm = input("⚠️ WARNING: LIVE MODE akan bertransaksi dengan UANG RIIL! Ketik 'CONFIRM' untuk melanjutkan: ")
        if confirm != "CONFIRM":
            logger.info("Live mode dibatalkan.")
            return
        config["binance"]["testnet"] = False
        logger.warning("🚨 MENJALANKAN MODE LIVE. HATI-HATI!")
        run_paper_trading(config, args.model)
        
    elif args.mode == "paper":
        config["binance"]["testnet"] = True
        run_paper_trading(config, args.model)
        
    elif args.mode in ["backtest", "retrain"]:
        run_backtest_flow(config, args.model)

if __name__ == "__main__":
    main()
