"""
Retrain production model with regularized hyperparameters
Uses high-quality stocks from NSE Universe with complete yfinance + screener data
"""

import pandas as pd
import numpy as np
from pe_prediction_model import PEPredictionModel
from datetime import datetime

print("="*70)
print("RETRAINING PRODUCTION MODEL WITH REGULARIZED HYPERPARAMETERS")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Initialize model
model = PEPredictionModel('indian_stocks_tickers.csv')

# Load high-quality dataset
print("Loading high-quality stocks dataset and fetching fresh data...")
existing_df = pd.read_csv('nse_high_quality_stocks.csv')
# Support both 'ticker' and 'Ticker' column naming
ticker_col = 'ticker' if 'ticker' in existing_df.columns else 'Ticker'
tickers = existing_df[ticker_col].tolist()

print(f"Fetching fresh data for {len(tickers)} high-quality stocks...")
fresh_data = []
for i, ticker in enumerate(tickers):
    if (i + 1) % 50 == 0:
        print(f"Progress: {i + 1}/{len(tickers)} stocks fetched...")
    data = model.fetch_stock_data(ticker)
    if data:
        fresh_data.append(data)

high_quality_df = pd.DataFrame(fresh_data)
# Save updated data back to csv so it gets committed by GitHub Actions
high_quality_df.to_csv('nse_high_quality_stocks.csv', index=False)
print(f"\nSaved {len(high_quality_df)} stocks with fresh data to nse_high_quality_stocks.csv\n")

# Prepare features
print("Preparing features...")
X, y = model.prepare_features(high_quality_df)
print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X)}\n")

# Train model with regularized configuration (already updated in pe_prediction_model.py)
print("="*70)
print("TRAINING MODEL WITH REGULARIZED CONFIGURATION")
print("="*70)
print("\nConfiguration:")
print("  n_estimators: 100")
print("  max_depth: 12 (reduced from 15)")
print("  min_samples_split: 10 (increased from 5)")
print("  min_samples_leaf: 4 (increased from 2)")
print("  max_features: 0.6 (changed from 'sqrt')")
print("\nExpected improvement:")
print("  - Reduce overfitting from 40.5% to ~25.8%")
print("  - Improve test R² by ~33%")
print("  - Better real-world prediction accuracy\n")

metrics = model.train_model(X, y, test_size=0.2, random_state=42)

# Calculate overfitting gap
overfitting_gap = metrics['train_r2'] - metrics['test_r2']

print("\n" + "="*70)
print("TRAINING RESULTS")
print("="*70)

print("\nTraining Set Performance:")
print(f"  R²:   {metrics['train_r2']:.4f}")
print(f"  MAE:  {metrics['train_mae']:.2f} P/E points")
print(f"  RMSE: {metrics['train_rmse']:.2f} P/E points")

print("\nTest Set Performance:")
print(f"  R²:   {metrics['test_r2']:.4f}")
print(f"  MAE:  {metrics['test_mae']:.2f} P/E points")
print(f"  RMSE: {metrics['test_rmse']:.2f} P/E points")

print("\nOverfitting Analysis:")
print(f"  Train-Test R² Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.1f}%)")

if overfitting_gap < 0.20:
    print("  Status: Low overfitting - Excellent!")
elif overfitting_gap < 0.30:
    print("  Status: Moderate overfitting - Good")
else:
    print("  Status: High overfitting - Needs improvement")

# Save production model
print("\n" + "="*70)
print("SAVING PRODUCTION MODEL")
print("="*70)
model.save_model('pe_prediction_model.pkl')
print("\nModel saved: pe_prediction_model.pkl")

# Generate visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)
model.visualize_model(metrics, output_dir='model_visualizations')

print("\n" + "="*70)
print("PRODUCTION MODEL UPDATE COMPLETE!")
print("="*70)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Training stocks:        {len(X) - int(len(X)*0.2)}")
print(f"Test stocks:            {int(len(X)*0.2)}")
print(f"Total features:         {X.shape[1]}")
print(f"Test R²:                {metrics['test_r2']:.4f}")
print(f"Test MAE:               {metrics['test_mae']:.2f} P/E points")
print(f"Overfitting gap:        {overfitting_gap*100:.1f}%")
print("\nFiles updated:")
print("  - pe_prediction_model.pkl (production model)")
print("  - model_visualizations/ (updated charts)")
print("\nReady for GitHub deployment!")
