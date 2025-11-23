# Indian Stock P/E Prediction App

An AI-powered stock valuation tool that predicts fair P/E ratios for Indian stocks using Random Forest machine learning and comprehensive fundamental analysis.

## Features

- **27 Fundamental Metrics**: ROE, ROCE, profit margins, growth rates, debt metrics, and more
- **Random Forest ML Model**: 80/20 train-test split with robust validation
- **Fair Value Calculation**: Predicts fair P/E, calculates target price, shows upside/downside
- **Interactive Streamlit UI**: Easy-to-use interface with charts and visualizations
- **500+ Indian Stocks**: Covers NSE and BSE listed companies
- **Real-time Data**: Fetches latest data from Yahoo Finance

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the Indian stocks ticker CSV file:
   - `indian_stocks_tickers.csv` should be in the same directory

## Usage

### Step 1: Train the Random Forest Model

```bash
python pe_prediction_model.py
```

This will:
- Fetch fundamental data for all stocks in the CSV
- Engineer 27 features including custom ratios
- Train a Random Forest model with 80/20 split
- Generate visualizations (feature importance, actual vs predicted, decision trees)
- Save the trained model to `pe_prediction_model.pkl`

**Output Files:**
- `pe_prediction_model.pkl` - Trained model
- `stock_fundamental_data.csv` - Raw dataset
- `model_visualizations/` - Folder with model visualization images

### Step 2: Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## How It Works

### 1. Data Collection
- Fetches fundamental data from Yahoo Finance for each stock
- Includes: financials, ratios, growth metrics, sector/industry info

### 2. Feature Engineering
The model uses 27 features:

**Financial Metrics:**
- Market Cap, Revenue, Net Income, EBITDA, Gross Profit

**Profitability Ratios:**
- ROE (Return on Equity)
- ROA (Return on Assets)
- ROCE (Return on Capital Employed)
- Profit Margins, Operating Margins, Gross Margins
- NPM/OPM ratios

**Cash Flow Metrics:**
- FCF Margin (Free Cash Flow / Revenue)
- FCF to Net Income ratio

**Efficiency Ratios:**
- Gross Profit / Total Assets
- Asset Turnover

**Growth Metrics:**
- Revenue Growth
- Earnings Growth
- Quarterly Earnings Growth

**Valuation Metrics:**
- Price to Book
- Debt to Equity
- Current Ratio, Quick Ratio

**Other:**
- Dividend Yield, Payout Ratio
- Beta
- Sector & Industry (encoded)

### 3. Model Training
- Algorithm: Random Forest Regressor
- Trees: 100
- Max Depth: 15
- Split: 80% training, 20% testing
- Target: Current P/E ratio

### 4. Prediction & Valuation

For any stock:
1. Fetches current fundamental data
2. Predicts fair P/E using the trained model
3. Calculates fair price: `Fair Price = Predicted P/E × EPS`
4. Computes upside/downside: `(Fair Price - Current Price) / Current Price × 100`

### 5. Interpretation

- **Positive Upside**: Stock is potentially undervalued
- **Negative Downside**: Stock is potentially overvalued
- **Near Zero**: Stock is fairly valued

## Model Visualizations

The training process generates several visualizations:

1. **Feature Importance**: Shows which metrics matter most
2. **Actual vs Predicted**: Model accuracy visualization
3. **Residual Plot**: Error distribution
4. **Sample Decision Tree**: Visual representation of how the model works
5. **Performance Summary**: Training vs testing metrics

## Project Structure

```
.
├── fetch_nse_bse_stocks.py          # Fetch Indian stock tickers
├── indian_stocks_tickers.csv        # List of Indian stocks
├── pe_prediction_model.py           # Model training script
├── streamlit_app.py                 # Streamlit web app
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── pe_prediction_model.pkl          # Trained model (generated)
├── stock_fundamental_data.csv       # Training dataset (generated)
└── model_visualizations/            # Visualization images (generated)
    ├── feature_importance.png
    ├── actual_vs_predicted.png
    ├── residuals.png
    ├── decision_tree_sample.png
    └── model_performance_summary.png
```

## Model Performance

The model's performance is evaluated using:
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more
- **R² Score**: Proportion of variance explained (higher is better)

View detailed metrics in the console output after training.

## Limitations & Disclaimers

1. **Educational Purpose Only**: This tool is for learning and research
2. **Not Investment Advice**: Always do your own due diligence
3. **Data Accuracy**: Depends on Yahoo Finance data quality
4. **Model Limitations**:
   - Past performance doesn't guarantee future results
   - Model trained on historical data
   - Doesn't account for market sentiment, news, or macro events
5. **Market Efficiency**: Stock prices may not always reach fair value

## Example Workflow

```bash
# 1. Generate ticker list (if not already done)
python fetch_nse_bse_stocks.py

# 2. Train the model
python pe_prediction_model.py

# 3. Launch the app
streamlit run streamlit_app.py
```

## Customization

### Adjust Model Parameters

In `pe_prediction_model.py`, modify:

```python
self.model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Maximum tree depth
    min_samples_split=5,   # Min samples to split node
    min_samples_leaf=2,    # Min samples in leaf
    random_state=42
)
```

### Change Train-Test Split

```python
metrics = model.train_model(X, y, test_size=0.2)  # 0.2 = 20% test
```

### Add More Features

Edit `fetch_stock_data()` method to include additional metrics from yfinance.

## Troubleshooting

**Issue**: Model not found error in Streamlit app
- **Solution**: Run `python pe_prediction_model.py` first to train the model

**Issue**: Ticker CSV not found
- **Solution**: Run `python fetch_nse_bse_stocks.py` to generate the ticker list

**Issue**: Yahoo Finance rate limiting
- **Solution**: The scripts include delays and error handling. Wait and retry if needed.

**Issue**: Missing data for specific stock
- **Solution**: Some stocks may not have complete fundamental data. Try another stock.

## Contributing

Feel free to enhance the model by:
- Adding more features
- Trying different algorithms (XGBoost, LightGBM)
- Implementing ensemble methods
- Adding technical indicators
- Creating sector-specific models

## License

This project is for educational purposes. Use at your own risk.

## Data Source

Market data provided by [Yahoo Finance](https://finance.yahoo.com/)
