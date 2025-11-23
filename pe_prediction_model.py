"""
P/E Ratio Prediction Model using Random Forest
Predicts fair P/E ratio for Indian stocks based on fundamental data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import joblib
import warnings
warnings.filterwarnings('ignore')


class PEPredictionModel:
    """
    Random Forest model to predict P/E ratios for Indian stocks
    """

    def __init__(self, tickers_csv='indian_stocks_tickers.csv'):
        """
        Initialize the model with ticker list

        Parameters:
        -----------
        tickers_csv : str
            Path to CSV file containing Indian stock tickers
        """
        self.tickers_df = pd.read_csv(tickers_csv)
        self.model = None
        self.sector_encoder = LabelEncoder()
        self.industry_encoder = LabelEncoder()
        self.feature_names = []
        self.training_data = None

    def fetch_stock_data(self, ticker):
        """
        Fetch fundamental data for a single stock from yfinance

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol

        Returns:
        --------
        dict : Dictionary containing fundamental metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get historical data for growth calculations
            hist = stock.history(period='3y')

            if hist.empty or not info:
                return None

            # Basic valuation metrics
            current_pe = info.get('trailingPE', np.nan)
            forward_pe = info.get('forwardPE', np.nan)

            # Skip if no P/E ratio available
            if pd.isna(current_pe) or current_pe <= 0 or current_pe > 200:
                return None

            # Financial metrics from yfinance
            market_cap = info.get('marketCap', np.nan)
            revenue = info.get('totalRevenue', np.nan)
            net_income = info.get('netIncomeToCommon', np.nan)
            total_assets = info.get('totalAssets', np.nan)
            total_equity = info.get('totalStockholderEquity', np.nan)
            operating_cash_flow = info.get('operatingCashFlow', np.nan)
            free_cash_flow = info.get('freeCashFlow', np.nan)
            total_debt = info.get('totalDebt', np.nan)
            ebitda = info.get('ebitda', np.nan)
            gross_profit = info.get('grossProfits', np.nan)

            # Pre-calculated ratios from yfinance
            roe = info.get('returnOnEquity', np.nan)
            roa = info.get('returnOnAssets', np.nan)
            profit_margins = info.get('profitMargins', np.nan)
            operating_margins = info.get('operatingMargins', np.nan)
            gross_margins = info.get('grossMargins', np.nan)

            # Book value and price to book
            book_value = info.get('bookValue', np.nan)
            price_to_book = info.get('priceToBook', np.nan)

            # Growth metrics
            revenue_growth = info.get('revenueGrowth', np.nan)
            earnings_growth = info.get('earningsGrowth', np.nan)
            earnings_quarterly_growth = info.get('earningsQuarterlyGrowth', np.nan)

            # Debt metrics
            debt_to_equity = info.get('debtToEquity', np.nan)
            current_ratio = info.get('currentRatio', np.nan)
            quick_ratio = info.get('quickRatio', np.nan)

            # Dividend metrics
            dividend_yield = info.get('dividendYield', np.nan)
            payout_ratio = info.get('payoutRatio', np.nan)

            # Beta and volatility
            beta = info.get('beta', np.nan)

            # Calculate additional ratios if data available
            # ROCE (Return on Capital Employed)
            roce = np.nan
            if not pd.isna(ebitda) and not pd.isna(total_assets) and not pd.isna(total_debt):
                capital_employed = total_assets - (total_assets - total_equity - total_debt)
                if capital_employed > 0:
                    roce = ebitda / capital_employed

            # FCF Margin
            fcf_margin = np.nan
            if not pd.isna(free_cash_flow) and not pd.isna(revenue) and revenue > 0:
                fcf_margin = free_cash_flow / revenue

            # FCF to Net Income ratio
            fcf_to_net_income = np.nan
            if not pd.isna(free_cash_flow) and not pd.isna(net_income) and net_income > 0:
                fcf_to_net_income = free_cash_flow / net_income

            # Gross Profit to Total Assets
            gp_to_assets = np.nan
            if not pd.isna(gross_profit) and not pd.isna(total_assets) and total_assets > 0:
                gp_to_assets = gross_profit / total_assets

            # Asset turnover
            asset_turnover = np.nan
            if not pd.isna(revenue) and not pd.isna(total_assets) and total_assets > 0:
                asset_turnover = revenue / total_assets

            # Get sector and industry
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')

            return {
                'ticker': ticker,
                'current_pe': current_pe,
                'forward_pe': forward_pe,
                'market_cap': market_cap,
                'revenue': revenue,
                'net_income': net_income,
                'ebitda': ebitda,
                'gross_profit': gross_profit,
                'roe': roe,
                'roa': roa,
                'roce': roce,
                'profit_margins': profit_margins,
                'operating_margins': operating_margins,
                'gross_margins': gross_margins,
                'fcf_margin': fcf_margin,
                'fcf_to_net_income': fcf_to_net_income,
                'gp_to_assets': gp_to_assets,
                'asset_turnover': asset_turnover,
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth,
                'earnings_quarterly_growth': earnings_quarterly_growth,
                'price_to_book': price_to_book,
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'quick_ratio': quick_ratio,
                'dividend_yield': dividend_yield,
                'payout_ratio': payout_ratio,
                'beta': beta,
                'sector': sector,
                'industry': industry
            }

        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def build_dataset(self, max_stocks=None):
        """
        Build training dataset from all available tickers

        Parameters:
        -----------
        max_stocks : int, optional
            Maximum number of stocks to fetch (for testing purposes)

        Returns:
        --------
        pd.DataFrame : DataFrame with all features and target variable
        """
        print("Building dataset from stock tickers...")
        print("=" * 70)

        tickers = self.tickers_df['Ticker'].tolist()

        if max_stocks:
            tickers = tickers[:max_stocks]

        data_list = []

        for i, ticker in enumerate(tickers):
            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{len(tickers)} stocks processed...")

            stock_data = self.fetch_stock_data(ticker)

            if stock_data:
                data_list.append(stock_data)

        df = pd.DataFrame(data_list)

        print("=" * 70)
        print(f"\nDataset built successfully!")
        print(f"Total stocks with valid data: {len(df)}")
        print(f"\nSample data:")
        print(df.head())

        return df

    def prepare_features(self, df):
        """
        Prepare features for model training

        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataset

        Returns:
        --------
        tuple : (X, y) features and target variable
        """
        # Encode categorical variables
        df['sector_encoded'] = self.sector_encoder.fit_transform(df['sector'])
        df['industry_encoded'] = self.industry_encoder.fit_transform(df['industry'])

        # Select features for training
        feature_columns = [
            'market_cap', 'revenue', 'net_income', 'ebitda', 'gross_profit',
            'roe', 'roa', 'roce',
            'profit_margins', 'operating_margins', 'gross_margins',
            'fcf_margin', 'fcf_to_net_income', 'gp_to_assets', 'asset_turnover',
            'revenue_growth', 'earnings_growth', 'earnings_quarterly_growth',
            'price_to_book', 'debt_to_equity',
            'current_ratio', 'quick_ratio',
            'dividend_yield', 'payout_ratio',
            'beta',
            'sector_encoded', 'industry_encoded'
        ]

        # Target variable
        target = 'current_pe'

        # Create feature matrix
        X = df[feature_columns].copy()
        y = df[target].copy()

        # Handle missing values - fill with median for numeric columns
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col].fillna(X[col].median(), inplace=True)

        self.feature_names = feature_columns

        print(f"\nFeatures prepared:")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Features: {feature_columns}")
        print(f"\nFeature statistics:")
        print(X.describe())

        return X, y

    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train Random Forest model with 80/20 train-test split

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        test_size : float
            Proportion of dataset for testing (default 0.2 for 80/20 split)
        random_state : int
            Random seed for reproducibility

        Returns:
        --------
        dict : Dictionary containing model performance metrics
        """
        print("\n" + "=" * 70)
        print("Training Random Forest Model...")
        print("=" * 70)

        # 80/20 train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"\nDataset split:")
        print(f"Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"Testing samples: {len(X_test)} ({test_size*100:.0f}%)")

        # Initialize Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )

        # Train model
        print("\nTraining Random Forest...")
        self.model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }

        print("\n" + "=" * 70)
        print("Model Training Complete!")
        print("=" * 70)
        print("\nTraining Set Performance:")
        print(f"  MAE:  {train_mae:.2f}")
        print(f"  RMSE: {train_rmse:.2f}")
        print(f"  R²:   {train_r2:.4f}")

        print("\nTesting Set Performance:")
        print(f"  MAE:  {test_mae:.2f}")
        print(f"  RMSE: {test_rmse:.2f}")
        print(f"  R²:   {test_r2:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

        return metrics

    def visualize_model(self, metrics, output_dir='model_visualizations'):
        """
        Create visualizations of the Random Forest model

        Parameters:
        -----------
        metrics : dict
            Dictionary containing model metrics and test data
        output_dir : str
            Directory to save visualization images
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating model visualizations...")

        # 1. Feature Importance Plot
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
        plt.title('Top 15 Feature Importances in Random Forest Model', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_dir}/feature_importance.png")
        plt.close()

        # 2. Actual vs Predicted Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(metrics['y_test'], metrics['y_test_pred'], alpha=0.6, edgecolors='k', linewidths=0.5)
        plt.plot([0, max(metrics['y_test'])], [0, max(metrics['y_test'])], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual P/E Ratio', fontsize=12)
        plt.ylabel('Predicted P/E Ratio', fontsize=12)
        plt.title('Actual vs Predicted P/E Ratios (Test Set)', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_dir}/actual_vs_predicted.png")
        plt.close()

        # 3. Residual Plot
        plt.figure(figsize=(10, 6))
        residuals = metrics['y_test'] - metrics['y_test_pred']
        plt.scatter(metrics['y_test_pred'], residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted P/E Ratio', fontsize=12)
        plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
        plt.title('Residual Plot', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/residuals.png', dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_dir}/residuals.png")
        plt.close()

        # 4. Single Decision Tree Visualization
        plt.figure(figsize=(20, 12))
        plot_tree(self.model.estimators_[0],
                  feature_names=self.feature_names,
                  filled=True,
                  rounded=True,
                  fontsize=8,
                  max_depth=3)  # Limit depth for readability
        plt.title('Sample Decision Tree from Random Forest (Depth=3)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/decision_tree_sample.png', dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_dir}/decision_tree_sample.png")
        plt.close()

        # 5. Model Performance Summary
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Train vs Test Performance
        metrics_data = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R²'],
            'Training': [metrics['train_mae'], metrics['train_rmse'], metrics['train_r2']],
            'Testing': [metrics['test_mae'], metrics['test_rmse'], metrics['test_r2']]
        })

        x = np.arange(len(metrics_data))
        width = 0.35
        axes[0].bar(x - width/2, metrics_data['Training'], width, label='Training', alpha=0.8)
        axes[0].bar(x + width/2, metrics_data['Testing'], width, label='Testing', alpha=0.8)
        axes[0].set_xlabel('Metrics', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Training vs Testing Performance', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics_data['Metric'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Prediction Error Distribution
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_performance_summary.png', dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved: {output_dir}/model_performance_summary.png")
        plt.close()

        print(f"\nAll visualizations saved to: {output_dir}/")

    def save_model(self, filepath='pe_prediction_model.pkl'):
        """
        Save trained model to disk

        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'sector_encoder': self.sector_encoder,
            'industry_encoder': self.industry_encoder
        }

        joblib.dump(model_data, filepath)
        print(f"\nModel saved to: {filepath}")

    def load_model(self, filepath='pe_prediction_model.pkl'):
        """
        Load trained model from disk

        Parameters:
        -----------
        filepath : str
            Path to load the model from
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.sector_encoder = model_data['sector_encoder']
        self.industry_encoder = model_data['industry_encoder']

        print(f"Model loaded from: {filepath}")

    def predict_pe(self, ticker):
        """
        Predict P/E ratio for a given stock

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol

        Returns:
        --------
        dict : Prediction results with fair value analysis
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Fetch stock data
        stock_data = self.fetch_stock_data(ticker)

        if not stock_data:
            return None

        # Prepare features
        stock_df = pd.DataFrame([stock_data])
        stock_df['sector_encoded'] = self.sector_encoder.transform([stock_data['sector']])[0]
        stock_df['industry_encoded'] = self.industry_encoder.transform([stock_data['industry']])[0]

        X = stock_df[self.feature_names].copy()

        # Fill missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col].fillna(X[col].median(), inplace=True)

        # Predict P/E
        predicted_pe = self.model.predict(X)[0]
        current_pe = stock_data['current_pe']

        # Get current stock price and EPS
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get('currentPrice', np.nan)
        eps = info.get('trailingEps', np.nan)

        # Calculate fair value
        fair_price = np.nan
        upside_downside = np.nan

        if not pd.isna(eps) and eps > 0:
            fair_price = predicted_pe * eps

            if not pd.isna(current_price) and current_price > 0:
                upside_downside = ((fair_price - current_price) / current_price) * 100

        return {
            'ticker': ticker,
            'company_name': info.get('longName', ''),
            'current_price': current_price,
            'current_pe': current_pe,
            'predicted_pe': predicted_pe,
            'eps': eps,
            'fair_price': fair_price,
            'upside_downside_pct': upside_downside,
            'sector': stock_data['sector'],
            'industry': stock_data['industry']
        }


def main():
    """
    Main function to build and train the model
    """
    # Initialize model
    model = PEPredictionModel('indian_stocks_tickers.csv')

    # Build dataset
    df = model.build_dataset()

    # Save raw dataset
    df.to_csv('stock_fundamental_data.csv', index=False)
    print(f"\nRaw dataset saved to: stock_fundamental_data.csv")

    # Prepare features
    X, y = model.prepare_features(df)

    # Train model
    metrics = model.train_model(X, y, test_size=0.2)

    # Visualize model
    model.visualize_model(metrics)

    # Save model
    model.save_model('pe_prediction_model.pkl')

    print("\n" + "=" * 70)
    print("Model training complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review visualizations in 'model_visualizations/' folder")
    print("2. Use the model in Streamlit app")
    print("3. Model file: pe_prediction_model.pkl")


if __name__ == "__main__":
    main()
