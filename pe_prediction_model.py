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

    def fetch_screener_data(self, ticker):
        """
        Fetch additional fundamental data from screener database

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (without .NS suffix)

        Returns:
        --------
        dict : Dictionary containing screener metrics
        """
        import sqlite3

        try:
            conn = sqlite3.connect('screener_data.db')

            # Remove .NS suffix if present
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')

            screener_data = {}

            # Get latest annual ratios (operational efficiency metrics)
            query = f"""
            SELECT debtor_days, inventory_days, days_payable,
                   cash_conversion_cycle, working_capital_days, roce_percent
            FROM annual_ratios
            WHERE ticker = '{clean_ticker}'
            ORDER BY year DESC LIMIT 1
            """
            df = pd.read_sql_query(query, conn)
            if len(df) > 0:
                screener_data['debtor_days'] = df['debtor_days'].iloc[0]
                screener_data['inventory_days'] = df['inventory_days'].iloc[0]
                screener_data['days_payable'] = df['days_payable'].iloc[0]
                screener_data['cash_conversion_cycle'] = df['cash_conversion_cycle'].iloc[0]
                screener_data['working_capital_days'] = df['working_capital_days'].iloc[0]
                screener_data['screener_roce'] = df['roce_percent'].iloc[0]

            # Get latest balance sheet data
            query = f"""
            SELECT equity_capital, reserves, borrowings, total_liabilities,
                   fixed_assets, investments, total_assets
            FROM balance_sheet
            WHERE ticker = '{clean_ticker}'
            ORDER BY year DESC LIMIT 1
            """
            df = pd.read_sql_query(query, conn)
            if len(df) > 0:
                equity = df['equity_capital'].iloc[0] if pd.notna(df['equity_capital'].iloc[0]) else 0
                reserves = df['reserves'].iloc[0] if pd.notna(df['reserves'].iloc[0]) else 0
                borrowings = df['borrowings'].iloc[0] if pd.notna(df['borrowings'].iloc[0]) else 0
                total_assets = df['total_assets'].iloc[0]
                fixed_assets = df['fixed_assets'].iloc[0]
                investments = df['investments'].iloc[0]

                # Calculate additional metrics
                if pd.notna(total_assets) and total_assets > 0:
                    screener_data['fixed_assets_ratio'] = fixed_assets / total_assets if pd.notna(fixed_assets) else np.nan
                    screener_data['investment_ratio'] = investments / total_assets if pd.notna(investments) else np.nan

                    total_equity = equity + reserves
                    if total_equity > 0:
                        screener_data['screener_debt_to_equity'] = borrowings / total_equity if pd.notna(borrowings) else np.nan

            # Get latest P&L data for OPM
            query = f"""
            SELECT opm_percent, dividend_payout_percent
            FROM annual_profit_loss
            WHERE ticker = '{clean_ticker}'
            ORDER BY year DESC LIMIT 1
            """
            df = pd.read_sql_query(query, conn)
            if len(df) > 0:
                screener_data['screener_opm'] = df['opm_percent'].iloc[0]
                screener_data['screener_dividend_payout'] = df['dividend_payout_percent'].iloc[0]

            conn.close()
            return screener_data

        except Exception as e:
            # Silently return empty dict if screener data not available
            return {}

    # ------------------------------------------------------------------ #
    #  Internal helpers for robust yfinance fetching                       #
    # ------------------------------------------------------------------ #
    _USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    ]

    def _make_session(self):
        """
        Return the best available HTTP session for yfinance.

        Priority:
        1. curl_cffi Session impersonating Chrome (bypasses TLS fingerprinting)
        2. requests.Session with urllib3 Retry adapter (plain fallback)

        Yahoo Finance uses TLS fingerprinting to block cloud servers.
        A user-agent string alone is NOT enough — the TLS handshake must also
        look like a real browser.  curl_cffi replicates Chrome's TLS/HTTP2
        fingerprint at the C layer, which is why it works on Streamlit Cloud
        when plain requests does not.
        """
        try:
            from curl_cffi import requests as curl_requests
            # Impersonate a recent Chrome version — yfinance accepts this natively
            session = curl_requests.Session(impersonate="chrome124")
            return session
        except ImportError:
            pass

        # Fallback: standard requests with retry adapter
        import random
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {"User-Agent": random.choice(self._USER_AGENTS)}
        )
        return session

    def _fetch_info_with_retry(self, ticker, max_retries=5):
        """
        Fetch yfinance .info dict with robust retry + backoff.
        Returns (info_dict, ticker_used) or (None, ticker_used) on failure.
        """
        import time as _time
        import random

        # Try the primary ticker, then .NS fallback for .BO tickers (and bare tickers)
        candidates = [ticker]
        if ticker.upper().endswith('.BO'):
            candidates.append(ticker[:-3] + '.NS')
        elif not ticker.upper().endswith('.NS') and '.' not in ticker:
            candidates.append(ticker + '.NS')

        meaningful_keys = {
            'currentPrice', 'regularMarketPrice', 'marketCap',
            'trailingPE', 'forwardPE', 'trailingEps',
            'returnOnEquity', 'profitMargins', 'totalRevenue'
        }

        for t in candidates:
            for attempt in range(max_retries):
                try:
                    session = self._make_session()
                    stock = yf.Ticker(t, session=session)
                    info = stock.info

                    if info and meaningful_keys.intersection(info.keys()):
                        return info, t   # success

                    # Stub / empty response — back off and retry
                    wait = (2 ** attempt) + random.uniform(0.5, 2.0)
                    print(f"[yfinance] Stub info for {t} (attempt {attempt+1}/{max_retries}), "
                          f"waiting {wait:.1f}s…")
                    _time.sleep(wait)

                except Exception as exc:
                    wait = (2 ** attempt) + random.uniform(0.5, 2.0)
                    print(f"[yfinance] Error for {t} attempt {attempt+1}: {exc}; "
                          f"waiting {wait:.1f}s…")
                    _time.sleep(wait)

        return None, ticker

    def _fetch_from_screener_db(self, ticker):
        """
        Pull all available fundamental data from the local screener SQLite DB.

        The screener DB is committed to the repo and is always available on both
        local and cloud environments — no network calls required.

        Returns a dict of fundamental fields (NaN where data is missing).
        """
        import sqlite3

        clean = ticker.replace('.NS', '').replace('.BO', '').upper()
        out = {}

        try:
            conn = sqlite3.connect('screener_data.db')

            # ── key_metrics: price, PE, ROE, ROCE, market cap, dividend yield ──
            row = conn.execute(
                "SELECT market_cap, current_price, stock_pe, roe_percent, "
                "roce_percent, dividend_yield "
                "FROM key_metrics WHERE ticker=? LIMIT 1", (clean,)
            ).fetchone()
            if row:
                out['market_cap_cr'] = row[0]   # crores; convert later
                out['current_price_db'] = row[1]
                out['current_pe_db'] = row[2]
                out['roe_db'] = row[3] / 100.0 if row[3] is not None else np.nan
                out['roce_db'] = row[4] / 100.0 if row[4] is not None else np.nan
                out['dividend_yield_db'] = row[5] / 100.0 if row[5] is not None else np.nan

            # ── companies: sector / industry ──
            row = conn.execute(
                "SELECT sector, industry, company_name FROM companies WHERE ticker=? LIMIT 1",
                (clean,)
            ).fetchone()
            if row:
                out['sector_db'] = row[0] or 'Unknown'
                out['industry_db'] = row[1] or 'Unknown'
                out['company_name_db'] = row[2] or clean

            # ── annual_profit_loss: revenue, net income, EPS, margins ──
            rows = conn.execute(
                "SELECT year, sales, net_profit, eps, opm_percent, dividend_payout_percent "
                "FROM annual_profit_loss WHERE ticker=? ORDER BY year DESC LIMIT 3",
                (clean,)
            ).fetchall()
            if rows:
                latest = rows[0]
                sales = latest[1]
                net_profit = latest[2]
                eps = latest[3]
                opm = latest[4]
                div_payout = latest[5]

                out['revenue_db'] = (sales * 1e7) if sales is not None else np.nan  # cr → INR
                out['net_income_db'] = (net_profit * 1e7) if net_profit is not None else np.nan
                out['eps_db'] = eps  # already per share (INR)
                out['profit_margins_db'] = (net_profit / sales) if (sales and net_profit) else np.nan
                out['operating_margins_db'] = opm / 100.0 if opm is not None else np.nan
                out['screener_dividend_payout_db'] = div_payout / 100.0 if div_payout is not None else np.nan

                # Revenue growth YoY
                if len(rows) >= 2 and rows[1][1] and rows[0][1]:
                    out['revenue_growth_db'] = (rows[0][1] - rows[1][1]) / abs(rows[1][1])
                # Earnings growth YoY
                if len(rows) >= 2 and rows[1][2] and rows[0][2]:
                    out['earnings_growth_db'] = (rows[0][2] - rows[1][2]) / abs(rows[1][2])

            # ── balance_sheet: debt/equity, assets ──
            row = conn.execute(
                "SELECT equity_capital, reserves, borrowings, total_assets, "
                "fixed_assets, investments "
                "FROM balance_sheet WHERE ticker=? ORDER BY year DESC LIMIT 1",
                (clean,)
            ).fetchone()
            if row:
                equity = (row[0] or 0) + (row[1] or 0)
                borrowings = row[2] or 0
                total_assets = row[3]
                fixed_assets = row[4]
                investments = row[5]

                if equity > 0:
                    out['debt_to_equity_db'] = borrowings / equity
                if total_assets and total_assets > 0:
                    out['total_assets_db'] = total_assets * 1e7
                    if fixed_assets is not None:
                        out['fixed_assets_ratio'] = fixed_assets / total_assets
                    if investments is not None:
                        out['investment_ratio'] = investments / total_assets
                    if out.get('revenue_db'):
                        out['asset_turnover_db'] = out['revenue_db'] / (total_assets * 1e7)
                    if out.get('net_income_db'):
                        out['roa_db'] = out['net_income_db'] / (total_assets * 1e7)

            # ── cash_flow ──
            row = conn.execute(
                "SELECT cash_from_operating_activity, cash_from_investing_activity "
                "FROM cash_flow WHERE ticker=? ORDER BY year DESC LIMIT 1",
                (clean,)
            ).fetchone()
            if row:
                op_cf = row[0]
                inv_cf = row[1]
                if op_cf is not None and inv_cf is not None:
                    fcf_cr = op_cf + inv_cf   # free cash flow in crores
                    out['fcf_db'] = fcf_cr * 1e7
                    if out.get('revenue_db'):
                        out['fcf_margin_db'] = (fcf_cr * 1e7) / out['revenue_db']
                    if out.get('net_income_db') and out['net_income_db'] != 0:
                        out['fcf_to_net_income_db'] = (fcf_cr * 1e7) / out['net_income_db']

            # ── annual_ratios (already fetched in fetch_screener_data) ──
            row = conn.execute(
                "SELECT debtor_days, inventory_days, days_payable, "
                "cash_conversion_cycle, working_capital_days, roce_percent "
                "FROM annual_ratios WHERE ticker=? ORDER BY year DESC LIMIT 1",
                (clean,)
            ).fetchone()
            if row:
                out['debtor_days'] = row[0]
                out['inventory_days'] = row[1]
                out['days_payable'] = row[2]
                out['cash_conversion_cycle'] = row[3]
                out['working_capital_days'] = row[4]
                out['screener_roce'] = (row[5] / 100.0) if row[5] is not None else np.nan

            conn.close()
        except Exception as e:
            print(f"[screener_db] Error fetching {clean}: {e}")

        return out

    def fetch_stock_data(self, ticker, max_retries=3):
        """
        Fetch fundamental data for a single stock.

        Strategy (cloud-proof):
        1. Always query the local screener DB first — never fails, no network.
        2. Try yfinance .info to supplement/override with fresher data.
        3. If yfinance is unavailable (throttled / cloud IP blocked), fall back
           entirely to screener DB values.
        4. Return None only if BOTH sources have no usable data.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g. RELIANCE.NS or RELIANCE.BO)
        max_retries : int
            Max yfinance retry attempts.

        Returns
        -------
        dict | None
        """
        # ── Step 1: screener DB (always works) ──
        db = self._fetch_from_screener_db(ticker)

        # ── Step 2: yfinance .info (best-effort, may fail on cloud) ──
        info, used_ticker = self._fetch_info_with_retry(ticker, max_retries=max_retries)
        if not info:
            print(f"[fetch_stock_data] yfinance unavailable for {ticker}; using screener DB only")
            info = {}
            used_ticker = ticker.replace('.BO', '.NS') if '.BO' in ticker.upper() else ticker

        # ── Step 3: merge — yfinance preferred where available, DB as fallback ──

        def _yf(key, default=np.nan):
            v = info.get(key, default)
            return default if v is None else v

        def _db(key, default=np.nan):
            v = db.get(key, default)
            return default if v is None else v

        # Require at least price OR fundamental DB data to proceed
        has_price = bool(info.get('currentPrice') or info.get('regularMarketPrice')
                         or db.get('current_price_db'))
        has_fundamentals = bool(db.get('revenue_db') or db.get('eps_db'))
        if not has_price and not has_fundamentals:
            print(f"[fetch_stock_data] No data from any source for {ticker}")
            return None

        # Valuation
        current_pe = _yf('trailingPE') if not pd.isna(_yf('trailingPE')) else _db('current_pe_db')
        forward_pe = _yf('forwardPE')

        # Financials (yfinance in native currency; screener DB in crores → convert)
        market_cap = _yf('marketCap') if not pd.isna(_yf('marketCap')) \
            else (_db('market_cap_cr') * 1e7 if not pd.isna(_db('market_cap_cr')) else np.nan)
        revenue = _yf('totalRevenue') if not pd.isna(_yf('totalRevenue')) else _db('revenue_db')
        net_income = _yf('netIncomeToCommon') if not pd.isna(_yf('netIncomeToCommon')) \
            else _db('net_income_db')
        total_assets = _yf('totalAssets') if not pd.isna(_yf('totalAssets')) \
            else _db('total_assets_db')
        total_equity = _yf('totalStockholderEquity')
        free_cash_flow = _yf('freeCashFlow') if not pd.isna(_yf('freeCashFlow')) \
            else _db('fcf_db')
        total_debt = _yf('totalDebt')
        ebitda = _yf('ebitda')
        gross_profit = _yf('grossProfits')

        # Ratios
        roe = _yf('returnOnEquity') if not pd.isna(_yf('returnOnEquity')) else _db('roe_db')
        roa = _yf('returnOnAssets') if not pd.isna(_yf('returnOnAssets')) else _db('roa_db')
        profit_margins = _yf('profitMargins') if not pd.isna(_yf('profitMargins')) \
            else _db('profit_margins_db')
        operating_margins = _yf('operatingMargins') if not pd.isna(_yf('operatingMargins')) \
            else _db('operating_margins_db')
        gross_margins = _yf('grossMargins')

        # Growth
        revenue_growth = _yf('revenueGrowth') if not pd.isna(_yf('revenueGrowth')) \
            else _db('revenue_growth_db')
        earnings_growth = _yf('earningsGrowth') if not pd.isna(_yf('earningsGrowth')) \
            else _db('earnings_growth_db')
        earnings_quarterly_growth = _yf('earningsQuarterlyGrowth')

        # Debt
        debt_to_equity = _yf('debtToEquity') if not pd.isna(_yf('debtToEquity')) \
            else _db('debt_to_equity_db')
        current_ratio = _yf('currentRatio')
        quick_ratio = _yf('quickRatio')

        # Dividends / beta
        dividend_yield = _yf('dividendYield') if not pd.isna(_yf('dividendYield')) \
            else _db('dividend_yield_db')
        payout_ratio = _yf('payoutRatio') if not pd.isna(_yf('payoutRatio')) \
            else _db('screener_dividend_payout_db')
        beta = _yf('beta')

        # Derived ratios
        roce = np.nan
        if not pd.isna(ebitda) and not pd.isna(total_assets) and not pd.isna(total_debt):
            cap_emp = total_assets - (total_assets - total_equity - total_debt) \
                if not pd.isna(total_equity) else total_assets
            if cap_emp and cap_emp > 0:
                roce = ebitda / cap_emp
        if pd.isna(roce):
            roce = _db('roce_db')

        fcf_margin = np.nan
        if not pd.isna(free_cash_flow) and not pd.isna(revenue) and revenue > 0:
            fcf_margin = free_cash_flow / revenue
        if pd.isna(fcf_margin):
            fcf_margin = _db('fcf_margin_db')

        fcf_to_net_income = np.nan
        if not pd.isna(free_cash_flow) and not pd.isna(net_income) and net_income > 0:
            fcf_to_net_income = free_cash_flow / net_income
        if pd.isna(fcf_to_net_income):
            fcf_to_net_income = _db('fcf_to_net_income_db')

        gp_to_assets = np.nan
        if not pd.isna(gross_profit) and not pd.isna(total_assets) and total_assets > 0:
            gp_to_assets = gross_profit / total_assets

        asset_turnover = np.nan
        if not pd.isna(revenue) and not pd.isna(total_assets) and total_assets > 0:
            asset_turnover = revenue / total_assets
        if pd.isna(asset_turnover):
            asset_turnover = _db('asset_turnover_db')

        # Sector / industry
        sector = (info.get('sector') or _db('sector_db', 'Unknown') or 'Unknown')
        industry = (info.get('industry') or _db('industry_db', 'Unknown') or 'Unknown')

        # Build _raw_info cache for predict_pe (price + EPS display)
        # Supplement with screener DB values if yfinance is missing them
        cached_info = dict(info)
        if not cached_info.get('currentPrice') and not cached_info.get('regularMarketPrice'):
            cached_info['currentPrice'] = db.get('current_price_db')
        if not cached_info.get('trailingEps'):
            cached_info['trailingEps'] = db.get('eps_db')
        if not cached_info.get('longName'):
            cached_info['longName'] = db.get('company_name_db', used_ticker)

        return {
            'ticker': used_ticker,
            '_raw_info': cached_info,
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
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'dividend_yield': dividend_yield,
            'payout_ratio': payout_ratio,
            'beta': beta,
            'sector': sector,
            'industry': industry,
            'debtor_days': _db('debtor_days'),
            'inventory_days': _db('inventory_days'),
            'days_payable': _db('days_payable'),
            'cash_conversion_cycle': _db('cash_conversion_cycle'),
            'working_capital_days': _db('working_capital_days'),
            'screener_roce': _db('screener_roce'),
            'fixed_assets_ratio': _db('fixed_assets_ratio'),
            'investment_ratio': _db('investment_ratio'),
            'screener_debt_to_equity': _db('debt_to_equity_db'),
            'screener_opm': _db('operating_margins_db'),
            'screener_dividend_payout': _db('screener_dividend_payout_db'),
        }

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

        # Select features for training (removed price_to_book to prevent data leakage)
        # Now includes additional screener database metrics for operational efficiency
        feature_columns = [
            # Core financial metrics (yfinance)
            'market_cap', 'revenue', 'net_income', 'ebitda', 'gross_profit',
            'roe', 'roa', 'roce',
            'profit_margins', 'operating_margins', 'gross_margins',
            'fcf_margin', 'fcf_to_net_income', 'gp_to_assets', 'asset_turnover',
            'revenue_growth', 'earnings_growth', 'earnings_quarterly_growth',
            'debt_to_equity',
            'current_ratio', 'quick_ratio',
            'dividend_yield', 'payout_ratio',
            'beta',
            # Operational efficiency metrics (screener database)
            'debtor_days', 'inventory_days', 'days_payable',
            'cash_conversion_cycle', 'working_capital_days',
            'screener_roce', 'fixed_assets_ratio', 'investment_ratio',
            'screener_debt_to_equity', 'screener_opm', 'screener_dividend_payout',
            # Categorical encodings
            'sector_encoded', 'industry_encoded'
        ]

        # Target variable
        target = 'current_pe'

        # Create feature matrix
        X = df[feature_columns].copy()
        y = df[target].copy()

        # Handle missing values - fill with median for numeric columns
        # Store medians for use at prediction time
        self.training_medians = {}
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                median_val = X[col].median()
                self.training_medians[col] = median_val
                X[col].fillna(median_val, inplace=True)

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

        # Initialize Random Forest (Regularized configuration to reduce overfitting)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,              # Reduced from 15 to limit tree complexity
            min_samples_split=10,      # Increased from 5 for more robust patterns
            min_samples_leaf=4,        # Increased from 2 for better generalization
            max_features=0.6,          # 60% of features per split (was 'sqrt')
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
        from datetime import datetime

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'sector_encoder': self.sector_encoder,
            'industry_encoder': self.industry_encoder,
            'training_medians': getattr(self, 'training_medians', {}),
            'training_date': datetime.now().isoformat(),
            'model_version': '1.1'
        }

        joblib.dump(model_data, filepath)
        print(f"\nModel saved to: {filepath}")
        print(f"Training date: {model_data['training_date']}")
        print(f"Training medians saved: {len(model_data['training_medians'])} features")

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
        self.training_medians = model_data.get('training_medians', {})

        print(f"Model loaded from: {filepath}")
        if self.training_medians:
            print(f"Training medians loaded: {len(self.training_medians)} features")

    def predict_pe(self, ticker):
        """
        Predict P/E ratio for a given stock.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g. RELIANCE.NS)

        Returns
        -------
        dict | None : Prediction results with fair value analysis.
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        # Fetch stock data (includes cached raw info to avoid a 2nd API call)
        stock_data = self.fetch_stock_data(ticker)

        if not stock_data:
            return None

        # Reuse the cached info dict — no second yfinance call needed
        info = stock_data.pop('_raw_info', {})

        # Prepare features
        stock_df = pd.DataFrame([stock_data])

        # Handle unknown sector/industry labels gracefully
        try:
            stock_df['sector_encoded'] = self.sector_encoder.transform([stock_data['sector']])[0]
        except (ValueError, KeyError):
            print(f"Unknown sector '{stock_data['sector']}', using fallback value -1")
            stock_df['sector_encoded'] = -1

        try:
            stock_df['industry_encoded'] = self.industry_encoder.transform([stock_data['industry']])[0]
        except (ValueError, KeyError):
            print(f"Unknown industry '{stock_data['industry']}', using fallback value -1")
            stock_df['industry_encoded'] = -1

        X = stock_df[self.feature_names].copy()

        # Fill missing values using training medians (not single-row median which = NaN)
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                if col in self.training_medians:
                    X[col].fillna(self.training_medians[col], inplace=True)
                else:
                    X[col].fillna(0, inplace=True)

        # Predict P/E
        predicted_pe = float(self.model.predict(X)[0])
        current_pe = stock_data['current_pe']

        # Price and EPS already in the cached info dict — no extra network call
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', np.nan)
        if current_price is None:
            current_price = np.nan
        eps = info.get('trailingEps', np.nan)
        if eps is None:
            eps = np.nan

        # Calculate fair value
        fair_price = np.nan
        upside_downside = np.nan

        if not pd.isna(eps) and eps > 0:
            fair_price = predicted_pe * eps
            if not pd.isna(current_price) and current_price > 0:
                upside_downside = ((fair_price - current_price) / current_price) * 100

        used_ticker = stock_data.get('ticker', ticker)
        return {
            'ticker': used_ticker,
            'company_name': info.get('longName', '') or used_ticker,
            'current_price': current_price,
            'current_pe': current_pe,
            'predicted_pe': predicted_pe,
            'eps': eps,
            'fair_price': fair_price,
            'upside_downside_pct': upside_downside,
            'sector': stock_data['sector'],
            'industry': stock_data['industry'],
            # Fundamental metrics for key contributors analysis
            'roe': stock_data.get('roe', np.nan),
            'roce': stock_data.get('roce', np.nan),
            'profit_margins': stock_data.get('profit_margins', np.nan),
            'revenue_growth': stock_data.get('revenue_growth', np.nan),
            'debt_to_equity': stock_data.get('debt_to_equity', np.nan)
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
