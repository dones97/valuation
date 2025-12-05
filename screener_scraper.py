"""
Screener.in Web Scraper

Extracts comprehensive fundamental data from screener.in for Indian stocks.
Stores data in SQLite database for efficient querying and periodic updates.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple
import re
import json
from pathlib import Path


class ScreenerScraper:
    """Scrape fundamental data from screener.in"""

    def __init__(self, db_path: str = 'screener_data.db'):
        """
        Initialize scraper

        Args:
            db_path: Path to SQLite database
        """
        self.base_url = "https://www.screener.in"
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table 1: Company info
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS companies (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                industry TEXT,
                last_updated TIMESTAMP,
                data_available BOOLEAN
            )
        ''')

        # Table 2: Quarterly results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quarterly_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                quarter_date TEXT,
                sales REAL,
                expenses REAL,
                operating_profit REAL,
                opm_percent REAL,
                other_income REAL,
                interest REAL,
                depreciation REAL,
                profit_before_tax REAL,
                tax_percent REAL,
                net_profit REAL,
                eps REAL,
                scraped_at TIMESTAMP,
                UNIQUE(ticker, quarter_date)
            )
        ''')

        # Table 3: Annual financials (P&L)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annual_profit_loss (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                year TEXT,
                sales REAL,
                expenses REAL,
                operating_profit REAL,
                opm_percent REAL,
                other_income REAL,
                interest REAL,
                depreciation REAL,
                profit_before_tax REAL,
                tax_percent REAL,
                net_profit REAL,
                eps REAL,
                dividend_payout_percent REAL,
                scraped_at TIMESTAMP,
                UNIQUE(ticker, year)
            )
        ''')

        # Table 4: Balance sheet
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS balance_sheet (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                year TEXT,
                equity_capital REAL,
                reserves REAL,
                borrowings REAL,
                other_liabilities REAL,
                total_liabilities REAL,
                fixed_assets REAL,
                cwip REAL,
                investments REAL,
                other_assets REAL,
                total_assets REAL,
                scraped_at TIMESTAMP,
                UNIQUE(ticker, year)
            )
        ''')

        # Table 5: Cash flow
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cash_flow (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                year TEXT,
                cash_from_operating_activity REAL,
                cash_from_investing_activity REAL,
                cash_from_financing_activity REAL,
                net_cash_flow REAL,
                scraped_at TIMESTAMP,
                UNIQUE(ticker, year)
            )
        ''')

        # Table 6: Annual ratios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annual_ratios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                year TEXT,
                debtor_days REAL,
                inventory_days REAL,
                days_payable REAL,
                cash_conversion_cycle REAL,
                working_capital_days REAL,
                roce_percent REAL,
                scraped_at TIMESTAMP,
                UNIQUE(ticker, year)
            )
        ''')

        # Table 7: Key metrics (current/latest)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS key_metrics (
                ticker TEXT PRIMARY KEY,
                market_cap REAL,
                current_price REAL,
                high_low TEXT,
                stock_pe REAL,
                book_value REAL,
                dividend_yield REAL,
                roce_percent REAL,
                roe_percent REAL,
                face_value REAL,
                scraped_at TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

        print(f"✅ Database initialized: {self.db_path}")

    def _clean_number(self, text: str) -> Optional[float]:
        """
        Convert scraped text to number

        Handles:
        - Commas: "1,234.56" -> 1234.56
        - Percentages: "12.5%" -> 12.5
        - Empty/missing: "" -> None
        """
        if not text or text.strip() in ['', '-', 'N/A']:
            return None

        # Remove commas and percentage signs
        text = text.replace(',', '').replace('%', '').strip()

        try:
            return float(text)
        except:
            return None

    def scrape_stock(self, ticker: str) -> Dict:
        """
        Scrape all available data for a stock

        Args:
            ticker: Stock symbol (e.g., 'RELIANCE')

        Returns:
            Dictionary with scraping results
        """
        url = f"{self.base_url}/company/{ticker}/"

        print(f"\n📊 Scraping: {ticker}")
        print(f"   URL: {url}")

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract all data categories
            company_info = self._scrape_company_info(soup, ticker)
            key_metrics = self._scrape_key_metrics(soup, ticker)
            quarterly = self._scrape_quarterly_results(soup, ticker)
            annual_pl = self._scrape_annual_profit_loss(soup, ticker)
            balance_sheet = self._scrape_balance_sheet(soup, ticker)
            cash_flow = self._scrape_cash_flow(soup, ticker)
            ratios = self._scrape_ratios(soup, ticker)

            # Save to database
            self._save_to_database(
                ticker, company_info, key_metrics, quarterly,
                annual_pl, balance_sheet, cash_flow, ratios
            )

            print(f"   ✅ Successfully scraped and saved {ticker}")

            return {
                'ticker': ticker,
                'status': 'success',
                'records': {
                    'quarterly': len(quarterly),
                    'annual_pl': len(annual_pl),
                    'balance_sheet': len(balance_sheet),
                    'cash_flow': len(cash_flow),
                    'ratios': len(ratios)
                }
            }

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"   ⚠️  {ticker} not found on screener.in")
                return {'ticker': ticker, 'status': 'not_found'}
            else:
                print(f"   ❌ HTTP error for {ticker}: {e}")
                return {'ticker': ticker, 'status': 'error', 'error': str(e)}

        except Exception as e:
            print(f"   ❌ Error scraping {ticker}: {str(e)}")
            return {'ticker': ticker, 'status': 'error', 'error': str(e)}

    def _scrape_company_info(self, soup: BeautifulSoup, ticker: str) -> Dict:
        """Extract company basic information"""
        info = {'ticker': ticker}

        # Company name
        name_elem = soup.find('h1', class_='company-name')
        if name_elem:
            info['company_name'] = name_elem.text.strip()

        return info

    def _scrape_key_metrics(self, soup: BeautifulSoup, ticker: str) -> Dict:
        """Extract key metrics from top ratios section"""
        metrics = {'ticker': ticker}

        ratios_section = soup.find('div', id='top-ratios')
        if ratios_section:
            ratio_items = ratios_section.find_all('li', class_='flex-row')

            for item in ratio_items:
                name_elem = item.find('span', class_='name')
                value_elem = item.find('span', class_='number')

                if name_elem and value_elem:
                    name = name_elem.text.strip()
                    value = self._clean_number(value_elem.text)

                    # Map to database columns
                    if 'Market Cap' in name:
                        metrics['market_cap'] = value
                    elif 'Current Price' in name:
                        metrics['current_price'] = value
                    elif 'Stock P/E' in name:
                        metrics['stock_pe'] = value
                    elif 'Book Value' in name:
                        metrics['book_value'] = value
                    elif 'Dividend Yield' in name:
                        metrics['dividend_yield'] = value
                    elif 'ROCE' in name:
                        metrics['roce_percent'] = value
                    elif 'ROE' in name:
                        metrics['roe_percent'] = value
                    elif 'Face Value' in name:
                        metrics['face_value'] = value

        return metrics

    def _scrape_quarterly_results(self, soup: BeautifulSoup, ticker: str) -> List[Dict]:
        """Extract quarterly results table"""
        results = []

        section = soup.find('section', id='quarters')
        if not section:
            return results

        table = section.find('table')
        if not table:
            return results

        # Get headers (quarters)
        headers = [th.text.strip() for th in table.find_all('th')[1:]]  # Skip first column

        # Get rows
        rows = table.find_all('tr')[1:]  # Skip header row

        row_mapping = {
            'Sales': 'sales',
            'Expenses': 'expenses',
            'Operating Profit': 'operating_profit',
            'OPM %': 'opm_percent',
            'Other Income': 'other_income',
            'Interest': 'interest',
            'Depreciation': 'depreciation',
            'Profit before tax': 'profit_before_tax',
            'Tax %': 'tax_percent',
            'Net Profit': 'net_profit',
            'EPS in Rs': 'eps'
        }

        # Create data structure
        quarterly_data = {header: {} for header in headers}

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue

            row_name = cols[0].text.strip()
            if row_name not in row_mapping:
                continue

            field_name = row_mapping[row_name]

            for i, col in enumerate(cols[1:]):
                if i < len(headers):
                    quarter = headers[i]
                    value = self._clean_number(col.text)
                    quarterly_data[quarter][field_name] = value

        # Convert to list of dicts
        for quarter, data in quarterly_data.items():
            if data:  # Only add if has data
                record = {
                    'ticker': ticker,
                    'quarter_date': quarter,
                    **data
                }
                results.append(record)

        return results

    def _scrape_annual_profit_loss(self, soup: BeautifulSoup, ticker: str) -> List[Dict]:
        """Extract annual P&L statement"""
        results = []

        section = soup.find('section', id='profit-loss')
        if not section:
            return results

        table = section.find('table')
        if not table:
            return results

        # Get headers (years)
        headers = [th.text.strip() for th in table.find_all('th')[1:]]

        # Similar to quarterly, extract rows
        rows = table.find_all('tr')[1:]

        row_mapping = {
            'Sales': 'sales',
            'Expenses': 'expenses',
            'Operating Profit': 'operating_profit',
            'OPM %': 'opm_percent',
            'Other Income': 'other_income',
            'Interest': 'interest',
            'Depreciation': 'depreciation',
            'Profit before tax': 'profit_before_tax',
            'Tax %': 'tax_percent',
            'Net Profit': 'net_profit',
            'EPS in Rs': 'eps',
            'Dividend Payout %': 'dividend_payout_percent'
        }

        annual_data = {header: {} for header in headers}

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue

            row_name = cols[0].text.strip()
            if row_name not in row_mapping:
                continue

            field_name = row_mapping[row_name]

            for i, col in enumerate(cols[1:]):
                if i < len(headers):
                    year = headers[i]
                    value = self._clean_number(col.text)
                    annual_data[year][field_name] = value

        for year, data in annual_data.items():
            if data:
                record = {
                    'ticker': ticker,
                    'year': year,
                    **data
                }
                results.append(record)

        return results

    def _scrape_balance_sheet(self, soup: BeautifulSoup, ticker: str) -> List[Dict]:
        """Extract balance sheet"""
        results = []

        section = soup.find('section', id='balance-sheet')
        if not section:
            return results

        table = section.find('table')
        if not table:
            return results

        headers = [th.text.strip() for th in table.find_all('th')[1:]]
        rows = table.find_all('tr')[1:]

        row_mapping = {
            'Equity Capital': 'equity_capital',
            'Reserves': 'reserves',
            'Borrowings': 'borrowings',
            'Other Liabilities': 'other_liabilities',
            'Total Liabilities': 'total_liabilities',
            'Fixed Assets': 'fixed_assets',
            'CWIP': 'cwip',
            'Investments': 'investments',
            'Other Assets': 'other_assets',
            'Total Assets': 'total_assets'
        }

        annual_data = {header: {} for header in headers}

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue

            row_name = cols[0].text.strip()
            if row_name not in row_mapping:
                continue

            field_name = row_mapping[row_name]

            for i, col in enumerate(cols[1:]):
                if i < len(headers):
                    year = headers[i]
                    value = self._clean_number(col.text)
                    annual_data[year][field_name] = value

        for year, data in annual_data.items():
            if data:
                record = {
                    'ticker': ticker,
                    'year': year,
                    **data
                }
                results.append(record)

        return results

    def _scrape_cash_flow(self, soup: BeautifulSoup, ticker: str) -> List[Dict]:
        """Extract cash flow statement"""
        results = []

        section = soup.find('section', id='cash-flow')
        if not section:
            return results

        table = section.find('table')
        if not table:
            return results

        headers = [th.text.strip() for th in table.find_all('th')[1:]]
        rows = table.find_all('tr')[1:]

        row_mapping = {
            'Cash from Operating Activity': 'cash_from_operating_activity',
            'Cash from Investing Activity': 'cash_from_investing_activity',
            'Cash from Financing Activity': 'cash_from_financing_activity',
            'Net Cash Flow': 'net_cash_flow'
        }

        annual_data = {header: {} for header in headers}

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue

            row_name = cols[0].text.strip()
            if row_name not in row_mapping:
                continue

            field_name = row_mapping[row_name]

            for i, col in enumerate(cols[1:]):
                if i < len(headers):
                    year = headers[i]
                    value = self._clean_number(col.text)
                    annual_data[year][field_name] = value

        for year, data in annual_data.items():
            if data:
                record = {
                    'ticker': ticker,
                    'year': year,
                    **data
                }
                results.append(record)

        return results

    def _scrape_ratios(self, soup: BeautifulSoup, ticker: str) -> List[Dict]:
        """Extract annual ratios"""
        results = []

        section = soup.find('section', id='ratios')
        if not section:
            return results

        table = section.find('table')
        if not table:
            return results

        headers = [th.text.strip() for th in table.find_all('th')[1:]]
        rows = table.find_all('tr')[1:]

        row_mapping = {
            'Debtor Days': 'debtor_days',
            'Inventory Days': 'inventory_days',
            'Days Payable': 'days_payable',
            'Cash Conversion Cycle': 'cash_conversion_cycle',
            'Working Capital Days': 'working_capital_days',
            'ROCE %': 'roce_percent'
        }

        annual_data = {header: {} for header in headers}

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue

            row_name = cols[0].text.strip()
            if row_name not in row_mapping:
                continue

            field_name = row_mapping[row_name]

            for i, col in enumerate(cols[1:]):
                if i < len(headers):
                    year = headers[i]
                    value = self._clean_number(col.text)
                    annual_data[year][field_name] = value

        for year, data in annual_data.items():
            if data:
                record = {
                    'ticker': ticker,
                    'year': year,
                    **data
                }
                results.append(record)

        return results

    def _save_to_database(self, ticker: str, company_info: Dict, key_metrics: Dict,
                          quarterly: List[Dict], annual_pl: List[Dict],
                          balance_sheet: List[Dict], cash_flow: List[Dict],
                          ratios: List[Dict]):
        """Save all scraped data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()

        try:
            # Company info
            cursor.execute('''
                INSERT OR REPLACE INTO companies (ticker, company_name, last_updated, data_available)
                VALUES (?, ?, ?, ?)
            ''', (ticker, company_info.get('company_name'), timestamp, True))

            # Key metrics
            if key_metrics:
                key_metrics['scraped_at'] = timestamp
                cols = ', '.join(key_metrics.keys())
                placeholders = ', '.join(['?' for _ in key_metrics])
                cursor.execute(f'''
                    INSERT OR REPLACE INTO key_metrics ({cols})
                    VALUES ({placeholders})
                ''', tuple(key_metrics.values()))

            # Quarterly results
            for record in quarterly:
                record['scraped_at'] = timestamp
                cols = ', '.join(record.keys())
                placeholders = ', '.join(['?' for _ in record])
                cursor.execute(f'''
                    INSERT OR REPLACE INTO quarterly_results ({cols})
                    VALUES ({placeholders})
                ''', tuple(record.values()))

            # Annual P&L
            for record in annual_pl:
                record['scraped_at'] = timestamp
                cols = ', '.join(record.keys())
                placeholders = ', '.join(['?' for _ in record])
                cursor.execute(f'''
                    INSERT OR REPLACE INTO annual_profit_loss ({cols})
                    VALUES ({placeholders})
                ''', tuple(record.values()))

            # Balance sheet
            for record in balance_sheet:
                record['scraped_at'] = timestamp
                cols = ', '.join(record.keys())
                placeholders = ', '.join(['?' for _ in record])
                cursor.execute(f'''
                    INSERT OR REPLACE INTO balance_sheet ({cols})
                    VALUES ({placeholders})
                ''', tuple(record.values()))

            # Cash flow
            for record in cash_flow:
                record['scraped_at'] = timestamp
                cols = ', '.join(record.keys())
                placeholders = ', '.join(['?' for _ in record])
                cursor.execute(f'''
                    INSERT OR REPLACE INTO cash_flow ({cols})
                    VALUES ({placeholders})
                ''', tuple(record.values()))

            # Ratios
            for record in ratios:
                record['scraped_at'] = timestamp
                cols = ', '.join(record.keys())
                placeholders = ', '.join(['?' for _ in record])
                cursor.execute(f'''
                    INSERT OR REPLACE INTO annual_ratios ({cols})
                    VALUES ({placeholders})
                ''', tuple(record.values()))

            conn.commit()

        except Exception as e:
            conn.rollback()
            print(f"   ❌ Database error for {ticker}: {str(e)}")
            raise

        finally:
            conn.close()

    def scrape_multiple_stocks(self, tickers: List[str], delay: float = 2.0) -> pd.DataFrame:
        """
        Scrape multiple stocks with rate limiting

        Args:
            tickers: List of stock symbols
            delay: Seconds to wait between requests (default: 2)

        Returns:
            DataFrame with scraping results
        """
        results = []

        print(f"\n{'='*80}")
        print(f"SCRAPING {len(tickers)} STOCKS")
        print(f"{'='*80}")

        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}]", end=" ")

            result = self.scrape_stock(ticker)
            results.append(result)

            # Rate limiting - be respectful
            if i < len(tickers):
                print(f"   ⏳ Waiting {delay}s...")
                time.sleep(delay)

        print(f"\n{'='*80}")
        print("SCRAPING COMPLETE")
        print(f"{'='*80}\n")

        df = pd.DataFrame(results)

        # Summary
        successful = sum(df['status'] == 'success')
        failed = sum(df['status'] == 'error')
        not_found = sum(df['status'] == 'not_found')

        print(f"✅ Successful: {successful}")
        print(f"⚠️  Not found: {not_found}")
        print(f"❌ Failed: {failed}")

        return df

    def get_stock_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Retrieve all data for a stock from database

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with DataFrames for each data category
        """
        conn = sqlite3.connect(self.db_path)

        data = {}

        # Key metrics
        data['key_metrics'] = pd.read_sql_query(
            "SELECT * FROM key_metrics WHERE ticker = ?",
            conn, params=(ticker,)
        )

        # Quarterly results
        data['quarterly'] = pd.read_sql_query(
            "SELECT * FROM quarterly_results WHERE ticker = ? ORDER BY quarter_date DESC",
            conn, params=(ticker,)
        )

        # Annual P&L
        data['annual_pl'] = pd.read_sql_query(
            "SELECT * FROM annual_profit_loss WHERE ticker = ? ORDER BY year DESC",
            conn, params=(ticker,)
        )

        # Balance sheet
        data['balance_sheet'] = pd.read_sql_query(
            "SELECT * FROM balance_sheet WHERE ticker = ? ORDER BY year DESC",
            conn, params=(ticker,)
        )

        # Cash flow
        data['cash_flow'] = pd.read_sql_query(
            "SELECT * FROM cash_flow WHERE ticker = ? ORDER BY year DESC",
            conn, params=(ticker,)
        )

        # Ratios
        data['ratios'] = pd.read_sql_query(
            "SELECT * FROM annual_ratios WHERE ticker = ? ORDER BY year DESC",
            conn, params=(ticker,)
        )

        conn.close()

        return data


def main():
    """Test the scraper"""
    scraper = ScreenerScraper()

    # Test with a few stocks
    test_tickers = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']

    print("Testing scraper with sample stocks...")
    results = scraper.scrape_multiple_stocks(test_tickers, delay=2.0)

    print(f"\n{'='*80}")
    print("SAMPLE DATA RETRIEVAL")
    print(f"{'='*80}")

    # Show sample data for first successful stock
    successful = results[results['status'] == 'success']
    if not successful.empty:
        sample_ticker = successful.iloc[0]['ticker']
        print(f"\nRetrieving data for {sample_ticker}...")

        data = scraper.get_stock_data(sample_ticker)

        for category, df in data.items():
            if not df.empty:
                print(f"\n{category.upper()}:")
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {list(df.columns[:10])}...")  # First 10 columns

    print(f"\n✅ Data saved to: {scraper.db_path}")


if __name__ == "__main__":
    main()
