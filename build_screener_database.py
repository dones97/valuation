"""
Build Complete Screener.in Database

Scrapes all stocks from NSE_Universe.csv and stores in SQLite database.
Run this once to build the initial database, then use the GitHub workflow for updates.
"""

import pandas as pd
from screener_scraper import ScreenerScraper
import time

def main():
    print("\n" + "="*80)
    print("BUILD SCREENER.IN DATABASE")
    print("="*80)

    # Load NSE universe
    print("\n[1] Loading NSE universe...")
    df = pd.read_csv('NSE_Universe.csv')
    # Remove .NS suffix if present, otherwise use as-is
    all_tickers = df['Ticker'].apply(lambda x: x.replace('.NS', '').replace('.BO', '')).tolist()

    print(f"    Total stocks to scrape: {len(all_tickers)}")
    print(f"    Estimated time: {(len(all_tickers) * 2.5) / 60:.1f} minutes")
    print(f"    (2 second delay per stock + ~0.5s scraping time)")

    # Initialize scraper
    print("\n[2] Initializing scraper...")
    scraper = ScreenerScraper(db_path='screener_data.db')
    print("    Database initialized: screener_data.db")

    # Check existing data
    import sqlite3
    conn = sqlite3.connect('screener_data.db')
    existing = pd.read_sql_query("SELECT ticker FROM companies WHERE data_available = 1", conn)
    conn.close()

    existing_tickers = set(existing['ticker'].tolist())
    print(f"    Already scraped: {len(existing_tickers)} stocks")

    # Determine what to scrape
    new_tickers = [t for t in all_tickers if t not in existing_tickers]

    if new_tickers:
        print(f"\n[3] Scraping {len(new_tickers)} new stocks...")
        to_scrape = new_tickers
    else:
        print(f"\n[3] All stocks already scraped. Refreshing all {len(all_tickers)} stocks...")
        to_scrape = all_tickers

    # Scrape with rate limiting
    print(f"\n    Starting scrape at {time.strftime('%H:%M:%S')}")
    print(f"    This will take approximately {(len(to_scrape) * 2.5) / 60:.1f} minutes")
    print(f"    You can safely stop (Ctrl+C) and resume later - progress is saved to database")
    print()

    start_time = time.time()

    try:
        results = scraper.scrape_multiple_stocks(to_scrape, delay=2.0)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Scraping stopped by user")
        print("Progress has been saved to database")
        return

    elapsed = time.time() - start_time

    # Save results summary
    results.to_csv('screener_scraping_results.csv', index=False)

    print(f"\n{'='*80}")
    print("SCRAPING COMPLETE")
    print(f"{'='*80}")
    print(f"\n[OK] Total time: {elapsed / 60:.1f} minutes")
    print(f"[OK] Database: screener_data.db")
    print(f"[OK] Results summary: screener_scraping_results.csv")

    # Final statistics
    conn = sqlite3.connect('screener_data.db')
    total_in_db = pd.read_sql_query("SELECT COUNT(*) as count FROM companies WHERE data_available = 1", conn)
    conn.close()

    print(f"\n[STATS] Total stocks in database: {total_in_db['count'].iloc[0]}")
    print(f"[STATS] Success rate: {sum(results['status'] == 'success') / len(results) * 100:.1f}%")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("""
    1. Your database is ready: screener_data.db
    2. Test data retrieval:
       python -c "from screener_scraper import ScreenerScraper; s=ScreenerScraper(); print(s.get_stock_data('RELIANCE')['quarterly'])"
    3. Integrate with your model training pipeline
    4. Set up GitHub Actions for monthly updates
    """)


if __name__ == "__main__":
    main()
