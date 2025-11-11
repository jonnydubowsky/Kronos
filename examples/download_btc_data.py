"""
Download BTC Historical Data for Kronos Prediction

This script downloads Bitcoin price data and formats it for use with Kronos.
It uses the yfinance library which is free and doesn't require API keys.
"""

import pandas as pd
from datetime import datetime, timedelta


def download_btc_yfinance(interval='1h', period_days=90):
    """
    Download BTC data using yfinance library (easiest method).

    Args:
        interval: Data interval - '1m', '5m', '15m', '1h', '1d', etc.
        period_days: Number of days of historical data

    Returns:
        DataFrame with BTC OHLCV data
    """
    try:
        import yfinance as yf
        print(f"Downloading BTC data from Yahoo Finance...")
        print(f"  Interval: {interval}")
        print(f"  Period: {period_days} days\n")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        # Download data
        btc = yf.download(
            'BTC-USD',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval=interval,
            progress=True,
            auto_adjust=True  # Explicitly set to avoid warning
        )

        if btc.empty:
            print("ERROR: No data downloaded")
            return None

        # Handle multi-level columns if present
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = btc.columns.get_level_values(0)

        # Prepare DataFrame in Kronos format
        df = pd.DataFrame({
            'timestamps': btc.index.to_numpy(),
            'open': btc['Open'].to_numpy().flatten(),
            'high': btc['High'].to_numpy().flatten(),
            'low': btc['Low'].to_numpy().flatten(),
            'close': btc['Close'].to_numpy().flatten(),
            'volume': btc['Volume'].to_numpy().flatten(),
        })

        # Calculate amount (volume * close price)
        df['amount'] = df['close'] * df['volume']

        # Remove any NaN values
        df = df.dropna()

        # Reset index
        df = df.reset_index(drop=True)

        print(f"\n✓ Downloaded {len(df)} BTC candles")
        print(f"  Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        return df

    except ImportError:
        print("ERROR: yfinance not installed")
        print("\nInstall it with:")
        print("  conda activate Kronos")
        print("  pip install yfinance")
        return None
    except Exception as e:
        print(f"ERROR downloading data: {e}")
        return None


def download_btc_ccxt(exchange='binance', symbol='BTC/USDT', timeframe='1h', limit=1000):
    """
    Download BTC data using ccxt library (supports many exchanges).

    Args:
        exchange: Exchange name (binance, coinbase, kraken, etc.)
        symbol: Trading pair (BTC/USDT, BTC/USD, etc.)
        timeframe: Candle timeframe ('1m', '5m', '1h', '1d', etc.)
        limit: Number of candles to fetch

    Returns:
        DataFrame with BTC OHLCV data
    """
    try:
        import ccxt

        print(f"Downloading BTC data from {exchange.upper()}...")
        print(f"  Symbol: {symbol}")
        print(f"  Timeframe: {timeframe}")
        print(f"  Limit: {limit} candles\n")

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange)
        exchange_instance = exchange_class({'enableRateLimit': True})

        # Fetch OHLCV data
        ohlcv = exchange_instance.fetch_ohlcv(symbol, timeframe, limit=limit)

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convert timestamp to datetime
        df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop('timestamp', axis=1)

        # Calculate amount
        df['amount'] = df['close'] * df['volume']

        # Reorder columns
        df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]

        print(f"✓ Downloaded {len(df)} BTC candles")
        print(f"  Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        return df

    except ImportError:
        print("ERROR: ccxt not installed")
        print("\nInstall it with:")
        print("  conda activate Kronos")
        print("  pip install ccxt")
        return None
    except Exception as e:
        print(f"ERROR downloading data: {e}")
        return None


def main():
    print("="*70)
    print("BTC DATA DOWNLOADER FOR KRONOS")
    print("="*70)
    print("\nThis script will download Bitcoin historical data and save it")
    print("in the format required by Kronos for price prediction.\n")

    # Configuration
    METHOD = "yfinance"  # Options: "yfinance" or "ccxt"
    INTERVAL = "1h"      # Options: "1m", "5m", "15m", "1h", "1d"
    DAYS = 180           # Number of days of historical data (for yfinance)

    print(f"Configuration:")
    print(f"  Method: {METHOD}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Historical period: {DAYS} days")
    print("="*70 + "\n")

    # Download data
    if METHOD == "yfinance":
        df = download_btc_yfinance(interval=INTERVAL, period_days=DAYS)
    elif METHOD == "ccxt":
        df = download_btc_ccxt(timeframe=INTERVAL, limit=1000)
    else:
        print(f"ERROR: Unknown method '{METHOD}'")
        return

    if df is None or df.empty:
        print("\nFailed to download BTC data. Please check your internet connection.")
        return

    # Save to CSV
    output_file = "./data/btc_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to: {output_file}")

    # Display sample
    print(f"\n{'='*70}")
    print("DATA PREVIEW")
    print("="*70)
    print(df.head(10).to_string())
    print(f"\n... ({len(df)} total rows)")

    # Display statistics
    print(f"\n{'='*70}")
    print("DATA STATISTICS")
    print("="*70)
    print(f"Total candles: {len(df)}")
    print(f"Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")
    print(f"Price statistics:")
    print(f"  Min:  ${df['close'].min():,.2f}")
    print(f"  Max:  ${df['close'].max():,.2f}")
    print(f"  Mean: ${df['close'].mean():,.2f}")
    print(f"  Last: ${df['close'].iloc[-1]:,.2f}")
    print(f"\nVolume statistics:")
    print(f"  Total: {df['volume'].sum():,.0f} BTC")
    print(f"  Mean:  {df['volume'].mean():,.0f} BTC")
    print("="*70)

    print(f"\n✓ Ready to run predictions!")
    print(f"\nNext step:")
    print(f"  python btc_prediction_example.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
