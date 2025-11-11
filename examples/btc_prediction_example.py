"""
BTC Price Prediction Example using Kronos
This script demonstrates how to predict Bitcoin prices using the Kronos model.

Requirements:
- BTC historical data in CSV format with columns: timestamps, open, high, low, close, volume
- You can download BTC data from: Binance, CoinGecko, or other crypto exchanges
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df, title="BTC Price Prediction"):
    """Plot BTC price predictions vs ground truth"""
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Actual BTC Price', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Predicted BTC Price', color='orange', linewidth=1.5)
    ax1.set_ylabel('BTC Price (USD)', fontsize=14)
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(volume_df['Ground Truth'], label='Actual Volume', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Predicted Volume', color='orange', linewidth=1.5)
    ax2.set_ylabel('Volume (BTC)', fontsize=14)
    ax2.set_xlabel('Time', fontsize=14)
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def fetch_btc_data_example():
    """
    Example function to show how to prepare BTC data.

    You can get BTC data from:
    1. Binance API: https://api.binance.com/api/v3/klines
    2. CoinGecko API: https://www.coingecko.com/en/api
    3. Yahoo Finance: using yfinance library
    4. Download CSV from exchanges

    Required format:
    - timestamps: datetime (e.g., "2024-01-01 00:00:00")
    - open: opening price
    - high: highest price
    - low: lowest price
    - close: closing price
    - volume: trading volume
    - amount: (optional) total trading amount
    """
    print("\n" + "="*60)
    print("HOW TO GET BTC DATA")
    print("="*60)
    print("\nOption 1: Use yfinance (easiest)")
    print("  pip install yfinance")
    print("  import yfinance as yf")
    print("  btc = yf.download('BTC-USD', start='2024-01-01', interval='1h')")
    print("  btc.to_csv('btc_data.csv')")

    print("\nOption 2: Download from Binance")
    print("  Visit: https://www.binance.com/en/landing/data")
    print("  Download historical BTCUSDT data")

    print("\nOption 3: Use ccxt library")
    print("  pip install ccxt")
    print("  import ccxt")
    print("  exchange = ccxt.binance()")
    print("  ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h')")
    print("="*60 + "\n")


def main():
    # Configuration
    BTC_DATA_FILE = "./data/btc_data.csv"  # Path to your BTC data CSV
    DEVICE = "mps"  # Use "mps" for Apple Silicon, "cuda:0" for NVIDIA GPU, "cpu" for CPU
    MODEL_SIZE = "small"  # Options: "mini", "small", "base"
    LOOKBACK = 400  # Number of historical candles to use (context window)
    PRED_LEN = 120  # Number of future candles to predict

    print("="*60)
    print("BTC PRICE PREDICTION WITH KRONOS")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: Kronos-{MODEL_SIZE}")
    print(f"Lookback: {LOOKBACK} periods")
    print(f"Prediction: {PRED_LEN} periods ahead")
    print("="*60 + "\n")

    # Check if BTC data file exists
    if not os.path.exists(BTC_DATA_FILE):
        print(f"ERROR: BTC data file not found at {BTC_DATA_FILE}")
        fetch_btc_data_example()
        print("\nPlease prepare your BTC data CSV file with the following columns:")
        print("  - timestamps (datetime)")
        print("  - open, high, low, close (prices)")
        print("  - volume (trading volume)")
        print("\nSave it as: ./data/btc_data.csv")
        print("\nThen run this script again!")
        return

    # 1. Load Model and Tokenizer
    print("Loading Kronos model and tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

    if MODEL_SIZE == "mini":
        model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    elif MODEL_SIZE == "small":
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    else:  # base
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

    print(f"✓ Model loaded: Kronos-{MODEL_SIZE}")

    # 2. Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=512)
    print(f"✓ Predictor initialized on device: {DEVICE}\n")

    # 3. Load and Prepare BTC Data
    print(f"Loading BTC data from {BTC_DATA_FILE}...")
    df = pd.read_csv(BTC_DATA_FILE)

    # Convert timestamps to datetime
    if 'timestamps' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamps'])
    elif 'date' in df.columns:
        df['timestamps'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['timestamps'] = pd.to_datetime(df['Date'])

    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        # Try uppercase versions
        for col in missing_cols[:]:
            if col.capitalize() in df.columns:
                df[col] = df[col.capitalize()]
                missing_cols.remove(col)

    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Add 'amount' column if not present (optional)
    if 'amount' not in df.columns:
        df['amount'] = df['close'] * df['volume']

    print(f"✓ Loaded {len(df)} BTC candles")
    print(f"  Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Check if we have enough data
    min_required = LOOKBACK + PRED_LEN
    if len(df) < min_required:
        print(f"\nERROR: Need at least {min_required} candles, but only have {len(df)}")
        print(f"Please reduce LOOKBACK or PRED_LEN, or get more historical data")
        return

    # Prepare data for prediction
    x_df = df.loc[:LOOKBACK-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:LOOKBACK-1, 'timestamps']
    y_timestamp = df.loc[LOOKBACK:LOOKBACK+PRED_LEN-1, 'timestamps']

    print(f"\n  Historical window: {x_timestamp.min()} to {x_timestamp.max()}")
    print(f"  Prediction window: {y_timestamp.min()} to {y_timestamp.max()}\n")

    # 4. Make Prediction
    print("Generating BTC price predictions...")
    print("(This may take a few minutes depending on your hardware)\n")

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,          # Temperature (higher = more random)
        top_p=0.9,      # Nucleus sampling
        sample_count=1, # Number of samples to average
        verbose=True
    )

    print("\n✓ Prediction complete!\n")

    # 5. Display Results
    print("="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print("\nFirst 10 predicted candles:")
    print(pred_df.head(10).to_string())

    # Calculate prediction statistics
    actual_prices = df.loc[LOOKBACK:LOOKBACK+PRED_LEN-1, 'close'].values
    predicted_prices = pred_df['close'].values

    last_historical_price = df.loc[LOOKBACK-1, 'close']
    first_predicted_price = pred_df.iloc[0]['close']
    last_predicted_price = pred_df.iloc[-1]['close']

    price_change = ((last_predicted_price - last_historical_price) / last_historical_price) * 100

    print(f"\n{'='*60}")
    print("PRICE ANALYSIS")
    print("="*60)
    print(f"Last historical price: ${last_historical_price:,.2f}")
    print(f"First predicted price: ${first_predicted_price:,.2f}")
    print(f"Last predicted price:  ${last_predicted_price:,.2f}")
    print(f"Predicted change:      {price_change:+.2f}%")

    if len(actual_prices) == len(predicted_prices):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

        print(f"\nPrediction Accuracy Metrics:")
        print(f"  MAE:  ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAPE: {mape:.2f}%")

    print("="*60 + "\n")

    # 6. Visualize Results
    print("Generating visualization...")
    kline_df = df.loc[:LOOKBACK+PRED_LEN-1]
    plot_prediction(kline_df, pred_df, title=f"BTC Price Prediction ({PRED_LEN} periods ahead)")

    # 7. Save predictions to CSV
    output_file = "./data/btc_predictions.csv"
    pred_df['timestamps'] = y_timestamp.values
    pred_df.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrediction interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
