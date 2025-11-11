"""
BTC Future Price Prediction - Predict from NOW
This script predicts future Bitcoin prices starting from the most recent data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction_future(historical_df, pred_df, title="BTC Future Price Prediction"):
    """Plot future BTC price predictions starting from now"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # Plot 1: Recent historical prices + future predictions
    recent_history = historical_df.tail(200)  # Last 200 candles for context

    ax1.plot(recent_history.index, recent_history['close'],
             label='Historical BTC Price', color='blue', linewidth=2)
    ax1.plot(pred_df.index, pred_df['close'],
             label='Predicted BTC Price', color='orange', linewidth=2, linestyle='--')

    # Mark the transition point
    ax1.axvline(x=historical_df.index[-1], color='red', linestyle=':', linewidth=1.5,
                label='NOW (Prediction Start)')

    ax1.set_ylabel('BTC Price (USD)', fontsize=14, fontweight='bold')
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Predicted volume
    ax2.plot(pred_df.index, pred_df['volume'],
             label='Predicted Volume', color='orange', linewidth=2)
    ax2.axvline(x=historical_df.index[-1], color='red', linestyle=':', linewidth=1.5)
    ax2.set_ylabel('Volume (BTC)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def generate_future_timestamps(last_timestamp, pred_len, interval_hours=1):
    """Generate future timestamps for predictions"""
    timestamps = []
    for i in range(1, pred_len + 1):
        future_time = last_timestamp + timedelta(hours=interval_hours * i)
        timestamps.append(future_time)
    return timestamps


def main():
    # Configuration
    BTC_DATA_FILE = "./data/btc_data.csv"
    DEVICE = "mps"  # Use "mps" for Apple Silicon, "cuda:0" for NVIDIA GPU, "cpu" for CPU
    MODEL_SIZE = "small"  # Options: "mini", "small", "base"
    LOOKBACK = 400  # Number of historical candles to use
    PRED_LEN = 120  # Number of future hours to predict
    INTERVAL_HOURS = 1  # Interval between candles (1 hour)

    print("="*70)
    print("BTC FUTURE PRICE PREDICTION - STARTING FROM NOW")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Model: Kronos-{MODEL_SIZE}")
    print(f"Using last {LOOKBACK} hours of data")
    print(f"Predicting next {PRED_LEN} hours ({PRED_LEN//24} days)")
    print("="*70 + "\n")

    # Check if data exists
    if not os.path.exists(BTC_DATA_FILE):
        print(f"ERROR: BTC data file not found at {BTC_DATA_FILE}")
        print("\nPlease run: python download_btc_data.py")
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

    # 3. Load BTC Data
    print(f"Loading BTC data from {BTC_DATA_FILE}...")
    df = pd.read_csv(BTC_DATA_FILE)
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    # Ensure required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns and col.capitalize() in df.columns:
            df[col] = df[col.capitalize()]

    if 'amount' not in df.columns:
        df['amount'] = df['close'] * df['volume']

    print(f"✓ Loaded {len(df)} BTC candles")
    print(f"  Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")

    last_timestamp = df['timestamps'].iloc[-1]
    last_price = df['close'].iloc[-1]

    print(f"\n📊 CURRENT BTC STATUS:")
    print(f"  Last update: {last_timestamp}")
    print(f"  Current price: ${last_price:,.2f}")
    print(f"  24h change: {((df['close'].iloc[-1] / df['close'].iloc[-24] - 1) * 100):+.2f}%")

    # Use the most recent data for prediction
    if len(df) < LOOKBACK:
        print(f"\nERROR: Need at least {LOOKBACK} candles, but only have {len(df)}")
        return

    # Take the last LOOKBACK candles as input
    start_idx = len(df) - LOOKBACK
    x_df = df.iloc[start_idx:][['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
    x_timestamp = df.iloc[start_idx:]['timestamps'].reset_index(drop=True)

    # Generate future timestamps
    y_timestamp = pd.Series(generate_future_timestamps(last_timestamp, PRED_LEN, INTERVAL_HOURS))

    print(f"\n  Historical window: {x_timestamp.iloc[0]} to {x_timestamp.iloc[-1]}")
    print(f"  Prediction window: {y_timestamp.iloc[0]} to {y_timestamp.iloc[-1]}")
    print(f"  (Predicting from {y_timestamp.iloc[0].strftime('%B %d, %Y %H:%M')} onwards)")

    # 4. Generate Predictions
    print("\n" + "="*70)
    print("GENERATING FUTURE PREDICTIONS...")
    print("="*70)
    print("(This may take a few minutes)\n")

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )

    print("\n✓ Prediction complete!\n")

    # 5. Analyze Predictions
    print("="*70)
    print("FUTURE PRICE PREDICTIONS")
    print("="*70)

    # Reset index to use integer positions
    pred_df_reset = pred_df.reset_index(drop=True)

    first_pred_price = pred_df_reset.iloc[0]['close']
    last_pred_price = pred_df_reset.iloc[-1]['close']
    pred_change = ((last_pred_price - last_price) / last_price) * 100

    # Find highest and lowest predicted prices
    max_pred_price = pred_df_reset['close'].max()
    min_pred_price = pred_df_reset['close'].min()
    max_pred_idx = pred_df_reset['close'].idxmax()
    min_pred_idx = pred_df_reset['close'].idxmin()
    max_pred_time = y_timestamp.iloc[max_pred_idx]
    min_pred_time = y_timestamp.iloc[min_pred_idx]

    print(f"\n📈 PRICE FORECAST:")
    print(f"  Current price:        ${last_price:,.2f}")
    print(f"  Next hour prediction: ${first_pred_price:,.2f} ({((first_pred_price/last_price-1)*100):+.2f}%)")
    print(f"  {PRED_LEN}h prediction:      ${last_pred_price:,.2f} ({pred_change:+.2f}%)")
    print(f"\n  Predicted HIGH: ${max_pred_price:,.2f} at {max_pred_time.strftime('%b %d, %H:%M')}")
    print(f"  Predicted LOW:  ${min_pred_price:,.2f} at {min_pred_time.strftime('%b %d, %H:%M')}")
    print(f"  Predicted range: ${min_pred_price:,.2f} - ${max_pred_price:,.2f}")
    print(f"  Volatility: {((max_pred_price - min_pred_price) / last_price * 100):.2f}%")

    # Show hourly predictions for next 24 hours
    print(f"\n📊 NEXT 24 HOURS DETAILED FORECAST:")
    print(f"{'Time':<25} {'Price':>12} {'Change':>10} {'Volume':>15}")
    print("-" * 70)

    hours_to_show = min(24, PRED_LEN)
    for i in range(0, hours_to_show, 3):  # Show every 3 hours
        time_str = y_timestamp.iloc[i].strftime('%b %d, %Y %H:%M')
        price = pred_df_reset.iloc[i]['close']
        change = ((price / last_price - 1) * 100)
        volume = pred_df_reset.iloc[i]['volume']
        print(f"{time_str:<25} ${price:>10,.2f} {change:>9.2f}% {volume:>14,.0f}")

    print("\n" + "="*70)

    # 6. Save Predictions
    output_file = "./data/btc_future_predictions.csv"
    pred_df_save = pred_df.copy()
    pred_df_save['timestamps'] = y_timestamp.values
    pred_df_save.to_csv(output_file, index=False)
    print(f"✓ Future predictions saved to: {output_file}")

    # 7. Visualize
    print("\nGenerating visualization...")

    # Prepare data for plotting
    historical_df = df.copy()
    historical_df.index = historical_df['timestamps']

    pred_df_plot = pred_df.copy()
    pred_df_plot.index = y_timestamp

    plot_prediction_future(historical_df, pred_df_plot,
                          title=f"BTC Price Forecast - Next {PRED_LEN} Hours")

    print(f"\n{'='*70}")
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Current BTC: ${last_price:,.2f}")
    print(f"Predicted in {PRED_LEN}h: ${last_pred_price:,.2f} ({pred_change:+.2f}%)")
    print(f"\nTrend: {'🚀 BULLISH' if pred_change > 0 else '📉 BEARISH'}")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrediction interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
