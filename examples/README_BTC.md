# Bitcoin Price Prediction with Kronos

This guide shows you how to predict Bitcoin prices using Kronos.

## Quick Start

### 1. Install yfinance (one-time setup)
```bash
conda activate Kronos
pip install yfinance scikit-learn
```

### 2. Download BTC data
```bash
cd /Users/jonnydubowsky/Kronos/examples
python download_btc_data.py
```

This will download ~180 days of hourly BTC data and save it to `./data/btc_data.csv`

### 3. Run BTC prediction
```bash
python btc_prediction_example.py
```

This will:
- Load Kronos-small model (~100MB, downloads automatically)
- Use 400 historical candles to predict next 120 candles
- Run on Apple Silicon GPU (MPS) for fast inference
- Display prediction charts
- Calculate accuracy metrics
- Save predictions to `./data/btc_predictions.csv`

## Customization

Edit the configuration section in `btc_prediction_example.py`:

```python
# Configuration
BTC_DATA_FILE = "./data/btc_data.csv"  # Your BTC data file
DEVICE = "mps"                          # "mps", "cuda:0", or "cpu"
MODEL_SIZE = "small"                    # "mini", "small", or "base"
LOOKBACK = 400                          # Historical window
PRED_LEN = 120                          # Future predictions
```

### Model Selection

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| mini  | Fastest | Good | Quick tests, CPU-only |
| small | Fast | Better | Recommended for most users |
| base  | Slower | Best | Maximum accuracy |

### Data Intervals

Modify `download_btc_data.py` to change the interval:

```python
INTERVAL = "1h"  # Options: "1m", "5m", "15m", "1h", "1d"
DAYS = 180       # Number of days to download
```

Note: Shorter intervals (1m, 5m) are limited to ~60 days by Yahoo Finance.

## Using Your Own BTC Data

If you have BTC data from another source, format it as CSV with these columns:

```csv
timestamps,open,high,low,close,volume
2024-01-01 00:00:00,42000.5,42500.0,41800.0,42300.0,1500.5
2024-01-01 01:00:00,42300.0,42800.0,42100.0,42600.0,1800.2
...
```

Then update `BTC_DATA_FILE` in `btc_prediction_example.py`.

## Advanced: Download from Exchanges

### Using ccxt (supports 100+ exchanges)

```bash
pip install ccxt
```

Edit `download_btc_data.py` and change:
```python
METHOD = "ccxt"  # Instead of "yfinance"
```

Supported exchanges: Binance, Coinbase, Kraken, Bitfinex, etc.

## Troubleshooting

**Error: "yfinance not installed"**
```bash
conda activate Kronos
pip install yfinance
```

**Error: "BTC data file not found"**
```bash
python download_btc_data.py  # Download data first
```

**Slow predictions**
- Use smaller model: `MODEL_SIZE = "mini"`
- Reduce prediction length: `PRED_LEN = 60`
- Check device is set correctly: `DEVICE = "mps"` for Apple Silicon

**Out of memory**
- Use `MODEL_SIZE = "mini"`
- Reduce `LOOKBACK` to 200
- Set `DEVICE = "cpu"`

## Understanding the Output

The script shows:

1. **Prediction Results**: First 10 predicted candles with OHLCV values
2. **Price Analysis**:
   - Last historical price
   - First/last predicted prices
   - Predicted price change %
3. **Accuracy Metrics** (if ground truth available):
   - MAE (Mean Absolute Error): Average price difference
   - RMSE (Root Mean Squared Error): Penalizes large errors
   - MAPE (Mean Absolute Percentage Error): Error as percentage
4. **Visualization**: Charts comparing predictions vs actual prices

## Example Output

```
==============================================================
BTC PRICE PREDICTION WITH KRONOS
==============================================================
Device: mps
Model: Kronos-small
Lookback: 400 periods
Prediction: 120 periods ahead
==============================================================

✓ Loaded 4320 BTC candles
  Date range: 2024-04-15 to 2024-10-15
  Price range: $52,000.00 - $68,000.00

Generating BTC price predictions...

✓ Prediction complete!

==============================================================
PRICE ANALYSIS
==============================================================
Last historical price: $62,450.00
First predicted price: $62,800.00
Last predicted price:  $64,200.00
Predicted change:      +2.80%

Prediction Accuracy Metrics:
  MAE:  $850.50
  RMSE: $1,250.75
  MAPE: 1.35%
==============================================================

✓ Predictions saved to: ./data/btc_predictions.csv
```

## What's Next?

- Try different intervals (5m, 15m, 1d)
- Compare mini/small/base models
- Adjust temperature (T) and top_p for different prediction styles
- Use predictions as input for trading strategies
- Fine-tune on your specific trading pairs

## Resources

- [Kronos Paper](https://arxiv.org/abs/2508.02739)
- [Full Setup Guide](../SETUP_GUIDE.md)
- [Quick Start Guide](../QUICK_START.md)
