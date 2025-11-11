# Kronos Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Activate Environment
```bash
conda activate Kronos
```

### Step 2: Run Your First Prediction
```bash
cd /Users/jonnydubowsky/Kronos/examples
python prediction_example.py
```

That's it! The model will download automatically and show you predictions.

---

## 🎯 Common Tasks

### Predict Bitcoin (BTC) Prices

**Quick Start:**
```bash
cd /Users/jonnydubowsky/Kronos/examples

# Step 1: Download BTC data (requires yfinance)
pip install yfinance
python download_btc_data.py

# Step 2: Run BTC prediction
python btc_prediction_example.py
```

The script will:
- Use MPS (Apple Silicon GPU) for acceleration
- Predict 120 periods ahead using 400 historical periods
- Display predictions vs actual prices
- Show accuracy metrics (MAE, RMSE, MAPE)
- Save predictions to CSV

**Customize the prediction:**
Edit [btc_prediction_example.py](examples/btc_prediction_example.py):
```python
DEVICE = "mps"       # Your device
MODEL_SIZE = "small" # "mini", "small", or "base"
LOOKBACK = 400       # Historical window
PRED_LEN = 120       # Future predictions
```

### Make a Simple Prediction
```python
from model import Kronos, KronosTokenizer, KronosPredictor
import pandas as pd

# Load model (downloads automatically first time)
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# Create predictor (use "mps" for Apple Silicon, "cuda:0" for NVIDIA GPU)
predictor = KronosPredictor(model, tokenizer, device="mps", max_context=512)

# Load your data (must have: open, high, low, close, timestamps)
df = pd.read_csv("your_data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# Predict next 120 periods using last 400 periods
pred_df = predictor.predict(
    df=df.iloc[:400][['open', 'high', 'low', 'close', 'volume']],
    x_timestamp=df.iloc[:400]['timestamps'],
    y_timestamp=df.iloc[400:520]['timestamps'],
    pred_len=120
)

print(pred_df.head())
```

### Start Web UI
```bash
cd /Users/jonnydubowsky/Kronos/webui
python run.py
# Open http://localhost:7070 in browser
```

### Use Different Models
```python
# Fastest (mini)
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")

# Balanced (small) - RECOMMENDED
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

# Best accuracy (base)
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
```

---

## 📊 Data Format

Your CSV needs these columns:
- `timestamps` (or `date`)
- `open`
- `high`
- `low`
- `close`
- `volume` (optional)
- `amount` (optional)

Example:
```csv
timestamps,open,high,low,close,volume
2024-01-01 09:30:00,100.5,101.2,99.8,100.9,1000000
2024-01-01 09:35:00,100.9,102.5,100.5,102.0,1200000
```

---

## 🔧 Key Parameters

### Prediction Parameters
- `lookback`: How many past periods to use (default: 400)
- `pred_len`: How many future periods to predict (default: 120)
- `T`: Temperature for sampling (higher = more random, default: 1.0)
- `top_p`: Nucleus sampling probability (default: 0.9)
- `sample_count`: Number of samples to average (default: 1)

### Device Selection
- `"cpu"`: CPU only (slowest)
- `"mps"`: Apple Silicon GPU (M1/M2/M3)
- `"cuda:0"`: NVIDIA GPU

---

## 📁 File Locations

| What | Where |
|------|-------|
| Your environment | `Kronos` conda env |
| Examples | [/Users/jonnydubowsky/Kronos/examples](examples/) |
| Web UI | [/Users/jonnydubowsky/Kronos/webui](webui/) |
| Your data | Put CSV files in [/Users/jonnydubowsky/Kronos/data](data/) |
| Configuration | [.env.template](.env.template) → `.env` |
| Full guide | [SETUP_GUIDE.md](SETUP_GUIDE.md) |

---

## ❓ FAQ

**Q: Do I need API keys?**
A: No! Kronos works without any API keys for basic usage.

**Q: Which model should I use?**
A: Start with `Kronos-small` - it's the best balance of speed and accuracy.

**Q: Can I use GPU?**
A: Yes! Use `device="mps"` for Apple Silicon or `device="cuda:0"` for NVIDIA.

**Q: How do I use my own data?**
A: Put your CSV in the `data/` folder with columns: timestamps, open, high, low, close, volume.

**Q: Where are models stored?**
A: Automatically cached in `~/.cache/huggingface/hub/`

**Q: Can I fine-tune on my data?**
A: Yes! See [finetune_csv/README.md](finetune_csv/README.md) for CSV data or [finetune/](finetune/) for Qlib data.

---

## 🆘 Need Help?

- **Full Documentation**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **GitHub Issues**: https://github.com/shiyu-coder/Kronos/issues
- **Paper**: https://arxiv.org/abs/2508.02739
- **Live Demo**: https://shiyu-coder.github.io/Kronos-demo/

---

## 📝 Environment Activation Reminder

Every time you open a new terminal:
```bash
conda activate Kronos
```

Check it's active:
```bash
which python
# Should show: /Users/jonnydubowsky/miniconda3/envs/Kronos/bin/python
```
