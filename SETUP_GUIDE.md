# Kronos Setup and Usage Guide

## Overview

**Kronos** is a foundation model for financial time series forecasting, specifically designed for candlestick (K-line) data. It can predict future price movements for stocks, cryptocurrencies, and other financial instruments.

### Key Features
- Pre-trained on data from 45+ global exchanges
- Supports OHLCV (Open, High, Low, Close, Volume) predictions
- Multiple model sizes: mini (4.1M), small (24.7M), base (102.3M)
- No API keys required for basic usage
- Web UI for easy visualization
- Fine-tuning support for custom datasets

---

## 1. Environment Setup ✅

### 1.1 Conda Environment (Recommended)

A conda environment has been created for you:

```bash
# Activate the Kronos environment
conda activate Kronos

# Verify installation
python --version  # Should show Python 3.12.x
pip list          # Should show torch, pandas, matplotlib, etc.
```

### 1.2 Required Dependencies

All dependencies are already installed in the `Kronos` conda environment:
- **PyTorch** 2.9.0 (with MPS support for Apple Silicon)
- **Pandas** 2.2.2 (data manipulation)
- **NumPy** 2.2.6 (numerical operations)
- **Matplotlib** 3.9.3 (visualization)
- **Hugging Face Hub** 0.33.1 (model downloads)
- **einops**, **safetensors**, **tqdm** (utilities)

---

## 2. Configuration (Optional)

### 2.1 Basic Usage - No Configuration Needed!

For inference and prediction, **no API keys or configuration files are required**. Models automatically download from Hugging Face Hub.

### 2.2 Optional Configuration (.env file)

If you want to customize paths or use experiment tracking:

```bash
# Copy the template
cp .env.template .env

# Edit .env with your preferred editor
nano .env
```

**What you might want to configure:**
- `DEVICE`: Set to "mps" for Apple Silicon GPU, "cuda:0" for NVIDIA GPU, or "cpu"
- `COMET_API_KEY`: Only if you want experiment tracking during fine-tuning
- `QLIB_DATA_PATH`: Only if fine-tuning on Chinese A-share market data

---

## 3. Quick Start - Making Your First Prediction

### 3.1 Basic Prediction Example

```bash
# Activate environment
conda activate Kronos

# Navigate to examples directory
cd /Users/jonnydubowsky/Kronos/examples

# Run the prediction example
python prediction_example.py
```

**What this does:**
1. Downloads Kronos-small model (~24.7M params) from Hugging Face
2. Downloads Kronos-Tokenizer-base automatically
3. Loads sample financial data (600977.csv)
4. Predicts next 120 time periods based on 400 historical periods
5. Displays visualization comparing predictions vs ground truth

### 3.2 Understanding the Example Code

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# 1. Load model and tokenizer (downloads automatically)
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. Create predictor
predictor = KronosPredictor(model, tokenizer, device="mps", max_context=512)

# 3. Load your data (must have: open, high, low, close columns)
df = pd.read_csv("your_data.csv")

# 4. Prepare input
lookback = 400  # Historical window
pred_len = 120  # Future predictions

x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.iloc[:lookback]['timestamps']
y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']

# 5. Generate predictions
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # Temperature (controls randomness)
    top_p=0.9,      # Nucleus sampling
    sample_count=1  # Number of samples to average
)
```

---

## 4. Available Examples

### 4.1 Standard Examples

| Example | Description | File |
|---------|-------------|------|
| Basic Prediction | Single time series forecast | [prediction_example.py](examples/prediction_example.py) |
| Batch Prediction | Multiple time series at once | [prediction_batch_example.py](examples/prediction_batch_example.py) |
| Without Volume | Prediction using only OHLC | [prediction_wo_vol_example.py](examples/prediction_wo_vol_example.py) |
| Chinese Markets | Daily predictions for A-shares | [prediction_cn_markets_day.py](examples/prediction_cn_markets_day.py) |

### 4.2 Running Examples

```bash
# Navigate to examples
cd /Users/jonnydubowsky/Kronos/examples

# Activate environment
conda activate Kronos

# Run any example
python prediction_example.py
python prediction_batch_example.py
python prediction_wo_vol_example.py
```

---

## 5. Web UI - Interactive Interface

### 5.1 Starting the Web UI

```bash
# Navigate to webui directory
cd /Users/jonnydubowsky/Kronos/webui

# Activate environment
conda activate Kronos

# Install webui dependencies (if not already installed)
pip install -r requirements.txt

# Start the web server
python run.py
```

**Access the UI:**
- Open browser to: http://localhost:7070
- Load a model (kronos-mini, kronos-small, or kronos-base)
- Upload your CSV data or use sample data
- Configure prediction parameters
- View interactive charts

### 5.2 Web UI Features

- **Model Selection**: Choose between mini/small/base models
- **Data Upload**: Load CSV files with financial data
- **Interactive Charts**: Plotly-based candlestick visualizations
- **Custom Time Windows**: Select specific date ranges
- **Parameter Tuning**: Adjust temperature, top_p, lookback, prediction length
- **Comparison Mode**: Overlay predictions with actual data

---

## 6. Working with Your Own Data

### 6.1 Data Format Requirements

Your CSV file must have these columns:
- `open` (required)
- `high` (required)
- `low` (required)
- `close` (required)
- `timestamps` or `date` (required - datetime format)
- `volume` (optional but recommended)
- `amount` (optional)

**Example CSV:**
```csv
timestamps,open,high,low,close,volume,amount
2024-01-01 09:30:00,100.5,101.2,99.8,100.9,1000000,100900000
2024-01-01 09:35:00,100.9,102.5,100.5,102.0,1200000,122400000
...
```

### 6.2 Data Preparation Tips

```python
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Convert timestamps to datetime
df['timestamps'] = pd.to_datetime(df['timestamps'])

# Ensure numeric columns
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove any NaN values
df = df.dropna()

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
```

---

## 7. Model Selection Guide

| Model | Parameters | Context Length | Use Case | Download Time |
|-------|-----------|----------------|----------|---------------|
| **Kronos-mini** | 4.1M | 2048 | Fast predictions, resource-constrained | ~20 MB |
| **Kronos-small** | 24.7M | 512 | Balanced performance/speed | ~100 MB |
| **Kronos-base** | 102.3M | 512 | Best accuracy, more compute | ~400 MB |

**Recommendations:**
- **Beginners**: Start with `Kronos-small`
- **CPU Only**: Use `Kronos-mini`
- **Apple Silicon (M1/M2/M3)**: Use `Kronos-base` with device="mps"
- **NVIDIA GPU**: Use `Kronos-base` with device="cuda:0"

---

## 8. Fine-Tuning on Custom Data

### 8.1 Prerequisites

```bash
# Install qlib (only for A-share market data)
conda activate Kronos
pip install pyqlib
```

### 8.2 Fine-Tuning Process

The fine-tuning workflow is in the `finetune/` directory:

1. **Configure**: Edit [finetune/config.py](finetune/config.py)
2. **Prepare Data**: Run `python finetune/qlib_data_preprocess.py`
3. **Train Tokenizer**: Run `torchrun --standalone --nproc_per_node=1 finetune/train_tokenizer.py`
4. **Train Predictor**: Run `torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py`
5. **Backtest**: Run `python finetune/qlib_test.py --device mps`

### 8.3 Fine-Tuning with CSV Data

For custom CSV data (non-Qlib), use the `finetune_csv/` directory:

```bash
cd /Users/jonnydubowsky/Kronos/finetune_csv

# See README for details
cat README.md
```

---

## 9. Repository Structure

```
/Users/jonnydubowsky/Kronos/
├── model/                      # Core model implementations
│   ├── kronos.py              # Main model classes
│   └── module.py              # Model components
├── examples/                   # Ready-to-run examples
│   ├── prediction_example.py
│   ├── prediction_batch_example.py
│   └── data/                  # Sample datasets
├── webui/                     # Web interface
│   ├── app.py                 # Flask backend
│   ├── run.py                 # Entry point
│   └── requirements.txt       # Web UI dependencies
├── finetune/                  # Fine-tuning (Qlib)
│   ├── config.py              # Configuration
│   ├── train_tokenizer.py     # Tokenizer training
│   ├── train_predictor.py     # Predictor training
│   └── qlib_test.py           # Backtesting
├── finetune_csv/              # Fine-tuning (CSV data)
│   └── README.md              # CSV fine-tuning guide
├── tests/                     # Unit tests
├── requirements.txt           # Core dependencies
├── .env.template              # Environment template
└── README.md                  # Project documentation
```

---

## 10. API Keys and Services

### 10.1 Required Services

**None required for basic usage!**

Kronos works out-of-the-box with no API keys.

### 10.2 Optional Services

| Service | Purpose | Required For | Sign Up |
|---------|---------|--------------|---------|
| **Comet.ml** | Experiment tracking | Fine-tuning (optional) | https://www.comet.com/ |
| **Hugging Face** | Model sharing | Uploading custom models | https://huggingface.co/ |
| **Qlib** | Chinese market data | A-share fine-tuning | https://github.com/microsoft/qlib |

---

## 11. Performance Optimization

### 11.1 Apple Silicon (M1/M2/M3)

```python
# Use MPS for GPU acceleration
predictor = KronosPredictor(model, tokenizer, device="mps", max_context=512)
```

### 11.2 NVIDIA GPU

```python
# Use CUDA
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
```

### 11.3 CPU Optimization

```python
# Use smaller model for faster CPU inference
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
```

---

## 12. Troubleshooting

### 12.1 Common Issues

**Issue**: Model download fails
```bash
# Solution: Check internet connection and Hugging Face access
ping huggingface.co
```

**Issue**: Out of memory
```bash
# Solution: Use smaller model or reduce batch size
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
```

**Issue**: Slow predictions
```bash
# Solution: Use GPU acceleration
# Apple Silicon: device="mps"
# NVIDIA: device="cuda:0"
```

### 12.2 Getting Help

- **GitHub Issues**: https://github.com/shiyu-coder/Kronos/issues
- **Paper**: https://arxiv.org/abs/2508.02739
- **Live Demo**: https://shiyu-coder.github.io/Kronos-demo/

---

## 13. Next Steps

### 13.1 Learn the Features

1. ✅ Run basic prediction example
2. ✅ Try the Web UI
3. ✅ Test with your own data
4. ⬜ Experiment with different models
5. ⬜ Fine-tune on custom dataset
6. ⬜ Build a trading strategy

### 13.2 Advanced Topics

- **Batch Prediction**: Process multiple time series efficiently
- **Probabilistic Forecasting**: Use multiple samples and temperature
- **Backtesting**: Evaluate predictions against historical data
- **Portfolio Optimization**: Combine predictions with risk management

---

## 14. Citation

If you use Kronos in your research:

```bibtex
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets},
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739},
}
```

---

## 15. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
