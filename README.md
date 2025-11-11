# LSTM Price Prediction for Nifty 50 (Intraday)

Forecast Nifty 50 intraday prices with an LSTM model trained on minute-level data (2015–2024). Includes training callbacks and a baseline configuration that runs on low-memory machines, plus guidance for scaling up on GPU. :contentReference[oaicite:0]{index=0}

---

## Repo Structure
- `LSTM PRICE PREDICTION.ipynb` — main training/forecasting notebook. :contentReference[oaicite:1]{index=1}  
- `LSTM price prediction.ipynb` — alternate notebook (same project). :contentReference[oaicite:2]{index=2}  
- `LICENSE` — MIT. :contentReference[oaicite:3]{index=3}

---

## Key Features
- Minute-interval pricing data from **2015 → 2024** (intraday). :contentReference[oaicite:4]{index=4}  
- Uses `ModelCheckpoint` and `ReduceLROnPlateau` callbacks (no EarlyStopping in this setup). :contentReference[oaicite:5]{index=5}  
- Baseline results (current low-resource settings): **MSE ≈ 0.0254**. :contentReference[oaicite:6]{index=6}

> **Note:** The notebook intentionally uses *small* sequence lengths, few LSTM layers, fewer epochs, and a short forecast horizon to fit limited RAM/VRAM. Increase these on a better GPU/CPU. :contentReference[oaicite:7]{index=7}

---

## Quick Start

### 1) Clone
```bash
git clone https://github.com/Mohammad-Hossein-sedghi/lstm-nifty50-intraday-prediction.git
cd lstm-nifty50-intraday-prediction
```
### 2) Environment
```python
python -m venv .venv
# Windows
. .venv/Scripts/activate
# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install jupyter numpy pandas scikit-learn matplotlib tensorflow
```
Optional: install tensorflow[and-cuda] if you have a compatible NVIDIA GPU.
### 3) Run the notebook
Open LSTM PRICE PREDICTION.ipynb in Jupyter/Lab, run all cells, and follow the inline comments to adjust paths/hyperparameters.
## How It Works (High Level)

1. Load & scale intraday price series.

2. Windowing: build sequences of length seq_len to predict the next step(s).

3. Model: stacked LSTM → Dense output for regression.

4. Train with ModelCheckpoint and ReduceLROnPlateau. (EarlyStopping disabled for this project.) 
GitHub

5. Evaluate (MSE) and plot predictions vs. ground truth.

6. Forecast for the next future_len minutes.

## Important Hyperparameters (tune these)

- seq_len (lookback window)

- future_len (forecast horizon) — currently kept short for resource limits. 

- n_lstm_layers, hidden_units — currently reduced for low RAM/VRAM. 

- batch_size, epochs — also kept small by default. 

- learning_rate

### Callbacks

- ModelCheckpoint: persist best weights.

- ReduceLROnPlateau: lower LR on plateau.

- (Not using EarlyStopping in this configuration.)

## Baseline Result

- Overall MSE ≈ 0.025405 on the included setup. Your score should improve with longer sequences, deeper networks, more epochs, and GPU training.

## Tips for Better Accuracy

- Increase seq_len, future_len, and model depth gradually.

- Try LayerNormalization or Dropout between LSTM layers.

- Add features: returns, rolling volatility, time-of-day, technical indicators.

- Walk-forward (rolling) validation to mimic live trading.

- Compare against naive baselines (last value; moving average).

- Consider 1D-CNN+LSTM or Transformer encoder as alternatives.

## Reproducibility

- Set a global seed (NumPy/TensorFlow) in the first notebook cell.

- Log versions of tensorflow, numpy, pandas, and your GPU/driver.

## Roadmap
- Export a pure Python script (train.py) with CLI args.

- Add requirements.txt and environment.yml.

- Optional Dockerfile + GPU instructions.

- Automated walk-forward backtests.

- Save/Load scalers and model via joblib/tf.saved_model.

## License

MIT. See `LICENSE` 

## Disclaimer

This project is for research/education. Not financial advice. Use at your own risk.
