# StockPricePrediction

## Stock price prediction 
🎯 Project Objective
To build a machine learning model that predicts whether the stock price will rise the next day (binary classification), using technical indicators and engineered features derived from historical stock data.

Target Variable:
Target = 1 if next day Close > today’s Close
Target = 0 otherwise

📊 Final Selected Features
python
Copy code
features = [
    'ratio', 'Close', 'Daily Return', 'Volume', 'Volume_Change',
    'MA20','Momentum','Momentum5', 'STD5', 'STD10', 'STD20',
    'Momentum10', 'Momentum15', 'Monthly_Volume_Avg',
    'Open_Close_Diff', 'lagVolume_Change', 'ATR14',
    'lagClose5','RSI', 'MACD', 'MACD_signal', 'price_vol_interaction'
]
These were selected based on performance metrics like F1 score and recall.

🏗️ Feature Engineering Process
  1. Price-Based Features
    Feature	Description
      Close	Daily closing price
      Daily Return	% change from previous day: Close.pct_change()
      Momentum, Momentum5, Momentum10, Momentum15	Measures how much price changed from prior days (e.g., Close - Close.shift(5))
      lagClose5	Price 5 days ago (Close.shift(5))
      MA5, MA10, MA20	Moving averages to capture trend (short/medium-term smoothing)
      STD5, STD10, STD20	Price volatility over different windows using rolling standard deviation
      Open_Close_Diff	Intraday momentum: Open - Close
  
  2. Volume-Based Features
    Feature	Description
      Volume	Raw daily volume
      Volume_Change	% change in volume from previous day
      Monthly_Volume_Avg	21-day rolling average of volume (monthly approximation)
      lagVolume_Change	5-day lag of volume change
      price_vol_interaction	Interaction between volume change and return: Volume_Change * Daily Return
  
  3. Technical Indicators
    Indicator	Description
      ATR14	Average True Range over 14 days → Measures volatility using high/low/close spread
      RSI	Relative Strength Index → Momentum oscillator from 0-100
      MACD	Moving Average Convergence Divergence → Short EMA - Long EMA (12 & 26)
      MACD_signal	9-day EMA of MACD
  
  4. Derived Ratios & Interaction Terms
    Feature	Description
      ratio	Ratio of 5-day avg Close (shifted) to current Open: (Close.rolling(5).mean().shift(1)) / Open
      price_vol_interaction	Captures days with both price movement and volume surge

🧼 Data Cleaning & Post-processing
  All .shift() or .rolling() operations naturally introduce NaN values at the beginning.
  
  These were handled using df.dropna(inplace=True) to ensure clean data before training.

📝 Why LightGBM & Stacking?
  Faster and memory-efficient than XGBoost for large datasets.
  Better with categorical & sparse data.
  Stacking improves robustness across different market conditions.

💡 Future Plans & Extensions
  Integrate LSTM-based deep learning models for sequence learning.
  AutoML optimization pipelines (Optuna, HyperOpt).
  Live API for streaming current stock prices & predictions.
  Support for multiple stocks & dynamic feature selection.
  Enhanced model explainability using SHAP or LIME.
  Model retraining with rolling windows for true online learning.

