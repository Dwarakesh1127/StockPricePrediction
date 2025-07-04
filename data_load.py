# data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np

def get_data(ticker="RELIANCE.NS", period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

def add_technical_indicators(df):
    df['Daily Return'] = df['Close'].pct_change()
    #df['Monthly Return'] = df['Close'].rolling(window=20).mean()
    #df['STD20'] = df['Close'].rolling(window=20).std() #Month with 21 days

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['Momentum5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum10'] = df['Close'] - df['Close'].shift(10)
    df['Momentum15'] = df['Close'] - df['Close'].shift(15)
    df['Momentum30'] = df['Close'] - df['Close'].shift(30)
    df['Momentum'] = df['Close'] - df['Close'].shift(20) #Month wise
    df['STD5'] = df['Close'].rolling(window=5).std()
    df['Upper'] = df['MA5'] + 2 * df['STD5']
    df['Lower'] = df['MA5'] - 2 * df['STD5']
    df['Volume_Change'] = df['Volume'].pct_change().replace([np.inf, -np.inf], 0)
    df['lagVolume_Change'] = df['Volume_Change'].shift(5)
    df['ratio'] = df['Close'].rolling(window=5).mean().shift(1)/df['Open'] #Ratio of prev 5 days close and curr day open
    df['lagClose5'] = df['Close'].shift(5)
    df['Monthly_Volume_Avg'] = df['Volume'].rolling(window=21).mean()
    df['Open_Close_Diff'] = df['Open'] - df['Close']
    df['STD10'] = df['Close'].rolling(window=10).std()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['sales'] = df['Volume_Change'] * df['Open_Close_Diff']  # Assuming 'Volume' is a column in df
    
 
        # ATR Calculation
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TrueRange'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR14'] = df['TrueRange'].rolling(window=10).mean()
 
        # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=10).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs)) 

        # MACD Calculation
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['price_vol_interaction'] = df['Volume_Change'] * df['Daily Return']
    df['directional_vol_change'] = df['Volume_Change'] * np.sign(df['Daily Return'])

    volume_spike_threshold = df['Volume_Change'].quantile(0.90)
    df['is_volume_spike'] = (df['Volume_Change'] > volume_spike_threshold).astype(int)
    df['volume_spike_streak'] = df['is_volume_spike'].rolling(window=10).sum()

    df.dropna(inplace=True)
    return df

#all momentums, all stds, all rolling means,   
