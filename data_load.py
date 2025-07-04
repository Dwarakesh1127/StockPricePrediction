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
    df['ratio'] = df['Close'].shift(5).mean()/df['Open'] #Ratio of prev 5 days close and curr day open
    df['lagClose5'] = df['Close'].shift(5)
    df['Monthly_Volume_Avg'] = df['Volume'].rolling(window=21).mean()
    df['Open_Close_Diff'] = df['Open'] - df['Close']
    df['STD10'] = df['Close'].rolling(window=10).std()
    df['STD20'] = df['Close'].rolling(window=20).std()
    #df['lag5transformed'] = df['Target'].shift(5)
    #can prev vol month sum afect current close? - Monthly_Volume_Avg
    df.dropna(inplace=True)
    return df

#all momentums, all stds, all rolling means, 