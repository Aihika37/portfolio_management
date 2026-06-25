import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_1 import RISK_FREE
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data_loader import top10_stocks

stocks=top10_stocks
starting_date = "2019-01-01"
ending_date = "2025-01-01"
NUM_OF_T_DAYS=252
RISK_FREE=0.05

def train_xgboost():
    features = ["Return", "MA20","MA50","Volatility","Momentum_1M",
                "Momentum_3M","Volume_Change"]
    #differentiating features and target and dividing into train and test data
    X = final_df[features]
    y = final_df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    model = XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,
        objective="reg:squarederror",random_state=42)#setting the xgboost model
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    predictions = {}
    #defining the features
    features = ["Return","MA20","MA50","Volatility","Momentum_1M"
        ,"Momentum_3M","Volume_Change"]
    for stock, df in stock_data.items():
        latest = df[features].iloc[-1:]
        pred = model.predict(latest)[0]
        predictions[stock] = pred
    top10 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
    scores = np.array([pred for _, pred in top10])
    weights = np.exp(scores - np.max(scores))
    weights /= weights.sum()
    close_prices = data_new["Close"]
    returns = ((close_prices - close_prices.shift(1)) / close_prices.shift(1))[1:]
    #finding the expected return and risk of portfolio
    expected_returns = returns.mean().to_numpy() * NUM_OF_T_DAYS
    covariance_matrix = returns.cov().to_numpy() * NUM_OF_T_DAYS
    portfolio_return = np.dot(expected_returns, weights)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe = (portfolio_return - RISK_FREE) / portfolio_risk
    return {
        "name": "USING_XGBOOST",
        "weights": weights,
        "return": portfolio_return,
        "risk": portfolio_risk,
        "sharpe": round(sharpe, 3)
    }

def create_features(data, stock):
    df = pd.DataFrame()
    #creating features in the data frame
    df["Close"] = data["Close"][stock]
    df["Volume"] = data["Volume"][stock]
    df["Return"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Return"].rolling(20).std()
    df["Momentum_1M"] = (df["Close"] / df["Close"].shift(21)) - 1
    df["Momentum_3M"] = (df["Close"] / df["Close"].shift(63)) - 1
    df["Volume_Change"] = (df["Volume"].pct_change())
    df["Target"] = (df["Close"].shift(-5)/ df["Close"]) - 1
    df.dropna(inplace=True)
    return df

data_new = yf.download(top10_stocks,start=starting_date, end=ending_date,
                   auto_adjust=True,threads=False)
stock_data = {}
all_data = []
for stock in top10_stocks:
    stock_data[stock] = create_features(data_new, stock)
for stock in top10_stocks:
    df = stock_data[stock].copy()
    df["Stock"] = stock
    all_data.append(df)
final_df = pd.concat(all_data, ignore_index=True)#creating final data frame


