import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data_loader import top10_stocks
from model_1 import RISK_FREE

stocks=top10_stocks
starting_date = "2019-01-01"
ending_date = "2025-01-01"
NUM_OF_T_DAYS=252
RISK_FREE=0.05

class PortfolioNN(nn.Module):#defining the structure of neutral network
    def __init__(self, input_dim):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        return self.model(x)

data_new = yf.download(top10_stocks,start=starting_date, end=ending_date,
                   auto_adjust=True,threads=False)#downloading data
predicted_returns={}

for stock in top10_stocks:
    close = data_new["Close"][stock]
    df = pd.DataFrame()
    #adding features to dataframe
    df["Return"] = close.pct_change()
    df["Volatility"] = (df["Return"].rolling(20).std())
    df["Momentum"] = (close / close.shift(20)) - 1
    df["Target"] = (close.shift(-5) / close) - 1
    df.dropna(inplace=True)
    X = df[["Return","Volatility", "Momentum"]].values
    y = df["Target"].values
    #converting to tensor objects
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    #splitting into train and test data sets
    split = int(0.8 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    #training the data
    model=PortfolioNN(X_train.shape[1])
    loss_fn=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(100):
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()#computing gradient
        optimizer.step()#updating weights
    latest = X[-1].reshape(1, -1)
    with torch.no_grad():
         prediction = model(latest).item()
    predicted_returns[stock] = prediction#predicting future returns

def model():
    top10 = sorted(predicted_returns.items(),key=lambda x: x[1],reverse=True)[:10]#giving weights considering the future returns
    scores = np.array([pred for _, pred in top10])
    weights = np.exp(scores - np.max(scores))
    weights = weights / weights.sum()
    close_prices = data_new["Close"]
    returns=((close_prices-close_prices.shift(1))/close_prices.shift(1))[1:]
    expected_returns = returns.mean().to_numpy() * NUM_OF_T_DAYS
    covariance_matrix = returns.cov().to_numpy() * NUM_OF_T_DAYS
    #finding portfolio return and risk
    portfolio_return = np.dot(expected_returns, weights)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe = (portfolio_return - RISK_FREE) / portfolio_risk
    return {
        "name": "USING_NEURAL_NETWORK",
        "weights": weights,
        "return": portfolio_return,
        "risk": portfolio_risk,
        "sharpe": round(sharpe, 3)
        
    }
