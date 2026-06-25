import yfinance as yf
import pandas as pd

stocks = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "RELIANCE.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
    "ITC.NS", "ADANIGREEN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "PIDILITIND.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "M&M.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

starting_date = "2019-01-01"
ending_date = "2025-01-01"
NUM_OF_T_DAYS=252
NUM_OF_STOCKS=10#final stocks after selecting from 50 stocks

def download_data():
    data = yf.download(stocks,start=starting_date,end=ending_date
                       ,auto_adjust=True,progress=True,threads=False)#downloading data
    return data


def momentum_score(close_prices):
    close_prices = close_prices.dropna(axis=1)
    if len(close_prices) < 126:
        raise ValueError("Not enough historical data for momentum calculation")
    latest = close_prices.iloc[-1]
    r1 = latest / close_prices.iloc[-21] - 1#1 month return
    r3 = latest / close_prices.iloc[-63] - 1#2 month return
    r6 = latest / close_prices.iloc[-126] - 1#3 month return
    score = 0.2 * r1 + 0.3 * r3 + 0.5 * r6 #finding weighted average of returns to find score
    return score

data = download_data()
close_prices = data["Close"]
scores = momentum_score(close_prices)
top10 = scores.sort_values(ascending=False).head(10)#taking top 10 stocks according to momemtum score
top10_stocks = top10.index.tolist()







