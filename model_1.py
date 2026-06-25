import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from data_loader import top10_stocks

stocks=top10_stocks
starting_date = "2019-01-01"
ending_date = "2025-01-01"
NUM_OF_T_DAYS=252
NUM_OF_STOCKS=10
RISK_FREE = 0.05

data_new = yf.download(top10_stocks,start=starting_date, end=ending_date,
                   auto_adjust=True,threads=False)
close_prices = data_new["Close"]
returns=((close_prices-close_prices.shift(1))/close_prices.shift(1))[1:]#finding returns
expected_returns=returns.mean()*NUM_OF_T_DAYS#finding mean over time for each stock
expected_returns=expected_returns.to_numpy()#return as vector of length 10
covariance_matrix=returns.cov()*NUM_OF_T_DAYS#finding covariance matrix
covariance_matrix=((covariance_matrix.to_numpy()))
e=np.ones(NUM_OF_STOCKS,dtype=int)
c_inverse=np.linalg.inv(covariance_matrix)
ret_all=expected_returns
cm=covariance_matrix
data=returns

def model1():#using optimization(weights can be negative-->short selling allowed)
    weights = (np.dot(c_inverse, e)) / (np.dot(e.T, np.dot(c_inverse, e)))
    s = 0
    for i in range(0, len(weights)):
        s += weights[i]
    ris = np.sqrt(np.dot(weights.T, np.dot(cm, weights)))#finding risk using weighted variance
    ret = np.dot(ret_all, weights)#return of portfolio
    sharpe = (ret - RISK_FREE) / ris#sharpe ratio
    return {
        "name": "GLOBAL_MEAN_VARIANCE_PORTFOLIO",
        "weights": weights,
        "return": ret.round(3),
        "risk": ris.round(3),
        "sharpe": round(sharpe,3)
    }


def model2(R):# finding weights using one more constraint as fixed return R
    #equations obtained by optimization
    num_1 = np.linalg.det(
        [[R, np.dot(expected_returns.T, (np.dot(c_inverse, e)))], [1, np.dot(e.T, np.dot(c_inverse, e))]])
    num_2 = np.linalg.det([[np.dot(expected_returns.T, np.dot(c_inverse, expected_returns)), R],
                           [np.dot(e.T, np.dot(c_inverse, expected_returns)), 1]])
    deno = np.linalg.det([[np.dot(expected_returns.T, np.dot(c_inverse, expected_returns)),
                           np.dot(expected_returns.T, (np.dot(c_inverse, e)))],
                          [np.dot(e.T, np.dot(c_inverse, expected_returns)), np.dot(e.T, np.dot(c_inverse, e))]])
    weights = num_1 / deno * np.dot(c_inverse, expected_returns) + num_2 / deno * np.dot(c_inverse, e)
    s = 0
    for i in range(0, len(weights)):
        s += weights[i]
    ris = np.sqrt(np.dot(weights.T, np.dot(cm, weights)))
    ret = np.dot(ret_all, weights)
    sharpe = (ret - RISK_FREE) / ris
    return {
        "name": "WITH_A_GIVEN_RETURN",
        "weights": weights,
        "return": ret.round(3),
        "risk": ris.round(3),
        "sharpe": round(sharpe, 3)
    }
    


def func(weights, data):
    covariance_matrix = data.cov() * NUM_OF_T_DAYS
    covariance_matrix = ((covariance_matrix.to_numpy()))
    ris = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return ris


def model3():#using inbuilt optimzation(with weights between 0 and 1 and their sum is 1)
    s = 0
    constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}#to ensure sum of weights is 1
    bound = tuple((0, 1) for _ in range(len(stocks)))#bounds on weights
    ini = np.zeros(len(stocks))
    min = minimize(fun=func, x0=ini, args=data, method='SLSQP', constraints=constraint, bounds=bound)
    weight = min['x'].round(3)
    for w in weight:
        s += w
    risk = min['fun'].round(3)
    ret = np.dot(ret_all, weight)
    sharpe = (ret - RISK_FREE) / risk
    return {
        "name": "LONG_ONLY_MIN_VARIANCE",
        "weights": weight,
        "return": ret.round(3),
        "risk": risk,
        "sharpe": round(sharpe, 3)
    }


def model4(R):#using inbuilt optimzation on finding atleast return R
    s = 0
    #defining the constraints
    constraint = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                  {'type': 'ineq', 'fun': lambda x: np.dot(x, ret_all) - R}]
    ini = np.zeros(len(stocks))# intial weights
    min = minimize(fun=func, x0=ini, args=data, method='SLSQP', constraints=constraint)
    weight = min['x'].round(3)
    for w in min['x']:
        s += w
    risk = min['fun'].round(3)
    ret = np.dot(ret_all, weight)
    sharpe = (ret - RISK_FREE) / risk
    return {
        "name": "WITH_RETURN_GREATER_THAN_SPECIFIC_VALUE",
        "weights": weight,
        "return": ret.round(3),
        "risk": risk,
        "sharpe": round(sharpe, 3)
    }








