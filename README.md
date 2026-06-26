# Portfolio Optimization Using Different Strategies

## Overview

This project combines **Modern Portfolio Theory (MPT)** with **Machine Learning** to construct optimized stock portfolios. Historical stock data is used to predict future returns, which are then fed into portfolio optimization models to generate risk-adjusted investment strategies.

The project compares traditional portfolio optimization techniques with machine learning-enhanced portfolio construction and visualizes the resulting risk-return tradeoffs.

---

## Features

### Portfolio Optimization
- Global Minimum Variance Portfolio (GMVP)
- Return-Constrained Portfolio Optimization
- Long-Only Portfolio Optimization
- Risk-Return Analysis
- Sharpe Ratio Calculation

### Machine Learning Models
- Neural Network (PyTorch) for stock return prediction
- XGBoost Regressor for stock return prediction
- Feature engineering using historical market data

### Visualization
- Risk vs Return scatter plots
- Random portfolio simulations
- Portfolio comparison plots
- Risk-return performance analysis

---

## Methodology

### 1. Data Collection

Historical stock price data is collected using Yahoo Finance.From selected stocks we used momentum score to identify the top 10 stocks..

### 2. Feature Engineering

Features used for prediction include:

- Daily Returns
- Rolling Volatility
- Moving Averages
- Momentum Indicators
- Historical Price Trends

### 3. Return Prediction

#### Neural Network

A Multi-Layer Perceptron (MLP) is trained using PyTorch to predict future stock returns and then arrange stocks to ensure high return stocks come first..

Architecture:

Input → 64 Neurons → ReLU → 32 Neurons → ReLU → Output

#### XGBoost

An XGBoost Regressor is trained on the same feature set to forecast future returns.

### 4. Portfolio Optimization

Predicted returns are used as inputs to Markowitz Mean-Variance Optimization.

Implemented models:

#### Model 1: Global Minimum Variance Portfolio

Minimizes portfolio variance subject to:

Σw = 1

#### Model 2: Fixed Return Portfolio

Minimizes portfolio variance while achieving a specified target return.

#### Model 3: Long-Only Minimum Variance Portfolio

Adds the constraint to the optimization (short selling not allowed, no fixed return)

0 ≤ wᵢ ≤ 1

#### Model 4: Return-Constrained Optimization

Finds the minimum-risk portfolio with return greater than or equal to a target return using scipy optimization.

---

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- PyTorch
- XGBoost
- Matplotlib
- yFinance

---

## Results

The project generates thousands of random portfolios and compares them against optimized portfolios.

Example results:

| Portfolio | Expected Return | Risk |
|------------|----------------|--------|
| Global Minimum Variance | ~24% | ~16% |
| Neural Network Portfolio | ~26% | ~18% |
| XGBoost Portfolio | ~26% | ~18% |

Machine learning-based portfolios achieved higher expected returns while maintaining competitive levels of risk.




This project is intended for educational and research purposes only and should not be considered financial advice.
