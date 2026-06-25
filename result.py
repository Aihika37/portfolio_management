import matplotlib.pyplot as plt
import numpy as np
from data_loader import top10_stocks
import model_1
import model_2
import model_3
from model_1 import cm,ret_all
iterations=10000

#comparing the models and determine which is best to compute weights
def compare_models(results):
    print("\nMODEL COMPARISON")
    print("-" * 100)
    for r in results:
        print(
            f"{r['name']:30s}"
            f" Return={r['return']:.4f}"
            f" Risk={r['risk']:.4f}"
            f" Sharpe={r['sharpe']:.4f}"
        )
        print("Weights:")
        for stock, weight in zip(top10_stocks, r["weights"]):
            print(f"  {stock:10s} {weight*100:6.2f}%")
        print("-" * 100)

    best = max(results, key=lambda x: x["sharpe"])
    print("\nBEST STRATEGY:", best["name"])
    return best

def display_portfolio(best):
    print("\nRECOMMENDED PORTFOLIO")
    print("-" * 50)
    for stock, weight in zip(
            top10_stocks,
            best["weights"]):
        print(f"{stock:20s}"f"{weight*100:.2f}%")
    print()
    print("Expected Return:",round(best["return"]*100,2),"%")
    print("Risk:",round(best["risk"],2))
    print("Sharpe:",round(best["sharpe"],2))

def visualize_models(results):
    plt.figure(figsize=(11, 6))

    # Random portfolios
    risks = []
    returns = []
    #plotting portfolio risk and return on the graph using random weights
    for _ in range(iterations):
        weight = np.random.rand(10)
        weight /= weight.sum()
        ret = np.dot(ret_all, weight)
        risk = np.sqrt(weight.T @ cm @ weight)
        returns.append(ret)
        risks.append(risk)
    #visualising the expected return and risk of portfolio on a graph between risk and return
    plt.scatter(risks,returns,s=10,alpha=0.3, label="Random Portfolios")
    markers = ['o', 's', '^', 'D', '*', 'P']
    colors = ['orange', 'green', 'red', 'mediumpurple', 'brown', 'hotpink']
    for r, marker, color in zip(results, markers, colors):
        plt.scatter(r["risk"],r["return"],marker=marker,color=color,s=160,
        edgecolors="black",linewidth=1,label=r["name"])
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Expected Return")
    plt.title("Portfolio Strategies")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

def main():
    results = []
    # Model 1A
    results.append(model_1.model1())
    # Model 1B
    results.append(model_1.model2(0.24))
    # Model 1C
    results.append(model_1.model3())
    # Model 1D
    results.append(model_1.model4(0.05))
    # Model 2
    results.append(model_2.model())
    # Model 3
    results.append(model_3.train_xgboost())
    best = compare_models(results)
    display_portfolio(best)
    visualize_models(results)

if __name__ == "__main__":
    main()