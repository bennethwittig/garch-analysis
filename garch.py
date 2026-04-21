import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import yfinance as yf 
from arch import arch_model

# -1- Load data
SYMBOL = "AAPL"

print(f"Loading data for {SYMBOL}...")
df = yf.download(SYMBOL, period="5y", auto_adjust=True)

df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1)) * 100
df.dropna(inplace=True)

print(f"Loaded {len(df)} trading days")
print(f"\nReturn Statistics:")
print(f"   Mean daily return:  {df['Log_Return'].mean():.3f}%")
print(f"   Std deviation:      {df['Log_Return'].std():.3f}%")
print(f"   Max daily gain:     {df['Log_Return'].max():.3f}%")
print(f"   Max daily loss:     {df['Log_Return'].min():.3f}%")

# -2- Votality Clustering
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
## --Chart1-- ##
ax1.plot(df.index, df["Log_Return"], color="blue", linewidth=0.5, alpha=0.7)
ax1.axhline(y=0, color="white", linewidth=0.5)
ax1.set_title(f"{SYMBOL} – Daily Log Returns (5 years)", fontsize=14)
ax1.set_ylabel("Log Return (%)")
ax1.grid(True, alpha=0.3)
## --Chart2-- ##
df["Rolling_Vol"] = df["Log_Return"].rolling(window=20).std()
ax2.plot(df.index, df["Rolling_Vol"], color="orange", linewidth=1)
ax2.fill_between(df.index, df["Rolling_Vol"], alpha=0.3, color="orange")
ax2.set_title("20-Day Rolling Volatility – Volatility Clustering visible!", fontsize=14)
ax2.set_ylabel("Volatility (%)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("volatility_clustering.png")
plt.show()

# -3- Garch Model Training
print("\nFitting Garch(1,1,) model...")

returns = df["Log_Return"].values
model = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
result = model.fit(disp="off")

print(result.summary())

# -4- Votality Forecast
conditional_vol = result.conditional_volatility
forecast = result.forecast(horizon=30)
future_vol = np.sqrt(forecast.variance.values[-1])

print(f"\n30-Day Volatility Forecast:")
print(f"   Current volatility:  {conditional_vol[-1]:.3f}%")
print(f"   In 5 days:           {future_vol[4]:.3f}%")
print(f"   In 15 days:          {future_vol[14]:.3f}%")
print(f"   In 30 days:          {future_vol[29]:.3f}%")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
## --Chart1-- ##
ax1.plot(df.index, conditional_vol, color="orange", linewidth=1, label="GARCH Volatility")
ax1.fill_between(df.index, conditional_vol, alpha=0.3, color="orange")
ax1.set_title(f"{SYMBOL} – GARCH(1,1) Conditional Volatility", fontsize=14)
ax1.set_ylabel("Volatility (%)")
ax1.legend()
ax1.grid(True, alpha=0.3)
## --Chart2-- ##
days = list(range(1, 31))
ax2.plot(days, future_vol, color="blue", linewidth=2, marker="o", markersize=4)
ax2.fill_between(days, future_vol * 0.8, future_vol * 1.2, alpha=0.2, color="blue", label="Confidence Band")
ax2.axhline(y=conditional_vol[-1], color="orange", linewidth=2, linestyle="--", label=f"Current: {conditional_vol[-1]:.3f}%")
ax2.set_title("30-Day Volatility Forecast", fontsize=14)
ax2.set_xlabel("Days Ahead")
ax2.set_ylabel("Forecasted Volatility (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("garch_forecast.png")
plt.show() 

# -5- Value at Risk (VaR)
current_price = float(df["Close"].iloc[-1].squeeze())
investment = 10000

current_vol = conditional_vol[-1] / 100

var_95 = investment * current_vol * 1.645
var_99 = investment * current_vol * 2.326
var_999 = investment * current_vol * 3.090

print(f"\nVlue at risk analysis")
print(f"=" * 45)
print(f"Investment:          €{investment:,.2f}")
print(f"Current Price:       ${current_price:.2f}")
print(f"Current Volatility:  {conditional_vol[-1]:.3f}% per day")
print(f"\n1-Day Value at Risk:")
print(f"   95% VaR:  €{var_95:.2f} (1 in 20 days)")
print(f"   99% VaR:  €{var_99:.2f} (1 in 100 days)")
print(f"   99.9% VaR: €{var_999:.2f} (1 in 1000 days)")
print(f"\nInterpretation:")
print(f"   With 95% probability, you won't lose more than €{var_95:.2f} tomorrow")
print(f"   With 99% probability, you won't lose more than €{var_99:.2f} tomorrow")
print(f"=" * 45)

# -6- Expected Shortfall (CVaR)
from scipy import stats

def expected_shortfall(investment, volatility, confidence):
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    es = investment * volatility * (stats.norm.pdf(z) / alpha)
    return es

es_95 = expected_shortfall(investment, current_vol, 0.95)
es_99 = expected_shortfall(investment, current_vol, 0.99)
es_999 = expected_shortfall(investment, current_vol, 0.999)

print(f"\nExpected shortfall (CVaR) analysis")
print(f"=" * 45)
print(f"Investment:          €{investment:,.2f}")
print(f"\nComparison VaR vs Expected Shortfall:")
print(f"{'Confidence':<12} {'VaR':>10} {'ES (CVaR)':>12} {'Difference':>12}")
print(f"-" * 45)
print(f"{'95%':<12} {'€'+f'{var_95:.2f}':>10} {'€'+f'{es_95:.2f}':>12} {'€'+f'{es_95-var_95:.2f}':>12}")
print(f"{'99%':<12} {'€'+f'{var_99:.2f}':>10} {'€'+f'{es_99:.2f}':>12} {'€'+f'{es_99-var_99:.2f}':>12}")
print(f"{'99.9%':<12} {'€'+f'{var_999:.2f}':>10} {'€'+f'{es_999:.2f}':>12} {'€'+f'{es_999-var_999:.2f}':>12}")
print(f"=" * 45)

print(f"\nInterpretation:")
print(f"   On the worst 5% of days, average loss: €{es_95:.2f}")
print(f"   On the worst 1% of days, average loss: €{es_99:.2f}")
print(f"   On the worst 0.1% of days, average loss: €{es_999:.2f}")

fig, ax = plt.subplots(figsize=(12, 6))

x = np.linspace(-investment * current_vol * 4, investment * current_vol * 4, 1000)
y = stats.norm.pdf(x, 0, investment * current_vol)

ax.plot(x, y, color="blue", linewidth=2, label="Return Distribution")
ax.axvline(x=-var_95, color="red", linewidth=2, linestyle="--", label=f"VaR Threshold")
ax.axvline(x=-es_95, color="darkred", linewidth=2, linestyle="-", label=f"ES (avg loss beyond VaR): €{es_95:.0f}")

ax.set_title(f"{SYMBOL} – VaR vs Expected Shortfall (€{investment:,} Investment)", fontsize=14)
ax.set_xlabel("Daily P&L (€)")
ax.set_ylabel("Probability Density")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("var_vs_es.png")
plt.show()
