# SPY-Hedging-strategies
trying multiple hedging strategies for SPY with cross assets
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# =====================================
# CONFIG
# =====================================

API_KEY = "YOURAPYKEY"  # <-- replace with your key if needed
START_DATE = "2019-01-01"  # Options data available from 2019; analysis focused on drawdowns since then
END_DATE   = "2025-12-31"

INITIAL_CAPITAL = 100_000
CONTRACT_SIZE = 100

OUTPUT_FILE = "SPY_OVERLAY_FULL_REPORT.xlsx"

# Transaction cost assumptions (tune for sensitivity analysis)
STOCK_SLIPPAGE_BPS = 0.001        # 10 bps one-way slippage on SPY (set 0.0 to disable)
OPTION_COMMISSION_PER_CONTRACT = 0.65  # USD per option contract (approx. round-trip charged at entry)

# =====================================
# SESSION WITH RETRY
# =====================================

session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://", adapter)

# =====================================
# DOWNLOAD SPY
# =====================================

print("Downloading SPY...")
spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)

if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

spy = spy.dropna().sort_index()

# Quarterly roll dates: first trading day of each quarter
roll_dates = (
    spy.groupby(spy.index.to_period("Q"))
       .apply(lambda x: x.index[0])
       .sort_values()
       .to_list()
)

print("Total roll dates:", len(roll_dates))

def get_spot_on_or_before(date):
    """
    Get SPY close on or BEFORE the given date.
    This avoids using a Monday price for a Saturday expiration.
    """
    data = spy.loc[spy.index <= date]
    if data.empty:
        return None
    return float(data.iloc[-1]["Close"])

# =====================================
# OPTION FETCH
# =====================================

def get_option_chain(trade_date, cp):
    """
    Fetch SPY options around ~90 DTE with a delta filter.
    cp = "P" for puts, "C" for calls.
    """
    url = "https://restapi.ivolatility.com/equities/eod/stock-opts-by-param"

    if cp == "P":
        delta_from, delta_to = -0.60, -0.05
    else:
        delta_from, delta_to = 0.05, 0.60

    params = {
        "apiKey": API_KEY,
        "symbol": "SPY",
        "dteFrom": 80,
        "dteTo": 100,
        "cp": cp,
        "tradeDate": trade_date.strftime("%Y-%m-%d"),
        "deltaFrom": delta_from,
        "deltaTo": delta_to,
        "region": "USA"
    }

    try:
        r = session.get(url, params=params, timeout=(5, 15))
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    if "data" not in data or len(data["data"]) == 0:
        return None

    df = pd.DataFrame(data["data"])

    required = ["expiration_date", "Bid", "Ask", "price_strike"]
    if not all(col in df.columns for col in required):
        return None

    df = df.dropna(subset=required).copy()
    df["expiration_date"] = pd.to_datetime(df["expiration_date"])

    # Ensure numeric
    for col in ["Bid", "Ask", "price_strike"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Bid", "Ask", "price_strike"])

    if df.empty:
        return None

    return df

# =====================================
# BACKTEST
# =====================================

spy_returns = []
put_returns = []
collar_returns = []
dates = []
alloc_equity_pct = []
alloc_put_pct = []
alloc_collar_net_pct = []

for start in roll_dates:

    spot_start = get_spot_on_or_before(start)
    if spot_start is None:
        continue

    puts = get_option_chain(start, "P")
    calls = get_option_chain(start, "C")

    if puts is None or calls is None:
        continue

    # -------- SELECT PUT (90–95% moneyness, target ~92.5%) --------
    puts["moneyness"] = puts["price_strike"] / spot_start
    puts = puts[(puts["moneyness"] >= 0.90) & (puts["moneyness"] <= 0.95)]
    if puts.empty:
        continue

    puts["target_diff"] = (puts["moneyness"] - 0.925).abs()
    put = puts.sort_values("target_diff").iloc[0]

    expiration = put["expiration_date"]
    spot_exp = get_spot_on_or_before(expiration)
    if spot_exp is None:
        continue

    strike_put = float(put["price_strike"])
    put_cost = float(put["Ask"]) * CONTRACT_SIZE  # per contract

    # -------- SELECT CALL (same expiry, 105–110% moneyness, target ~107.5%) --------
    calls = calls[calls["expiration_date"] == expiration].copy()
    if calls.empty:
        continue

    calls["moneyness"] = calls["price_strike"] / spot_start
    calls = calls[(calls["moneyness"] >= 1.05) & (calls["moneyness"] <= 1.10)]
    if calls.empty:
        continue

    calls["target_diff"] = (calls["moneyness"] - 1.075).abs()
    call = calls.sort_values("target_diff").iloc[0]

    strike_call = float(call["price_strike"])
    call_premium = float(call["Bid"]) * CONTRACT_SIZE  # per contract

    # =============================================
    # CAPITAL ALLOCATION (per-contract sizing with costs)
    # =============================================

    # Stock cost per contract including slippage
    stock_cost_per_contract = spot_start * CONTRACT_SIZE * (1.0 + STOCK_SLIPPAGE_BPS)

    # Overlay costs per contract (option premium + commission)
    put_overlay_per_contract = put_cost + OPTION_COMMISSION_PER_CONTRACT
    collar_overlay_per_contract = put_cost - call_premium + OPTION_COMMISSION_PER_CONTRACT

    # Cost per collar and per protective put (per contract)
    cost_per_contract_put = stock_cost_per_contract + put_overlay_per_contract
    cost_per_contract_collar = stock_cost_per_contract + collar_overlay_per_contract

    # Use the more expensive one for sizing so both strategies fit
    cost_per_contract = max(cost_per_contract_put, cost_per_contract_collar)

    contracts = int(INITIAL_CAPITAL // cost_per_contract)
    if contracts == 0:
        continue

    shares = contracts * CONTRACT_SIZE

    # Total stock cost including slippage
    total_stock_cost = shares * spot_start * (1.0 + STOCK_SLIPPAGE_BPS)

    # Total overlay costs
    total_put_overlay_cost = contracts * put_overlay_per_contract
    total_collar_overlay_cost = contracts * collar_overlay_per_contract

    # Cash for each strategy after paying all entry costs
    cash_spy = INITIAL_CAPITAL - total_stock_cost
    cash_put = INITIAL_CAPITAL - (total_stock_cost + total_put_overlay_cost)
    cash_collar = INITIAL_CAPITAL - (total_stock_cost + total_collar_overlay_cost)

    # =============================================
    # PAYOFFS AT EXPIRATION
    # =============================================

    equity_value = shares * spot_exp

    # SPY
    final_spy = equity_value + cash_spy
    spy_ret = final_spy / INITIAL_CAPITAL - 1.0

    # PROTECTIVE PUT
    put_payoff = contracts * max(strike_put - spot_exp, 0.0) * CONTRACT_SIZE
    final_put = equity_value + put_payoff + cash_put
    put_ret = final_put / INITIAL_CAPITAL - 1.0

    # COLLAR
    call_payoff = contracts * max(spot_exp - strike_call, 0.0) * CONTRACT_SIZE
    final_collar = equity_value + put_payoff - call_payoff + cash_collar
    collar_ret = final_collar / INITIAL_CAPITAL - 1.0

    spy_returns.append(spy_ret)
    put_returns.append(put_ret)
    collar_returns.append(collar_ret)
    dates.append(start)

    # Allocation: equity vs derivatives as % of initial capital
    alloc_equity_pct.append((shares * spot_start) / INITIAL_CAPITAL)
    alloc_put_pct.append((put_cost * contracts) / INITIAL_CAPITAL)
    alloc_collar_net_pct.append((put_cost - call_premium) * contracts / INITIAL_CAPITAL)

    # small pause to be gentle with the API
    time.sleep(0.5)

# =====================================
# DATAFRAME
# =====================================

df = pd.DataFrame({
    "Start": dates,
    "SPY": spy_returns,
    "SPY+Put": put_returns,
    "SPY+Collar": collar_returns
})

df["SPY_Cum"] = (1 + df["SPY"]).cumprod()
df["SPY+Put_Cum"] = (1 + df["SPY+Put"]).cumprod()
df["SPY+Collar_Cum"] = (1 + df["SPY+Collar"]).cumprod()

# =====================================
# PERFORMANCE METRICS
# =====================================

def performance_stats(returns):
    returns = np.array(returns, dtype=float)

    if len(returns) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Quarterly returns -> annualized
    n_years = len(returns) / 4.0
    cumulative = np.prod(1.0 + returns)

    cagr = cumulative ** (1.0 / n_years) - 1.0 if cumulative > 0 else np.nan

    mean_q = np.mean(returns)
    vol_q = np.std(returns, ddof=1) if len(returns) > 1 else np.nan
    vol_ann = vol_q * np.sqrt(4.0) if not np.isnan(vol_q) else np.nan

    sharpe = (mean_q / vol_q) * np.sqrt(4.0) if vol_q not in (0, np.nan) else np.nan

    curve = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(curve)
    drawdown = (curve - peak) / peak
    max_dd = float(drawdown.min())

    return round(cagr, 4), round(vol_ann, 4), round(sharpe, 3), round(max_dd, 4)

stats = pd.DataFrame(
    [
        performance_stats(df["SPY"]),
        performance_stats(df["SPY+Put"]),
        performance_stats(df["SPY+Collar"])
    ],
    columns=["CAGR", "Vol", "Sharpe", "MaxDD"],
    index=["SPY", "SPY+Put", "SPY+Collar"]
)

# Additional distribution statistics (skewness, kurtosis)
dist_rows = []
for col in ["SPY", "SPY+Put", "SPY+Collar"]:
    series = df[col].dropna()
    dist_rows.append(
        {
            "Strategy": col,
            "Mean_Q": series.mean(),
            "Std_Q": series.std(ddof=1),
            "Skew_Q": series.skew(),
            "Kurtosis_Excess_Q": series.kurt(),  # pandas returns excess kurtosis by default
        }
    )

dist_stats_df = pd.DataFrame(dist_rows)

# =====================================
# REGIME / SUBPERIOD ANALYSIS (from 2019 onwards)
# =====================================

regimes = [
    ("Late-cycle_2019", "2019-01-01", "2019-12-31"),
    ("COVID_and_QE_2020_2021", "2020-01-01", "2021-12-31"),
    ("Tightening_2022_2023", "2022-01-01", "2023-12-31"),
    ("Recent_2024_2025", "2024-01-01", "2025-12-31"),
]

regime_rows = []

for regime_name, start_date, end_date in regimes:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    mask = (df["Start"] >= start_ts) & (df["Start"] <= end_ts)
    sub = df.loc[mask]

    if sub.empty:
        continue

    for col, strat_name in [("SPY", "SPY"), ("SPY+Put", "SPY+Put"), ("SPY+Collar", "SPY+Collar")]:
        cagr, vol, sharpe, max_dd = performance_stats(sub[col])
        regime_rows.append(
            {
                "Regime": regime_name,
                "Strategy": strat_name,
                "CAGR": cagr,
                "Vol": vol,
                "Sharpe": sharpe,
                "MaxDD": max_dd,
                "N_Quarters": len(sub),
            }
        )

regime_stats_df = pd.DataFrame(regime_rows)

# =====================================
# CRISIS WINDOW ANALYSIS (peak-to-trough on quarterly series)
# =====================================

def identify_peak_to_trough_episodes(dates, equity_curve, threshold=0.15):
    """
    Identify peak-to-trough drawdown episodes using the same frequency
    as the backtest (quarterly). Episodes trigger once drawdown <= -threshold
    and end when a new high is reached (drawdown returns to 0).
    """
    curve = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(curve)
    dd = curve / running_max - 1.0

    episodes = []
    in_episode = False
    peak_idx = None
    trough_idx = None
    min_dd = None

    for i in range(len(curve)):
        if not in_episode:
            if dd[i] <= -threshold:
                # peak is the last index where we hit the running max for this level
                peak_candidates = np.where(curve[: i + 1] == running_max[i])[0]
                peak_idx = int(peak_candidates[-1]) if len(peak_candidates) else 0
                trough_idx = i
                min_dd = float(dd[i])
                in_episode = True
        else:
            if float(dd[i]) < min_dd:
                min_dd = float(dd[i])
                trough_idx = i

            # episode ends once we recover to a new high (dd back to ~0)
            if dd[i] >= -1e-12:
                episodes.append(
                    {
                        "Peak": pd.to_datetime(dates[peak_idx]),
                        "Trough": pd.to_datetime(dates[trough_idx]),
                        "Depth": float(min_dd),
                        "PeakIdx": int(peak_idx),
                        "TroughIdx": int(trough_idx),
                    }
                )
                in_episode = False
                peak_idx = None
                trough_idx = None
                min_dd = None

    # If we end the sample mid-episode, still record the peak-to-current-trough
    if in_episode and peak_idx is not None and trough_idx is not None:
        episodes.append(
            {
                "Peak": pd.to_datetime(dates[peak_idx]),
                "Trough": pd.to_datetime(dates[trough_idx]),
                "Depth": float(min_dd),
                "PeakIdx": int(peak_idx),
                "TroughIdx": int(trough_idx),
            }
        )

    return episodes


# Build quarterly equity curves (starting at 1.0) for each strategy
equity_q = pd.DataFrame(
    {
        "Start": df["Start"],
        "SPY": (1.0 + df["SPY"]).cumprod(),
        "SPY+Put": (1.0 + df["SPY+Put"]).cumprod(),
        "SPY+Collar": (1.0 + df["SPY+Collar"]).cumprod(),
    }
).set_index("Start")

# Identify crises on the benchmark SPY curve so they match your drawdown plot frequency
episodes = identify_peak_to_trough_episodes(
    dates=equity_q.index.values,
    equity_curve=equity_q["SPY"].values,
    threshold=0.15,  # 15% peak-to-trough threshold
)

crisis_rows = []

for j, ep in enumerate(episodes, start=1):
    peak_dt = ep["Peak"]
    trough_dt = ep["Trough"]
    crisis_name = f"DD{j}_{peak_dt.date()}_{trough_dt.date()}"

    sub = df.loc[(df["Start"] >= peak_dt) & (df["Start"] <= trough_dt)]
    if sub.empty:
        continue

    for col, strat_name in [("SPY", "SPY"), ("SPY+Put", "SPY+Put"), ("SPY+Collar", "SPY+Collar")]:
        rets = sub[col].values
        cum_ret = float(np.prod(1.0 + rets) - 1.0)

        curve = np.cumprod(1.0 + rets)
        peak = np.maximum.accumulate(curve)
        dd = (curve - peak) / peak
        max_dd = float(dd.min())

        crisis_rows.append(
            {
                "Crisis": crisis_name,
                "Peak": peak_dt,
                "Trough": trough_dt,
                "Depth_SPY": ep["Depth"],
                "Strategy": strat_name,
                "CumRet": cum_ret,
                "MaxDD": max_dd,
                "N_Quarters": len(sub),
            }
        )

crisis_stats_df = pd.DataFrame(crisis_rows)

# =====================================
# WORST / BEST QUARTERS (by SPY leg)
# =====================================

worst = df.nsmallest(5, "SPY")
best  = df.nlargest(5, "SPY")

# =====================================
# PLOTS (saved as PNG)
# =====================================

# Equity curves
equity = df.set_index("Start")[["SPY_Cum", "SPY+Put_Cum", "SPY+Collar_Cum"]]
plt.figure(figsize=(10, 6))
for col in equity.columns:
    plt.plot(equity.index, equity[col], label=col)
plt.title("Cumulative Performance (Starting at 1.0)")
plt.xlabel("Date")
plt.ylabel("Cumulative Wealth (normalised)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("equity_curves.png", dpi=150)
plt.close()

# Drawdown curves
plt.figure(figsize=(10, 6))
for col in ["SPY", "SPY+Put", "SPY+Collar"]:
    curve = (1.0 + df[col]).cumprod()
    peak = np.maximum.accumulate(curve)
    dd = (curve / peak) - 1.0
    plt.plot(df["Start"], dd, label=col)
plt.title("Quarterly Drawdowns")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("drawdowns.png", dpi=150)
plt.close()

# Boxplot of quarterly returns
plt.figure(figsize=(8, 6))
plt.boxplot(
    [df["SPY"].dropna(), df["SPY+Put"].dropna(), df["SPY+Collar"].dropna()],
    labels=["SPY", "SPY+Put", "SPY+Collar"],
)
plt.title("Distribution of Quarterly Returns")
plt.ylabel("Quarterly Return")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("quarterly_returns_boxplot.png", dpi=150)
plt.close()

# =====================================
# ALLOCATION: SPY vs DERIVATIVES
# =====================================

alloc_df = pd.DataFrame({
    "Start": dates,
    "Equity_Pct": alloc_equity_pct,
    "Put_Cost_Pct": alloc_put_pct,
    "Collar_Net_Pct": alloc_collar_net_pct,
})

# Combined graph: average allocation per strategy
fig, ax = plt.subplots(figsize=(8, 5))
strategies = ["SPY", "SPY+Put", "SPY+Collar"]
equity_avg = [1.0, np.mean(alloc_equity_pct), np.mean(alloc_equity_pct)]
deriv_avg = [0.0, np.mean(alloc_put_pct), np.mean(alloc_collar_net_pct)]

x = np.arange(len(strategies))
w = 0.35
bars1 = ax.bar(x - w/2, equity_avg, w, label="SPY (equity)", color="steelblue")
bars2 = ax.bar(x + w/2, deriv_avg, w, label="Derivatives (net cost)", color="coral")
ax.set_ylabel("% of Initial Capital")
ax.set_title("Average Portfolio Allocation: SPY vs Derivatives")
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("allocation_spy_vs_derivatives.png", dpi=150)
plt.close()

# =====================================
# PAYOFF TABLE (derivatives at expiry)
# =====================================

# Illustrative: S = terminal SPY, K_put = 92.5%, K_call = 107.5% of initial spot
S0 = 400  # illustrative spot
K_put = S0 * 0.925
K_call = S0 * 1.075

S_levels = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
payoff_rows = []
for m in S_levels:
    S = S0 * m
    put_payoff = max(K_put - S, 0)
    call_payoff_short = -max(S - K_call, 0)
    payoff_rows.append({
        "Moneyness_S": m,
        "S": S,
        "Put_Payoff": put_payoff,
        "Short_Call_Payoff": call_payoff_short,
        "Collar_Net_Payoff": put_payoff + call_payoff_short,
    })

payoff_table_df = pd.DataFrame(payoff_rows)

# =====================================
# PAYOFF DIAGRAMS (dissertation figures)
# =====================================

def payoff_protective_put():
    """Payoff at expiry: Long SPY + Long Put (K_put = 92.5%)"""
    S = np.linspace(S0 * 0.7, S0 * 1.2, 200)
    equity = S
    put_payoff = np.maximum(K_put - S, 0)
    total = equity + put_payoff
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(S, equity, "k--", alpha=0.5, label="Long SPY")
    ax.plot(S, put_payoff, "b--", alpha=0.5, label="Long Put")
    ax.plot(S, total, "b-", lw=2, label="Protective Put (combined)")
    ax.axvline(K_put, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("SPY at Expiry (S)")
    ax.set_ylabel("Payoff")
    ax.set_title("Protective Put: Long SPY + Long OTM Put")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(S0 * 0.7, S0 * 1.2)
    plt.tight_layout()
    plt.savefig("payoff_protective_put.png", dpi=150)
    plt.close()

def payoff_collar():
    """Payoff at expiry: Long SPY + Long Put + Short Call"""
    S = np.linspace(S0 * 0.7, S0 * 1.2, 200)
    equity = S
    put_payoff = np.maximum(K_put - S, 0)
    call_payoff = -np.maximum(S - K_call, 0)
    total = equity + put_payoff + call_payoff
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(S, equity, "k--", alpha=0.5, label="Long SPY")
    ax.plot(S, put_payoff, "b--", alpha=0.5, label="Long Put")
    ax.plot(S, call_payoff, "r--", alpha=0.5, label="Short Call")
    ax.plot(S, total, "g-", lw=2, label="Collar (combined)")
    ax.axvline(K_put, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(K_call, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("SPY at Expiry (S)")
    ax.set_ylabel("Payoff")
    ax.set_title("Collar: Long SPY + Long Put + Short Call")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(S0 * 0.7, S0 * 1.2)
    plt.tight_layout()
    plt.savefig("payoff_collar.png", dpi=150)
    plt.close()

payoff_protective_put()
payoff_collar()

# =====================================
# GREEKS: Delta (Black-Scholes)
# =====================================

def bs_delta_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1

def bs_delta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

# Delta vs moneyness (S/S0) for put and call, T=0.25, r=0.04, sigma=0.2
S_vals = np.linspace(S0 * 0.8, S0 * 1.2, 100)
T, r, sigma = 0.25, 0.04, 0.20
delta_put = [bs_delta_put(s, K_put, T, r, sigma) for s in S_vals]
delta_call = [bs_delta_call(s, K_call, T, r, sigma) for s in S_vals]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(S_vals / S0, delta_put, "b-", lw=2, label="Put Delta (K=92.5%)")
ax.plot(S_vals / S0, delta_call, "r-", lw=2, label="Call Delta (K=107.5%)")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Moneyness (S / S₀)")
ax.set_ylabel("Delta")
ax.set_title("Option Delta vs Spot (Black-Scholes, T=3 months)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("greek_delta.png", dpi=150)
plt.close()

# =====================================
# EXPORT
# =====================================

with pd.ExcelWriter(OUTPUT_FILE) as writer:
    df.to_excel(writer, sheet_name="Quarterly_Returns", index=False)
    stats.to_excel(writer, sheet_name="Performance")
    dist_stats_df.to_excel(writer, sheet_name="Distribution_Stats", index=False)
    regime_stats_df.to_excel(writer, sheet_name="Regime_Performance", index=False)
    crisis_stats_df.to_excel(writer, sheet_name="Crisis_Windows", index=False)
    alloc_df.to_excel(writer, sheet_name="Allocation_SPY_vs_Deriv", index=False)
    payoff_table_df.to_excel(writer, sheet_name="Derivative_Payoffs", index=False)
    worst.to_excel(writer, sheet_name="Worst_SPY_Quarters", index=False)
    best.to_excel(writer, sheet_name="Best_SPY_Quarters", index=False)

print("\nBacktest complete.")
print(stats)
print("\nTrades executed:", len(df))
print("First backtest date:", df["Start"].min())
print("Last  backtest date:", df["Start"].max())