# SPY-Hedging-strategies
trying multiple hedging strategies for SPY with cross assets
Strategy Summary
# 1.⁠ ⁠SPY (Benchmark)
Long‑only position in SPDR S&P 500 ETF (SPY).
No derivatives; fully exposed to equity upside and downside.
Serves as the unhedged reference portfolio.

# 2.⁠ ⁠SPY + Protective Put
Long SPY plus a long put option on SPY.
Every quarter we buy a new put with ~80–100 days to expiry and strike 90–95% of spot, i.e. roughly −0.20 to −0.30 delta.
Objective: pay a recurring premium to cap downside beyond ~5–10% falls while retaining most upside.

# 3.⁠ ⁠SPY + Collar (Put + Short Call)
Long SPY, long the same 90–95% put (≈ −0.20 / −0.30 delta), and short a call on SPY.
The call has the same expiry, with strike 105–110% of spot, i.e. roughly +0.20 to +0.30 delta.
Objective: use the call premium to partially finance the put, reducing hedge cost, in exchange for capping upside beyond ~5–10% gains above the entry level.

# will later update with more cross assets: Gold, VIX, XOVER, EUROSTOXX, ETFS and indexes...

   this code will download graphic representations and an excel summary of all the results
   backtest goes from 2019 to 2025, needs iVolatility account to scrape the data
