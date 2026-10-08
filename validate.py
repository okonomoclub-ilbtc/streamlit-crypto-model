"""Harness de validacion walk-forward (rapido, sklearn, semilla fija).

Compara un modelo Ridge contra baselines naives: "manana = hoy" y media movil de 5.
Uso: python validate.py [TICKER]   (default BTC-USD)

Nota sobre la construccion de los baselines: la fila de fecha d tiene como objetivo
el cierre de d+1. Por tanto el baseline "manana = hoy" debe ser el cierre de d
(NO el de d-1), y la MA5 debe promediar los cierres que terminan en d.
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge

SEED = 7
TICKER = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
TEST_DAYS = 365

df = yf.download(TICKER, period="5y", progress=False, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.dropna()
close = df["Close"].astype(float)

feat = pd.DataFrame(index=df.index)
feat["ret1"] = close.pct_change(1)
feat["ret5"] = close.pct_change(5)
feat["rsi"] = 100 - 100 / (1 + close.diff().clip(lower=0).rolling(14).mean()
                           / (-close.diff().clip(upper=0).rolling(14).mean()).replace(0, np.nan))
feat["dist_ema20"] = close / close.ewm(span=20, adjust=False).mean() - 1
feat["vol20"] = feat["ret1"].rolling(20).std()
feat["target"] = close.shift(-1)  # cierre de manana
data = feat.dropna()
print(f"ticker={TICKER}  filas={len(data)}  rango={data.index[0].date()} -> {data.index[-1].date()}", flush=True)

Xcols = ["ret1", "ret5", "rsi", "dist_ema20", "vol20"]
test = data.iloc[-TEST_DAYS:]
preds, dates = [], []
for i in range(0, len(test), 30):  # reentrena cada mes
    block = test.iloc[i:i + 30]
    train = data.loc[:block.index[0]].iloc[:-1]
    mu, sd = train[Xcols].mean(), train[Xcols].std().replace(0, 1)
    model = Ridge(random_state=SEED)
    model.fit(((train[Xcols] - mu) / sd).values, train["target"].values)
    preds.extend(model.predict(((block[Xcols] - mu) / sd).values))
    dates.extend(block.index)

res = pd.DataFrame({"real": test["target"].values[:len(preds)], "modelo": preds}, index=dates)
# Baseline "manana = hoy": el forecast para el objetivo de d es el cierre de d.
res["naive"] = close.reindex(res.index)
# Media movil de 5 cierres que terminan en d (incluye d).
res["ma5"] = [close.loc[:d].iloc[-5:].mean() for d in res.index]
res = res.dropna()

print(f"\nn={len(res)} observaciones fuera de muestra\n")
level = float(close.reindex(res.index).mean())
print(f"{'modelo':10s} {'MAE':>10s} {'MAE %':>8s}   (menor = mejor)")
for col in ("modelo", "naive", "ma5"):
    mae = float((res["real"] - res[col]).abs().mean())
    print(f"{col:10s} {mae:10.1f} {100 * mae / level:7.2f}%")

# Acierto direccional solo tiene sentido para-columnas que predicen un movimiento
# distinto de cero. El baseline "naive" predice literalmente el precio de hoy, asi
# que su signo siempre es 0: medirlo seria reportar la frecencia de dias alcistas,
# no skill del modelo. Se excluye a proposito.
closes = close.reindex(res.index)
up = (res["real"] - closes) > 0
print()
for col in ("modelo", "ma5"):
    acc = float((up == ((res[col] - closes) > 0)).mean())
    print(f"acierto direccional {col:10s} = {acc:.1%}")
print(f"(naive excluido: su forecast es el cierre de hoy, signo 0 por definicion)")
print(f"frecuencia de dias alcistas en el test = {up.mean():.1%}  <- referencia de azar")

mae_modelo = float((res["real"] - res["modelo"]).abs().mean())
mae_naive = float((res["real"] - res["naive"]).abs().mean())
print()
if mae_modelo > mae_naive:
    print(f"VEREDICTO: el modelo NO supera al baseline naive ({mae_modelo:.0f} vs {mae_naive:.0f}).")
else:
    print(f"VEREDICTO: el modelo supera al baseline naive en MAE.")