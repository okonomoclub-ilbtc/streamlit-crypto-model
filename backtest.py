"""Backtest real de las reglas de la Sentinel V10 Pro.

Simula exactamente lo que describe el manual operativo y lo que implementa
streamlit-crypto-model.py:

    ENTRADA  (señal evaluada sobre el cierre del día T)
        LONG : Close[T] > EMA(span)[T]  y  ret3D[T] <= -umbral
        SHORT: Close[T] < EMA(span)[T]  y  ret3D[T] >= +umbral
        Ejecución al día siguiente: Open[T+1]

    SALIDA   (time-exit, 24 h ≈ 1 vela diaria)
        Close[T+1]   (cierre del mismo día en que se entró)

    DIMENSIONADO
        nocional = riesgo_pct * capital / (ATR14[T] / Close[T])
        nocional = min(nocional, apalancamiento_max * capital)

No hay stop loss, igual que en la app. Por eso el riesgo "0.5%" NO acota la
pérdida: una vela adversa mayor que 1 ATR la excede. El backtest modela eso en
lugar de ocultarlo, y por eso el nocional se recorta cuando la pérdida supera el
patrimonio (liquidación).

Uso:
    python backtest.py                      # BTC-USD, todos los activos
    python backtest.py ETH-USD
    python backtest.py BTC-USD --fee 0.001
"""
import argparse

import numpy as np
import pandas as pd
import yfinance as yf

FEE_POR_LADO = 0.0005   # 5 bps por lado: comisión + slippage
RIESGO_PCT = 0.005      # 0.5% del balance, igual que el default de la app
APALANCAMIENTO_MAX = 2.0
EMA_SPAN = 50
MOM_PCT = 3.0


def descargar(ticker, periodo="max"):
    df = yf.download(ticker, period=periodo, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def indicadores(df, ema_span=EMA_SPAN, mom_pct=MOM_PCT):
    out = df.copy()
    out["ema"] = out["Close"].ewm(span=ema_span, adjust=False).mean()
    out["ret3d"] = out["Close"].pct_change(3) * 100
    prev_close = out["Close"].shift(1)
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - prev_close).abs(),
        (out["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    return out.dropna()


def senales(ind, mom_pct=MOM_PCT):
    """Devuelve la serie de señales: 1 long, -1 short, 0 sin posición."""
    s = pd.Series(0, index=ind.index, dtype=int)
    s[(ind["Close"] > ind["ema"]) & (ind["ret3d"] <= -mom_pct)] = 1
    s[(ind["Close"] < ind["ema"]) & (ind["ret3d"] >= mom_pct)] = -1
    return s


def simular(ind, fee=FEE_POR_LADO, riesgo_pct=RIESGO_PCT,
            apalancamiento=APALANCAMIENTO_MAX, capital_inicial=100_000.0,
            senal=None, rng=None):
    """Itera sobre las velas. Sin lookahead: la señal de T se ejecuta en T+1.

    `senal` permite inyectar una serie de señales alternativa (para los baselines
    aleatorios). Si es None se usan las reglas de la estrategia.
    """
    if senal is None:
        senal = senales(ind)
    capital = capital_inicial
    pico = capital
    dd_max = 0.0
    operaciones = []
    equity_curve = []

    fechas = list(ind.index)
    for i in range(len(fechas) - 1):
        equity_curve.append((fechas[i], capital))
        f_salida, f_entrada = fechas[i], fechas[i + 1]
        lado = int(senal.iloc[i])
        if lado == 0:
            continue

        px_entrada = float(ind["Open"].loc[f_entrada])       # Open[T+1]
        px_salida = float(ind["Close"].loc[f_entrada])        # Close[T+1], 24 h después
        atr = float(ind["atr"].loc[f_salida])
        if not np.isfinite(atr) or atr <= 0 or px_entrada <= 0:
            continue

        # Dimensionado idéntico al de la app.
        nocional = (capital * riesgo_pct) / (atr / px_entrada)
        nocional = min(nocional, apalancamiento * capital)

        bruto = px_salida / px_entrada - 1
        neto = (bruto if lado == 1 else -bruto) - 2 * fee    # entrada y salida
        pnl = nocional * neto
        # Sin stop loss: si la pérdida supera el patrimonio, es liquidación total.
        pnl = max(pnl, -capital)
        capital += pnl

        pico = max(pico, capital)
        dd_max = max(dd_max, (pico - capital) / pico)

        operaciones.append({
            "entrada": f_entrada, "salida": f_entrada, "lado": lado,
            "px_entrada": px_entrada, "px_salida": px_salida,
            "ret_neto": neto, "nocional": nocional, "pnl": pnl,
            "equity": capital,
        })

    equity_curve.append((fechas[-1], capital))
    return pd.DataFrame(operaciones), pd.Series(
        dict(equity_curve)).sort_index(), dd_max


def metricas(ops, equity, dd_max, capital_inicial=100_000.0):
    if ops.empty:
        return {"trades": 0}
    pnl = ops["pnl"]
    Wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    m = {
        "trades": len(ops),
        "win_rate": float((pnl > 0).mean()),
        "payoff": float(Wins.mean() / abs(losses.mean())) if len(losses) else np.nan,
        "media_operacion": float(pnl.mean()),
        "mediana_operacion": float(pnl.median()),
        "mejor": float(pnl.max()),
        "peor": float(pnl.min()),
        "dd_max": float(dd_max),
        "retorno_total": float((equity.iloc[-1] - capital_inicial) / capital_inicial),
        "final_equity": float(equity.iloc[-1]),
        "exposicion_pct": float((ops["ret_neto"].abs() > 0).mean()),
    }
    m["profit_factor"] = float(Wins.sum() / abs(losses.sum())) if len(losses) else np.inf
    m["recovery_factor"] = (
        float(pnl.sum() / (dd_max * capital_inicial)) if dd_max > 0 else np.inf
    )
    años = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    m["cagr"] = float((equity.iloc[-1] / capital_inicial) ** (1 / años) - 1)
    return m


def baseline_buy_hold(ind, capital_inicial=100_000.0):
    px0, px1 = float(ind["Close"].iloc[0]), float(ind["Close"].iloc[-1])
    final = capital_inicial * px1 / px0
    equity = capital_inicial * ind["Close"] / px0
    pico = equity.cummax()
    dd = float(((pico - equity) / pico).max())
    años = max((ind.index[-1] - ind.index[0]).days / 365.25, 1e-9)
    return {"trades": 0, "retorno_total": final / capital_inicial - 1,
            "dd_max": dd, "cagr": (final / capital_inicial) ** (1 / años) - 1}


def baseline_aleatorio(ind, n_operaciones, fee=FEE_POR_LADO, repeticiones=2000,
                       seed=7, capital_inicial=100_000.0):
    """Baseline decisivo: entradas ALEATORIAS con el MISMO número de operaciones y el
    MISMO dimensionamiento que la estrategia.

    Si la estrategia no supera esta distribución, su resultado se explica por el
    dimensionamiento y el límite de operaciones, no por las reglas EMA+momentum.
    """
    rng = np.random.default_rng(seed)
    reales = np.flatnonzero(senales(ind).to_numpy() != 0)
    n = len(reales)
    if n == 0:
        return None
    ret = []
    for _ in range(repeticiones):
        idx = rng.choice(len(ind) - 2, size=n, replace=False)
        s = pd.Series(0, index=ind.index, dtype=int)
        for j in idx:
            s.iloc[j] = 1 if rng.random() < 0.5 else -1
        o, e, _ = simular(ind, fee=fee, senal=s, rng=rng, capital_inicial=capital_inicial)
        ret.append(float(o["pnl"].sum()))
    return np.array(ret)


def linea(nombre, m):
    if m.get("trades", 0) == 0 and "retorno_total" not in m:
        return f"  {nombre:22s} sin operaciones"
    r = m["retorno_total"]
    d = m["dd_max"]
    extra = ""
    if m.get("trades"):
        extra = f" | trades {m['trades']:4d} | win {m['win_rate']:6.1%} | payoff {m['payoff']:.2f}"
    return (f"  {nombre:22s} retorno {r:+8.2%} | dd_max {d:6.2%}"
            f" | cagr {m['cagr']:+7.2%}{extra}")


def significatividad(ops, n_boot=10000, seed=7):
    """Bootstrap y t-estadístico sobre los retornos por trade.

    AVISO: las operaciones son diarias y pueden ser consecutivas, así que sus
    retornos tienen autocorrelación y NO son iid. El intervalo bootstrap asume
    independencia, luego es optimista. Se reporta también la autocorrelación de
    orden 1 para poder juzgarlo.
    """
    r = ops["ret_neto"].to_numpy()
    n = len(r)
    if n < 5:
        return None
    rng = np.random.default_rng(seed)
    medias = np.array([rng.choice(r, size=n, replace=True).mean() for _ in range(n_boot)])
    ic = np.percentile(medias, [2.5, 97.5])
    t_stat = float(r.mean() / (r.std(ddof=1) / np.sqrt(n)))
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if n > 3 else float("nan")
    # Qué fracción del retorno depende de unas pocas operaciones ganadoras.
    top5 = np.sort(r)[-5:].sum()
    return {"n": n, "media": float(r.mean()), "ic": (float(ic[0]), float(ic[1])),
            "t": t_stat, "ac1": ac1, "share_top5": float(top5 / r.sum()) if r.sum() != 0 else float("nan")}


def breakeven_fee(ind, max_fee=0.01, tolerancia=0.00005):
    """Comisión por lado a la que el retorno cae a cero. 0 = nunca es rentable."""
    o, e, d = simular(ind, fee=0.0)
    if metricas(o, e, d).get("retorno_total", -1) <= 0:
        return 0.0
    lo, hi = 0.0, max_fee
    for _ in range(20):
        mid = (lo + hi) / 2
        o, e, d = simular(ind, fee=mid)
        if metricas(o, e, d).get("retorno_total", -1) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tolerancia:
            break
    return lo


def barrido_costas(ind):
    """¿En qué comisión deja de ser rentable la estrategia?"""
    print(f"\n  BARRIDO DE COSTES (la ventaja es pequeña: los costes importan)")
    print(f"  {'fee/lado':>9s} {'coste/trade':>12s} {'retorno':>10s} {'pnl':>12s}")
    for fee in (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01):
        o, e, d = simular(ind, fee=fee)
        m = metricas(o, e, d)
        if m.get("trades", 0) == 0:
            continue
        print(f"  {fee:8.4%} {2 * fee:11.4%} {m['retorno_total']:+9.2%} {m['final_equity'] - 100_000:>+11,.0f}")
    be = breakeven_fee(ind)
    if be == 0.0:
        print("  → Nunca es rentable: ni con comisión cero el retorno bruto es positivo.")
    else:
        print(f"  → Punto de equilibrio: {be:.4%} por lado ({2 * be:.4%} por trade).")
        print(f"    Por debajo de eso la estrategia gana; por encima, pierde dinero.")


def por_ano(ind, fee=FEE_POR_LADO):
    """El edge, ¿es consistente cada año o está concentrado en unos pocos?"""
    print(f"\n  DESGLOSE POR AÑO (indicadores calculados sobre toda la historia)")
    print(f"  {'año':>6s} {'ops':>5s} {'retorno':>10s} {'win':>7s} {'payoff':>7s}")
    for año in sorted(set(ind.index.year)):
        sub = ind[ind.index.year == año]
        if len(sub) < 30:
            continue
        o, e, d = simular(sub, fee=fee)
        m = metricas(o, e, d)
        if m.get("trades", 0) == 0:
            print(f"  {año:6d} {0:5d}   sin operaciones")
            continue
        print(f"  {año:6d} {m['trades']:5d} {m['retorno_total']:+9.2%} {m['win_rate']:6.1%} {m['payoff']:6.2f}x")


def run_ticker(ticker, fee=FEE_POR_LADO, mom=MOM_PCT, ema=EMA_SPAN, simulaciones=2000,
               periodo="max", analisis_extra=True):
    print(f"\n{'=' * 78}\n{ticker}   EMA{ema} / momentum ±{mom}% / "
          f"fee {fee:.3%} por lado / riesgo {RIESGO_PCT:.1%} / lev {APALANCAMIENTO_MAX}x\n{'=' * 78}")
    df = descargar(ticker, periodo)
    ind = indicadores(df, ema, mom)
    print(f"  datos: {ind.index[0].date()} -> {ind.index[-1].date()}  ({len(ind)} velas, period={periodo})")

    ops, equity, dd = simular(ind, fee=fee)
    m = metricas(ops, equity, dd)
    print("\n  ESTRATEGIA")
    print(linea("Sentinel V10", m))

    bh = baseline_buy_hold(ind)
    print("\n  BASELINES (para no confundir azar con edge)")
    print(linea("buy & hold", bh))
    n = len(ind)
    largos = (ind["Close"].pct_change().fillna(0) > 0).mean()
    print(f"  {'dirección al azar':22s} {largos:6.1%} de velas alcistas -> "
          f"{(2 * largos - 1) * 100:+.2f}% de aciertos esperados")

    # El test decisivo: mismas reglas de dinero, entradas elegidas al azar.
    print(f"\n  BASELINE ALEATORIO ({simulaciones} simulaciones, mismo nº de operaciones")
    print("  y mismo dimensionamiento que la estrategia)")
    ale = baseline_aleatorio(ind, m["trades"], fee=fee, repeticiones=simulaciones)
    if ale is not None:
        pnl_real = float(ops["pnl"].sum())
        pct = float((ale < pnl_real).mean())
        print(f"    pnl estrategia      : ${pnl_real:+,.0f}")
        print(f"    pnl aleatorio       : mediana ${np.median(ale):+,.0f}  "
              f"rango ${ale.min():,.0f} … ${ale.max():,.0f}")
        print(f"    percentil de la estrategia entre entradas aleatorias: {pct:.1%}")
        if pct >= 0.95:
            print("    → La estrategia supera al azar con margen: las reglas aportan valor.")
        elif pct >= 0.80:
            print("    → Supera al azar, pero sin margen amplio. Evidencia débil.")
        else:
            print("    → NO supera al azar: el resultado se explica por el dimensionamiento,")
            print("      no por las reglas EMA+momentum.")

    # Split cronológico: mitad de la muestra para in-sample, resto out-of-sample.
    corte = ind.index[len(ind) // 2]
    print(f"\n  SPLIT CRONOLÓGICO (corte en {corte.date()})")
    for etiqueta, sub in (("in-sample ", ind[ind.index <= corte]), ("out-sample", ind[ind.index > corte])):
        if len(sub) < 60:
            print(f"  {etiqueta}: muestra demasiado corta ({len(sub)} velas)")
            continue
        o, e, d = simular(sub, fee=fee)
        print(linea(etiqueta, metricas(o, e, d)))

    if analisis_extra and not ops.empty:
        sig = significatividad(ops)
        if sig:
            print(f"\n  SIGNIFICATIVIDAD (bootstrap 10.000, sobre {sig['n']} operaciones)")
            print(f"    retorno medio por trade : {sig['media']:+.4%}")
            print(f"    IC 95% del retorno medio: [{sig['ic'][0]:+.4%}, {sig['ic'][1]:+.4%}]")
            print(f"    t-estadístico          : {sig['t']:+.2f}")
            print(f"    autocorrelación lag-1  : {sig['ac1']:+.3f}  "
                  f"({'los trades NO son independientes' if abs(sig['ac1']) > 0.1 else 'casi independientes'})")
            print(f"    peso de las 5 mejores   : {sig['share_top5']:.0%} del retorno total")
            if sig["ic"][0] <= 0 <= sig["ic"][1]:
                print("    → El IC 95% incluye el cero: no se puede rechazar que el retorno")
                print("      medio sea nulo. La ventaja no es estadísticamente significante.")
            elif sig["t"] < 2:
                print("    → IC por encima de cero pero |t| < 2: evidencia débil.")
            else:
                print("    → IC excluye el cero y |t| > 2: diferencia significativa,")
                print("      sujeta a la cautela por la autocorrelación de los trades.")

        por_ano(ind)
        barrido_costas(ind)

    return ind, m


def sensibilidad(ticker):
    """¿El resultado depende de los parámetros? Un edge real sobrevive a vecinos."""
    print(f"\n{'=' * 78}\nSENSIBILIDAD DE PARÁMETROS — {ticker}\n{'=' * 78}")
    print("  Un edge real se mantiene al mover los parámetros. Si solo gana en un")
    print("  punto exacto, es sobreajuste (curve fitting), no una ventaja.\n")
    df = descargar(ticker, "max")
    filas = []
    print(f"  {'EMA':>5s} {'mom':>5s} | {'trades':>7s} {'retorno':>10s} {'dd_max':>8s} {'payoff':>7s}")
    for ema in (20, 50, 100):
        for mom in (2.0, 3.0, 4.0):
            ind = indicadores(df, ema, mom)
            o, e, d = simular(ind)
            m = metricas(o, e, d)
            if m.get("trades", 0) == 0:
                print(f"  {ema:5d} {mom:5.1f} | {'0':>7s}  (sin operaciones)")
                continue
            filas.append((ema, mom, m["retorno_total"]))
            print(f"  {ema:5d} {mom:5.1f} | {m['trades']:7d} {m['retorno_total']:+9.2%} "
                  f"{m['dd_max']:7.2%} {m['payoff']:6.2f}x")
    if filas:
        positivos = [r for _, _, r in filas if r > 0]
        negativos = [r for _, _, r in filas if r <= 0]
        print(f"\n  Configuraciones con retorno positivo: {len(positivos)}/{len(filas)}")
        if len(negativos) == 0:
            print("  → Todas las configuraciones ganan. El resultado no depende de los")
            print("    parámetros: no es curve fitting.")
        elif len(positivos) == 0:
            print("  → Todas las configuraciones PIERDEN, de forma consistente. La")
            print("    estrategia no tiene edge en este activo con estos costes.")
        else:
            print("  → El signo cambia al mover los parámetros: el resultado es frágil y")
            print("    no constituye evidencia de ventaja.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=["BTC-USD", "ETH-USD", "SOL-USD"])
    ap.add_argument("--fee", type=float, default=FEE_POR_LADO)
    ap.add_argument("--simulaciones", type=int, default=2000,
                    help="réplicas del baseline aleatorio")
    ap.add_argument("--period", default="max", help="max | 10y | 5y | 2y")
    ap.add_argument("--rapido", action="store_true",
                    help="omite significatividad, desglose anual y barrido de costes")
    ap.add_argument("--sensibilidad", action="store_true", help="barrido de parámetros")
    args = ap.parse_args()

    tickers = args.tickers or ["BTC-USD", "ETH-USD", "SOL-USD"]
    print(f"Backtest Sentinel V10 Pro | coste {args.fee:.3%} por lado "
          f"(entrada + salida = {2 * args.fee:.3%} por trade) | "
          f"riesgo nominal {RIESGO_PCT:.1%} | apalancamiento máx {APALANCAMIENTO_MAX}x")
    print("Sin stop loss: el riesgo nominal NO acota la pérdida por trade.")

    for t in tickers:
        run_ticker(t, fee=args.fee, simulaciones=args.simulaciones, periodo=args.period, analisis_extra=not args.rapido)

    if args.sensibilidad:
        for t in tickers:
            sensibilidad(t)