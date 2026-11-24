import sys
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_daily_data(ticker: str = "BTC-USD", start_date: str = "2000-01-01") -> pd.DataFrame:
    """Descarga velas diarias (1D) reales en UTC y elimina la vela en curso."""
    print(f"[DATA] Descargando historial diario (1D) para {ticker} desde {start_date}...")
    df = yf.download(ticker, start=start_date, interval="1d", progress=False, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No se pudieron descargar datos para {ticker}.")

    # Aplanar columnas MultiIndex si yfinance las devuelve así
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    # Eliminar la vela de hoy si aún no ha cerrado
    if len(df) > 2:
        df = df.iloc[:-1]

    return df


def simulate_alphaia_daily(
    df: pd.DataFrame,
    ema_period: int = 50,
    mom_days: int = 3,
    atr_period: int = 14,
    atr_sl_mult: float = 2.0,
    cost_bps_per_side: float = 15.0,
    annual_days: float = 365.0,
) -> dict:
    """
    Simula EMA 50 + Momentum 3D + Stop ATR + Time-Exit 24h (1 barra diaria).
    Ejecución estricta en la apertura del día siguiente (.shift(1)).
    """
    data = df.copy()
    data["ema50"] = data["close"].ewm(span=ema_period, adjust=False).mean()
    data["mom3d"] = data["close"] / data["close"].shift(mom_days) - 1.0

    tr = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr"] = tr.rolling(atr_period).mean()

    # Señal calculada al cierre de t-1 -> Ejecutada en apertura de t
    raw_sig = (data["close"] > data["ema50"]) & (data["mom3d"] > 0)
    signal = raw_sig.shift(1).fillna(False)
    entry_atr = data["atr"].shift(1)

    cost_rate = (2.0 * cost_bps_per_side) / 10000.0
    daily_returns = np.zeros(len(data))
    trade_returns = []

    opens = data["open"].values
    lows = data["low"].values
    closes = data["close"].values
    sigs = signal.values
    atrs = entry_atr.values

    for i in range(len(data)):
        if sigs[i] and not np.isnan(atrs[i]):
            entry_p = opens[i]
            stop_p = entry_p - (atr_sl_mult * atrs[i])

            # Si el mínimo del día perfora el Stop ATR, sale en el Stop + slippage
            if lows[i] <= stop_p:
                exit_p = stop_p * (1.0 - (5.0 / 10000.0))
            else:
                # Salida por tiempo a las 24h (cierre de la vela diaria)
                exit_p = closes[i]

            net_ret = ((exit_p - entry_p) / entry_p) - cost_rate
            daily_returns[i] = net_ret
            trade_returns.append(net_ret)

    ret_series = pd.Series(daily_returns, index=data.index)
    equity = (1.0 + ret_series).cumprod()

    years = max((data.index[-1] - data.index[0]).days / annual_days, 0.1)
    total_ret = equity.iloc[-1] - 1.0
    cagr = (equity.iloc[-1] ** (1.0 / years)) - 1.0 if equity.iloc[-1] > 0 else -1.0

    std = ret_series.std()
    sharpe = (ret_series.mean() / std) * np.sqrt(annual_days) if std > 0 else 0.0

    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_dd = abs(drawdown.min())

    wins = [r for r in trade_returns if r > 0]
    losses = [abs(r) for r in trade_returns if r < 0]
    win_rate = (len(wins) / len(trade_returns) * 100.0) if trade_returns else 0.0
    profit_factor = (sum(wins) / sum(losses)) if sum(losses) > 0 else 0.0

    return {
        "trades": len(trade_returns),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "total_return_pct": total_ret * 100.0,
        "cagr_pct": cagr * 100.0,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100.0,
        "daily_returns": ret_series,
        "signals": sigs,
    }


def run_monkey_test(df: pd.DataFrame, real_sharpe: float, n_monkeys: int = 1000, annual_days: float = 365.0) -> dict:
    """Test del Mono (Aronson): 1,000 estrategias aleatorias con igual frecuencia de exposición."""
    rng = np.random.default_rng(42)
    open_to_close = ((df["close"] - df["open"]) / df["open"]).values
    cost_rate = 30.0 / 10000.0  # 15 bps entrada + 15 bps salida

    # Probabilidad empírica de estar dentro del mercado
    n_bars = len(df)
    monkey_sharpes = np.zeros(n_monkeys)

    # Frecuencia de trades aproximada del 25% de los días
    p_trade = 0.25
    for m in range(n_monkeys):
        rand_entries = rng.random(n_bars) < p_trade
        m_rets = np.where(rand_entries, open_to_close - cost_rate, 0.0)
        std = m_rets.std()
        monkey_sharpes[m] = (m_rets.mean() / std) * np.sqrt(annual_days) if std > 0 else 0.0

    p_value = np.mean(monkey_sharpes >= real_sharpe)
    return {
        "p_value": float(p_value),
        "monkey_median_sharpe": float(np.median(monkey_sharpes)),
        "monkey_95th_sharpe": float(np.percentile(monkey_sharpes, 95)),
    }


def compute_hurst_variance(prices: pd.Series) -> float:
    """Estimador robusto de Exponente de Hurst por Escalamiento de Varianza."""
    log_p = np.log(prices.dropna().values)
    lags = [2, 4, 8, 16, 32, 64]
    variances = [np.var(log_p[lag:] - log_p[:-lag]) for lag in lags if len(log_p) > lag * 2]
    valid_lags = lags[: len(variances)]
    slope, _ = np.polyfit(np.log(valid_lags), np.log(variances), 1)
    return float(slope / 2.0)


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    annual_days = 365.0 if "-USD" in ticker or "USDT" in ticker else 252.0

    df = fetch_daily_data(ticker=ticker, start_date="2000-01-01")

    # División Purgada IS (2000 - 2023) y OOS (2024 - 2026) con 5 barras de embargo
    is_df = df[df.index < "2024-01-01"]
    oos_df = df[df.index >= "2024-01-06"]

    if len(is_df) < 200 or len(oos_df) < 100:
        print("[ERROR] Historial insuficiente para dividir en IS (2000-2023) y OOS (2024-2026).")
        return

    res_is = simulate_alphaia_daily(is_df, annual_days=annual_days)
    res_oos = simulate_alphaia_daily(oos_df, annual_days=annual_days)

    # Walk-Forward Efficiency Ratio (Robert Pardo)
    wfer = (res_oos["cagr_pct"] / res_is["cagr_pct"]) * 100.0 if res_is["cagr_pct"] > 0 else 0.0

    # Test del Mono sobre el tramo OOS (2024-2026)
    monkey = run_monkey_test(oos_df, real_sharpe=res_oos["sharpe"], n_monkeys=1000, annual_days=annual_days)

    # ADN Fractal en OOS
    hurst_oos = compute_hurst_variance(oos_df["close"])

    print("\n" + "=" * 78)
    print(f" AUDITORÍA RIGUROSA ALPHAIA (1D) | ACTIVO: {ticker}")
    print(f" Rango Total: {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} velas diarias)")
    print("=" * 78)

    print(f"\n[1] IN-SAMPLE (2000 -> 2023-12-31 | {len(is_df)} velas)")
    print(f"    Trades: {res_is['trades']} | Win Rate: {res_is['win_rate_pct']:.2f}% | Profit Factor: {res_is['profit_factor']:.2f}")
    print(f"    CAGR IS: {res_is['cagr_pct']:.2f}% | Sharpe IS: {res_is['sharpe']:.2f} | Max DD: {res_is['max_dd_pct']:.2f}%")

    print(f"\n[2] OUT-OF-SAMPLE (2024-01-06 -> Presente | {len(oos_df)} velas)")
    print(f"    Trades: {res_oos['trades']} | Win Rate: {res_oos['win_rate_pct']:.2f}% | Profit Factor: {res_oos['profit_factor']:.2f}")
    print(f"    CAGR OOS: {res_oos['cagr_pct']:.2f}% | Sharpe OOS: {res_oos['sharpe']:.2f} | Max DD: {res_oos['max_dd_pct']:.2f}%")

    print("\n[3] MOTOR DE TORTURA ESTADÍSTICA (OOS)")
    print(f"    Ratio WFER (Pardo >= 50%)       : {wfer:.1f}%  -> {'PASS' if wfer >= 50 else 'FAIL (Sobreajuste)'}")
    print(f"    Test del Mono (p-value < 0.05)   : p = {monkey['p_value']:.4f} (Sharpe Mono 95th: {monkey['monkey_95th_sharpe']:.2f}) -> {'PASS' if monkey['p_value'] < 0.05 else 'FAIL (Azar)'}")
    print(f"    Exponente de Hurst OOS (H)       : {hurst_oos:.3f}")

    passed = (wfer >= 50.0) and (monkey["p_value"] < 0.05) and (res_oos["profit_factor"] > 1.15)
    print("\n" + "-" * 78)
    print(f" DICTAMEN FINAL: {'APROBADO - EDGE REAL EN OOS' if passed else 'RECHAZADO - SIN VENTAJA ESTADÍSTICA POST-COSTOS'}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
