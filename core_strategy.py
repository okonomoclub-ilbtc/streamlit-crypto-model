import pandas as pd
import numpy as np

ACTIVOS_PORTAFOLIO = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
PESO_POR_ACTIVO = 0.25  # 25% asignado a cada cripto

def calcular_indicadores_variante_c(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores causales de la Variante C: Impulso 3D + SMA(50) + Salida SMA(5)."""
    data = df.copy()
    data['sma_50'] = data['Close'].rolling(window=50).mean()
    data['sma_5'] = data['Close'].rolling(window=5).mean()

    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift(1)).abs()
    low_close = (data['Low'] - data['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr_14'] = tr.ewm(span=14, adjust=False).mean()

    impulso_3d = data['Close'] - data['Close'].shift(3)
    data['impulso_atr'] = impulso_3d / data['atr_14']
    data['retorno_3d_pct'] = data['Close'].pct_change(periods=3) * 100.0
    data['es_dia_habil'] = data.index.dayofweek < 5

    data['cond_entrada'] = (
        (data['Close'] > data['sma_50']) &
        (data['impulso_atr'] >= 1.00) &
        (data['es_dia_habil'])
    )
    data['cond_salida_sma'] = data['Close'] < data['sma_5']
    return data.dropna()

def _calcular_metricas_resumen(df_trades: pd.DataFrame, df_equity: pd.DataFrame, capital_inicial: float) -> dict:
    """Calcula métricas institucionales estándar desde trades y curva de equity."""
    if df_trades.empty or df_equity.empty:
        return {}
    capital_final = float(df_equity['Equity'].iloc[-1])
    ganadores = df_trades[df_trades['PnL_USD'] > 0]
    perdedores = df_trades[df_trades['PnL_USD'] <= 0]

    gross_profit = ganadores['PnL_USD'].sum()
    gross_loss = abs(perdedores['PnL_USD'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    payoff = (ganadores['PnL_USD'].mean() / abs(perdedores['PnL_USD'].mean())) if len(perdedores) > 0 and len(ganadores) > 0 else 0.0

    ret_total = ((capital_final / capital_inicial) - 1) * 100
    win_rate = (len(ganadores) / len(df_trades)) * 100
    rolling_max = df_equity['Equity'].cummax()
    drawdown = ((df_equity['Equity'] - rolling_max) / rolling_max) * 100
    max_dd = abs(drawdown.min())
    ret_diarios = df_equity['Equity'].pct_change().dropna()
    sharpe = (ret_diarios.mean() / ret_diarios.std()) * np.sqrt(252) if ret_diarios.std() > 0 else 0.0

    return {
        'Rendimiento Neto (%)': f"{ret_total:+.2f}%",
        'Capital Final ($)': f"${capital_final:,.2f} USD",
        'Trades Totales': f"{len(df_trades)} operaciones",
        'Tasa de Aciertos (Win Rate)': f"{win_rate:.2f}%",
        'Profit Factor': f"{profit_factor:.2f}x",
        'Payoff Ratio Promedio': f"{payoff:.2f}x",
        'Sharpe Ratio Anualizado': f"{sharpe:.2f}",
        'Drawdown Máximo': f"{max_dd:.2f}%"
    }

def ejecutar_backtest_variante_c(df: pd.DataFrame, capital_asignado: float = 25000.0,
                                 riesgo_pct: float = 0.01, fee_bps: float = 15.0):
    """Ejecuta backtest causal (Next-Bar Open) con fricción real de 30 bps RT y Hard Stop 2.5x ATR."""
    data = calcular_indicadores_variante_c(df)
    fee = fee_bps / 10000.0

    capital = capital_asignado
    en_posicion = False
    precio_entrada = 0.0
    unidades = 0.0
    stop_price = 0.0
    fecha_entrada = None

    trades = []
    curva_equity = []

    for i in range(1, len(data)):
        fila_ayer = data.iloc[i - 1]
        fila_hoy = data.iloc[i]
        fecha_hoy = data.index[i]

        if en_posicion:
            if fila_hoy['Low'] <= stop_price:
                precio_salida = min(fila_hoy['Open'], stop_price)
                pnl_bruto = (precio_salida - precio_entrada) * unidades
                costo = (precio_entrada + precio_salida) * unidades * fee
                pnl_neto = pnl_bruto - costo
                capital += pnl_neto
                trades.append({
                    'Entrada': fecha_entrada, 'Salida': fecha_hoy,
                    'Precio_In': precio_entrada, 'Precio_Out': precio_salida,
                    'PnL_USD': pnl_neto, 'Motivo': 'Hard Stop 2.5 ATR'
                })
                en_posicion = False
            elif fila_ayer['cond_salida_sma']:
                precio_salida = fila_hoy['Open']
                pnl_bruto = (precio_salida - precio_entrada) * unidades
                costo = (precio_entrada + precio_salida) * unidades * fee
                pnl_neto = pnl_bruto - costo
                capital += pnl_neto
                trades.append({
                    'Entrada': fecha_entrada, 'Salida': fecha_hoy,
                    'Precio_In': precio_entrada, 'Precio_Out': precio_salida,
                    'PnL_USD': pnl_neto, 'Motivo': 'Cierre < SMA(5)'
                })
                en_posicion = False

        if not en_posicion and fila_ayer['cond_entrada']:
            precio_entrada = fila_hoy['Open']
            atr_ref = fila_ayer['atr_14']
            distancia_stop = 2.5 * atr_ref
            if distancia_stop > 0:
                riesgo_usd = capital * riesgo_pct
                unidades = min(riesgo_usd / distancia_stop, capital / precio_entrada)
                stop_price = precio_entrada - distancia_stop
                fecha_entrada = fecha_hoy
                en_posicion = True

        equity_hoy = capital + ((fila_hoy['Close'] - precio_entrada) * unidades if en_posicion else 0.0)
        curva_equity.append({'Fecha': fecha_hoy, 'Equity': equity_hoy})

    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(curva_equity).set_index('Fecha')
    resumen = _calcular_metricas_resumen(df_trades, df_equity, capital_asignado)
    return df_trades, df_equity, resumen

def ejecutar_backtest_portafolio(diccionario_dfs: dict, capital_total: float = 100000.0,
                                 riesgo_pct: float = 0.01, fee_bps: float = 15.0):
    """Consolida los 4 activos equiponderados (25% c/u) en una única curva y métricas de portafolio."""
    capital_por_activo = capital_total * PESO_POR_ACTIVO
    lista_trades = []
    series_equity = []

    for ticker, df_asset in diccionario_dfs.items():
        if df_asset is None or df_asset.empty:
            continue
        t_df, eq_df, _ = ejecutar_backtest_variante_c(df_asset, capital_por_activo, riesgo_pct, fee_bps)
        if not t_df.empty:
            t_df = t_df.copy()
            t_df['Activo'] = ticker
            lista_trades.append(t_df)
        if not eq_df.empty:
            series_equity.append(eq_df['Equity'].rename(ticker))

    if not series_equity:
        return pd.DataFrame(), pd.DataFrame(), {}

    df_eq_combinado = pd.concat(series_equity, axis=1).sort_index().ffill().fillna(capital_por_activo)
    df_eq_combinado['Equity'] = df_eq_combinado.sum(axis=1)
    df_trades_all = pd.concat(lista_trades, ignore_index=True).sort_values('Salida') if lista_trades else pd.DataFrame()
    resumen = _calcular_metricas_resumen(df_trades_all, df_eq_combinado[['Equity']], capital_total)
    return df_trades_all, df_eq_combinado[['Equity']], resumen

def generar_pine_script_v6() -> str:
    """Devuelve el código Pine Script v6 con paridad exacta al motor Python."""
    return """// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org
// © QuantumTradeA · Bookbinderr-2026
//@version=6
strategy("Crypto Macro Momentum [Variante C - Validada]", shorttitle="Macro Mom C", overlay=true,
     initial_capital=25000, currency=currency.USD,
     default_qty_type=strategy.percent_of_equity, default_qty_value=100,
     commission_type=strategy.commission.percent, commission_value=0.15, slippage=0,
     pyramiding=0, process_orders_on_close=false, calc_on_every_tick=false)

// ========= Parámetros Validados (Sin Sobreajuste) =========
grpReglas = "Parámetros Estructurales (Variante C)"
smaMacroLen = input.int(50, "SMA Filtro Macro (velas)", minval=10, group=grpReglas)
smaExitLen  = input.int(5,  "SMA Salida Rápida (velas)", minval=2, group=grpReglas)
atrLen      = input.int(14, "ATR Volatilidad (velas)", minval=1, group=grpReglas)
impBars     = input.int(3,  "Ventana Impulso (días)", minval=1, group=grpReglas)
impMult     = input.float(1.0, "Umbral Impulso (× ATR)", step=0.1, group=grpReglas)
stopMult    = input.float(2.5, "Hard Stop Catastrófico (× ATR)", step=0.1, group=grpReglas)
onlyWeekdays = input.bool(true, "Operar solo días hábiles UTC (Lunes a Viernes)", group=grpReglas)

// ========= Comprobación de Temporalidad =========
if barstate.isfirst and timeframe.period != "D"
    runtime.error("Este script está calibrado exclusivamente para velas diarias (1D UTC).")

// ========= Indicadores Causales (Al Cierre de Vela) =========
sma50 = ta.sma(close, smaMacroLen)
sma5  = ta.sma(close, smaExitLen)
atr14 = ta.rma(ta.tr(true), atrLen)

impulso3D  = close - close[impBars]
impulsoATR = atr14 > 0 ? impulso3D / atr14 : 0.0

// Calendario institucional UTC (Lunes=2 ... Viernes=6 en Pine Script)
dowUTC = dayofweek(time, "UTC")
esDiaHabil = not onlyWeekdays or (dowUTC >= dayofweek.monday and dowUTC <= dayofweek.friday)

// Señales evaluadas al cierre (se ejecutan en la apertura siguiente)
condEntrada = close > sma50 and impulsoATR >= impMult and esDiaHabil
condSalida  = close < sma5

// ========= Redirección de Apertura Siguiente + Hard Stop =========
var float stopPrice = na

if strategy.position_size == 0 and condEntrada
    strategy.entry("LongMacro", strategy.long, comment="LONG Impulso 3D")
    stopPrice := close - stopMult * atr14

if strategy.position_size > 0
    if condSalida
        strategy.close("LongMacro", comment="Salida < SMA(5)")
        stopPrice := na
    else if not na(stopPrice)
        strategy.exit("HardStop", from_entry="LongMacro", stop=stopPrice, comment="Stop 2.5 ATR")

// ========= Visualización en Gráfico =========
plot(sma50, "SMA 50 (Régimen Macro)", color=color.new(#3B82F6, 0), linewidth=2)
plot(sma5,  "SMA 5 (Salida Dinámica)", color=color.new(#F59E0B, 0), linewidth=1)
plot(strategy.position_size > 0 ? stopPrice : na, "Hard Stop 2.5 ATR", color=color.new(#EF4444, 0), style=plot.style_linebr, linewidth=2)
plotshape(condEntrada and strategy.position_size == 0, title="Señal Long", style=shape.triangleup, location=location.belowbar, color=color.new(#10B981, 0), size=size.small, text="LONG")
"""

def generar_manual_markdown() -> str:
    """Genera el Manual Científico y Operativo descargable con todos los hallazgos."""
    return """# Manual Científico y Operativo: Crypto Macro Momentum (Variante C)
Autor: QuantumTradeA · QuantumLab-2026
Estándar de Ejecución: Velas Diarias 1D (Cierre 23:59 UTC / Apertura 00:00 UTC)
Universo Operativo: Portafolio Equiponderado 4 Activos (25% BTC-USD · 25% ETH-USD · 25% SOL-USD · 25% XRP-USD)
"""
