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

    if df_trades.empty:
        return df_trades, df_equity, {}

    ganadores = df_trades[df_trades['PnL_USD'] > 0]
    perdedores = df_trades[df_trades['PnL_USD'] <= 0]
    gross_profit = ganadores['PnL_USD'].sum()
    gross_loss = abs(perdedores['PnL_USD'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    payoff = (ganadores['PnL_USD'].mean() / abs(perdedores['PnL_USD'].mean())) if len(perdedores) > 0 and len(ganadores) > 0 else 0.0

    ret_total = ((capital / capital_asignado) - 1) * 100
    win_rate = (len(ganadores) / len(df_trades)) * 100
    rolling_max = df_equity['Equity'].cummax()
    drawdown = ((df_equity['Equity'] - rolling_max) / rolling_max) * 100
    max_dd = abs(drawdown.min())
    ret_diarios = df_equity['Equity'].pct_change().dropna()
    sharpe = (ret_diarios.mean() / ret_diarios.std()) * np.sqrt(252) if ret_diarios.std() > 0 else 0.0

    resumen = {
        'Rendimiento Neto (%)': f"{ret_total:+.2f}%",
        'Capital Final ($)': f"${capital:,.2f} USD",
        'Trades Totales': f"{len(df_trades)} operaciones",
        'Tasa de Aciertos (Win Rate)': f"{win_rate:.2f}%",
        'Profit Factor': f"{profit_factor:.2f}x",
        'Payoff Ratio Promedio': f"{payoff:.2f}x",
        'Sharpe Ratio Anualizado': f"{sharpe:.2f}",
        'Drawdown Máximo': f"{max_dd:.2f}%"
    }
    return df_trades, df_equity, resumen
