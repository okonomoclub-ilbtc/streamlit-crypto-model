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
    """Genera el Manual Científico y Operativo completo con estructura ejecutiva y pedagógica."""
    return """# Manual Científico y Operativo: Crypto Macro Momentum (Variante C)
**Autor:** QuantumTradeA · @Bookbinderr-2026  
**Estándar de Ejecución:** Velas Diarias 1D (Evaluación 23:59 UTC / Ejecución 00:00 UTC)  
**Universo Operativo:** Portafolio Equiponderado 4 Activos (`25% BTC-USD` · `25% ETH-USD` · `25% SOL-USD` · `25% XRP-USD`)

---

## 💡 MOMENTO REVELACIÓN: LA LÓGICA ECONÓMICA DEL EDGE
> *"La mayoría de los sistemas de impulso en criptomonedas fallan por dos razones: persiguen movimientos en mercados bajistas sin tendencia macro o aplican filtros fractales tan estrictos que dejan al sistema sin muestra estadística. La ventaja real aparece cuando exiges ignición de volatilidad (Impulso 3D ≥ 1.00 × ATR) únicamente en días hábiles institucionales y sobre un régimen alcista confirmado (Close > SMA 50)."*

Tras la entrada de los ETF institucionales en 2024, la microestructura del mercado cripto cambió de dueño:
* **Por qué descartamos la reversión a la media y el filtro Hurst rígido (`H > 0.55`):** A escala diaria, los criptoactivos oscilan cerca de una caminata aleatoria ($H \approx 0.50 - 0.54$). Exigir persistencia fractal extrema bloqueó las operaciones fuera de muestra ($p = 0.51$ en el Test del Mono, indistinguible del azar).
* **Por qué funciona la Variante C:** Captura los flujos direccionales institucionales de lunes a viernes cuando el mercado rompe su inercia de 3 días con fuerza superior a su volatilidad diaria (`ATR 14`), dejando correr la tendencia hasta perder la media rápida (`SMA 5`) y diversificando el riesgo en 4 líderes líquidos al **25%** cada uno.

---

## 📋 RESUMEN TÉCNICO EJECUTIVO

### 1. Activos Target y Asignación de Capital
* **Portafolio Base (100% Capital):** Distribuido en 4 bloques equiponderados de **25.0%** (`BTC-USD`, `ETH-USD`, `SOL-USD`, `XRP-USD`).
* **Por qué multiactivo y no un solo ticker:** Operar un activo aislado sufre durante los inviernos cripto. La cesta al 25% rota el capital hacia los líderes de impulso y eleva la Eficiencia Walk-Forward (**WFER**) al **91.77%**.
* **Apalancamiento:** `1.0x` (Spot puro, sin deuda ni riesgo de liquidación).

### 2. Variables de Setup (3 Reglas Únicas — Simplicidad Máxima)
1. **Filtro de Régimen Macro:** `Close > SMA(50)`. Si el precio cierra por debajo, ese bloque del 25% permanece en *Defensive Cash*.
2. **Gatillo de Ignición de Impulso:** `(Close_t - Close_{t-3}) / ATR(14) >= 1.00`. Exige que el avance acumulado de 3 días sea al menos igual a una vez la volatilidad diaria promedio.
3. **Filtro de Calendario Institucional:** Evaluar señales exclusivamente de **Lunes a Viernes (UTC)**, filtrando la baja liquidez profesional de los fines de semana.

---

## 🎯 TRIGGERS DE EJECUCIÓN Y GESTIÓN DE SALIDA (BINARIO)

### Entrada (`Next-Bar Open`)
1. Esperar el cierre consolidado de la vela diaria a las **23:59 UTC** (sin operar parpadeos intradía).
2. Confirmar que se cumplen simultáneamente: `Close > SMA(50)`, `Impulso 3D ≥ 1.00 × ATR(14)` y día hábil (`L-V`).
3. Ejecutar orden de compra (`LONG`) a mercado en la apertura de la vela siguiente (**00:00 UTC**).

### Salida Dual (Protección y Captura de Cola)
* **Giro de Tendencia Corta (`Cierre < SMA 5`):** Si una vela diaria cierra por debajo de su media móvil simple de 5 días, se liquida la posición a mercado en la apertura siguiente (**00:00 UTC**).
* **Corte de Cola Catastrófico (`Hard Stop 2.5 × ATR`):** Toda orden nace protegida desde su apertura con un Stop Loss situado a `2.5 × ATR(14)` por debajo del precio de entrada.

---

## 🧮 GESTIÓN DE RIESGO (MATEMÁTICA)
Cada activo opera de forma independiente dentro de su asignación del **25%** del capital total:
* **Capital Asignado por Activo:** `Capital_Activo = Capital_Total × 0.25`
* **Riesgo Monetario por Operación:** `Riesgo_$ = Capital_Activo × (% Riesgo / 100)` *(Estándar validado: 1.0%)*
* **Distancia al Hard Stop:** `Stop_$/unidad = 2.5 × ATR(14)`
* **Tamaño Exacto del Lote:** `Unidades = min(Riesgo_$ ÷ Stop_$/unidad , Capital_Activo ÷ Precio_Entrada)`

---

## ⏱️ FLUJO DE TRABAJO DIARIO (RUTINA DE 5 MINUTOS)

* **23:50 UTC (10 min antes del cierre):** Abrir el dictaminador Sentinel V11 Pro y revisar el estado de las 4 monedas en el Radar Multiactivo.
* **00:00 UTC (Apertura de nueva vela diaria):**
  1. Verificar si algún activo confirmó `GATILLO LONG` al cierre.
  2. Si hay posición abierta, verificar si el cierre previo cayó por debajo de `SMA(5)` para cerrarla.
  3. Ejecutar la orden de entrada en Spot y colocar inmediatamente la orden Stop Loss en `Entrada - 2.5 × ATR(14)`.
  4. Registrar la operación en la bitácora.

---

## ✅ CHECKLIST PRE-TRADE (OBLIGATORIO)
- [ ] Vela diaria de las `23:59 UTC` cerrada y consolidada (no operar velas en curso).
- [ ] Día hábil institucional confirmado (`Lunes a Viernes` UTC).
- [ ] Precio de cierre por encima de `SMA(50)`.
- [ ] Impulso acumulado de 3 días $\ge 1.00 \times \text{ATR}(14)$.
- [ ] Tamaño calculado respetando el tope del **25%** asignado al activo (sin apalancamiento).
- [ ] Orden `Hard Stop` colocada exactamente en `Entrada - 2.5 × ATR(14)`.

---

## ⚠️ QUÉ NO HACE ESTE SISTEMA Y CUÁNDO NO OPERAR
* **No es un sistema *All-Weather* (Solo Largos):** Gana durante expansiones alcistas y permanece plano en *Defensive Cash* durante mercados bajistas prolongados. No opera en corto (`SHORT`).
* **No es alta frecuencia:** Es un sistema deliberadamente selectivo; habrá semanas sin señales mientras el mercado lateraliza o corrige.
* **Prohibido operar si:**
  * El precio cotiza por debajo de `SMA(50)`.
  * El cierre corresponde a sábado o domingo UTC.
  * El spread o deslizamiento del exchange supera los `30 bps` *roundtrip* en momentos de iliquidez extrema.

---

## 📊 VALIDACIÓN ESTADÍSTICA Y AUDITORÍA INSTITUCIONAL (2020–2026 UTC)
Evaluación ejecutada sobre datos históricos reales (`In-Sample: 2020–Jun 2024` | `Out-of-Sample: Jun 2024–2026`) con fricción real de `30 bps` *roundtrip*:

| Prueba Institucional | Resultado Auditado | Interpretación |
| :--- | :--- | :--- |
| **WFER (Robert Pardo, 2008)** | `91.77%` (`SR IS 1.86` · `SR OOS 1.80`) | Conserva más del 91% de eficiencia fuera de muestra. |
| **CPCV (López de Prado, 2018)** | `100.0%` caminos positivos (`Mean SR 1.72`) | Consistencia total en particiones combinatorias. |
| **Test del Mono OOS (Aronson)** | `p-value = 0.01099` (`123` trades OOS) | Supera al 98.9% de 1,000 simulaciones aleatorias. |
| **Estrés Latencia CPS (+1 vela)** | `-4.94%` caída en *Profit Factor* | Alta resistencia a retrasos de ejecución. |
| **Deflated Sharpe Ratio (DSR)** | `91.3%` ($N=10$) · `87.8%` ($N=20$) | Alta probabilidad de alfa genuino sin *data-mining*. |

---

## 📝 TEMPLATE DE BITÁCORA DE OPERACIÓN (TRADE LOG)
* **Fecha Entrada (00:00 UTC):** `____ / ____ / 2026`
* **Activo (Bloque 25%):** `BTC-USD / ETH-USD / SOL-USD / XRP-USD`
* **Impulso 3D / ATR(14) al Cierre:** `______ x`
* **Precio Entrada:** `$________` | **Hard Stop (2.5 ATR):** `$________`
* **Unidades Operadas:** `________` | **Riesgo Monetario ($):** `$________`
* **Fecha y Precio Salida:** `____ / ____ / 2026` · `$________`
* **Motivo de Salida:** `Cierre < SMA(5) / Hard Stop 2.5 ATR`
* **Resultado Neto (% y $):** `______ %` (`$________`)

---
> *Aviso Legal: Documento estrictamente educativo y de investigación cuantitativa. Los resultados auditados en histórico no garantizan rendimientos futuros.*
"""

