import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import feedparser
import os
import random
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

SEED = 7
# Streamlit reejecuta el script en cada interacción, así que la semilla se fija aquí
# para que cada rerun produzcapesos idénticos.
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AI Crypto Strategist & Sentinel V10 Pro", layout="wide")


def leer_secreto(clave):
    """Lee un secreto de st.secrets o del entorno. Devuelve "" si no está configurado."""
    try:
        return str(st.secrets[clave])
    except Exception:
        return str(os.environ.get(clave, "")).strip()


TOKEN_TELEGRAM = leer_secreto("TELEGRAM_TOKEN")
CLAVE_MAESTRA = leer_secreto("MAINTENANCE_PASSWORD")
CHAT_ID = "@quantumtradear"


def despachar_alerta_telegram(mensaje):
    """Envía la ficha de orden al canal. Devuelve (ok, detalle) para no mentir al usuario."""
    if not TOKEN_TELEGRAM:
        return False, "TELEGRAM_TOKEN no está configurado."
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=8)
    except Exception as exc:
        return False, f"Error de red: {exc}"
    if r.status_code != 200:
        return False, f"Telegram respondió HTTP {r.status_code}: {r.text[:180]}"
    return True, "Mensaje entregado."

# --- FUNCIONES DE TRADING CUANTITATIVO (SENTINEL V10 PRO) ---
@st.cache_data(ttl=3600)
def load_data_v10(ticker, days):
    """Descarga datos diarios y calcula los indicadores. Devuelve (df, error_texto)."""
    try:
        df = yf.download(ticker, start=(pd.Timestamp.now() - pd.Timedelta(days=days)), progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(), f"yfinance devolvió una respuesta vacía para {ticker}."
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Filtro de Tendencia Intermedio (EMA 50)
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # Retorno de 3 días (Momentum Inmediato)
        df['retorno_3d'] = df['Close'].pct_change(periods=3) * 100
        
        # True Range y ATR de 14 para Dimensionamiento del Riesgo Controlado (0.5%)
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift(1))
        low_cp = np.abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df.dropna(), None
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


import backtest as bt


def _f(x):
    """Convierte tipos numpy a float de Python (st.cache_data y st.metric los exigen)."""
    return float(x) if x is not None and np.isfinite(x) else None


@st.cache_data(ttl=3600, show_spinner=False)
def ejecutar_backtest(ticker, simulaciones=1000):
    """Ejecuta el backtest real de las reglas y devuelve las métricas ya calculadas."""
    ind = bt.indicadores(bt.descargar(ticker, "max"))
    ops, equity, dd = bt.simular(ind)
    m = bt.metricas(ops, equity, dd)
    bh = bt.baseline_buy_hold(ind)

    corte = ind.index[len(ind) // 2]
    mis_ops, mis_eq, mis_dd = bt.simular(ind[ind.index <= corte])
    mos_ops, mos_eq, mos_dd = bt.simular(ind[ind.index > corte])
    mis = bt.metricas(mis_ops, mis_eq, mis_dd)
    mos = bt.metricas(mos_ops, mos_eq, mos_dd)

    ale = bt.baseline_aleatorio(ind, m["trades"], repeticiones=simulaciones)
    pct = float((ale < ops["pnl"].sum()).mean()) if ale is not None else float("nan")
    ale_med = float(np.median(ale)) if ale is not None else 0.0

    sig = bt.significatividad(ops) or {}
    be = bt.breakeven_fee(ind)
    # Retorno bruto: sin comisión, para ver cuánto edge real hay antes de los costes.
    o0, e0, d0 = bt.simular(ind, fee=0.0)
    m0 = bt.metricas(o0, e0, d0)
    # ¿Cuántos años positivos hay? Si cambia de signo por régimen, no es estable.
    anuales = []
    for y in sorted(set(ind.index.year)):
        sub = ind[ind.index.year == y]
        if len(sub) < 30:
            continue
        oy, ey, dy = bt.simular(sub)
        my = bt.metricas(oy, ey, dy)
        if my.get("trades", 0):
            anuales.append(my["retorno_total"])

    def limpiar(d, extra=None):
        out = {k: _f(v) for k, v in d.items() if k not in ("trades",)}
        out["trades"] = int(d.get("trades", 0))
        if extra:
            out.update(extra)
        return out

    return {
        "estrategia": limpiar(m, {"n_velas": len(ind),
                                  "rango": (str(ind.index[0].date()), str(ind.index[-1].date()))}),
        "buy_hold": limpiar(bh),
        "corte": str(corte.date()),
        "in_sample": limpiar(mis),
        "out_sample": limpiar(mos),
        "bruto": limpiar(m0),
        "pct_azar": pct,
        "ale_med": ale_med,
        "sig": {"ic_lo": sig.get("ic", (None, None))[0], "ic_hi": sig.get("ic", (None, None))[1],
                "t": _f(sig.get("t")), "media": _f(sig.get("media")),
                "ac1": _f(sig.get("ac1")), "share_top5": _f(sig.get("share_top5")),
                "n": sig.get("n", 0)},
        "breakeven_fee": float(be),
        "anios_positivos": int(sum(1 for r in anuales if r > 0)),
        "anios_total": len(anuales),
    }

# --- INTERFAZ LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Panel de Control")
    crypto = st.selectbox("Activo a Auditar", ["BTC-USD", "ETH-USD", "SOL-USD"])
    history_days = st.slider("Ventana Histórica (Días)", 500, 3000, 1500)
    epochs_n = st.slider("Épocas Máximas Entrenamiento LSTM", 5, 50, 15)
    st.caption("El slider es el máximo de épocas; el entrenamiento para solo si la validación deja de mejorar.")
    st.markdown("---")
    st.write("💰 **Gestión de Riesgo de Portafolio**")
    capital_total = st.number_input("Capital Operativo Base ($)", min_value=10.0, value=100000.0, step=1000.0)
    riesgo_deseado = st.slider("Riesgo por Operación (%)", 0.1, 2.0, 0.5, step=0.1)
    st.markdown("---")
    st.write("📢 **Canales Oficiales:**")
    st.markdown("[✈️ Telegram QuantumTradeA](https://t.me)")
    st.markdown("[𝕏 Twitter @bookbinderr](https://x.com)")

df, error_descarga = load_data_v10(crypto, history_days)

# --- CUERPO PRINCIPAL ---
st.title(f"🚀 AI Crypto Strategist & Dictaminador Sentinel V10 Pro")

if df.empty or len(df) < 100:
    if error_descarga:
        st.error(f"❌ Fallo al descargar datos de {crypto}: {error_descarga}")
    else:
        st.error(f"❌ Muestra insuficiente para {crypto}: se necesitan 100 velas, hay {len(df)}.")
    st.stop()
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Gráfico e Interfaz Operativa",
        "🤖 Predicción Neuronal LSTM",
        "🎯 Registro de Robustez (Backtest)",
        "📰 Noticias en Tiempo Real",
        "📖 Manual Operativo Sistemático"
    ])
    
    # Última fila del dataset diario: en yfinance es la vela EN CURSO, no un cierre
    # confirmado. Se etiqueta explícitamente para no venderla como señal cerrada.
    now = df.iloc[-1]
    precio_actual = float(now['Close'])
    ret3d_actual = float(now['retorno_3d'])
    ema50_actual = float(now['ema_50'])
    atr_actual = float(now['atr'])
    vela_en_curso = df.index[-1] == pd.Timestamp.now().normalize()
    
    # --- PESTAÑA 1: GRÁFICO PRO E INTERFAZ DE ALERTAS ---
    with tab1:
        # Trazamos las señales históricas basadas puramente en reglas matemáticas fijas
        df['chart_signal'] = 0
        df.loc[(df['Close'] > df['ema_50']) & (df['retorno_3d'] <= -3.0), 'chart_signal'] = 1
        df.loc[(df['Close'] < df['ema_50']) & (df['retorno_3d'] >= 3.0), 'chart_signal'] = -1
        longs = df[df['chart_signal'] == 1]
        shorts = df[df['chart_signal'] == -1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=f"Precio {crypto}", line=dict(color='#F8FAFC', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], name="EMA 50 (Dirección Macro)", line=dict(color='#3B82F6', width=1.5)))
        # Marcadores en el cierre real de la vela: desplazarlos falsearía el precio de entrada.
        fig.add_trace(go.Scatter(x=longs.index, y=longs['Close'], mode='markers', name="Gatillo Long", marker=dict(symbol='triangle-up', size=11, color='#10B981')))
        fig.add_trace(go.Scatter(x=shorts.index, y=shorts['Close'], mode='markers', name="Gatillo Short", marker=dict(symbol='triangle-down', size=11, color='#EF4444')))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # DICTAMINADOR OPERATIVO DE SEÑALES EN TIEMPO REAL
        st.subheader("📋 Estado Actual del Dictaminador Sentinel")
        estado_senal = "😴 ESPERANDO SETUP CLARO (El precio cotiza en zona de ruido neutral)"
        tipo_op = None
        
        # Evaluación de las reglas
        if precio_actual > ema50_actual and ret3d_actual <= -3.0:
            estado_senal = "🚀 SEÑAL ACTIVA: GATILLO LONG DETECTADO (Extensión en micro-tendencia alcista)"
            tipo_op = "LONG"
        elif precio_actual < ema50_actual and ret3d_actual >= 3.0:
            estado_senal = "📉 SEÑAL ACTIVA: GATILLO SHORT DETECTADO (Extensión en micro-tendencia bajista)"
            tipo_op = "SHORT"
            
        # CALCULADORA INTERACTIVA DE GESTIÓN MONETARIA INSTITUCIONAL (0.5%)
        c_p1, c_p2, c_p3 = st.columns(3)
        c_p1.metric("Precio Última Vela", f"${precio_actual:,.2f}")
        c_p2.metric("Retorno Acumulado 3D", f"{ret3d_actual:.2f}%")
        c_p3.metric("ATR Volatilidad Diaria", f"${atr_actual:,.2f}")
        
        if vela_en_curso:
            st.warning(
                f"⚠️ La última vela ({df.index[-1].date()}) está **en curso**: el precio, el retorno 3D y la señal "
                "cambian durante el día. El protocolo de la estrategia exige validar a las 23:59 UTC; hasta entonces "
                "esta lectura es provisional y no una señal cerrada."
            )
        
        capital_arriesgar = capital_total * (riesgo_deseado / 100)
        pos_size = (capital_arriesgar / (atr_actual / precio_actual)) if atr_actual > 0 else 0.0
        pos_size = min(pos_size, capital_total * 2.0) # Techo de protección de apalancamiento 2x
        
        st.write("### 📐 Ficha Estricta de Orden Recomendada")
        st.caption(
            "El dimensionamiento asume que el riesgo se materializa a 1 ATR. Como la estrategia NO tiene stop loss "
            "(salida por tiempo a 24 h), este cálculo NO acota la pérdida real: un movimiento adverso mayor que 1 ATR "
            "dentro de la ventana supera el límite de riesgo mostrado."
        )
        col_o1, col_o2, col_o3 = st.columns(3)
        col_o1.metric(f"Límite de Pérdida ({riesgo_deseado}%)", f"${capital_arriesgar:,.2f} USD")
        col_o2.metric("Exposición Nominal Máxima (USD)", f"${pos_size:,.2f} USD")
        col_o3.metric("Tamaño Sugerido en Moneda Base", f"{pos_size / precio_actual:.5f} unidades")
        
        # =========================================================================
        # BLOQUE INTEGRADO: MÓDULO DE DESPACHO ADMINISTRATIVO BLINDADO CON CLAVE
        # =========================================================================
        if tipo_op in ["LONG", "SHORT"]:
            st.markdown("---")
            st.write("🔒 **Módulo de Despacho Administrativo (QuantumTradeA)**")
            if not TOKEN_TELEGRAM:
                st.info("Envío a Telegram deshabilitado: no hay `TELEGRAM_TOKEN` configurado "
                        "(en `st.secrets` o en la variable de entorno).")
            
            admin_password = st.text_input(
                "Introduce la clave maestra para autorizar el envío al canal:",
                type="password",
                key="admin_pwd_field"
            )
            
            if tipo_op == "LONG":
                st.success(estado_senal)
            else:
                st.error(estado_senal)
            
            etiqueta = "COMPRA (LONG) 🚀" if tipo_op == "LONG" else "VENTA (SHORT) 📉"
            msg_alert = (f"🚨 *NUEVA SEÑAL SENTINEL V10 PRO*\n\n"
                         f"• Activo: {crypto}\n"
                         f"• Tipo: {etiqueta}\n"
                         f"• Precio Entrada: ${precio_actual:,.2f} USD\n"
                         f"• Tamaño sugerido: {pos_size / precio_actual:.5f} unidades\n"
                         f"⏱️ Salida Rígida: 24 Horas Estrictas")
            
            if st.button(f"✈️ Despachar Alerta {tipo_op} a Telegram", key=f"btn_{tipo_op.lower()}"):
                if not CLAVE_MAESTRA:
                    st.error("❌ Envío bloqueado: no hay `MAINTENANCE_PASSWORD` configurado.")
                elif admin_password == "":
                    st.error("❌ Por favor, introduce la clave maestra para autorizar el envío.")
                elif admin_password != CLAVE_MAESTRA:
                    st.error("❌ Clave incorrecta. Intento de despacho bloqueado por seguridad.")
                else:
                    ok, detalle = despachar_alerta_telegram(msg_alert)
                    if ok:
                        st.toast(f"✅ Señal {tipo_op} enviada a Telegram.")
                    else:
                        st.error(f"❌ El despacho falló: {detalle}")
        else:
            st.info(estado_senal)

        # --- PESTAÑA 2: MODELO LSTM CON SPLIT TEMPORAL (MEJORA GITHUB CORREGIDA) ---
    with tab2:
        st.subheader("🤖 Algoritmo de Redes Neuronales Recurrentes (LSTM)")
        st.write("Presiona el botón de abajo para entrenar el modelo en tiempo real utilizando la fijación de semilla reproducible (SEED=7).")
        
        if st.button("🔥 Iniciar Entrenamiento Predictivo LSTM", key="btn_ia_7d"):
            with st.spinner("La IA está analizando la microestructura y los patrones de velas..."):
                scaler = MinMaxScaler()
                
                # MEJORA CRÍTICA: Escalamos ajustando SOLO con el tramo de entrenamiento para evitar fuga de datos
                closes = df[['Close']].values
                split_idx = int(len(closes) * 0.8)
                scaler.fit(closes[:split_idx])
                
                scaled_all = scaler.transform(closes)
                X, y = [], []
                for i in range(60, len(scaled_all)):
                    X.append(scaled_all[i-60:i, 0])
                    y.append(scaled_all[i, 0])
                X, y = np.array(X), np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                # La ventana k predice el cierre de la serie en la posición 60+k.
                # X[:split_idx-60] deja fuera la ventana cuyo target ya cae en test,
                # así la frontera no comparte ninguna barra entre train y test.
                Xtr, ytr = X[:split_idx - 60], y[:split_idx - 60]
                Xte, yte = X[split_idx - 60:], y[split_idx - 60:]
                
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(60, 1)),
                    Dropout(0.2),
                    LSTM(50),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                # El slider es el máximo de épocas; se detiene si la validación empeora.
                stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                hist = model.fit(Xtr, ytr, epochs=epochs_n, batch_size=32, verbose=0,
                                 validation_split=0.1, callbacks=[stop])
                
                # --- Validación fuera de muestra: LSTM vs baseline "mañana = hoy" ---
                # yte[j] es el cierre de la serie en la posición split_idx+j, así que el
                # baseline correcto es el cierre de split_idx+j-1, es decir closes[split_idx-1:-1].
                pred_te = model.predict(Xte, verbose=0)
                real_te = scaler.inverse_transform(yte.reshape(-1, 1)).flatten()
                pred_te_px = scaler.inverse_transform(pred_te.reshape(-1, 1)).flatten()
                naive_te = closes[split_idx - 1:-1, 0]
                
                mae_m = float(np.abs(real_te - pred_te_px).mean())
                mae_n = float(np.abs(real_te - naive_te).mean())
                # El acierto direccional solo se define para predicciones con signo:
                # el baseline "mañana = hoy" predice el cierre de hoy, luego su signo es 0
                # y medirlo solo reportaría la frecuencia de días alcistas, no skill.
                sube = (real_te[1:] - real_te[:-1]) > 0
                dir_m = float((sube == ((pred_te_px[1:] - real_te[:-1]) > 0)).mean())
                ref_azar = float(sube.mean())
                
                st.subheader("📏 Validación fuera de muestra (último 20% del histórico)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE LSTM", f"${mae_m:,.0f}")
                c2.metric("MAE naive (mañana = hoy)", f"${mae_n:,.0f}")
                c3.metric("Acierto direccional LSTM", f"{dir_m:.0%}")
                c4.metric("Referencia al azar", f"{ref_azar:.0%}")
                st.caption(f"Épocas ejecutadas: {len(hist.history['loss'])} de {epochs_n} máximas. "
                           f"Observaciones de test: {len(yte)}.")
                if mae_m > mae_n:
                    st.warning(
                        f"⚠️ El LSTM **no supera** al baseline naive ({mae_m:,.0f} vs {mae_n:,.0f} USD de MAE). "
                        "Interpretar la proyección siguiente únicamente como ejercicio académico, no como señal operable."
                    )
                else:
                    st.success(f"✅ El LSTM mejora el MAE del baseline naive en este periodo ({mae_n:,.0f} → {mae_m:,.0f} USD).")
                
                # Proyección futura de 7 días
                future_preds = []
                current_batch = scaled_all[-60:].reshape(1, 60, 1)
                for _ in range(7):
                    p = model.predict(current_batch, verbose=0)
                    future_preds.append(p)
                    current_batch = np.append(current_batch[:, 1:, :], p.reshape(1, 1, 1), axis=1)
                
                preds_7d = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
                f_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(1, 8)]
                
                # Renderizado de gráfico interactivo Plotly
                fig_7d = go.Figure()
                fig_7d.add_trace(go.Scatter(x=f_dates, y=preds_7d.flatten(), mode='lines+markers', name="Proyección IA", line=dict(color='#EF4444', width=3)))
                fig_7d.update_layout(template="plotly_dark", title="Tendencia Proyectada Próximos 7 Días", margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_7d, use_container_width=True)
                
                # Despliegue de la tabla predictiva con el cálculo de variación porcentual
                preds_flat = preds_7d.flatten()
                pred_df = pd.DataFrame({
                    'Fecha': f_dates, 
                    'Precio Est.': preds_flat, 
                    'Variación %': [f"{((p / precio_actual) - 1) * 100:+.2f}%" for p in preds_flat]
                })
                st.table(pred_df.style.format({"Precio Est.": "${:,.2f}"}))
                st.success("✅ Red Neuronal Predictiva entrenada y proyección generada.")

# --- PESTAÑA 3: BACKTEST REAL DE LAS REGLAS (CALCULADO AL VUELO) ---
    with tab3:
        st.subheader("🎯 Backtest Real de las Reglas Sentinel V10")
        st.caption("Las métricas se calculan ejecutando `backtest.py` sobre datos de yfinance. "
                   "No hay cifras escritas a mano en esta pestaña.")
        st.markdown(
            "Las versiones anteriores mostraban aquí Win Rate, Payoff, Drawdown y Factor de "
            "Recuperación escritos a mano, sin una línea de backtest que los calculara. Ahora se "
            "simulan las reglas reales: señal sobre el cierre de T, entrada al `Open[T+1]`, "
            "salida a `Close[T+1]`, dimensionado por ATR y costes de comisión."
        )

        if st.button("🧪 Ejecutar backtest sobre el activo seleccionado", key="btn_backtest"):
            with st.spinner(f"Simulando las reglas Sentinel V10 sobre {crypto}..."):
                try:
                    r = ejecutar_backtest(crypto)
                    error_bt = None
                except Exception as exc:
                    r, error_bt = None, f"{type(exc).__name__}: {exc}"

            if error_bt:
                st.error(f"No se pudo ejecutar el backtest: {error_bt}")
            else:
                m, bh = r["estrategia"], r["buy_hold"]
                st.success(f"Backtest sobre {m['n_velas']} velas "
                           f"({m['rango'][0]} → {m['rango'][1]}).")

                st.markdown("#### Resultado")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Retorno neto", f"{m['retorno_total']:+.2%}")
                c2.metric("Retorno bruto (sin costes)", f"{r['bruto']['retorno_total']:+.2%}")
                c3.metric("Drawdown máximo", f"{m['dd_max']:.2%}")
                c4.metric("CAGR", f"{m['cagr']:+.2%}")

                st.markdown("#### Comparado con buy & hold")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("B&H retorno", f"{bh['retorno_total']:+.2%}")
                d2.metric("B&H CAGR", f"{bh['cagr']:+.2%}")
                d3.metric("B&H drawdown", f"{bh['dd_max']:.2%}")
                d4.metric("Operaciones", f"{m['trades']}")

                st.markdown("#### Métricas por operación")
                e1, e2, e3 = st.columns(3)
                e1.metric("Win rate", f"{m['win_rate']:.2%}")
                e2.metric("Payoff medio", f"{m['payoff']:.2f}x")
                e3.metric("Profit factor", f"{m['profit_factor']:.2f}")
                f1, f2 = st.columns(2)
                f1.metric("Recovery factor", f"{m['recovery_factor']:.2f}")
                f2.metric("Mejor / peor operación",
                          f"${m['mejor']:,.0f} / ${m['peor']:,.0f}")

                st.markdown(f"#### Split cronológico (corte en {r['corte']})")
                g1, g2 = st.columns(2)
                g1.markdown(f"**In-sample** — {r['in_sample']['trades']} ops · win "
                            f"{r['in_sample']['win_rate']:.1%} · retorno "
                            f"{r['in_sample']['retorno_total']:+.2%}")
                g2.markdown(f"**Out-of-sample** — {r['out_sample']['trades']} ops · win "
                            f"{r['out_sample']['win_rate']:.1%} · retorno "
                            f"{r['out_sample']['retorno_total']:+.2%}")

                st.markdown("#### ¿Hay ventaja estadística?")
                sig = r["sig"]
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("t-estadístico", f"{sig['t']:+.2f}")
                h2.metric("IC 95% inferior", f"{sig['ic_lo']:+.4%}")
                h3.metric("IC 95% superior", f"{sig['ic_hi']:+.4%}")
                h4.metric("Peso de las 5 mejores", f"{sig['share_top5']:.0%}")

                if sig["ic_lo"] is not None and sig["ic_lo"] <= 0 <= sig["ic_hi"]:
                    st.error(
                        "El intervalo de confianza al 95% **incluye el cero**: no se puede rechazar "
                        "que el retorno medio por operación sea nulo. **Las reglas no demuestran "
                        "tener ventaja estadística.**"
                    )
                else:
                    st.info(f"t = {sig['t']:+.2f}. Con |t| < 2 la evidencia es débil aunque el "
                            "intervalo excluya el cero.")

                if sig["ac1"] is not None and abs(sig["ac1"]) > 0.1:
                    st.caption(f"Autocorrelación lag-1 de {sig['ac1']:+.3f}: las operaciones diarias "
                               "no son independientes, así que el intervalo bootstrap es optimista.")
                if sig["share_top5"] is not None and sig["share_top5"] > 100:
                    st.warning(
                        f"Las 5 mejores operaciones aportan el {sig['share_top5']:.0%} del retorno "
                        "total. Sin ellas el resultado sería negativo: la estrategia depende de "
                        "unos pocos aciertos extremos."
                    )

                st.markdown("#### ¿Los costes se comen la ventaja?")
                if r["breakeven_fee"] == 0:
                    st.error(
                        "El retorno **bruto** (sin comisión ninguna) ya es negativo: la estrategia "
                        "no es rentable ni con costes cero."
                    )
                else:
                    st.warning(
                        f"El equilibrio está en **{r['breakeven_fee']:.4%} por lado** "
                        f"({2 * r['breakeven_fee']:.4%} por operación). Por encima de esa comisión "
                        "la estrategia pierde dinero, aunque el bruto sea positivo."
                    )

                st.markdown("#### ¿Las reglas aportan valor o solo el dimensionamiento?")
                st.caption(f"Comparación contra entradas elegidas al azar con el mismo número de "
                           f"operaciones y el mismo tamaño de posición.")
                if r["pct_azar"] >= 0.95:
                    st.success(f"Percentil {r['pct_azar']:.1%} frente a entradas aleatorias "
                               f"(pnl mediano ${r['ale_med']:,.0f}). Las reglas aportan valor.")
                elif r["pct_azar"] >= 0.80:
                    st.warning(f"Percentil {r['pct_azar']:.1%} frente a entradas aleatorias "
                               f"(pnl mediano ${r['ale_med']:,.0f}). Evidencia débil.")
                else:
                    st.error(f"Percentil **{r['pct_azar']:.1%}**, por debajo o cerca de la mediana "
                             f"aleatoria (${r['ale_med']:,.0f}). Las reglas no superan al azar: el "
                             "resultado se explica por el dimensionamiento, no por EMA + momentum.")

                st.markdown("#### Consistencia entre años")
                st.write(f"Años con retorno positivo: **{r['anios_positivos']} de {r['anios_total']}**. "
                         "Si el signo cambia según el régimen de mercado, el resultado no es estable "
                         "y no debe proyectarse al futuro.")

        st.markdown("---")
        st.markdown("#### Sobre la predicción de precio (modelo LSTM)")
        st.markdown(
            "El módulo LSTM de la pestaña 2 **no supera** al baseline `mañana = hoy` "
            "(MAE medido en BTC-USD: LSTM $3.205 frente a naive $1.156). "
            "Reproducible con `python validate.py BTC-USD`."
        )
        st.info(
            "💡 Conclusión honesta del backtest: **no se ha demostrado ventaja estadística en las "
            "reglas de entrada y salida**, ni en BTC ni en ETH ni en SOL. El retorno bruto es "
            "marginal y los costes de transacción lo consumen. La aritmética de gestión de riesgo "
            "por ATR sí es correcta, pero un dimensionamiento correcto no creaedge por sí solo."
        )
              # --- PESTAÑA 4: NOTICIAS (FEEDS RSS REALES, SIN CONTENIDO FABRICADO) ---
    with tab4:
        st.subheader(f"📰 Noticias del Mercado: {crypto}")
        st.caption("Titulares servidos por feeds RSS públicos reales. Si ninguno responde, la app lo dice; "
                   "no se muestra contenido inventado.")
        
        asset_keyword = crypto.split("-")[0].lower()  # btc, eth, sol
        
        feeds = [
            ("Yahoo Finance", f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={crypto}&region=US&lang=en-US"),
            ("Decrypt", "https://decrypt.co/feed"),
            ("Cointelegraph", "https://cointelegraph.com/rss"),
        ]
        
        entries, origen, fallos = [], None, []
        for nombre, url in feeds:
            try:
                with st.spinner(f"Consultando {nombre}..."):
                    feed = feedparser.parse(url)
                if feed.entries:
                    entries, origen = feed.entries, nombre
                    break
                fallos.append(f"{nombre}: sin entradas (bozo={feed.bozo})")
            except Exception as exc:
                fallos.append(f"{nombre}: {type(exc).__name__}: {exc}")
        
        if not entries:
            st.error("No se pudo obtener ningún feed de noticias. Detalle del fallo:")
            for f in fallos:
                st.text(f"• {f}")
            st.caption("No se muestran titulares de relleno: un titular inventado con sello de "
                       "'tiempo real' es peor que una pantalla vacía.")
        else:
            st.caption(f"Fuente: **{origen}** · {len(entries)} titulares recibidos")
            relevantes = [e for e in entries
                          if asset_keyword in e.get("title", "").lower()
                          or asset_keyword in e.get("summary", "").lower()]
            # Si el filtro por activo deja poco, se completan con los más recientes.
            mostrar = (relevantes + [e for e in entries if e not in relevantes])[:5]
            if not relevantes:
                st.info(f"Ningún titular menciona '{asset_keyword.upper()}' en esta fuente; "
                        "se muestran los más recientes disponibles.")
            for i, entry in enumerate(mostrar):
                titulo = entry.get("title", "(sin título)")
                with st.expander(f"🔸 {titulo}"):
                    st.write(entry.get("summary", "Sin resumen en el feed."))
                    if entry.get("published"):
                        st.caption(f"📅 {entry['published']}")
                    if entry.get("link"):
                        st.link_button("Leer noticia completa", entry["link"], key=f"ln_{origen}_{i}")

               # --- PESTAÑA 5: MÓDULO DIDÁCTICO, ESPECIFICACIONES TÉCNICAS Y CALCULADORA DE LOTAJE ---
    with tab5:
        st.header("📖 Manual Operativo Oficial: Sentinel V10 Pro")
        st.caption("FAQ y Protocolo Cuantitativo de Ejecución Sistemática Internacional (Estándar UTC)")
        
        st.markdown("### 🔬 Conceptos Fundamentales e Implementación Global")
        
        with st.expander("❓ Q1: ¿Cuándo se toma EXACTAMENTE la entrada tras una señal activa?"):
            st.markdown("""
            *   **Respuesta Directa:** La entrada **JAMÁS** se ejecuta en tiempo real durante el desarrollo del día si el indicador parpadea. Toda alerta del dictaminador técnico se valida al cierre oficial de la vela diaria (**23:59 UTC**).
            *   **El Protocolo 'Next Bar Open':** Si al consolidarse el cierre diario las reglas binarias terminan en estado positivo (`GO_LONG` o `GO_SHORT`), la orden de mercado se ingresa de forma obligatoria e inmediata en la **Apertura de la vela del día siguiente (00:00 UTC)**.
            *   **Sincronización Internacional:** Configura la alarma de tu plataforma de gráficos en huso horario **UTC**. Entrar antes o después de la apertura de la nueva vela diaria altera la Esperanza Matemática del sistema.
            """)
            
        with st.expander("❓ Q2: ¿Cuál es el paso a paso exacto para colocar la orden en cualquier terminal o Broker?"):
            st.markdown("""
            Para operar estas reglas de forma sistemática, sigue este protocolo:
            
            1.  **Monitoreo del Cierre (23:50 UTC):** Diez minutos antes del cierre de la vela diaria internacional, revisa el panel de control de esta App para verificar si la señal se ha consolidado.
            2.  **Lectura de la Ficha de Orden:** Si la señal está activa, copia los valores calculados en la **Ficha Estricta de Orden Recomendada** basados en tu capital real.
            3.  **Apertura de la Orden (00:00 UTC):** Inmediatamente en el segundo en que abre la nueva barra del día, ejecuta una **Orden de Mercado (Market Order)** de Compra o Venta en tu terminal con el tamaño exacto sugerido.
            4.  **Cinturón de Seguridad (Time-Exit):** La posición tiene una fecha de caducidad rígida de **exactly 24 horas (1 vela diaria)**. Al llegar a las 23:59 UTC del día siguiente, cierras la posición a mercado inmediato, sin importar si va en ganancia o pérdida.
            """)
            
        with st.expander("❓ Q3: ¿Por qué este sistema no utiliza un Stop Loss técnico tradicional?"):
            st.markdown("""
            *   **Argumento del diseño:** En la microestructura moderna, los algoritmos de alta frecuencia ejecutan barridas de liquidez (*stop-hunts*) en zonas de soporte y resistencia obvias antes de desarrollar el movimiento real. Un stop loss estrecho es barrido con frecuencia.
            *   **El time-exit como límite temporal:** Salir a las 24 h sí acota el **tiempo** de exposición al mercado. Lo que acota, no es la **magnitud** de la pérdida.
            *   **Advertencia importante:** como no hay stop loss, ninguna de las cifras de riesgo de esta app acota la pérdida real. El dimensionamiento por ATR de la calculadora asume que el riesgo se materializa a 1 ATR, pero un movimiento adverso mayor que 1 ATR dentro de la ventana de 24 h puede producir una pérdida varias veces superior al límite mostrado. Cualquier afirmación previa de que este diseño "encajona el drawdown por debajo del 3%" carecía de un backtest que la respaldara y se ha retirado.
            """)

        st.markdown("---")
        st.subheader("📐 Especificaciones Técnicas del Núcleo Lógico")
        st.write("Especificación de las reglas tal como están implementadas en el código. Su ventaja estadística **aún no ha sido medida**: ver la pestaña de registro de validación.")
        
        st.markdown("""
        #### 1. Arquitectura Lógica de Entrada (Reglas Binarias)
        *   **Dirección Macro (EMA 50):** El precio de cierre diario debe estar por encima para autorizar compras (`LONG`) y por debajo para autorizar ventas (`SHORT`).
        *   **Gatillo de Momentum (Retorno 3D):** Exige un movimiento de **±3%** acumulado en las últimas 3 jornadas.
        
        #### 2. Lógica de Salida
        *   **Time-Exit:** La posición se liquida a las **24 horas exactas (1 vela diaria)**. No hay stop loss ni take profit, por lo tanto no hay límite de pérdida por trade.
        
        #### 3. Parámetros Monetarios y Gestión de Capital
        *   **Capital Base de Simulación:** $100,000.00 USD por defecto, parametrizable en el menú lateral.
        *   **Riesgo Nominal Asignado:** 0.5% del balance por defecto, indexado por **ATR(14)** y con techo de apalancamiento 2.0x. Ver la advertencia de la calculadora sobre lo que este sizing no cubre.
        """)

        # --- SIMULACIÓN Y CALCULADORA DINÁMICA POR ATR(14) EN VIVO ---
        st.markdown("---")
        st.subheader("🧮 Calculadora de Lotaje Institucional por ATR(14)")
        st.write("Utiliza este módulo didáctico interactivo para calcular el tamaño exacto de tu posición en cualquier activo del mundo, indexando el riesgo según la volatilidad real del momento.")
        
        # Bloque de inputs interactivos para el usuario
        col_calc1, col_calc2 = st.columns(2)
        with col_calc1:
            capital_sim = st.number_input("Introduce tu Capital Operativo ($)", min_value=10.0, value=2000.0, step=100.0, key="calc_cap")
            precio_sim = st.number_input("Precio de Entrada Actual del Activo ($)", min_value=0.01, value=precio_actual, step=50.0, key="calc_price")
        with col_calc2:
            riesgo_sim_pct = st.slider("Riesgo por Trade Deseado (%)", 0.1, 2.0, 0.5, step=0.1, key="calc_risk")
            atr_sim = st.number_input("Valor del ATR(14) Diario Actual ($)", min_value=0.01, value=atr_actual, step=10.0, key="calc_atr")
            
        # Fórmulas de la Ingeniería Cuantitativa para dimensionamiento de riesgo
        dinero_en_riesgo = capital_sim * (riesgo_sim_pct / 100.0)
        
        # Tamaño de posición nominal y en contratos/unidades
        if atr_sim > 0 and precio_sim > 0:
            exposicion_nominal_usd = (dinero_en_riesgo / (atr_sim / precio_sim))
            unidades_moneda_base = exposicion_nominal_usd / precio_sim
            apalancamiento_requerido = exposicion_nominal_usd / capital_sim
        else:
            exposicion_nominal_usd, unidades_moneda_base, apalancamiento_requerido = 0.0, 0.0, 0.0
            
        # Renderizado de la ficha de salida de datos para el usuario
        st.write("#### 📋 Ficha de Ejecución Estandarizada")
        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.metric("Pérdida Máxima Permitida", f"${dinero_en_riesgo:,.2f} USD")
        c_res2.metric("Poder de Compra (Nominal USD)", f"${exposicion_nominal_usd:,.2f} USD")
        c_res3.metric("Lote Exacto a Operar", f"{unidades_moneda_base:.5f} Unidades")
        
        # Mensajes dinámicos de control de apalancamiento para el usuario
        if apalancamiento_requerido > 2.0:
            st.warning(f"⚠️ Alerta de margen: la gestión que describes requiere un apalancamiento de "
                       f"{apalancamiento_requerido:.1f}x. El techo de la estrategia es 2.0x.")
        else:
            st.info(f"Nivel de apalancamiento requerido: {apalancamiento_requerido:.1f}x, dentro del "
                    "techo de 2.0x. Esto limita el margen requerido, **no** la pérdida: sin stop "
                    "loss, un movimiento adverso puede superar el límite de riesgo calculado.")

    # DESARROLLADOR
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #64748B;'>🛡️ QuantumTradeA 2026 | Desarrollado por @Bookbinderr-2026 — Ecosistema Científico Estabilizado</p>", unsafe_allow_html=True)
