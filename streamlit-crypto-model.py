# --- 1. INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS ---
import subprocess
import sys

def check_dependencies():
    packages = ["streamlit", "yfinance", "pandas", "numpy", "plotly", "scikit-learn", "tensorflow-cpu", "feedparser"]
    for package in packages:
        try:
            __import__(package.replace("-cpu", ""))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

check_dependencies()

# Now import all libraries after ensuring they are installed
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import feedparser
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- 2. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AI Crypto Strategist & Sentinel 2026", layout="wide")

# --- 3. FUNCIONES DE DATOS (OPTIMIZADA CON SENTINEL V4) ---
@st.cache_data(ttl=3600)
def load_data(ticker, days):
    try:
        df = yf.download(ticker, start=(pd.Timestamp.now() - pd.Timedelta(days=days)), progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Indicadores Base de la Estrategia
        df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['ema_200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # TR y ATR para Canales de Keltner
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        df['tr'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['upper_k'] = df['ema_20'] + (1.8 * df['atr'])
        df['lower_k'] = df['ema_20'] - (1.8 * df['atr'])

        # Z-Score y ADX Profesional
        df['std_20'] = df['Close'].rolling(20).std()
        df['z_score'] = (df['Close'] - df['ema_20']) / df['std_20']

        plus_di = 100 * (df['High'].diff().clip(lower=0).rolling(14).mean() / df['atr'])
        minus_di = 100 * (df['Low'].diff().clip(upper=0).abs().rolling(14).mean() / df['atr'])
        df['adx'] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(14).mean()

        return df.dropna()
    except Exception:
        return pd.DataFrame()

# --- 4. INTERFAZ LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Panel de Control")
    crypto = st.selectbox("Criptomoneda", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"])
    history_days = st.slider("Historial (Días)", 500, 3000, 1500)
    epochs_n = st.slider("Épocas de Entrenamiento IA", 10, 100, 25)

    st.markdown("---")
    st.write("💰 **Gestión de Capital Inicial**")
    capital_total = st.number_input("Tu Capital Operativo ($", min_value=10.0, value=1000.0, step=100.0)
    riesgo_deseado = st.slider("Riesgo por Operación (%)", 0.5, 5.0, 1.5, step=0.5)
    vix_index = st.number_input("Índice de Volatilidad VIX", min_value=0.0, value=22.0, step=1.0)

    # Filtro geopolítico dinámico automático
    if vix_index > 30:
        st.warning("⚠️ Riesgo reducido al 50% por VIX elevado.")
        riesgo_deseado = riesgo_deseado / 2

    st.markdown("---")
    st.write("📢 **Compartir Análisis:**")
    share_msg = f"Analizando {crypto} con mi motor Sentinel V4 y LSTM."
    st.markdown(f'[✈️ Telegram](https://t.me{share_msg})')
    st.markdown(f'[X (Twitter)](https://twitter.com{share_msg})')

df = load_data(crypto, history_days)

# --- 5. CUERPO PRINCIPAL ---
st.title(f"🚀 AI Crypto Strategist & Dictaminador Sentinel: {crypto}")

if df.empty or len(df) < 60:
    st.error(f"❌ No se pudieron obtener suficientes datos para {crypto}. Intenta aumentar el rango de días.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Gráfico Pro & Estrategia", "🤖 Predicción IA 7 Días", "🎯 Tabla de Robustez", "📰 Noticias"])

    # ÚLTIMA VELA CERRADA PARA DICTAMINAR EL ESTADO ACTUAL
    now = df.iloc[-1]
    precio_actual = now['Close']
    adx_actual = now['adx']
    z_actual = now['z_score']
    ema200_actual = now['ema_200']
    sl_dinamico = now['ema_20']

    # --- PESTAÑA 1: GRÁFICO PRO Y CALCULO DE ENTRADAS ---
    with tab1:
        # Marcado dinámico de señales históricas en el gráfico
        df['chart_signal'] = 0
        df.loc[(df['Close'] > df['upper_k']) & (df['z_score'] > 2) & (df['Close'] > df['ema_200']) & (df['adx'] > 20), 'chart_signal'] = 1
        df.loc[(df['Close'] < df['lower_k']) & (df['z_score'] < -2) & (df['Close'] < df['ema_200']) & (df['adx'] > 20), 'chart_signal'] = -1

        longs = df[df['chart_signal'] == 1]
        shorts = df[df['chart_signal'] == -1]

        # CONSTRUCCIÓN DEL GRÁFICO
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Precio", line=dict(color='#ffffff', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['upper_k'], name="Keltner Sup", line=dict(color='rgba(255, 0, 128, 0.4)', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['lower_k'], name="Keltner Inf", line=dict(color='rgba(0, 255, 255, 0.4)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(0, 255, 255, 0.01)'))
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_200'], name="EMA 200 (Filtro Tendencia)", line=dict(color='#ffaa00', width=1.5)))

        fig.add_trace(go.Scatter(x=longs.index, y=longs['Close'] * 0.98, mode='markers', name="LONG 🚀", marker=dict(symbol='triangle-up', size=12, color='#00ff88')))
        fig.add_trace(go.Scatter(x=shorts.index, y=shorts['Close'] * 1.02, mode='markers', name="SHORT 📉", marker=dict(symbol='triangle-down', size=12, color='#ff3333')))

        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # DIAGNÓSTICO DEL DICTAMINADOR OPERATIVO
        st.subheader("📋 Estado del Dictaminador Técnico")
        estado_senal = "😴 ESPERANDO SETUP CLARO (Sin ventaja estadística actual)"
        tipo_op = None

        if precio_actual > now['upper_k'] and z_actual > 2 and adx_actual > 20:
            if precio_actual > ema200_actual:
                estado_senal = "🚀 SEÑAL ACTIVA: LONG (Momentum Alcista Confirmado)"
                tipo_op = "LONG"
            else:
                estado_senal = "⚠️ RUPTURA ALCISTA BLOQUEADA: El precio cotiza por debajo de la EMA 200."
        elif precio_actual < now['lower_k'] and z_actual < -2 and adx_actual > 20:
            if precio_actual < ema200_actual:
                estado_senal = "📉 SEÑAL ACTIVA: SHORT (Momentum Bajista Confirmado)"
                tipo_op = "SHORT"
            else:
                estado_senal = "⚠️ RUPTURA BAJISTA BLOQUEADA: Riesgo alto de rebote sobre la EMA 200."

        if "🚀" in estado_senal: st.success(estado_senal)
        elif "⚠️" in estado_senal: st.warning(estado_senal)
        else: st.info(estado_senal)

        # MÓDULO DE GESTIÓN DE RIESGO INTERACTIVO
        c_p1, c_p2, c_p3 = st.columns(3)
        c_p1.metric("Precio en Pantalla", f"${precio_actual:,.2f}")
        c_p2.metric("ADX (Fuerza de Tendencia)", f"{adx_actual:.2f}")
        c_p3.metric("Z-Score (Desviación)", f"{z_actual:.2f}")

        if tipo_op:
            capital_arriesgar = capital_total * (riesgo_deseado / 100)
            distancia_sl = abs(precio_actual - sl_dinamico)
            pos_size = capital_arriesgar / distancia_sl if distancia_sl > 0 else 0
            precio_tp = precio_actual + (distancia_sl * 4.0) if tipo_op == "LONG" else precio_actual - (distancia_sl * 4.0)

            st.write("### 📐 Ficha de Orden Recomendada")
            col_o1, col_o2, col_o3 = st.columns(3)
            col_o1.metric("Stop Loss Dinámico (EMA 20)", f"${sl_dinamico:,.2f}")
            col_o2.metric("Objetivo Take Profit (~4x Payoff)", f"${precio_tp:,.2f}")
            col_o3.metric("Tamaño Sugerido Operación", f"{pos_size:.5f} unidades")
            st.info(f"Riesgo financiero controlado: Arriesgando máximo **${capital_arriesgar:,.2f}** para buscar un beneficio estimado de **${capital_arriesgar * 4.0:,.2f}**.")

        csv_data = df.to_csv().encode('utf-8')
        st.download_button("📥 Descargar Historial Completo (CSV)", data=csv_data, file_name=f"{crypto}_sentinel_data.csv")

    # --- PESTAÑA 2: MODELO RED NEURONAL LSTM ---
    with tab2:
        if st.button("🔥 Iniciar Entrenamiento e IA 7 Días"):
            with st.spinner("La IA está analizando los patrones de velas..."):
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[['Close']].values)
                X, y = [], []
                for i in range(60, len(scaled_data)):
                    X.append(scaled_data[i-60:i, 0])
                    y.append(scaled_data[i, 0])
                X, y = np.array(X), np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))

                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(60, 1)),
                    Dropout(0.2),
                    LSTM(50),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                train_history = model.fit(X, y, epochs=epochs_n, batch_size=32, verbose=0)
                st.session_state['train_loss_history'] = train_history.history['loss']

                future_preds = []
                current_batch = scaled_data[-60:].reshape(1, 60, 1)
                for _ in range(7):
                    p = model.predict(current_batch, verbose=0)
                    future_preds.append(p)
                    current_batch = np.append(current_batch[:, 1:, :], p.reshape(1, 1, 1), axis=1)

                st.session_state['preds_7d'] = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
                st.success("✅ Red Neuronal Predictiva entrenada.")

                if 'preds_7d' in st.session_state:
                    f_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(1, 8)]
                    fig_7d = go.Figure()
                    fig_7d.add_trace(go.Scatter(x=f_dates, y=st.session_state['preds_7d'].flatten(), mode='lines+markers', name="Proyección IA", line=dict(color='red', width=3)))
                    fig_7d.update_layout(template="plotly_dark", title="Tendencia Proyectada Próximos 7 Días")
                    st.plotly_chart(fig_7d, use_container_width=True)
                    preds_flat = st.session_state['preds_7d'].flatten()
                    pred_df = pd.DataFrame({'Fecha': f_dates, 'Precio Est.': preds_flat, 'Variación %': [f"{((p / precio_actual) - 1) * 100:+.2f}%" for p in preds_flat]})
                    st.table(pred_df.style.format({"Precio Est.": "${:,.2f}"}))

    # --- PESTAÑA 3: MÉTRICAS DE ROBUSTEZ ---
    with tab3:
        st.subheader("📋 Registro de Validación Estadística del Algoritmo")
        st.write("Datos técnicos verificados mediante backtesting de entornos históricos:")
        # Tabla comparativa extraída directamente de tu documento de desarrollo
        data_robustez = {"Escenario Operativo": ["Backtest de Control Largo", "Prueba Fuera de Muestra (OOS)", "Stress Test Extremo (Cisne Negro)"],
                         "Régimen de Mercado": ["Ciclos Históricos Varios (5 años)", "Bear Market Técnico / Conflicto Bélico", "Volatilidad +20% / Slippage de Pánico 0.5%"],
                         "Win Rate": ["46.15%", "33.33%", "37.50%"],
                         "Payoff Ratio (G/P)": ["1.01", "6.71", "5.61"],
                         "Rendimiento Neto": ["-14.79% (Falsas señales)", "+13.19% (Selectivo)", "+10.16% (Inmunidad Defensiva)"],
                         "Robustez Evaluada": ["⚠️ FRÁGIL / SOBRE-OPERADO", "✅ ALTA ROBUSTEZ ESTRUCTURAL", "🛡️ ULTRA-ROBUSTO (Supervivencia)"]}
        st.table(pd.DataFrame(data_robustez))
        st.info("💡 Conclusión del sistema: Los datos prueban que la ventaja matemática de Sentinel V4 radica en su excelente Ratio de Payoff. No requiere ganar muchas operaciones (Win Rate bajo) ya que, cuando captura una tendencia real en momentos de crisis, compensa con creces las pérdidas controladas.")

    # --- PESTAÑA 4: NOTICIAS ---
    with tab4:
        st.subheader(f"📰 Noticias del Mercado en Tiempo Real: {crypto}")
        rss_url = f"https://yahoo.com{crypto}&region=US〈=en-US"
        feed = feedparser.parse(rss_url)
        if feed.entries:
            for entry in feed.entries[:5]:
                with st.expander(f"🔹 {entry.title}"):
                    st.write(getattr(entry, 'summary', 'Descripción no disponible.'))
                    st.caption(f"Publicado: {entry.published}")
                    st.link_button("Leer Noticia Completa", entry.link)
        else:
            st.info("No se localizaron despachos de noticias recientes para este activo.")

# PIE DE PÁGINA
st.markdown("---")
st.markdown("Desarrollado por: @Bookbinderr-2026App Profesional Cripto - Motor Sentinel V4 Estabilizado", unsafe_allow_html=True)
