import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import feedparser
import math
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# --- FIJACIÓN DE SEMILLA DE REPRODUCIBILIDAD (MEJORA GITHUB) ---
np.random.seed(7)
tf.random.set_seed(7)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AI Crypto Strategist & Sentinel V10 Pro", layout="wide")

# --- CONFIGURACIÓN DE CREDENCIALES OCULTAS (SEGURIDAD DE PRODUCCIÓN CON FALLBACK) ---
try:
    TOKEN_TELEGRAM = st.secrets.get("TELEGRAM_TOKEN", "DUMMY_TOKEN")
    CLAVE_MAESTRA = st.secrets.get("MAINTENANCE_PASSWORD", "PRO_PASS_2026")
except Exception:
    TOKEN_TELEGRAM = "DUMMY_TOKEN"
    CLAVE_MAESTRA = "PRO_PASS_2026"

CHAT_ID = "@quantumtradear"

def despachar_alerta_telegram(mensaje):
    """Envía notificaciones de rupturas matemáticas al canal de QuantumTradeA."""
    if TOKEN_TELEGRAM == "DUMMY_TOKEN":
        st.warning("⚠️ Telegram no configurado. Token ficticio detectado.")
        return
    url = f"https://telegram.org{TOKEN_TELEGRAM}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass

# --- FUNCIONES DE TRADING CUANTITATIVO (SENTINEL V10 PRO CORREGIDA) ---
@st.cache_data(ttl=3600)
def load_data_v10(ticker, days):
    try:
        # Descarga elástica de datos históricos diarios sin desfase de zona horaria
        df = yf.download(ticker, start=(pd.Timestamp.now() - pd.Timedelta(days=days)), progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Filtro de Tendencia Intermedio (EMA 50)
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # CORRECCIÓN OFF-BY-ONE: Cálculo preciso del Retorno de 3 velas cerradas consecutivas
        df['retorno_3d'] = df['Close'].pct_change(periods=3) * 100

        # True Range y ATR de 14 para Dimensionamiento del Riesgo Controlado (0.5%)
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift(1))
        low_cp = np.abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        return df.dropna()
    except Exception:
        return pd.DataFrame()

# --- INTERFAZ LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Panel de Control")
    crypto = st.selectbox("Activo a Auditar", ["BTC-USD", "ETH-USD", "SOL-USD"])
    history_days = st.slider("Ventana Histórica (Días)", 500, 3000, 1500)
    epochs_n = st.slider("Épocas Entrenamiento LSTM", 5, 50, 15)
    st.markdown("---")
    st.write("💰 **Gestión de Riesgo de Portafolio**")
    capital_total = st.number_input("Capital Operativo Base ($)", min_value=10.0, value=100000.0, step=1000.0)
    riesgo_deseado = st.slider("Riesgo por Operación (%)", 0.1, 2.0, 0.5, step=0.1)
    st.markdown("---")
    st.write("📢 **Canales Oficiales:**")
    st.markdown("[✈️ Telegram QuantumTradeA](https://t.me)")
    st.markdown("[𝕏 Twitter @bookbinderr](https://x.com)")

df = load_data_v10(crypto, history_days)

# --- CUERPO PRINCIPAL ---
st.title(f"🚀 AI Crypto Strategist & Dictaminador Sentinel V10 Pro")

if df.empty or len(df) < 100:
    st.error(f"❌ Muestra estadística insuficiente para simular {crypto}.")
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Gráfico e Interfaz Operativa",
        "🤖 Predicción Neuronal LSTM",
        "🎯 Registro de Robustez (Backtest)",
        "📰 Noticias en Tiempo Real",
        "📖 Manual Operativo Sistemático"
    ])

    # =========================================================================
    # CORRECCIÓN DE SEGURIDAD CRÍTICA: FILTRO DE CONSOLIDACIÓN RÍGIDA
    # =========================================================================
    # Cambiamos iloc[-1] por iloc[-2] para evaluar ÚNICAMENTE la vela cerrada de ayer
    now = df.iloc[-2]
    precio_actual = float(now['Close'])
    ret3d_actual = float(now['retorno_3d'])
    ema50_actual = float(now['ema_50'])
    atr_actual = float(now['atr'])

    # --- PESTAÑA 1: GRÁFICO PRO E INTERFAZ DE ALERTAS ---
    with tab1:
        df['chart_signal'] = 0
        df.loc[(df['Close'] > df['ema_50']) & (df['retorno_3d'] <= -3.0), 'chart_signal'] = 1
        df.loc[(df['Close'] < df['ema_50']) & (df['retorno_3d'] >= 3.0), 'chart_signal'] = -1
        longs = df[df['chart_signal'] == 1]
        shorts = df[df['chart_signal'] == -1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Precio Real", line=dict(color='#F8FAFC', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], name="EMA 50", line=dict(color='#3B82F6', width=1.5)))
        fig.add_trace(go.Scatter(x=longs.index, y=longs['Close'] * 0.96, mode='markers', name="Gatillo Long 🚀", marker=dict(symbol='triangle-up', size=11, color='#10B981')))
        fig.add_trace(go.Scatter(x=shorts.index, y=shorts['Close'] * 1.04, mode='markers', name="Gatillo Short 📉", marker=dict(symbol='triangle-down', size=11, color='#EF4444')))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Estado Actual del Dictaminador Sentinel")
        estado_senal = "😴 ESPERANDO SETUP CLARO (El precio cotiza en zona de ruido neutral)"
        tipo_op = None

        if precio_actual > ema50_actual and ret3d_actual <= -3.0:
            estado_senal = "🚀 SEÑAL ACTIVA: GATILLO LONG DETECTADO"
            tipo_op = "LONG"
        elif precio_actual < ema50_actual and ret3d_actual >= 3.0:
            estado_senal = "📉 SEÑAL ACTIVA: GATILLO SHORT DETECTADO"
            tipo_op = "SHORT"

        c_p1, c_p2, c_p3 = st.columns(3)
        c_p1.metric("Precio de Cierre Evaluado (Ayer)", f"${precio_actual:,.2f}")
        c_p2.metric("Retorno Acumulado 3D", f"{ret3d_actual:.2f}%")
        c_p3.metric("ATR Volatilidad Diaria", f"${atr_actual:,.2f}")

        capital_arriesgar = capital_total * (riesgo_deseado / 100)
        pos_size = (capital_arriesgar / (atr_actual / precio_actual)) if atr_actual > 0 else 0.0
        pos_size = min(pos_size, capital_total * 2.0)

        st.write("### 📐 Ficha Estricta de Orden Recomendada")
        col_o1, col_o2, col_o3 = st.columns(3)
        col_o1.metric("Límite de Pérdida Monetario (0.5%)", f"${capital_arriesgar:,.2f} USD")
        col_o2.metric("Exposición Nominal Máxima (USD)", f"${pos_size:,.2f} USD")
        col_o3.metric("Tamaño Sugerido en Moneda Base", f"{pos_size / precio_actual:.5f} unidades")

        if tipo_op in ["LONG", "SHORT"]:
            st.markdown("---")
            st.write("🔒 **Módulo de Despacho Administrativo (QuantumTradeA)**")
            admin_password = st.text_input("Introduce la clave maestra para autorizar el envío:", type="password", key="admin_pwd_field")

            if tipo_op == "LONG":
                st.success(estado_senal)
                msg_alert = f"🚨 *NUEVA SEÑAL SENTINEL V10 PRO*\n\n• Activo: {crypto}\n• Tipo: LONG 🚀\n• Precio Entrada: ${precio_actual:,.2f} USD\n⏱️ Salida Rígida: 24h"
                if st.button("✈️ Despachar Alerta LONG a Telegram", key="btn_long"):
                    if admin_password == CLAVE_MAESTRA:
                        despachar_alerta_telegram(msg_alert)
                        st.toast("✅ ¡Autorizado! Señal LONG enviada a Telegram.")
                    else:
                        st.error("❌ Credenciales inválidas.")
            elif tipo_op == "SHORT":
                st.error(estado_senal)
                msg_alert = f
        "</p>", 
        unsafe_allow_html=True
    )
