import streamlit as st
import subprocess
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import feedparser
from datetime import datetime

# --- 1. INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS ---
def check_dependencies():
    packages = ["streamlit", "yfinance", "pandas", "numpy", "plotly", "scikit-learn", "tensorflow-cpu", "feedparser"]
    for package in packages:
        try:
            __import__(package.replace("-cpu", ""))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

check_dependencies()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- 2. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AI Crypto Strategist 2026", layout="wide")

# --- 3. FUNCIONES DE DATOS ---
@st.cache_data(ttl=3600)
def load_data(ticker, days):
    try:
        df = yf.download(ticker, start=(pd.Timestamp.now() - pd.Timedelta(days=days)), progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        
        # Indicadores Técnicos
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
        return df.dropna()
    except Exception:
        return pd.DataFrame()

# --- 4. INTERFAZ LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Panel de Control")
    crypto = st.selectbox("Criptomoneda", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD"])
    history_days = st.slider("Historial (Días)", 500, 3000, 1500)
    epochs_n = st.slider("Épocas de Entrenamiento", 10, 100, 25)
    chart_style = st.radio("Tipo de Gráfico Principal", ["Líneas 📈", "Barras 📊"])
    
    st.markdown("---")
    st.write("📢 **Compartir Análisis:**")
    share_msg = f"Analizando {crypto} con mi IA LSTM. Predicción a 7 días disponible."
    st.markdown(f'[✈️ Telegram](https://t.me{share_msg})')
    st.markdown(f'[X (Twitter)](https://twitter.com{share_msg})')

df = load_data(crypto, history_days)

# --- 5. CUERPO PRINCIPAL ---
st.title(f"🚀 AI Crypto Strategist: {crypto}")

# VALIDACIÓN DE DATOS ANTES DE RENDERIZAR
if df.empty or len(df) < 60:
    st.error(f"❌ No se pudieron obtener suficientes datos para {crypto}. Intenta aumentar el rango de días o verifica tu conexión.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis Técnico", "🤖 Predicción 7 Días", "🎯 Confianza & Backtesting", "📰 Noticias"])

    with tab1:
        fig = go.Figure()
        if "Líneas" in chart_style:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Precio", line=dict(color='#00ff88')))
        else:
            fig.add_trace(go.Bar(x=df.index, y=df['Close'], name="Precio", marker_color='#00ff88'))
        
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="Media 20", line=dict(dash='dot', color='cyan')))
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🛠️ Diagnóstico de Mercado")
        last_p = df['Close'].iloc[-1]
        ma20 = df['MA20'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio Actual", f"${last_p:,.2f}")
        c2.metric("RSI (14d)", f"{rsi:.2f}")
        
        if last_p > ma20 and rsi < 70:
            c3.success("SEÑAL: COMPRA 🚀")
        elif rsi > 75:
            c3.error("SEÑAL: SOBRECOMPRA / VENTA 📉")
        else:
            c3.warning("SEÑAL: PRECAUCIÓN / LATERAL ⚖️")

        csv = df.to_csv().encode('utf-8')
        st.download_button("📥 Descargar Datos Históricos (CSV)", data=csv, file_name=f"{crypto}_data.csv")

    with tab2:
        if st.button("🔥 Iniciar Entrenamiento e IA 7 Días"):
            with st.spinner("La IA está aprendiendo de los patrones históricos..."):
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

                # Predicción recursiva
                future_preds_scaled = []
                current_batch = scaled_data[-60:].reshape(1, 60, 1)
                for _ in range(7):
                    p = model.predict(current_batch, verbose=0)
                    future_preds_scaled.append(p)
                    p_reshaped = p.reshape(1, 1, 1) 
                    current_batch = np.append(current_batch[:, 1:, :], p_reshaped, axis=1)
                
                st.session_state['preds_7d'] = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))
                st.success("✅ Modelo entrenado y proyección completada.")

        if 'preds_7d' in st.session_state:
            f_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(1, 8)]
            fig_7d = go.Figure()
            fig_7d.add_trace(go.Scatter(x=f_dates, y=st.session_state['preds_7d'].flatten(), mode='lines+markers', name="Proyección IA", line=dict(color='red', width=3)))
            fig_7d.update_layout(template="plotly_dark", title="Tendencia Proyectada Próximos 7 Días")
            st.plotly_chart(fig_7d, use_container_width=True)
            
            preds_flat = st.session_state['preds_7d'].flatten()
            pred_df = pd.DataFrame({
                'Fecha': f_dates, 
                'Precio Est.': preds_flat,
                'Variación %': [f"{((p / last_p) - 1) * 100:+.2f}%" for p in preds_flat]
            })
            st.table(pred_df.style.format({"Precio Est.": "${:,.2f}"}))

    with tab3:
        if 'train_loss_history' in st.session_state:
            final_l = st.session_state['train_loss_history'][-1]
            score = max(0, 100 - (final_l * 1000))
            col_a, col_b = st.columns(2)
            col_a.metric("Puntuación de Confianza IA", f"{score:.2f}%")
            st.write("### Curva de Aprendizaje")
            fig_loss = go.Figure(go.Scatter(y=st.session_state['train_loss_history'], line=dict(color='orange')))
            fig_loss.update_layout(template="plotly_dark")
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("⚠️ Ejecuta la predicción en la pestaña anterior para generar el análisis.")

    with tab4:
        st.subheader(f"📰 Noticias en Tiempo Real: {crypto}")
        rss_url = f"https://yahoo.com{crypto}&region=US&lang=en-US"
        feed = feedparser.parse(rss_url)
        if feed.entries:
            for entry in feed.entries[:5]:
                with st.expander(f"🔹 {entry.title}"):
                    st.write(getattr(entry, 'summary', 'Descripción no disponible.'))
                    st.caption(f"Publicado: {entry.published}")
                    st.link_button("Leer Noticia Completa", entry.link)
        else:
            st.info("No se encontraron noticias recientes.")

# PIE DE PÁGINA
st.markdown("---")
st.markdown("""<div style='text-align: center; color: #888888;'><small>Desarrollado por: <b>@Bookbinderr-2026</b></small><br><small>App de Predicción de Criptomonedas con LSTM</small></div>""", unsafe_allow_html=True)
