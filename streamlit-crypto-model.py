import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Crypto Predictor LSTM", layout="wide")
st.title("🚀 Predicción de Criptomonedas con IA (LSTM)")

# 1. Obtención de datos enriquecidos (Tu lógica original)
@st.cache_data(ttl=3600)
def get_enriched_data(symbol, start_date):
    df = yf.download(symbol, start=start_date, interval="1d", progress=False)
    if df.empty:
        return None
    
    # Asegurar que Close sea una serie simple (evitar problemas de MultiIndex)
    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].iloc[:, 0]
    
    # Indicadores Técnicos (Tu lógica original)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volume'] = df['Volume'].astype(float)
    
    df.dropna(inplace=True)
    return df

# 2. Preparación del dataset Multivariante
def create_multivariate_dataset(data, window_size):
    # Selección de características según tu código original
    features = data[['Close', 'RSI', 'Volatility', 'Volume']].values
    target = data['Close'].values.reshape(-1, 1)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaled_features = scaler_x.fit_transform(features)
    scaled_target = scaler_y.fit_transform(target)

    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(scaled_features[i-window_size:i])
        y.append(scaled_target[i, 0])

    return np.array(X), np.array(y), scaler_x, scaler_y

# Interfaz Lateral
st.sidebar.header("Configuración")
symbol = st.sidebar.text_input("Símbolo (ej: BTC-USD)", value="BTC-USD")
start_date = st.sidebar.date_input("Fecha de inicio", datetime(2020, 1, 1))
window_size = st.sidebar.slider("Días de historial (Window)", 30, 100, 60)
epochs = st.sidebar.slider("Épocas de entrenamiento", 5, 50, 15)

if st.button('🚀 Entrenar y Predecir'):
    df = get_enriched_data(symbol, start_date)
    
    if df is not None and len(df) > window_size:
        with st.spinner('Entrenando Red Neuronal...'):
            X, y, scaler_x, scaler_y = create_multivariate_dataset(df, window_size)
            
            # Modelo LSTM Bidireccional (Tu arquitectura original)
            model = Sequential([
                Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(window_size, X.shape[2])),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            model.fit(X, y, batch_size=32, epochs=epochs, verbose=0)

            # Predicción a 7 días (Lógica Multi-paso)
            last_sequence = X[-1].reshape(1, window_size, X.shape[2])
            future_preds_scaled = []
            current_seq = last_sequence.copy()

            for _ in range(7):
                pred = model.predict(current_seq, verbose=0)
                future_preds_scaled.append(pred[0,0])
                
                # Actualizar secuencia para la siguiente predicción
                new_row = current_seq[0, -1, :].copy()
                new_row[0] = pred[0,0] # Reemplazar precio con la predicción
                current_seq = np.append(current_seq[:, 1:, :], [[new_row]], axis=1)

            future_preds = scaler_y.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)

            # Visualización
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Gráfico de Predicción")
                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(df.index[-60:], df['Close'].iloc[-60:], label="Histórico")
                ax.plot(future_dates, future_preds, label="IA Predicción", color='red', marker='o')
                plt.xticks(rotation=45)
                ax.legend()
                st.pyplot(fig)

            with col2:
                st.subheader("Tendencia Próximos 7 Días")
                last_price = df['Close'].iloc[-1]
                preds_flat = future_preds.flatten()
                
                res_list = []
                temp_last = last_price
                for d, p in zip(future_dates, preds_flat):
                    trend = "🚀 ALZA" if p > temp_last else "📉 BAJA"
                    res_list.append({"Fecha": d.strftime('%Y-%m-%d'), "Precio": f"${p:,.2f}", "Tendencia": trend})
                    temp_last = p
                
                st.table(pd.DataFrame(res_list))
    else:
        st.error("No hay suficientes datos. Intenta con una fecha de inicio más antigua o un símbolo válido.")

# Desarrollado por: [@Bookbinderr]
# App de Predicción de Criptomonedas con LSTM (streamlit-crypto-model)
