import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

# Configuración de la página
st.set_page_config(page_title="Predicción BTC LSTM", layout="wide")
st.title("🚀 Analizador de Tendencia Bitcoin (LSTM)")

# 1. Obtención de datos
@st.cache_data(ttl=3600)
def get_enriched_data(symbol="BTC-USD", start_date="2020-01-01"):
    df = yf.download(symbol, start=start_date, interval="1d", progress=False)
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().gt(0).rolling(14).mean() / df['Close'].diff().lt(0).rolling(14).mean()))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df.dropna(inplace=True)
    return df

# 2. Preparación del dataset
def create_multivariate_dataset(data, window_size=60):
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

# Barra lateral para configuración
symbol = st.sidebar.text_input("Símbolo Ticker", value="BTC-USD")
window_size = st.sidebar.slider("Ventana de tiempo (días)", 30, 100, 60)
epochs = st.sidebar.slider("Épocas de entrenamiento", 5, 50, 10)

if st.button('Entrenar Modelo y Predecir'):
    with st.spinner('Descargando datos y entrenando modelo...'):
        df_btc = get_enriched_data(symbol)
        X, y, scaler_x, scaler_y = create_multivariate_dataset(df_btc, window_size)
        
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = Sequential([
            Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(window_size, X.shape[2])),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)

        # Predicción a 7 días
        last_sequence = X[-1].reshape(1, window_size, X.shape[2])
        future_preds_scaled = []
        current_sequence = last_sequence.copy()
        for _ in range(7):
            pred = model.predict(current_sequence, verbose=0)
            future_preds_scaled.append(pred[0, 0])
            new_row = current_sequence[0, -1, :].copy()
            new_row[0] = pred[0, 0]
            current_sequence = np.append(current_sequence[:, 1:, :], [[new_row]], axis=1)

        future_preds = scaler_y.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))
        future_dates = pd.date_range(start=df_btc.index[-1] + pd.Timedelta(days=1), periods=7)

        # Mostrar Resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gráfico de Predicción")
            fig, ax = plt.subplots()
            ax.plot(df_btc.index[-50:], df_btc['Close'].iloc[-50:], label="Histórico")
            ax.plot(future_dates, future_preds, label="Predicción", marker='o', color='red')
            plt.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Tendencia Próximos 7 Días")
            last_p = df_btc['Close'].iloc[-1]
            res_df = pd.DataFrame({'Fecha': future_dates.strftime('%Y-%m-%d'), 'Precio': future_preds.flatten()})
            st.table(res_df)

# Desarrollado por: [@Bookbinderr]
# App de Predicción de Criptomonedas con LSTM (streamlit-crypto-model)
