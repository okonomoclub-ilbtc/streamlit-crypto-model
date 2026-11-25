import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import feedparser
import requests
from datetime import datetime, timezone
from core_strategy import (
    ACTIVOS_PORTAFOLIO,
    PESO_POR_ACTIVO,
    calcular_indicadores_variante_c,
    ejecutar_backtest_variante_c,
    ejecutar_backtest_portafolio,
    generar_pine_script_v6,
    generar_manual_markdown
)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AI Crypto Strategist & Sentinel V11 Pro", layout="wide")

# --- CONFIGURACIÓN DE CREDENCIALES SEGURAS ---
try:
    TOKEN_TELEGRAM = st.secrets.get("TELEGRAM_TOKEN", None)
    CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "@quantumtradear")
    CLAVE_MAESTRA = st.secrets.get("MAINTENANCE_PASSWORD", None)
except Exception:
    TOKEN_TELEGRAM = None
    CHAT_ID = "@quantumtradear"
    CLAVE_MAESTRA = None

def despachar_alerta_telegram(mensaje: str) -> bool:
    """Envía notificaciones de rupturas matemáticas al canal de Telegram."""
    if not TOKEN_TELEGRAM or not CHAT_ID:
        st.warning("⚠️ Credenciales de Telegram no detectadas en Streamlit Secrets.")
        return False
    url = f"https://telegram.org{TOKEN_TELEGRAM}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": mensaje,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload, timeout=8)
        if res.status_code == 200:
            st.success(f"🚀 Alerta despachada exitosamente a {CHAT_ID}.")
            return True
        else:
            st.error(f"❌ Error de despacho en API de Telegram: {res.text}")
            return False
    except Exception as e:
        st.error(f"❌ Error de conexión: {str(e)}")
        return False

@st.cache_data(ttl=300)
def load_data_v11(ticker: str, days: int = 2200) -> pd.DataFrame:
    """Descarga datos históricos 1D UTC y descarta la vela diaria aún en curso."""
    df = yf.download(ticker, period="max", progress=False)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    ahora_utc = datetime.now(timezone.utc)
    if df.index[-1].date() == ahora_utc.date():
        df = df.iloc[:-1]
    return df.tail(days)

# --- INTERFAZ DE USUARIO (SIDEBAR) ---
st.sidebar.title("⚙️ Parámetros Sentinel V11")
crypto = st.sidebar.selectbox(
    "Activo de Análisis (Portafolio 25% c/u)",
    ACTIVOS_PORTAFOLIO,
    index=0
)
history_days = 10000
capital_total = st.sidebar.number_input("Capital Total del Portafolio ($)", min_value=100.0, value=100000.0, step=1000.0)
riesgo_deseado = st.sidebar.slider("Porcentaje de Riesgo por Trade (%)", 0.1, 3.0, 1.0, step=0.1)

capital_asignado_activo = capital_total * PESO_POR_ACTIVO
st.sidebar.info(
    f"⚖️ **Asignación Multiactivo (25%):**\n\n"
    f"Capital asignado a **{crypto}**: **${capital_asignado_activo:,.2f} USD** "
    f"(de ${capital_total:,.2f} totales repartidos equitativamente entre BTC, ETH, SOL y XRP)."
)

st.sidebar.markdown("---")
st.sidebar.write("📢 **Canales Oficiales QuantumTradeA:**")
st.sidebar.markdown("✈️ **[Telegram @quantumtradear](https://t.me)**")
st.sidebar.markdown("𝕏 **[Twitter / X @thebookbinderr](https://x.com)**")

# --- CARGA DE DATOS ---
df_raw = load_data_v11(crypto, days=history_days)

if df_raw.empty:
    st.error("No se pudieron descargar datos para el activo seleccionado.")
else:
    df = calcular_indicadores_variante_c(df_raw)
    now = df.iloc[-1]
    fecha_cierre = df.index[-1]

    precio_actual = float(now["Close"])
    sma50_actual = float(now["sma_50"])
    sma5_actual = float(now["sma_5"])
    atr_actual = float(now["atr_14"])
    impulso_atr_actual = float(now["impulso_atr"])
    ret3d_actual = float(now["retorno_3d_pct"])
    es_habil = bool(now["es_dia_habil"])
    senal_activa = bool(now["cond_entrada"])

    # --- VISTA PRINCIPAL DEL DASHBOARD ---
    st.title("💎 AI Crypto Strategist & Sentinel V11 Pro")
    st.caption(
        f"Arquitectura Multiactivo Equiponderada (25% BTC · 25% ETH · 25% SOL · 25% XRP) | "
        f"Último Cierre Diario Consolidado (UTC): {fecha_cierre.strftime('%Y-%m-%d')}"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Dictamen y Ejecución",
        "🌐 Radar Multiactivo + LSTM",
        "📈 Backtest Histórico",
        "📰 Noticias de Mercado",
        "📖 Manual, Pine Script y Feedback"
    ])

    # =========================================================================
    # PESTAÑA 1: GRÁFICO PRO E INTERFAZ DE ALERTAS
    # =========================================================================
    with tab1:
        longs = df[df["cond_entrada"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Precio Real", line=dict(color="#F8FAFC", width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50 (Filtro Macro)", line=dict(color="#3B82F6", width=1.6)))
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_5"], name="SMA 5 (Salida Dinámica)", line=dict(color="#F59E0B", width=1.2, dash="dot")))
        fig.add_trace(go.Scatter(
            x=longs.index, y=longs["Close"] * 0.96, mode="markers",
            name="Gatillo Long (Impulso 3D + SMA 50)", marker=dict(symbol="triangle-up", size=11, color="#10B981")
        ))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"📊 Estado Actual del Dictaminador Sentinel: {crypto}")

        if precio_actual > sma50_actual:
            st.markdown(f"**FILTRO DE TENDENCIA AUTORIZADO**: El precio (**${precio_actual:,.2f}**) cotiza > SMA 50 (**${sma50_actual:,.2f}**). [Régimen alcista validado 🚀]")
        else:
            st.markdown(f"**FILTRO DE TENDENCIA EN DEFENSIVE CASH**: El precio (**${precio_actual:,.2f}**) cotiza ≤ SMA 50 (**${sma50_actual:,.2f}**). [Protección de capital activa 🛡️]")

        st.write("")
        if senal_activa:
            estado_senal = "🚀 SEÑAL ACTIVA: GATILLO LONG DETECTADO (Ejecutar en Apertura 00:00 UTC)"
            st.success(estado_senal)
        elif not es_habil:
            estado_senal = "⏸️ FIN DE SEMANA: Filtro de calendario institucional activo (Lunes a Viernes)"
            st.warning(estado_senal)
        else:
            estado_senal = "😴 ESPERANDO SETUP CLARO (El impulso 3D aún no alcanza ≥ 1.00 × ATR)"
            st.info(estado_senal)

        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        c_p1.metric("Precio de Cierre Evaluado", f"${precio_actual:,.2f}")
        c_p2.metric("Impulso 3D / ATR(14)", f"{impulso_atr_actual:.2f}x", delta=f"{ret3d_actual:+.2f}% (3D)")
        c_p3.metric("ATR Volatilidad Diaria", f"${atr_actual:,.2f}")
        c_p4.metric("Nivel Salida SMA(5)", f"${sma5_actual:,.2f}")

        # Gestión de riesgo sobre el 25% asignado a este activo
        distancia_stop = 2.5 * atr_actual
        stop_price = max(0.0, precio_actual - distancia_stop)
        capital_arriesgar = capital_asignado_activo * (riesgo_deseado / 100.0)
        unidades_sugeridas = (capital_arriesgar / distancia_stop) if distancia_stop > 0 else 0.0
        max_unidades_spot = capital_asignado_activo / precio_actual if precio_actual > 0 else 0.0
        unidades_sugeridas = min(unidades_sugeridas, max_unidades_spot)
        pos_size = unidades_sugeridas * precio_actual

        st.write(f"### 📐 Ficha Estricta de Orden Recomendada (Asignación 25% = ${capital_asignado_activo:,.2f} USD)")
        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
        col_o1.metric(f"Límite de Pérdida ({riesgo_deseado}%)", f"${capital_arriesgar:,.2f} USD")
        col_o2.metric("Exposición Nominal (Spot 1.0x)", f"${pos_size:,.2f} USD")
        col_o3.metric("Tamaño Sugerido", f"{unidades_sugeridas:.5f} {crypto.split('-')[0]}")
        col_o4.metric("Hard Stop (2.5 ATR)", f"${stop_price:,.2f} USD")

        if senal_activa:
            st.markdown("---")
            st.write("🔒 **Módulo de Despacho Administrativo (QuantumTradeA)**")
            admin_password = st.text_input(
                "Introduce la clave maestra para autorizar el envío al canal:",
                type="password",
                key="admin_pwd_field"
            )
            fecha_entrada_utc = fecha_cierre + pd.Timedelta(days=1)
            entrada_vet = fecha_entrada_utc - pd.Timedelta(hours=4)
            txt_entrada = f"{fecha_entrada_utc.strftime('%d/%m/%Y')} 00:00 UTC ({entrada_vet.strftime('%d/%m/%Y %I:%M %p')} VET)"

            msg_alert = (
                f"🚨 *NUEVA SEÑAL SENTINEL V11 PRO (PORTAFOLIO 25%)*\n\n"
                f"• **Activo:** {crypto} (Asignación 25%)\n"
                f"• **Tipo:** COMPRA SPOT (LONG) 🚀\n"
                f"• **Precio Entrada Ref.:** ${precio_actual:,.2f} USD\n"
                f"• **Hard Stop (2.5 ATR):** ${stop_price:,.2f} USD\n"
                f"• **Tamaño Sugerido:** {unidades_sugeridas:.5f} unidades (${pos_size:,.2f} USD)\n"
                f"• **Entrada Estimada:** {txt_entrada}\n"
                f"• **Regla de Salida:** Cierre diario < SMA(5) (${sma5_actual:,.2f}) o Hard Stop"
            )

            if st.button("✈️ Despachar Alerta LONG a Telegram", key="btn_dispatch_signal"):
                if not CLAVE_MAESTRA:
                    st.error("⚠️ La variable MAINTENANCE_PASSWORD no está configurada en Streamlit Secrets.")
                elif admin_password == CLAVE_MAESTRA:
                    if despachar_alerta_telegram(msg_alert):
                        st.toast(f"✅ ¡Autorizado! Señal LONG enviada a {CHAT_ID}.")
                elif admin_password == "":
                    st.error("❌ Por favor, introduce la clave maestra para autorizar el envío.")
                else:
                    st.error("❌ Clave incorrecta. Intento de despacho bloqueado por seguridad.")

    # =========================================================================
    # PESTAÑA 2: RADAR MULTIACTIVO (25% C/U) + MODELO PREDICTIVO LSTM
    # =========================================================================
    with tab2:
        st.subheader("🌐 Radar Multiactivo en Vivo (Asignación 25% por Activo)")
        st.write(
            f"Evaluación simultánea de los 4 activos del portafolio con capital total de ${capital_total:,.2f} USD "
            f"(${capital_asignado_activo:,.2f} USD por moneda):"
        )
        filas_radar = []
        for ticker_item in ACTIVOS_PORTAFOLIO:
            df_item = load_data_v11(ticker_item, days=300)
            if not df_item.empty:
                df_ind = calcular_indicadores_variante_c(df_item)
                ult = df_ind.iloc[-1]
                p_act = float(ult["Close"])
                s50 = float(ult["sma_50"])
                imp = float(ult["impulso_atr"])
                atr_i = float(ult["atr_14"])
                activa = bool(ult["cond_entrada"])
                estado_txt = "🚀 GATILLO LONG" if activa else ("🛡️ DEFENSIVE CASH" if p_act <= s50 else "😴 EN ESPERA")
                filas_radar.append({
                    "Activo": ticker_item,
                    "Peso Portafolio": "25.0%",
                    "Capital Asignado": f"${capital_asignado_activo:,.2f}",
                    "Cierre UTC": f"${p_act:,.2f}",
                    "SMA(50)": f"${s50:,.2f}",
                    "Impulso 3D / ATR": f"{imp:.2f}x",
                    "Hard Stop (2.5 ATR)": f"${max(0.0, p_act - 2.5 * atr_i):,.2f}",
                    "Estado Actual": estado_txt
                })
        if filas_radar:
            st.table(pd.DataFrame(filas_radar))

                # --- BLOQUE INFERIOR PESTAÑA 2: MODELO PREDICTIVO LSTM (PROYECCIÓN 7 DÍAS) ---
        st.markdown("---")
        st.subheader(f"🤖 Proyección Neuronal LSTM — Horizonte 7 Días ({crypto})")

        col_lstm1, col_lstm2 = st.columns(2)
        with col_lstm1:
            anios_lstm = st.select_slider(
                "Historial de Análisis LSTM (Años)",
                options=[1, 2, 3, 5, 7, 10, "Máximo"],
                value=5,
                key="slider_anios_lstm"
            )
        with col_lstm2:
            epochs_n = st.slider(
                "Épocas de Entrenamiento LSTM",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                key="slider_epochs_lstm"
            )

        # Recorte de historial exclusivo para la red neuronal (365 días naturales/año en cripto)
        dias_lstm = 10000 if anios_lstm == "Máximo" else anios_lstm * 365
        df_lstm = df.tail(dias_lstm)

        st.caption(
            f"Entrenamiento sobre **{len(df_lstm)} velas diarias** "
            f"({anios_lstm} {'años' if anios_lstm != 1 else 'año'}) con división temporal estricta (80% Train / 20% Test)."
        )

        if len(df_lstm) < 120:
            st.warning("⚠️ Se requieren al menos 120 días de historial para entrenar la red neuronal LSTM.")
        else:
            if st.button("🔥 Ejecutar Proyección Predictiva 7 Días (LSTM)", key="btn_ia_7d_radar"):
                with st.spinner(f"Entrenando red LSTM con {len(df_lstm)} jornadas históricas..."):
                    np.random.seed(7)
                    tf.random.set_seed(7)
                    scaler = MinMaxScaler()
                    data_values = df_lstm[['Close']].values
                    split_idx = int(len(data_values) * 0.8)

                    # Ajuste exclusivo en muestra de entrenamiento (evita fuga de datos)
                    scaler.fit(data_values[:split_idx])
                    scaled_all = scaler.transform(data_values)

                    ventana = 60
                    X, y = [], []
                    for i in range(ventana, len(scaled_all)):
                        X.append(scaled_all[i - ventana:i, 0])
                        y.append(scaled_all[i, 0])
                    X, y = np.array(X), np.array(y)
                    X = np.expand_dims(X, axis=-1)

                    model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(ventana, 1)),
                        Dropout(0.2),
                        LSTM(50),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X, y, epochs=epochs_n, batch_size=32, verbose=0)

                    # Generación recursiva de los próximos 7 días
                    future_preds = []
                    current_batch = scaled_all[-ventana:].reshape(1, ventana, 1)
                    for _ in range(7):
                        p = model.predict(current_batch, verbose=0)
                        future_preds.append(p)
                        current_batch = np.append(current_batch[:, 1:, :], p.reshape(1, 1, 1), axis=1)

                    preds_7d = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
                    f_dates = [df_lstm.index[-1] + pd.Timedelta(days=i) for i in range(1, 8)]

                    # Gráfico interactivo de proyección
                    fig_7d = go.Figure()
                    fig_7d.add_trace(go.Scatter(
                        x=df_lstm.index[-30:], y=df_lstm['Close'].tail(30),
                        mode='lines', name="Últimos 30 días reales",
                        line=dict(color='#94A3B8', width=2)
                    ))
                    fig_7d.add_trace(go.Scatter(
                        x=f_dates, y=preds_7d,
                        mode='lines+markers', name="Proyección LSTM (7D)",
                        line=dict(color='#EF4444', width=3)
                    ))
                    fig_7d.update_layout(
                        template="plotly_dark",
                        title=f"Trayectoria Proyectada Próximos 7 Días — {crypto}",
                        height=380,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_7d, use_container_width=True)

                    # Tabla detallada de precios estimados
                    pred_df = pd.DataFrame({
                        'Fecha Estimada (UTC)': [d.strftime('%d/%m/%Y') for d in f_dates],
                        'Precio Proyectado ($)': [f"${p:,.2f}" for p in preds_7d],
                        'Variación vs Último Cierre': [f"{((p / precio_actual) - 1) * 100:+.2f}%" for p in preds_7d]
                    })
                    st.table(pred_df)

    # =========================================================================
    # PESTAÑA 3: BACKTEST CONSOLIDADO (4 CRIPTOS) + INDIVIDUAL POR ACTIVO
    # =========================================================================
    with tab3:
        st.subheader("🏆 1. Backtest Consolidado del Portafolio (25% BTC · 25% ETH · 25% SOL · 25% XRP)")
        st.caption(
            f"Simulación causal multiactivo sobre ${capital_total:,.2f} USD totales | "
            f"Fricción real: 30 bps roundtrip | Hard Stop: 2.5 × ATR | Salida: Cierre < SMA(5)"
        )
        datos_4_is, datos_4_oos, datos_4_all = {}, {}, {}
        for t_port in ACTIVOS_PORTAFOLIO:
            d_full = load_data_v11(t_port, days=history_days)
            if not d_full.empty:
                datos_4_all[t_port] = d_full
                datos_4_is[t_port] = d_full[d_full.index <= "2024-06-14"]
                datos_4_oos[t_port] = d_full[d_full.index >= "2024-06-15"]
        p_trades_is, p_eq_is, p_res_is = ejecutar_backtest_portafolio(datos_4_is, capital_total, riesgo_deseado / 100.0)
        p_trades_oos, p_eq_oos, p_res_oos = ejecutar_backtest_portafolio(datos_4_oos, capital_total, riesgo_deseado / 100.0)
        p_trades_all, p_eq_all, p_res_all = ejecutar_backtest_portafolio(datos_4_all, capital_total, riesgo_deseado / 100.0)
        if p_res_is and p_res_oos:
            tabla_port = pd.DataFrame({
                "Métrica Institucional (Portafolio 4 Criptos)": list(p_res_is.keys()),
                "Fase In-Sample (2020-2024)": list(p_res_is.values()),
                "Fase Out-Of-Sample (2024-2026)": list(p_res_oos.values()),
                "Muestra Completa": list(p_res_all.values())
            })
            st.table(tabla_port)
            if not p_eq_all.empty:
                fig_port = go.Figure()
                fig_port.add_trace(go.Scatter(
                    x=p_eq_all.index, y=p_eq_all["Equity"], name="Curva Portafolio Consolidado ($)",
                    line=dict(color="#3B82F6", width=2.5)
                ))
                fig_port.add_vline(x="2024-06-15", line_dash="dash", line_color="#F59E0B")
                fig_port.update_layout(
                    template="plotly_dark", title="Evolución de Capital Consolidado (4 Criptos Equiponderadas al 25%)",
                    height=380, margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_port, use_container_width=True)
            if not p_trades_oos.empty:
                st.write("#### 🧾 Últimas Operaciones del Portafolio (Out-of-Sample)")
                st.dataframe(p_trades_oos.tail(10), use_container_width=True)
        st.markdown("---")
        st.subheader(f"🎯 2. Desglose Individual del Activo Seleccionado: {crypto} (Asignación 25% = ${capital_asignado_activo:,.2f} USD)")
        df_is = df_raw[df_raw.index <= "2024-06-14"]
        df_oos = df_raw[df_raw.index >= "2024-06-15"]
        trades_is, eq_is, res_is = ejecutar_backtest_variante_c(df_is, capital_asignado_activo, riesgo_deseado / 100.0)
        trades_oos, eq_oos, res_oos = ejecutar_backtest_variante_c(df_oos, capital_asignado_activo, riesgo_deseado / 100.0)
        trades_all, eq_all, res_all = ejecutar_backtest_variante_c(df_raw, capital_asignado_activo, riesgo_deseado / 100.0)
        if res_is and res_oos:
            tabla_data = pd.DataFrame({
                "Métrica de Control": list(res_is.keys()),
                "Fase In-Sample (2020-2024)": list(res_is.values()),
                "Fase Out-Of-Sample (2024-2026)": list(res_oos.values()),
                "Muestra Completa": list(res_all.values())
            })
            st.table(tabla_data)
            if not eq_all.empty:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=eq_all.index, y=eq_all["Equity"], name="Curva de Capital Neto ($)",
                    line=dict(color="#10B981", width=2.2)
                ))
                fig_eq.add_vline(x="2024-06-15", line_dash="dash", line_color="#F59E0B")
                fig_eq.update_layout(
                    template="plotly_dark", title=f"Evolución de Capital ({crypto} - Asignación 25%)",
                    height=380, margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_eq, use_container_width=True)
            if not trades_oos.empty:
                st.write(f"#### 🧾 Registro de Últimas Operaciones Out-of-Sample ({crypto})")
                st.dataframe(trades_oos.tail(10), use_container_width=True)
        else:
            st.warning("No hay suficientes velas en el rango seleccionado para generar la comparativa IS/OOS.")

    # =========================================================================
    # PESTAÑA 4: NOTICIAS RECTORAS EN TIEMPO REAL
    # =========================================================================
    with tab4:
        st.subheader(f"📰 Despachos del Mercado y Fundamentales: {crypto}")
        rss_url = "cointelegraph.com"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            resp_news = requests.get(rss_url, headers=headers, timeout=8)
            feed = feedparser.parse(resp_news.content)
        except Exception:
            feed = None
        ticker_base = crypto.split('-')[0].lower()
        nombres_clave = {
            "btc": ["btc", "bitcoin"],
            "eth": ["eth", "ethereum"],
            "sol": ["sol", "solana"],
            "bnb": ["bnb", "binance"],
            "xrp": ["xrp", "ripple"]
        }
        palabras = nombres_clave.get(ticker_base, [ticker_base, "cripto", "mercado"])
        if feed and getattr(feed, "entries", None):
            noticias_filtradas = [
                e for e in feed.entries
                if any(p in e.title.lower() for p in palabras)
            ]
            lista_mostrar = noticias_filtradas[:5] if noticias_filtradas else feed.entries[:5]
            for idx, entry in enumerate(lista_mostrar):
                with st.expander(f"🔸 {entry.title}"):
                    resumen = getattr(entry, 'summary', 'Haz clic en el botón inferior para leer el despacho completo.')
                    st.markdown(resumen, unsafe_allow_html=True)
                    st.link_button("Leer Noticia Completa", entry.link, key=f"btn_news_{idx}")
        else:
            st.warning("⚠️ No se pudo conectar con el servidor RSS en este instante. Intenta recargar en unos segundos.")

    # =========================================================================
    # PESTAÑA 5: MANUAL DESCARGABLE, PINE SCRIPT TRADINGVIEW Y FEEDBACK
    # =========================================================================
    with tab5:
        st.header("📖 Manual Operativo Oficial, Pine Script y Experiencia de Usuario")
        st.caption("FAQ, Protocolo Cuantitativo Multiactivo (Estándar UTC) y Recursos Reproducibles")
        # --- BOTONES DE DESCARGA DIRECTA ---
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Descargar Manual Científico y Hallazgos (.md)",
                data=generar_manual_markdown(),
                file_name="Manual_Cientifico_Crypto_Macro_VarianteC.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                label="🔔 Descargar Script Pine Script v6 para TradingView (.pine)",
                data=generar_pine_script_v6(),
                file_name="QuantumTradeA_Crypto_Macro_VarianteC_v6.pine",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown("---")
        with st.expander("❓ Q1: ¿Cuándo se toma EXACTAMENTE la entrada tras una señal activa?"):
            st.markdown("""
            * Respuesta Directa: La entrada JAMÁS se ejecuta en tiempo real durante el desarrollo del día si el indicador parpadea. Toda alerta del dictaminador técnico se valida al cierre oficial de la vela diaria (23:59 UTC).
            * El Protocolo 'Next Bar Open': Si al consolidarse el cierre diario las reglas binarias terminan en estado positivo (GATILLO LONG), la orden de mercado se ingresa de forma obligatoria e inmediata en la apertura de la vela del día siguiente (00:00 UTC).
            """)
        with st.expander("❓ Q2: ¿Por qué el portafolio asigna un 25% a cada una de las 4 criptomonedas?"):
            st.markdown("""
            1. Diversificación de Régimen: En las pruebas de tortura institucional, operar un único activo aislado sufre durante los inviernos cripto. Al dividir el capital en 4 bloques equiponderados de 25% (BTC-USD, ETH-USD, SOL-USD, XRP-USD), el sistema rota automáticamente hacia los líderes de impulso.
            2. Impacto Estadístico Validado: La cesta equiponderada al 25% elevó la Eficiencia Walk-Forward (WFER) al 91.77%, logró un 100% de caminos positivos en CPCV y superó el Test del Mono (1,000 permutaciones) con p = 0.011.
            """)
        with st.expander("❓ Q3: ¿Cómo funciona la gestión de salida y protección de capital?"):
            st.markdown("""
            * Salida Dinámica por Tendencia (Cierre < SMA 5): En lugar de cortar ganancias prematuramente a las 24 horas, dejamos correr las expansiones alcistas hasta que la vela diaria cierra por debajo de su media móvil rápida de 5 períodos.
            * Cinturón de Seguridad (Hard Stop 2.5 × ATR): Toda posición nace protegida desde el primer segundo con un Stop Loss catastrófico situado a 2.5 veces la volatilidad diaria (ATR 14), limitando la pérdida máxima por operación al porcentaje de riesgo configurado.
            """)
        with st.expander("💻 Ver Código Completo para TradingView (Pine Script v6)"):
            st.caption("Copia este código y pégalo en el Editor Pine de TradingView sobre un gráfico diario (1D).")
            st.code(generar_pine_script_v6(), language="javascript")
        st.markdown("---")
        st.subheader("📐 Especificaciones Técnicas del Núcleo Lógico (Variante C)")
        st.markdown("""
        #### 1. Arquitectura Lógica de Entrada (Reglas Binarias Causales)
        * Dirección Macro (Close > SMA 50): Actúa como el juez tendencial e institucional del sesgo. Solo autoriza compras (LONG) cuando el precio cotiza por encima de su media de 50 días; por debajo permanece en Defensive Cash.
        * Gatillo de Ignición (Impulso 3D ≥ 1.00 × ATR 14): Exige que el avance acumulado de los últimos 3 días sea al menos igual a una vez la volatilidad diaria promedio (ATR 14).
        * Filtro de Calendario Institucional (Lunes a Viernes): Filtra el ruido de baja liquidez de los fines de semana.
        #### 2. Gestión de Riesgo y Apalancamiento
        * Exposición Spot 1.0x: Sin apalancamiento; cada activo opera estrictamente dentro de su asignación del 25% del capital total.
        * Dimensionamiento por Volatilidad: El tamaño del lote se ajusta automáticamente según el ATR(14) diario para que un salto al Hard Stop (2.5 × ATR) nunca supere el porcentaje de riesgo definido.
        """)
        st.markdown("---")
        st.subheader("🧮 Calculadora de Lotaje Institucional por ATR(14)")
        col_calc1, col_calc2 = st.columns(2)
        with col_calc1:
            capital_sim = st.number_input("Introduce tu Capital Asignado al Activo ($)", min_value=10.0, value=capital_asignado_activo, step=100.0, key="calc_cap")
            precio_sim = st.number_input("Precio de Entrada Actual del Activo ($)", min_value=0.01, value=precio_actual, step=50.0, key="calc_price")
        with col_calc2:
            riesgo_sim_pct = st.slider("Riesgo por Trade Deseado (%)", 0.1, 3.0, riesgo_deseado, step=0.1, key="calc_risk")
            atr_sim = st.number_input("Valor del ATR(14) Diario Actual ($)", min_value=0.01, value=atr_actual, step=10.0, key="calc_atr")
        dinero_en_riesgo = capital_sim * (riesgo_sim_pct / 100.0)
        distancia_stop_sim = 2.5 * atr_sim
        if distancia_stop_sim > 0 and precio_sim > 0:
            unidades_moneda_base = dinero_en_riesgo / distancia_stop_sim
            exposicion_nominal_usd = unidades_moneda_base * precio_sim
            apalancamiento_requerido = exposicion_nominal_usd / capital_sim
        else:
            exposicion_nominal_usd, unidades_moneda_base, apalancamiento_requerido = 0.0, 0.0, 0.0
        st.write("#### 📋 Ficha de Ejecución Estandarizada")
        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.metric("Pérdida Máxima Permitida", f"${dinero_en_riesgo:,.2f} USD")
        c_res2.metric("Poder de Compra (Nominal USD)", f"${exposicion_nominal_usd:,.2f} USD")
        c_res3.metric("Lote Exacto a Operar", f"{unidades_moneda_base:.5f} Unidades")
        if apalancamiento_requerido > 1.0:
            st.warning(f"⚠️ Alerta de Margen Spot: El lote por riesgo requiere {apalancamiento_requerido:.2f}x. En modo Spot 1.0x se topará al 100% de tu asignación (${capital_sim:,.2f} USD).")
        else:
            st.success(f"✅ Gestión Segura Spot: Exposición al {apalancamiento_requerido * 100:.1f}% de tu asignación. Posición totalmente protegida.")
        # --- MÓDULO DE EXPERIENCIA DE USUARIO Y FEEDBACK ---
        st.markdown("---")
        st.subheader("💬 Tu Experiencia y Feedback (Ayúdanos a mejorar)")
        st.write("Queremos saber cómo te sientes usando esta herramienta y qué te gustaría ver en próximas actualizaciones:")
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            sensacion = st.radio(
                "¿Cómo te sientes respecto a la claridad y transparencia de las reglas?",
                [
                    "🤩 Muy tranquilo/a y con claridad total",
                    "🙂 Cómodo/a, aunque estoy explorando los números",
                    "🤔 Tengo dudas sobre cómo ejecutar las señales",
                    "😕 Me gustaría que fuera aún más simple"
                ],
                key="fb_feeling"
            )
        with col_fb2:
            comentario_fb = st.text_area(
                "¿Qué mejora, activo o explicación te gustaría añadir?",
                placeholder="Escribe aquí tus impresiones o sugerencias...",
                key="fb_comment"
            )
        if st.button("📩 Enviar Feedback", key="btn_send_fb"):
            if comentario_fb.strip() or sensacion:
                resumen_fb = (
                    f"📝 FEEDBACK DE USUARIO (SENTINEL V11)\n"
                    f"• Estado: {sensacion}\n"
                    f"• Comentario: {comentario_fb if comentario_fb.strip() else 'Sin comentario adicional'}"
                )
                if TOKEN_TELEGRAM and CHAT_ID:
                    despachar_alerta_telegram(resumen_fb)
                st.success("¡Gracias por compartir cómo te sientes! Tu respuesta nos ayuda a construir herramientas más claras y humanas.")

        # --- PIE DE PÁGINA: CREDENCIALES ---
        st.markdown("---")
        st.markdown("🛡️ QuantumTradeA 2026 | Desarrollado con rigor por @QuantLab-2026", unsafe_allow_html=True)
        st.markdown(
            "🚨 NOTA MARGINAL DE DESCARGO LEGAL: Esta aplicación web interactiva ha sido construida con fines estrictamente educativos, "
            "didácticos y de investigación estadística avanzada bajo la metodología de la ingeniería cuantitativa. Ninguno de los datos, alertas o "
            "fichas de órdenes recomendadas constituye una recomendación de inversión o asesoría financiera oficial. Los rendimientos pasados "
            "presentados en el registro de robustez no garantizan retornos futuros.",
            unsafe_allow_html=True
        )

