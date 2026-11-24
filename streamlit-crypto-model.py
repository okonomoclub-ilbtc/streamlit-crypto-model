import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import feedparser
import requests
from datetime import datetime, timezone
from core_strategy import calcular_indicadores_variante_c, ejecutar_backtest_variante_c

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AI Crypto Strategist & Sentinel V11 Pro", layout="wide")

# --- CONFIGURACIÓN DE CREDENCIALES (DESDE STREAMLIT SECRETS) ---
TOKEN_TELEGRAM = st.secrets.get("TELEGRAM_TOKEN", None)
CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "@quantumtradear")
CLAVE_MAESTRA = st.secrets.get("MAINTENANCE_PASSWORD", None)

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
    "Activo del Portafolio Macro",
    ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
    index=0
)
history_days = st.sidebar.slider("Historial de Análisis (Días)", 500, 2500, 2200, step=100)
capital_total = st.sidebar.number_input("Capital Total Operativo ($)", min_value=100.0, value=10000.0, step=500.0)
riesgo_deseado = st.sidebar.slider("Riesgo por Operación (%)", 0.25, 3.0, 1.0, step=0.25)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Nota de Portafolio:** Esta estrategia fue validada institucionalmente "
    "(WFER 91.77%, Mono p=0.011) operando en cesta equiponderada (25% c/u) sobre BTC, ETH, SOL y XRP."
)
st.sidebar.markdown("---")
st.sidebar.write("📢 **Canales Oficiales QuantumTradeA:**")
st.sidebar.markdown("✈️ **[Telegram @quantumtradear](https://t.me)**")
st.sidebar.markdown("𝕏 **[Twitter / X @thebookbinderr](https://x.com)**")

# --- CARGA Y PROCESAMIENTO CON CORE_STRATEGY ---
df_raw = load_data_v11(crypto, days=history_days)

if df_raw.empty:
    st.error("No se pudieron descargar datos históricos para el activo seleccionado.")
else:
    df = calcular_indicadores_variante_c(df_raw)
    now = df.iloc[-1]
    fecha_cierre = df.index[-1]

    precio_actual = float(now['Close'])
    sma50_actual = float(now['sma_50'])
    sma5_actual = float(now['sma_5'])
    atr_actual = float(now['atr_14'])
    impulso_atr_actual = float(now['impulso_atr'])
    es_habil = bool(now['es_dia_habil'])
    senal_activa = bool(now['cond_entrada'])

    # --- ENCABEZADO PRINCIPAL ---
    st.title("💎 AI Crypto Strategist & Sentinel V11 Pro")
    st.caption(
        f"Motor Validado: Variante C (Impulso 3D ≥ 1.00 ATR + Close > SMA 50) | "
        f"Último Cierre Evaluado (UTC): {fecha_cierre.strftime('%Y-%m-%d')}"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Dictamen y Ficha de Orden",
        "📈 Backtest Dinámico (IS vs OOS)",
        "📰 Noticias de Mercado",
        "📖 Manual y Ficha Científica"
    ])

    # =========================================================================
    # PESTAÑA 1: GRÁFICO Y DICTAMEN EN VIVO
    # =========================================================================
    with tab1:
        longs = df[df['cond_entrada']]
        salidas = df[df['cond_salida_sma'] & (df['cond_entrada'].shift(1).fillna(False))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Precio Cierre", line=dict(color='#F8FAFC', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name="SMA 50 (Filtro Macro)", line=dict(color='#3B82F6', width=1.6)))
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_5'], name="SMA 5 (Salida)", line=dict(color='#F59E0B', width=1.1, dash='dot')))
        fig.add_trace(go.Scatter(
            x=longs.index, y=longs['Close'] * 0.96, mode='markers',
            name="Gatillo LONG (Variante C)", marker=dict(symbol='triangle-up', size=11, color='#10B981')
        ))
        fig.update_layout(template="plotly_dark", height=440, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Estado Actual del Dictaminador")

        c_reg1, c_reg2 = st.columns(2)
        with c_reg1:
            if precio_actual > sma50_actual:
                st.success(f"✅ **Régimen Alcista Activo:** Cierre (${precio_actual:,.2f}) > SMA 50 (${sma50_actual:,.2f})")
            else:
                st.warning(f"🛡️ **Defensive Cash:** Cierre (${precio_actual:,.2f}) ≤ SMA 50 (${sma50_actual:,.2f})")
        with c_reg2:
            if es_habil:
                st.info(f"📅 **Calendario Institucional:** Día hábil válido ({fecha_cierre.strftime('%A')})")
            else:
                st.warning(f"⏸️ **Filtro Fin de Semana:** Señal bloqueada en sábado/domingo")

        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Cierre Evaluado (UTC)", f"${precio_actual:,.2f}")
        c_m2.metric("Impulso 3D / ATR", f"{impulso_atr_actual:.2f}x", delta="Umbral ≥ 1.00x")
        c_m3.metric("ATR(14) Diario", f"${atr_actual:,.2f}")
        c_m4.metric("Nivel Salida SMA(5)", f"${sma5_actual:,.2f}")

        # Cálculo de tamaño de posición Spot 1.0x basado en riesgo ATR
        distancia_stop = 2.5 * atr_actual
        stop_sugerido = max(0.0, precio_actual - distancia_stop)
        capital_riesgo_usd = capital_total * (riesgo_deseado / 100.0)
        unidades_sugeridas = (capital_riesgo_usd / distancia_stop) if distancia_stop > 0 else 0.0
        # Tope Spot 1.0x (sin apalancamiento)
        max_unidades_spot = capital_total / precio_actual if precio_actual > 0 else 0.0
        unidades_sugeridas = min(unidades_sugeridas, max_unidades_spot)
        exposicion_usd = unidades_sugeridas * precio_actual

        st.markdown("---")
        if senal_activa:
            st.success("🚀 **SEÑAL ACTIVA: GATILLO LONG DETECTADO (Ejecutar en Apertura 00:00 UTC)**")
        else:
            st.info("😴 **ESPERANDO SETUP:** Condiciones de impulso y tendencia aún no alineadas.")

        st.write("### 📐 Ficha Estricta de Orden (Spot 1.0x)")
        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
        col_o1.metric("Riesgo Máximo ($)", f"${capital_riesgo_usd:,.2f} USD")
        col_o2.metric("Exposición Nominal", f"${exposicion_usd:,.2f} USD")
        col_o3.metric("Tamaño Sugerido", f"{unidades_sugeridas:.5f} {crypto.split('-')[0]}")
        col_o4.metric("Hard Stop (2.5 ATR)", f"${stop_sugerido:,.2f} USD")

        # Módulo de despacho a Telegram
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
                f"🚨 *NUEVA SEÑAL SENTINEL V11 PRO (VARIANTE C)*\n\n"
                f"• **Activo:** {crypto}\n"
                f"• **Dirección:** COMPRA SPOT (LONG) 🚀\n"
                f"• **Precio Ref. Cierre:** ${precio_actual:,.2f} USD\n"
                f"• **Hard Stop (2.5 ATR):** ${stop_sugerido:,.2f} USD\n"
                f"• **Tamaño Sugerido:** {unidades_sugeridas:.5f} unidades (${exposicion_usd:,.2f} USD)\n"
                f"• **Ejecución Apertura:** {txt_entrada}\n"
                f"• **Regla de Salida:** Cierre diario < SMA(5) (${sma5_actual:,.2f}) o Hard Stop"
            )

            if st.button("✈️ Despachar Alerta LONG a Telegram", key="btn_dispatch_signal"):
                if not CLAVE_MAESTRA:
                    st.error("⚠️ La variable MAINTENANCE_PASSWORD no está configurada en Streamlit Secrets.")
                elif admin_password == CLAVE_MAESTRA:
                    if despachar_alerta_telegram(msg_alert):
                        st.toast(f"✅ Señal enviada a {CHAT_ID}.")
                else:
                    st.error("❌ Clave incorrecta. Despacho bloqueado.")

    # =========================================================================
    # PESTAÑA 2: BACKTEST DINÁMICO EN VIVO (IN-SAMPLE vs OUT-OF-SAMPLE)
    # =========================================================================
    with tab2:
        st.subheader(f"📈 Auditoría Histórica Dinámica en Vivo: {crypto}")
        st.caption("Simulación causal con ejecución Next-Bar Open, Hard Stop 2.5 ATR y fricción real de 30 bps roundtrip.")

        df_is = df_raw[df_raw.index <= "2024-06-14"]
        df_oos = df_raw[df_raw.index >= "2024-06-15"]
        trades_is, eq_is, res_is = ejecutar_backtest_variante_c(df_is, capital_total, riesgo_deseado / 100.0)
        trades_oos, eq_oos, res_oos = ejecutar_backtest_variante_c(df_oos, capital_total, riesgo_deseado / 100.0)
        trades_all, eq_all, res_all = ejecutar_backtest_variante_c(df_raw, capital_total, riesgo_deseado / 100.0)
        
        if res_is and res_oos:
            comparativa = pd.DataFrame({
                "Métrica Institucional": list(res_is.keys()),
                "In-Sample (2020 - Jun 2024)": list(res_is.values()),
                "Out-of-Sample (Jun 2024 - Hoy)": list(res_oos.values()),
                "Muestra Completa": list(res_all.values())
            })
            st.table(comparativa)
            
            if not eq_all.empty:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=eq_all.index, y=eq_all['Equity'], name="Curva de Capital ($)",
                    line=dict(color='#10B981', width=2.2)
                ))
                fig_eq.add_vline(x="2024-06-15", line_dash="dash", line_color="#F59E0B")
                fig_eq.update_layout(
                    template="plotly_dark", title="Evolución de Capital Neto (Línea punteada = Inicio Out-of-Sample)",
                    height=380, margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_eq, use_container_width=True)
                
            if not trades_oos.empty:
                st.write("#### 🧾 Últimas Operaciones Auditadas (Out-of-Sample)")
                st.dataframe(trades_oos.tail(10), use_container_width=True)
        else:
            st.warning("No hay suficientes velas en alguno de los subperíodos para calcular la comparativa.")

    # =========================================================================
    # PESTAÑA 3: NOTICIAS DE MERCADO
    # =========================================================================
    with tab3:
        st.subheader(f"📰 Despachos del Mercado: {crypto}")
        rss_url = "coindesk.com"
        feed = feedparser.parse(rss_url)
        ticker_base = crypto.split('-')[0].lower()
        if feed.entries:
            mostradas = 0
            for entry in feed.entries:
                if ticker_base in entry.title.lower() or "crypto" in entry.title.lower() or "bitcoin" in entry.title.lower():
                    with st.expander(f"🔸 {entry.title}"):
                        st.write(getattr(entry, 'summary', 'Despacho oficial disponible en el enlace.'))
                        st.link_button("Leer Noticia Completa", entry.link, key=f"news_{mostradas}")
                    mostradas += 1
                    if mostradas >= 5:
                        break
        else:
            st.info("No se pudieron cargar titulares en este momento.")

    # =========================================================================
    # PESTAÑA 4: MANUAL OPERATIVO Y FICHA CIENTÍFICA
    # =========================================================================
    with tab4:
        st.header("📖 Manual Operativo y Ficha Científica: Variante C")
        st.markdown("""
        #### 1. Reglas Matemáticas del Sistema
        * Filtro de Régimen Macro (Close > SMA 50): Solo autoriza compras cuando el precio de cierre diario está por encima de su media móvil simple de 50 días. En mercados bajistas, el sistema permanece en Defensive Cash.
        * Gatillo de Ignición (Impulso 3D ≥ 1.00 × ATR 14): Exige que el avance acumulado de los últimos 3 días sea al menos igual a una vez la volatilidad diaria promedio (ATR 14).
        * Filtro de Calendario Institucional (Lunes a Viernes): Filtra el ruido de baja liquidez institucional de los fines de semana.
        * Salida Dinámica (Cierre < SMA 5 o Hard Stop 2.5 × ATR): Deja correr las tendencias explosivas hasta que el precio pierde su media rápida de 5 días o alcanza el stop de protección catastrófica.
        #### 2. Dictamen de Auditoría Institucional (Portafolio Equiponderado 4 Activos)
        * WFER (Robert Pardo, 2008): 91.77% (Eficiencia Walk-Forward fuera de muestra).
        * CPCV (López de Prado, 2018): 100.0% de caminos combinatorios positivos (Mean SR = 1.72).
        * Test del Mono OOS (Aronson, 1,000 permutaciones): p-value = 0.01099 (Significancia estadística > 98.9%).
        * Estrés de Latencia CPS (+1 vela): -4.94% de caída en Profit Factor (alta tolerancia a retrasos de ejecución).
        * Deflated Sharpe Ratio (DSR): 91.3% (N=10) / 87.8% (N=20).
        """)
        st.markdown("---")
        st.markdown(
            "🛡️ QuantumTradeA 2026 | Desarrollado con rigor por @thebookbinderr-2026 | "
            "Herramienta estrictamente educativa y de investigación cuantitativa. No constituye asesoría financiera."
        )

# --- PIE DE PÁGINA: CREDENCIALES ---
st.markdown("---")
st.markdown("🛡️ QuantumTradeA 2026 | Desarrollado con rigor por @Bookbinderr-2026", unsafe_allow_html=True)
st.markdown("🚨 NOTA MARGINAL DE DESCARGO LEGAL: Esta aplicación web interactiva ha sido construida con fines estrictamente educativos, didácticos y de investigación estadística avanzada bajo la metodología de la ingeniería cuantitativa. Ninguno de los datos, alertas o fichas de órdenes recomendadas constituye una recomendación de inversión o asesoría financiera oficial. Los rendimientos pasados presentados en el registro de robustez no garantizan retornos futuros.", unsafe_allow_html=True)
