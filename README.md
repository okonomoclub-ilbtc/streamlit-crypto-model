# 🚀 Estratega de Criptomonedas con IA y Dictaminador Sentinel V11 Pro

Entorno web cuantitativo avanzado para la auditoría estadística, radar multiactivo y proyección predictiva de criptoactivos (**BTC-USD, ETH-USD, SOL-USD y XRP-USD**) en tiempo real. Este ecosistema fue diseñado bajo estándares institucionales para eliminar el sesgo discrecional, erradicar cualquier dato estático y validar la robustez matemática de las reglas antes de asignar capital.

**Desarrollado por:** QuantumLab [@thebookbinderr(https://x.com)] | **Comunidad Oficial:** [QuantumTradeA en Telegram](https://t.me)

---

## 🔬 Arquitectura Core: Sentinel V11 Pro (Variante C Validada)

A diferencia de los sistemas minoristas sobreajustados o basados en reversión a la media tradicional, esta versión explota la microestructura del mercado institucional moderno (**Post-ETFs**) mediante un motor modular desacoplado (`core_strategy.py`) que opera sobre una **cesta equiponderada (25% asignado a cada activo)**:

* **Filtro de Dirección Macro (`Close > SMA 50`):** Define el sesgo tendencial institucional al cierre diario (`23:59 UTC`). Solo autoriza compras en expansión alcista; por debajo permanece en *Defensive Cash*.
* **Gatillo de Ignición de Impulso (`Impulso 3D ≥ 1.00 × ATR 14`):** Exige que el avance acumulado de las últimas 3 jornadas supere al menos una vez la volatilidad diaria promedio (`ATR 14`), filtrando entradas en mercados laterales sin recorrido.
* **Filtro de Calendario Institucional (`Lunes a Viernes`):** Opera exclusivamente en días hábiles UTC para evitar el ruido de baja liquidez profesional de los fines de semana.
* **Salida Dinámica y Cinturón de Seguridad (`Cierre < SMA 5` / `Hard Stop 2.5 × ATR`):** Deja correr las expansiones tendenciales hasta que el precio pierde su media rápida de 5 días, protegiendo cada posición desde su apertura con un *Hard Stop* catastrófico a `2.5 × ATR(14)`.
* **Ejecución Causal Realista (`Next-Bar Open`):** Evaluación estricta al cierre consolidado (`23:59 UTC`), ejecución en la apertura siguiente (`00:00 UTC`), exposición Spot `1.0x` (sin apalancamiento) y fricción simulada de `30 bps` *roundtrip* (`15 bps` por lado).

---

## 📊 Resultados de Auditoría y Batería de Tortura (2020–2026 UTC)

Validación ejecutada en vivo sobre datos históricos reales (`In-Sample: 2020–Jun 2024` | `Out-of-Sample: Jun 2024–2026`) bajo portafolio equiponderado al 25%:

| Prueba Institucional | Resultado Auditado | Dictamen |
| :--- | :--- | :--- |
| **WFER (Robert Pardo, 2008)** | `91.77%` (`SR IS = 1.86` · `SR OOS = 1.80`) | **PASS** |
| **CPCV (López de Prado, 2018)** | `100.0%` caminos positivos (`Mean SR = 1.72`) | **PASS** |
| **Test del Mono OOS (Aronson, 1,000 perm.)** | `p-value = 0.01099` (`123` operaciones OOS) | **PASS** |
| **Estrés de Latencia CPS (+1 vela)** | `-4.94%` variación en *Profit Factor* | **PASS** |
| **Deflated Sharpe Ratio (DSR)** | `91.3%` ($N=10$) · `87.8%` ($N=20$) | **PASS** |

---

## 🤖 Componentes de Inteligencia Artificial, Recursos Abiertos y Backend

* **Red Neuronal Recurrente (`LSTM` — Horizonte 7 Días):** Integrada en la pestaña de Radar Multiactivo con selector configurable de historial y épocas, utilizando división temporal estricta (`80% Train / 20% Test`) donde el `MinMaxScaler` se ajusta exclusivamente sobre el tramo de entrenamiento para evitar fuga de datos (*data leakage*), con semillas fijas (`SEED = 7`) para reproducibilidad exacta.
* **Backtest Forense Dinámico:** Simulación en tiempo real tanto del portafolio consolidado (4 criptos al 25% c/u) como del desglose individual por activo, descartando automáticamente la vela diaria en curso.
* **Paridad TradingView (`Pine Script v6`) y Manual Descargable:** Descarga directa desde la interfaz del manual científico (`.md`) y del script equivalente en **Pine Script v6** (`.pine`) con exactitud matemática en `RMA ATR(14)` y calendario UTC.
* **Canal Informativo y Alertas:** Lectura en caché de feeds RSS oficiales (**CoinDesk** con respaldo **Google News**) y despacho administrativo de señales hacia **Telegram** mediante `st.secrets`.

---

## 💻 Instrucciones para Despliegue Local

1. **Clona este repositorio en tu máquina local:**
   ```bash
   git clone https://github.com
   cd streamlit-crypto-model
   ```

2. **Instala las dependencias estandarizadas:**
   ```bash
   pip install -r requirements.txt
   ```

3. *(Opcional)* **Configura tus credenciales locales** creando `.streamlit/secrets.toml` (protegido por `.gitignore`):
   ```toml
   TELEGRAM_TOKEN = "tu_token_de_telegram"
   TELEGRAM_CHAT_ID = "@quantumtradear"
   MAINTENANCE_PASSWORD = "tu_clave_maestra"
   ```

4. **Ejecuta la aplicación interactiva de Streamlit:**
   ```bash
   streamlit run streamlit-crypto-model.py
   ```

---

> 💡 **Aviso Legal de Investigación:** Este software ha sido construido con fines estrictamente educativos, didácticos y de investigación estadística bajo metodologías cuantitativas. Ninguno de los datos, métricas o fichas de órdenes constituye asesoría financiera ni recomendación de inversión. Los rendimientos históricos auditados no garantizan resultados futuros.

