# 🚀 AI Crypto Strategist & Sentinel V10 Pro

Interfaz Streamlit para auditar las **reglas de entrada** de la estrategia Sentinel V10
(EMA 50 + momentum 3D + salida por tiempo a 24 h) sobre BTC, ETH y SOL, calcular el
dimensionamiento por ATR y contrastar un modelo LSTM contra un baseline naive.

**Desarrollado por:** [@Bookbinderr-2026](https://x.com) | **Comunidad:** [QuantumTradeA Telegram](https://t.me)

---

## ⚠️ Estado real del proyecto (léelo antes que nada)

Una versión anterior de esta app mostraba una tabla de Win Rate, Payoff, Drawdown Máximo y
Factor de Recuperación con cifras escritas a mano que el código nunca calculaba, seguida de
un dictamen de "robustez matemática impecable". Ese contenido se ha retirado y sustituido por
mediciones reales.

| Qué | Cómo | Resultado |
|---|---|---|
| Reglas de entrada/salida | `backtest.py` y pestaña 3 | **Sin ventaja demostrable** |
| Predicción del nivel de precio (LSTM) | `validate.py` y pestaña 2 | **No supera al naive** |

### Las reglas de trading: no se ha demostrado ventaja

`backtest.py` simula las reglas exactas de la app (señal sobre el cierre de T, entrada al
`Open[T+1]`, salida a `Close[T+1]`, dimensionado por ATR, 5 bps por lado) sobre **toda la
historia disponible**, y las compara contra entradas aleatorias con el mismo número de
operaciones y el mismo tamaño de posición.

| Activo | Período | Retorno neto | Bruto (sin costes) | DD máx | Ops | t | IC 95% media/trade |
|---|---|---|---|---|---|---|---|
| BTC-USD | 2014-2026 | **−2.4%** | +4.0% | 13.3% | 582 | +0.12 | [−0.31%, +0.34%] |
| ETH-USD | 2017-2026 | **−9.5%** | negativo | 13.3% | 560 | +0.03 | [−0.39%, +0.39%] |
| SOL-USD | 2020-2026 | **−5.4%** | negativo | 9.0% | 521 | −0.01 | [−0.54%, +0.52%] |

**En los tres activos el intervalo de confianza al 95% incluye el cero.** No se puede
rechazar que el retorno medio por operación sea nulo. En ETH y SOL la estrategia pierde
dinero incluso con comisión cero.

Tres cosas que hay que mirar antes de creer cualquier cifra de esta estrategia:

1. **El resultado depende de la ventana temporal.** Con los últimos 5 años (2021-2026) BTC
   da +11.3% y queda en el percentil 99.7 frente al azar. Con los 12 años completos da
   −2.4% y percentil 72. El +11.3% era un artefacto de haber mirado solo la década alcista.
2. **Los costes se comen todo el edge.** El equilibrio de BTC está en **3.1 bps por lado**,
   por debajo de la comisión taker típica de un exchange. ETH y SOL no tienen equilibrio:
   el bruto ya es negativo.
3. **Unas pocas operaciones explican el resultado.** En BTC, las 5 mejores operaciones suman
   el 482% del retorno total; en ETH, el 1990%. Sin ellas el resultado es negativo.

Consistencia por años (BTC): negativo de 2014 a 2020, positivo de 2021 a 2026. ETH y SOL
alternan signo sin patrón. Es dependencia de régimen, no una ventaja estable.

> Nota sobre método: las operaciones son diarias y pueden ser consecutivas, así que sus
> retornos tienen autocorrelación (−0.10 en BTC) y no son iid. El intervalo bootstrap
> asume independencia, luego es optimista. Aun así, el margen es tan pequeño que la
> conclusión no cambia.

### La predicción de precio tampoco tiene edge

```
python validate.py BTC-USD
modelo            MAE    MAE %
modelo        21132.3   26.71%
naive          1294.9    1.64%   <- "mañana = hoy"
ma5            1994.9    2.52%
VEREDICTO: el modelo NO supera al baseline naive (21132 vs 1295).
```

Un modelo lineal sobre features de precio es ~16x peor que decir "mañana valdrá lo mismo que
hoy". El LSTM de la pestaña 2 hace la misma comprobación y avisa cuando no supera al
baseline.

---

## 🔬 Reglas implementadas (Sentinel V10)

1. **Filtro de dirección macro:** EMA 50. Solo se buscan compras con el precio por encima, ventas por debajo.
2. **Gatillo de momentum:** retorno de 3 jornadas ≥ ±3%.
3. **Salida por tiempo:** liquidación a las 24 h exactas, sin stop loss ni take profit.
4. **Dimensionamiento:** riesgo nominal configurable (0.5% por defecto) escalado por ATR(14), con techo de apalancamiento 2.0x.

### Limitaciones conocidas, documentadas en la propia app

- **El dimensionamiento no acota la pérdida real.** Está calculado asumiendo que el riesgo se
  materializa a 1 ATR, pero la estrategia no tiene stop loss. Un movimiento adverso mayor
  que 1 ATR dentro de la ventana de 24 h supera con holgura el límite de riesgo mostrado.
- **La señal se evalúa sobre la vela en curso.** yfinance entrega el último día incompleto,
  así que precio, retorno 3D y señal cambian durante la jornada. El protocolo exige validar a
  las 23:59 UTC; la app lo advierte con un banner explícito.
- **El backtest no modela slippage ni profundidad de libro**, solo una comisión plana. Con un
  edge tan fino, esto es material: el equilibrio está en 3.1 bps por lado para BTC.
- **Muestra estadística corta.** 9-12 años y ~550 operaciones dan poco poder estadístico:
  el intervalo de confianza del retorno medio es de ±0.3%, mayor que el propio retorno.

---

## 🤖 Validación del LSTM

- **Split temporal 80/20 sin fuga:** el `MinMaxScaler` se ajusta solo con el tramo de
  entrenamiento; las ventanas de train y test no comparten ninguna barra.
- **Early stopping** sobre `val_loss` con `restore_best_weights`, así que el slider de
  épocas es un máximo, no un objetivo.
- **Contraste honesto contra baseline:** MAE del LSTM frente a "mañana = hoy", más acierto
  direccional junto a la frecuencia de días alcistas como referencia de azar. El acierto
  direccional del baseline naive se omite a propósito: su forecast es el cierre de hoy,
  siempre con signo 0, así que medirlo solo reportaría la frecuencia de días alcistas.
- **Semillas fijas** (`SEED=7`) en `random`, NumPy y TensorFlow.

---

## 💻 Despliegue local

```bash
git clone https://github.com/okonomoclub-ilbtc/streamlit-crypto-model.git
cd streamlit-crypto-model
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit-crypto-model.py
```

La app arranca **sin configurar nada más**. Telegram es opcional; sin credenciales el
botón de despacho se deshabilita con un aviso.

### Telegram (opcional)

Crea `.streamlit/secrets.toml` (no se versiona, está en `.gitignore`) o exporta las
variables de entorno:

```toml
TELEGRAM_TOKEN = "123456:ABC..."
MAINTENANCE_PASSWORD = "tu-clave"
```

| Variable | Entorno alternativo |
|---|---|
| `TELEGRAM_TOKEN` | `TELEGRAM_TOKEN` |
| `MAINTENANCE_PASSWORD` | `MAINTENANCE_PASSWORD` |

### Validación del modelo

```bash
python validate.py BTC-USD   # o ETH-USD, SOL-USD
```

Walk-forward: reentrena cada 30 días sobre el último año y compara Ridge contra los dos
baselines. No necesita TensorFlow, así que corre en segundos.

### Backtest de las reglas de trading

```bash
python backtest.py                                  # los tres activos, toda la historia
python backtest.py BTC-USD --sensibilidad           # barrido EMA 20/50/100 x momentum ±2/3/4%
python backtest.py ETH-USD --period 5y              # acortar la ventana (y ver cómo cambia el veredicto)
python backtest.py BTC-USD --fee 0.001              # otros costes
python backtest.py BTC-USD --rapido                 # sin bootstrap, desglose anual ni barrido de costes
```

Sin lookahead: la señal se evalúa sobre el cierre de T y se ejecuta al `Open[T+1]`, con
salida a `Close[T+1]`. Cada corrida muestra retorno, baselines, split cronológico,
significación (bootstrap + t-estadístico), desglose por año, barrido de costes con punto de
equilibrio, y el baseline aleatorio (2.000 réplicas por defecto, ajustable con
`--simulaciones`), que es la comparación que realmente importa. La pestaña 3 de la app
ejecuta este mismo backtest al vuelo sobre el activo que selecciones.

### Tests

```bash
python smoke_test.py
```

Script standalone (no es un módulo de pytest). Arranca la app en headless con el
framework de testing de Streamlit y comprueba 27 puntos: arranque sin secrets
configurados, ausencia de cifras y titulares fabricados, etiqueta de vela en curso,
entrenamiento LSTM, coherencia del veredicto frente a las métricas mostradas y la
aritmética de dimensionamiento por ATR. Devuelve código de salida 1 si algo falla, así
que sirve tal cual en CI.

Abre red (yfinance + RSS) y entrena un LSTM con 3 épocas, así que tarda del orden de
30 s con la caché de Streamlit tibia; el primer arranque en frío, con la descarga de
datos y la importación de TensorFlow, es más lento.

---

## 📰 Noticias

Titulares de feeds RSS públicos reales (Yahoo Finance, Decrypt, Cointelegraph), con
reintento en cascada. Si ninguno responde, la app muestra el detalle del fallo y no
muestra titulares de relleno.

---

## Licencia

GPL-3.0. Ver [LICENSE](LICENSE).

💡 *Software educativo. Ninguna cifra de este repositorio constituye una recomendación
de inversión, y no hay garantías de resultados futuros.*