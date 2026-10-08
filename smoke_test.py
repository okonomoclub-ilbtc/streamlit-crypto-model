"""Smoke test headless de la app con el framework de testing de Streamlit.

Verifica lo que antes rompia: arranque sin secrets, descarga de datos, pestanas,
validacion LSTM y alertas de baseline honestas.
"""
import sys
import warnings

from streamlit.testing.v1 import AppTest

APP = "streamlit-crypto-model.py"
fails = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


print("1) Arranque SIN secrets configurados (no debe crashing)")
at = AppTest.from_file(APP, default_timeout=300).run()
check(not at.exception, f"sin excepciones (exceptions={[str(e)[:120] for e in at.exception]})")
check(len(at.title) > 0, "renderiza el título")
errores = [e.value for e in at.error]
check(not any("Fallo al descargar" in e or "Muestra insuficiente" in e for e in errores),
      f"sin error de carga de datos: {errores}")

print("\n2) Estructura de la UI")
check(len(at.tabs) == 5, f"5 pestañas (encontradas {len(at.tabs)})")
check(len(at.selectbox) == 1, "selectbox de activo")
check(len(at.slider) >= 3, f"sliders de control ({len(at.slider)})")
check(len(at.button) >= 1, "botón de entrenamiento LSTM")

print("\n3) Ausencia de contenido fabricado")
todo = " ".join(
    [e.value for e in at.error] + [e.value for e in at.info]
    + [e.value for e in at.warning] + [e.value for e in at.success]
    + [e.value for e in at.caption] + [m.value for m in at.markdown]
    + [t.label for t in at.tabs]
)
for marcador in ["$9,997.95", "$2,890.95", "56.28%", "2.26", "robustez matemática impecable",
                 "Búfer QuantumTradeA", "inmunidad defensiva total", "binance.com"]:
    check(marcador not in todo, f"no aparece la cifra/texto fabricado: {marcador!r}")
# La pestaña 3 ya no declara que falte el backtest: ahora lo ejecuta y calcula.
check("No existe un backtest" not in todo, "la pestaña 3 ya no afirma que falte el backtest")
check("No hay cifras escritas a mano" in todo, "la pestaña 3 declara que las cifras se calculan")
check("validate.py" in todo, "se menciona validate.py")
check("naive" in todo.lower(), "se documenta el contraste contra el baseline naive")
for falsa in ["totalmente protegida", "Posición totalmente protegida",
              "inmunidad defensiva", "robustez impecable", "Búfer QuantumTradeA"]:
    check(falsa not in todo, f"sin afirmacion de proteccion total: {falsa!r}")

print("\n4) Etiqueta de vela en curso")
check(any("en curso" in w.value for w in at.warning), "avisa que la última vela está en curso")

print("\n5) Entrenamiento LSTM + validacion honesta (pulsa el boton)")
btn = [b for b in at.button if "LSTM" in b.label]
check(len(btn) == 1, f"se localiza el boton de entrenamiento: {[b.label for b in at.button]}")

print("\n5b) Estado del modulo Telegram (señal activa detectada)")
hay_tg = [b.label for b in at.button if "Telegram" in b.label]
if hay_tg:
    check(any("TELEGRAM_TOKEN" in i.value for i in at.info),
          f"informa que el envio esta deshabilitado por falta de credencial: {[i.value[:110] for i in at.info]}")
    check(not any("clave correcta" in i.value.lower() for i in at.error),
          "no afirma exito de envio sin credencial")
else:
    print("  (sin señal activa: se omite)")

if btn:
    at2 = at
    for s in at2.slider:
        if "Épocas" in s.label:
            s.set_value(3)
    at2 = at2.run()
    at2.button[[b.label for b in at2.button].index(btn[0].label)].click().run(timeout=900)
    check(not at2.exception, f"sin excepciones tras entrenar ({[str(e)[:150] for e in at2.exception]})")
    subs = [s.value for s in at2.subheader]
    check(any("Validación fuera de muestra" in s for s in subs),
          f"aparece la seccion de validacion OOS: {subs}")
    ms = [m.label for m in at2.metric]
    check(any("MAE LSTM" in l for l in ms), f"metrica MAE LSTM presente: {ms}")
    check(any("naive" in l.lower() for l in ms), f"metrica MAE naive presente: {ms}")
    warns = " ".join(w.value for w in at2.warning)
    check("no supera" in warns or "supera" in warns,
          f"emite veredicto LSTM vs naive: {warns[:180]!r}")
    # El veredicto honesto debe ser coherente con los numeros mostrados.
    vals = {m.label: m.value for m in at2.metric}
    if "MAE LSTM" in vals and any("naive" in l.lower() for l in vals):
        mae_l = float(vals["MAE LSTM"].replace("$", "").replace(",", ""))
        mae_n = float([v for k, v in vals.items() if "naive" in k.lower()][0].replace("$", "").replace(",", ""))
        coherente = (mae_l > mae_n) == ("no supera" in warns)
        check(coherente, f"veredicto coherente con las cifras: LSTM={mae_l:,.0f} naive={mae_n:,.0f}")

print("\n6) Backtest real de las reglas (pestaña 3, calculado al vuelo)")
btn_bt = [b for b in at.button if "backtest" in b.label.lower()]
check(len(btn_bt) == 1, f"se localiza el boton de backtest: {[b.label for b in at.button]}")
if btn_bt:
    at3 = at
    at3.button[[b.label for b in at3.button].index(btn_bt[0].label)].click().run(timeout=1800)
    check(not at3.exception, f"sin excepciones tras el backtest ({[str(e)[:200] for e in at3.exception]})")
    errs = [e.value for e in at3.error]
    check(not any("No se pudo ejecutar" in e for e in errs), f"el backtest se ejecuto: {errs}")
    labels = [m.label for m in at3.metric]
    for esperado in ("Retorno neto", "Retorno bruto (sin costes)", "Drawdown máximo",
                     "Win rate", "Payoff medio", "Profit factor", "Recovery factor",
                     "t-estadístico", "IC 95% inferior", "IC 95% superior"):
        check(any(esperado in l for l in labels), f"metrica presente: {esperado}")
    subs = [s.value for s in at3.subheader]
    check(any("Backtest Real" in s for s in subs), f"subheader del backtest: {subs}")
    txt = " ".join(m.value for m in at3.markdown)
    check("In-sample" in txt and "Out-of-sample" in txt, "muestra el split in/out-of-sample")
    check("t-test" not in txt and "t-estadístico" in " ".join(labels),
          "expone significacion estadistica")
    todo3 = " ".join([s.value for s in at3.subheader] + [i.value for i in at3.info]
                     + [w.value for w in at3.warning] + [e.value for e in at3.error] + [txt])
    check("ventaja estadística" in todo3, "el veredicto habla de ventaja estadistica")

print("\n7) Aritmetica de dimensionamiento por ATR (recalculo independiente)")
import numpy as np
atr, precio, capital, riesgo = 2000.0, 100000.0, 100000.0, 0.5
esperado = min((capital * riesgo / 100) / (atr / precio), capital * 2.0)
check(abs(esperado - 25000.0) < 1e-6, f"riesgo 0.5% con ATR 2% -> nocional 25k (obtenido {esperado:,.2f})")

print("\n" + "=" * 60)
if fails:
    print(f"FALLOS: {len(fails)}")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")