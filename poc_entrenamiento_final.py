# -*- coding: utf-8 -*-
"""
PoC 11 - Entrenamiento final del modelo predictivo
OBJETIVO UNICO: entrenar y evaluar el modelo tal como lo describe la seccion III-B 3) del paper,
con el diccionario congelado de 9 variables, y compararlo contra modelos de referencia.

Configuracion (fijada ANTES de correr, la misma de las PoC 07 a 10):
  - Poblacion: mujeres (CODE_GENDER == "F") con prestamos Cash loans de Home Credit.
  - Etiqueta: TARGET (proxy del sobreendeudamiento, Res. SBS 6941-2008).
  - Particion estratificada 70 % entrenamiento / 15 % calibracion / 15 % prueba, semilla 42.
  - LightGBM: 600 arboles, tasa 0.03, 31 hojas, 200 registros minimos por hoja, 0.8 filas, 0.9 variables,
    restriccion monotonica creciente SOLO en las 4 variables SBS.
  - Calibracion isotonica con la particion de calibracion.
  - Ajuste a la tasa SBS (Saerens et al., 2002): deterioro a 12 meses de creditos nuevos de consumo
    no revolvente = 4.7 % (SBS, IESF noviembre 2025).
  - Punto de corte para recall/precision/F1: el que maximiza el F1 en la particion de CALIBRACION
    (nunca se elige mirando la prueba).
  - Modelos de referencia, con el mismo procedimiento: regresion logistica, Random Forest y LightGBM
    sin restricciones monotonicas.

Metricas en la particion de PRUEBA, para todas y para emprendedoras:
  AUC-ROC, AUC-PR, recall, precision, F1 y Brier (probabilidad calibrada).
  Ademas: AUC-ROC por validacion cruzada de 5 particiones en toda la poblacion (comparable con PoC 09 y 10).
El Brier de la probabilidad ajustada a la tasa SBS NO se evalua con etiquetas de Home Credit, porque esa
probabilidad esta en la escala peruana (4.7 %) y las etiquetas en la de Home Credit (~7 %).

Uso: python poc_entrenamiento_final.py "C:\\ruta\\home-credit-default-risk"
"""
import json
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class _Tee:
    """Escribe en la consola y en salida_consola.txt (UTF-8) a la vez."""
    def __init__(self, ruta):
        self.archivo = open(ruta, "w", encoding="utf-8")
        self.consola = sys.stdout
    def write(self, s):
        self.consola.write(s)
        self.archivo.write(s)
    def flush(self):
        self.consola.flush()
        self.archivo.flush()


sys.stdout = _Tee("salida_consola.txt")

RUTA = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\alexa\Escritorio\UPC\TP1\home-credit-default-risk"
SEMILLA = 42
TASA_SBS = 0.047

# Diccionario congelado de 9 variables: orden y restriccion monotonica (+1 = nunca baja el riesgo)
COLUMNAS = ["ratio_cuota_ingreso", "ratio_deuda_ingreso", "ratio_monto_ingreso", "n_otros_creditos",
            "atraso_historial", "edad", "nivel_educativo", "n_hijos", "anios_trabajo_negocio"]
MONOTONIA = [1, 1, 0, 1, 1, 0, 0, 0, 0]
EDUCACION = {"Lower secondary": 0, "Secondary / secondary special": 1, "Incomplete higher": 2,
             "Higher education": 3, "Academic degree": 3}
PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=SEMILLA,
              n_jobs=-1, verbose=-1)


def csv(nombre):
    for n in (f"dataset-{nombre}.csv", f"{nombre}.csv"):
        if os.path.exists(os.path.join(RUTA, n)):
            return os.path.join(RUTA, n)
    sys.exit(f"No encuentro {nombre}.csv en {RUTA}")


t0 = time.time()
# ---------------------------------------------------------------- 1. Datos (misma construccion que PoC 07-10)
app = pd.read_csv(csv("application_train"), usecols=[
    "SK_ID_CURR", "TARGET", "CODE_GENDER", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
    "AMT_CREDIT", "DAYS_BIRTH", "NAME_EDUCATION_TYPE", "CNT_CHILDREN", "NAME_INCOME_TYPE",
    "ORGANIZATION_TYPE", "DAYS_EMPLOYED"])
w = app[(app.CODE_GENDER == "F") & (app.NAME_CONTRACT_TYPE == "Cash loans")].reset_index(drop=True)

bur = pd.read_csv(csv("bureau"), usecols=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "AMT_CREDIT_SUM_DEBT"])
en_buro = w.SK_ID_CURR.isin(bur.SK_ID_CURR)
act = (bur[bur.CREDIT_ACTIVE == "Active"].assign(d=lambda x: x.AMT_CREDIT_SUM_DEBT.fillna(0))
       .groupby("SK_ID_CURR").agg(n=("SK_ID_BUREAU", "count"), deuda=("d", "sum")))
w = w.merge(act, left_on="SK_ID_CURR", right_index=True, how="left")

bb = pd.read_csv(csv("bureau_balance"), usecols=["SK_ID_BUREAU", "STATUS"],
                 dtype={"SK_ID_BUREAU": "int64", "STATUS": "category"})
ids_bb = bb.SK_ID_BUREAU.unique()
ids_atr = bb.loc[bb.STATUS.isin(["1", "2", "3", "4", "5"]), "SK_ID_BUREAU"].unique()
del bb
c_bb = bur.loc[bur.SK_ID_BUREAU.isin(ids_bb), "SK_ID_CURR"].unique()
c_atr = bur.loc[bur.SK_ID_BUREAU.isin(ids_atr), "SK_ID_CURR"].unique()

X = pd.DataFrame(index=w.index)
X["ratio_cuota_ingreso"] = w.AMT_ANNUITY / w.AMT_INCOME_TOTAL
X["ratio_deuda_ingreso"] = (w.AMT_CREDIT + np.where(en_buro, w.deuda.fillna(0), np.nan)) / w.AMT_INCOME_TOTAL
X["ratio_monto_ingreso"] = w.AMT_CREDIT / w.AMT_INCOME_TOTAL
X["n_otros_creditos"] = np.where(en_buro, w.n.fillna(0), np.nan)
X["atraso_historial"] = np.where(w.SK_ID_CURR.isin(c_atr), 1.0, np.where(w.SK_ID_CURR.isin(c_bb), 0.0, np.nan))
X["edad"] = -w.DAYS_BIRTH / 365.25
X["nivel_educativo"] = w.NAME_EDUCATION_TYPE.map(EDUCACION)
X["n_hijos"] = w.CNT_CHILDREN
X["anios_trabajo_negocio"] = -w.DAYS_EMPLOYED.where(w.DAYS_EMPLOYED != 365243) / 365.25
X = X[COLUMNAS]
y = w.TARGET.values
emp = ((w.NAME_INCOME_TYPE == "Commercial associate") | (w.ORGANIZATION_TYPE == "Self-employed")).values

print("=" * 78)
print("PoC 11 - ENTRENAMIENTO FINAL | 9 variables | mujeres con Cash loans")
print("=" * 78)
print(f"Poblacion: {len(X):,} | emprendedoras: {emp.sum():,} | tasa TARGET: {y.mean():.4f}")
print("Faltantes por variable (%):", {c: round(float(X[c].isna().mean()) * 100, 2) for c in COLUMNAS})

# ---------------------------------------------------------------- 2. AUC-ROC por CV (comparable con PoC 09 y 10)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
cv_todas, cv_emp = [], []
for tr, te in skf.split(X, y):
    m = lgb.LGBMClassifier(monotone_constraints=MONOTONIA, **PARAMS).fit(X.iloc[tr], y[tr])
    p = m.predict_proba(X.iloc[te])[:, 1]
    cv_todas.append(roc_auc_score(y[te], p))
    cv_emp.append(roc_auc_score(y[te][emp[te]], p[emp[te]]))
cv = {"auc_todas": round(float(np.mean(cv_todas)), 4), "sd_todas": round(float(np.std(cv_todas)), 4),
      "auc_emprendedoras": round(float(np.mean(cv_emp)), 4), "sd_emprendedoras": round(float(np.std(cv_emp)), 4)}
print(f"\nCV 5 particiones, LightGBM final: AUC todas {cv['auc_todas']} ({cv['sd_todas']}) | "
      f"emprendedoras {cv['auc_emprendedoras']} ({cv['sd_emprendedoras']})")

# ---------------------------------------------------------------- 3. Particion 70/15/15
idx = np.arange(len(X))
i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEMILLA)
i_ca, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEMILLA)
pi_train = float(y[i_tr].mean())
print(f"\nParticion: entrenamiento {len(i_tr):,} | calibracion {len(i_ca):,} | prueba {len(i_te):,} | "
      f"tasa en entrenamiento {pi_train:.4f}")


def ajuste_sbs(p, pi=pi_train, pi_nueva=TASA_SBS):
    """Saerens, Latinne y Decaestecker (2002): cambia la tasa base sin alterar el orden."""
    a = p * pi_nueva / pi
    b = (1 - p) * (1 - pi_nueva) / (1 - pi)
    return a / (a + b)


def evaluar(nombre, modelo, Xm):
    t = time.time()
    modelo.fit(Xm.iloc[i_tr], y[i_tr])
    seg = time.time() - t
    p_ca = modelo.predict_proba(Xm.iloc[i_ca])[:, 1]
    p_te = modelo.predict_proba(Xm.iloc[i_te])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_ca, y[i_ca])
    c_ca, c_te = iso.predict(p_ca), iso.predict(p_te)
    # punto de corte que maximiza F1 en CALIBRACION
    grilla = np.unique(np.round(c_ca, 4))
    f1s = [f1_score(y[i_ca], c_ca >= u, zero_division=0) for u in grilla]
    corte = float(grilla[int(np.argmax(f1s))])
    res = {"segundos_entrenamiento": round(seg, 1), "corte_f1_calibracion": round(corte, 4)}
    for grupo, msk in (("todas", np.ones(len(i_te), bool)), ("emprendedoras", emp[i_te])):
        yt, pt, ct = y[i_te][msk], p_te[msk], c_te[msk]
        pred = ct >= corte
        res[grupo] = {
            "n": int(msk.sum()), "tasa": round(float(yt.mean()), 4),
            "auc_roc": round(roc_auc_score(yt, pt), 4),
            "auc_pr": round(average_precision_score(yt, pt), 4),
            "recall": round(recall_score(yt, pred, zero_division=0), 4),
            "precision": round(precision_score(yt, pred, zero_division=0), 4),
            "f1": round(f1_score(yt, pred, zero_division=0), 4),
            "brier_sin_calibrar": round(brier_score_loss(yt, pt), 4),
            "brier_calibrado": round(brier_score_loss(yt, ct), 4),
        }
    print(f"\n[{nombre}]  entrenamiento {seg:.1f} s | corte (F1 max. en calibracion) = {corte:.4f}")
    for g in ("todas", "emprendedoras"):
        r = res[g]
        print(f"  {g:<14} AUC-ROC {r['auc_roc']:.4f} | AUC-PR {r['auc_pr']:.4f} | recall {r['recall']:.4f} | "
              f"precision {r['precision']:.4f} | F1 {r['f1']:.4f} | Brier calibrado {r['brier_calibrado']:.4f}")
    return res, modelo, iso


imputar = lambda: SimpleImputer(strategy="median", add_indicator=True)
modelos = {
    "LightGBM final (4 restricciones SBS)": (lgb.LGBMClassifier(monotone_constraints=MONOTONIA, **PARAMS), X),
    "LightGBM sin restricciones": (lgb.LGBMClassifier(**PARAMS), X),
    "Regresion logistica": (make_pipeline(imputar(), StandardScaler(), LogisticRegression(max_iter=2000)), X),
    "Random Forest": (make_pipeline(imputar(), RandomForestClassifier(
        n_estimators=300, min_samples_leaf=50, n_jobs=-1, random_state=SEMILLA)), X),
}
resultados, final = {}, None
for nombre, (m, Xm) in modelos.items():
    resultados[nombre], mod, iso = evaluar(nombre, m, Xm)
    if final is None:
        final = (mod, iso)

# ---------------------------------------------------------------- 4. Ajuste a la tasa SBS (modelo final)
mod, iso = final
c_te = iso.predict(mod.predict_proba(X.iloc[i_te])[:, 1])
s_te = ajuste_sbs(c_te)
ajuste = {"tasa_sbs": TASA_SBS, "tasa_entrenamiento": round(pi_train, 4),
          "media_calibrada_prueba": round(float(c_te.mean()), 4),
          "media_ajustada_sbs_prueba": round(float(s_te.mean()), 4),
          "ejemplo_0_20": round(float(ajuste_sbs(np.array([0.20]))[0]), 4),
          "auc_roc_calibrada": round(roc_auc_score(y[i_te], c_te), 4),
          "auc_roc_ajustada": round(roc_auc_score(y[i_te], s_te), 4)}
print(f"\nAjuste SBS ({TASA_SBS:.3f}): media calibrada {ajuste['media_calibrada_prueba']} -> ajustada "
      f"{ajuste['media_ajustada_sbs_prueba']} | 20 % pasa a {ajuste['ejemplo_0_20']:.1%} | AUC sin cambio: "
      f"{ajuste['auc_roc_calibrada']} = {ajuste['auc_roc_ajustada']}")

# ---------------------------------------------------------------- 5. Artefactos y salida
os.makedirs("artefactos", exist_ok=True)
mod.booster_.save_model("artefactos/modelo_lgbm.txt")
with open("artefactos/calibrador.pkl", "wb") as fh:
    pickle.dump(iso, fh)
salida = {
    "poc": "11_entrenamiento_final",
    "versiones": {"lightgbm": lgb.__version__, "scikit_learn": sklearn.__version__,
                  "pandas": pd.__version__, "numpy": np.__version__},
    "poblacion": {"filas": int(len(X)), "emprendedoras": int(emp.sum()), "tasa_target": round(float(y.mean()), 4)},
    "particion": {"entrenamiento": len(i_tr), "calibracion": len(i_ca), "prueba": len(i_te)},
    "variables": COLUMNAS, "monotonia": MONOTONIA, "parametros_lightgbm": PARAMS,
    "cv_5_lightgbm_final": cv, "modelos": resultados, "ajuste_sbs": ajuste,
    "segundos_totales": round(time.time() - t0, 1),
}
with open("resultado_poc11.json", "w", encoding="utf-8") as fh:
    json.dump(salida, fh, indent=2, ensure_ascii=False)
print(f"\nTiempo total: {salida['segundos_totales']} s | Guardado: resultado_poc11.json y artefactos/")
