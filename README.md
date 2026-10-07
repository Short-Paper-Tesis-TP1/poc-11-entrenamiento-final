# PoC 11: entrenamiento final del modelo predictivo

Parte del proyecto de tesis "Plataforma web para la detección temprana del riesgo de sobreendeudamiento en mujeres emprendedoras" (UPC, Taller de Proyecto I, 2026-20).

## Objetivo único
Entrenar y evaluar el modelo exactamente como lo describe la sección III-B 3) del paper, con el diccionario congelado de 9 variables, y compararlo contra modelos de referencia.

## Datos
Home Credit Default Risk (Kaggle, 2018): `application_train`, `bureau` y `bureau_balance`. Los CSV no se incluyen en el repositorio por las reglas de la competencia.

- **Población:** mujeres con préstamos Cash loans.
- **Subgrupo emprendedoras:** clientas con `Commercial associate` o `Self-employed`. Se reporta por separado.

Las variables se construyen igual que en las PoC 07 a 10.

## Configuración (fijada antes de correr)
- **Partición:** estratificada 70/15/15 (entrenamiento, calibración y prueba), con semilla 42.
- **LightGBM:** la misma configuración de las PoC 07 a 10, con restricción monotónica creciente solo en las 4 variables de la Res. SBS 6941-2008.
- **Calibración:** isotónica, con la partición de calibración.
- **Ajuste a la tasa SBS:** procedimiento de Saerens et al. (2002) con 4.7 %, la tasa de deterioro a 12 meses de créditos nuevos de consumo no revolvente (SBS, Informe de Estabilidad del Sistema Financiero, noviembre 2025).
- **Punto de corte para recall, precisión y F1:** el que maximiza el F1 en la partición de calibración. Nunca se elige mirando la prueba.
- **Modelos de referencia,** con el mismo procedimiento: regresión logística, Random Forest y LightGBM sin restricciones monotónicas.
- **Métricas en la partición de prueba:** AUC-ROC, AUC-PR, recall, precisión, F1 y Brier, para todas y para emprendedoras.
- **AUC-ROC por validación cruzada de 5 particiones:** se calcula además, para compararlo con las PoC 09 y 10.

El Brier de la probabilidad ajustada a la tasa SBS no se evalúa contra las etiquetas de Home Credit. Esa probabilidad está en la escala peruana (4.7 %) y las etiquetas en la de Home Credit, que es de alrededor de 7 %.

## Cómo correrla
```
pip install -r requirements.txt
python poc_entrenamiento_final.py "C:\ruta\home-credit-default-risk"
```

## Resultado
**Población:** 182,800 mujeres con Cash loans, de las cuales 55,133 son emprendedoras. Tasa de TARGET: 0.0718.

**Partición:** entrenamiento 127,960, calibración 27,420 y prueba 27,420.

**AUC-ROC por validación cruzada de 5 particiones (LightGBM final):**

| Grupo | Media | Desviación estándar |
|---|---|---|
| Todas | 0.6558 | 0.0019 |
| Emprendedoras | 0.6437 | 0.0110 |

**Todas** (partición de prueba, 27,420 mujeres, tasa 0.0718)

| Modelo | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Brier calibrado | Punto de corte |
|---|---|---|---|---|---|---|---|
| LightGBM final (4 restricciones SBS) | 0.6584 | 0.1376 | 0.4530 | 0.1297 | 0.2017 | 0.0650 | 0.1024 |
| LightGBM sin restricciones | 0.6642 | 0.1410 | 0.3575 | 0.1530 | 0.2143 | 0.0648 | 0.0964 |
| Regresion logistica | 0.6538 | 0.1377 | 0.4835 | 0.1243 | 0.1977 | 0.0650 | 0.0884 |
| Random Forest | 0.6590 | 0.1335 | 0.4749 | 0.1268 | 0.2001 | 0.0651 | 0.1008 |

**Emprendedoras** (partición de prueba, 8,261 mujeres, tasa 0.0777)

| Modelo | AUC-ROC | AUC-PR | Recall | Precisión | F1 | Brier calibrado | Punto de corte |
|---|---|---|---|---|---|---|---|
| LightGBM final (4 restricciones SBS) | 0.6310 | 0.1255 | 0.4829 | 0.1200 | 0.1922 | 0.0706 | 0.1024 |
| LightGBM sin restricciones | 0.6414 | 0.1295 | 0.3863 | 0.1431 | 0.2088 | 0.0704 | 0.0964 |
| Regresion logistica | 0.6284 | 0.1264 | 0.5374 | 0.1151 | 0.1896 | 0.0706 | 0.0884 |
| Random Forest | 0.6255 | 0.1183 | 0.5218 | 0.1170 | 0.1912 | 0.0710 | 0.1008 |

El AUC-ROC y el AUC-PR de las tablas se calculan con la probabilidad sin calibrar. El recall, la precisión y el F1 usan la probabilidad calibrada y el punto de corte que maximiza el F1 en la partición de calibración.

**Ajuste a la tasa SBS (modelo final, partición de prueba):**

| Medida | Valor |
|---|---|
| Tasa SBS | 0.0470 |
| Tasa en entrenamiento | 0.0718 |
| Media de la probabilidad calibrada | 0.0718 |
| Media de la probabilidad ajustada a la tasa SBS | 0.0475 |
| Ejemplo: 0.2000 pasa a | 0.1375 |
| AUC-ROC de la probabilidad calibrada | 0.6565 |
| AUC-ROC de la probabilidad ajustada | 0.6565 |

## Archivos
- `poc_entrenamiento_final.py`: script.
- `resultado_poc11.json`: salida estructurada.
- `salida_consola.txt`: salida completa de la consola.
- `artefactos/`: modelo y calibrador. Son reproducibles con el script, así que no se suben al repositorio.
