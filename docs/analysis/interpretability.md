# Interpretabilidad del artefacto

> Medición de registro, reproducible con `uv run python scripts/run_interpretability.py`.
> Medida el 2026-10-20 sobre el artefacto `artifacts/fraud_detection_pipeline.pkl` (sha256
> `41869ee0…d4ebd`, `fraud-detection-pipeline` versión 1, ADR 0010), leyendo solo el periodo
> de referencia (`step <= 400`). El diseño y sus razones están en el ADR 0011; las figuras, en
> `reports/interpretability/`, que no está versionado. Dos corridas completas dieron las
> mismas cifras; solo cambiaron los tiempos.
>
> **Todo lo que mide este documento es in-sample:** el artefacto se ajustó con las mismas filas
> de referencia que aquí se explican. Dice qué usa el modelo, no cuánto generaliza.

---

## 1 · La compuerta: SHAP encuentra la firma

**Muestra:** 24.477 filas = los 4.477 positivos de referencia y 20.000 negativos tomados con
la semilla 42 (`settings.RANDOM_STATE`), la del gráfico 6 del notebook 01.

**Explicador:** `TreeExplainer` path-dependent. Valor base −9,5829 log-odds. Los valores SHAP
de cada fila suman su salida cruda con un error máximo de 1,8e-13 log-odds.

| Puesto | Feature | \|SHAP\| medio | Parte del total |
|---|---|---|---|
| 1 | `amount_equals_origin_balance` | 6,1968 | 79,8% |
| 2 | `oldbalanceOrg` | 0,4702 | 6,1% |
| 3 | `oldbalanceDest` | 0,3581 | 4,6% |
| 4 | `amount` | 0,2621 | 3,4% |
| 5 | `type_CASH_OUT` | 0,2326 | 3,0% |
| 6 | `hour_sin` | 0,0846 | 1,1% |
| 7 | `type_TRANSFER` | 0,0703 | 0,9% |
| 8 | `type_PAYMENT` | 0,0354 | 0,5% |
| 9 | `hour_cos` | 0,0309 | 0,4% |
| 10 | `type_CASH_IN` | 0,0239 | 0,3% |
| 11 | `type_DEBIT` | 0,0000 | 0,0% |

**Veredicto: pasa.** Las tres features de la firma suman el **89,2%** (el piso es más de 50%),
y la bandera es **primera** (el piso es entre las tres primeras).

El reparto del crédito entre las tres —79,8%, 6,1% y 3,4%— es lo esperado de features
perfectamente dependientes bajo path-dependent, no un hallazgo (ADR 0011).

**Figura:** `shap_beeswarm.png`. La bandera forma dos cúmulos con la firma, cerca de +23 y de
+35 log-odds. La bandera se usa en un solo corte (sección 2), así que la diferencia no viene de
otros árboles: en path-dependent, el valor esperado sin la bandera recorre la rama falsa
siguiendo las demás features de la fila, y donde esas features ya apuntan a fraude la bandera
agrega menos.

---

## 2 · Las tres ordenaciones, lado a lado

### Importancia por permutación

Sobre la métrica de decisión, recall con precisión ≥ 0,99, en las 5.787.030 filas de
referencia, al nivel de las 11 features del modelo, con 10 permutaciones por feature
(semillas 42 a 51). Sin permutar, la métrica da **0,997543**, in-sample; la regla
(`SignatureRule`) da 0,979451 sobre las mismas filas (4.385 de 4.477). La caída se expresa
también en fraudes: un fraude es 0,000223 de recall.

| Feature | Caída media | Desv. estándar | Mínimo | Máximo | ≈ Fraudes |
|---|---|---|---|---|---|
| `amount_equals_origin_balance` | 0,997521 | 0,000071 | 0,997320 | 0,997543 | 4.466 |
| `amount` | 0,104780 | 0,000819 | 0,103864 | 0,106321 | 469 |
| `oldbalanceOrg` | 0,039848 | 0,001316 | 0,038642 | 0,042216 | 178 |
| `oldbalanceDest` | 0,006143 | 0,000353 | 0,005584 | 0,006701 | 27,5 |
| `type_CASH_OUT` | 0,005361 | 0,000000 | 0,005361 | 0,005361 | 24 |
| `hour_cos` | 0,003417 | 0,000408 | 0,002904 | 0,004021 | 15,3 |
| `hour_sin` | 0,002613 | 0,000334 | 0,002234 | 0,003350 | 11,7 |
| `type_TRANSFER` | 0,000424 | 0,000071 | 0,000223 | 0,000447 | 1,9 |
| `type_PAYMENT` | 0,000000 | 0,000182 | −0,000223 | 0,000223 | 0 |
| `type_CASH_IN` | 0 | 0 | 0 | 0 | 0 |
| `type_DEBIT` | 0 | 0 | 0 | 0 | 0 |

La desviación es la muestral (ddof = 1) entre las 10 permutaciones. `type_PAYMENT` marca el
piso de ruido: un fraude de más o de menos según la semilla. `type_CASH_OUT` cae exactamente
24 fraudes en las 10 permutaciones; el mecanismo de esa invariancia no está determinado.

Advertencias del ADR 0011: permutar un one-hot de `type` suelto crea filas con dos tipos o
ninguno, y permutar la bandera, `amount` u `oldbalanceOrg` por separado crea filas donde la
bandera y los valores se contradicen.

### Las cuatro medidas

| Feature | SHAP | Puesto | Permutación | Puesto | Split | Puesto | Gain | Puesto |
|---|---|---|---|---|---|---|---|---|
| `amount_equals_origin_balance` | 6,197 | 1 | 0,9975 | 1 | 1 | 10 | 1,926e8 | 1 |
| `oldbalanceOrg` | 0,4702 | 2 | 0,03985 | 3 | 873 | 3 | 2,411e6 | 2 |
| `oldbalanceDest` | 0,3581 | 3 | 0,006143 | 4 | 1.181 | 2 | 1,580e5 | 4 |
| `amount` | 0,2621 | 4 | 0,1048 | 2 | 1.227 | 1 | 2,829e5 | 3 |
| `type_CASH_OUT` | 0,2326 | 5 | 0,005361 | 5 | 84 | 7 | 2,985e4 | 5 |
| `hour_sin` | 0,08459 | 6 | 0,002613 | 7 | 471 | 4 | 2,332e4 | 6 |
| `type_TRANSFER` | 0,07034 | 7 | 0,0004244 | 8 | 90 | 6 | 1.638 | 8 |
| `type_PAYMENT` | 0,03545 | 8 | 0,0000 | 9 | 16 | 8 | 727,1 | 9 |
| `hour_cos` | 0,03086 | 9 | 0,003417 | 6 | 390 | 5 | 1,961e4 | 7 |
| `type_CASH_IN` | 0,02385 | 10 | 0 | 10 | 16 | 8 | 214,5 | 10 |
| `type_DEBIT` | 0 | 11 | 0 | 10 | 0 | 11 | 0 | 11 |

Los puestos empatados comparten el mejor. Split es el número de nodos que cortan sobre la
feature en los 98 árboles; gain, la ganancia total de esos cortes; las dos son la importancia
interna de LightGBM.

### Dónde coinciden

- SHAP, permutación y gain ponen **primera a la bandera**.
- Las mismas cuatro features —la bandera, `oldbalanceOrg`, `oldbalanceDest` y `amount`—
  ocupan los cuatro primeros puestos en SHAP, permutación y gain, en distinto orden.
- `type_CASH_OUT` es quinta en SHAP, permutación y gain.

### Dónde discrepan

Se reporta sin resolver a favor de ninguna.

1. **Split pone a la bandera décima, con un solo corte.** Medido en el volcado de los
   árboles: la bandera se usa una sola vez, en la raíz del árbol 0, con ganancia 1,926e8. Su
   rama verdadera es una hoja con las 4.385 filas con firma y valor +27,7476; la rama falsa
   lleva 5.782.645 filas con valor medio −7,1713. Son 34,92 log-odds de diferencia en un
   corte, y ningún otro árbol vuelve a cortar sobre ella. Split cuenta cortes, no lo que hace
   cada uno.
2. **`amount`:** segunda por permutación (469 fraudes), primera por split, tercera por gain
   y cuarta por SHAP. La permutación de `amount` deja intacta la bandera, así que construye
   filas donde la bandera dice "se lleva todo el saldo" y el monto dice otra cosa. Cuánto de
   la caída viene de esas filas no está medido.
3. **`oldbalanceDest`:** segunda por split, tercera por SHAP, cuarta por permutación y por
   gain.
4. **La hora:** `hour_sin` es cuarta por split y sexta o séptima en el resto; `hour_cos` es
   quinta por split, sexta por permutación, séptima por gain y novena por SHAP. Ninguna de
   las dos llega a 0,005 de caída, la significancia práctica del proyecto (ADR 0009).

---

## 3 · ICE y PDP

Se varían tres columnas crudas y cada fila modificada pasa por el artefacto completo, así que
la bandera se recalcula en cada punto (ADR 0011). La grilla son 50 cuantiles, del 0,01 al
0,99, de la distribución de referencia de cada columna; los repetidos se funden. Las curvas
se calculan sobre las 24.477 filas de la muestra; la figura dibuja 100 por grupo (las 92 del
grupo chico) y el PDP de cada grupo. La salida es el log-odds; el umbral (sección 4) está en
−0,2910.

**Figura:** `ice_pdp.png`.

| Columna | Grilla | PDP por grupo, mín. a máx. (log-odds) |
|---|---|---|
| `oldbalanceOrg` | 34 puntos, 0 a 16.400.301,89 | Fraude cubierto por la regla: −10,56 a −5,72 · Fraude fuera de la regla: −12,74 a **+7,03** · Legítima: −10,77 a −8,81 · Toda la muestra: −10,49 a −8,24 |
| `oldbalanceDest` | 30 puntos, 0 a 12.197.274,78 | Fraude cubierto: **+24,23 a +24,97** · Fraude fuera: +5,24 a +10,77 · Legítima: −9,74 a −8,41 · Toda la muestra: **−3,59 a −2,37** |
| `amount` | 50 puntos, 454,22 a 1.621.796,86 | Fraude cubierto: −10,01 a −8,88 · Fraude fuera: −9,72 a −8,31 · Legítima: −9,63 a −8,35 · Toda la muestra: −9,67 a −8,45 |

Filas cuyo propio valor cae fuera de la grilla, donde la curva nunca evalúa al modelo en lo
que la fila tiene:

| Columna | Fraude cubierto por la regla | Fraude fuera de la regla | Legítima |
|---|---|---|---|
| `oldbalanceOrg` | 0 de 4.385 | 27 de 92 | 184 de 20.000 |
| `oldbalanceDest` | 21 de 4.385 | 0 de 92 | 188 de 20.000 |
| `amount` | 1.037 de 4.385 | **62 de 92** | 379 de 20.000 |

### Qué agrega ICE sobre el promedio

- **`oldbalanceDest`: el PDP de toda la muestra no describe a ninguna fila.** Va de −3,59 a
  −2,37 porque promedia tres comportamientos que no se parecen: los fraudes con firma, arriba
  de +24 y casi planos; los que la regla no ve, entre +5 y +11; las legítimas, cerca de −9. El
  promedio cae entre los grupos, lejos de los tres. Además, la muestra tiene 18% de
  positivos, no el 0,077% de la población, así que su PDP global tampoco es el de la población.
- **`oldbalanceOrg`, fraudes con firma: el modelo sin la firma.** Al mover el saldo, la
  igualdad con el monto se rompe y la bandera pasa a 0. Su PDP queda entre −10,56 y −5,72,
  debajo del umbral en toda la grilla: en promedio, sin la firma, el modelo los puntúa como
  legítimas. Es la limitación declarada en el ADR 0011, leída como resultado.
- **`oldbalanceOrg`, fraudes fuera de la regla:** su PDP recorre casi 20 log-odds, de −12,74 a
  +7,03, y sube recién en el extremo alto de la grilla. 27 de los 92 tienen un saldo por encima
  del último punto.
- **`amount` no dice nada sobre los fraudes fuera de la regla.** La grilla termina en
  1.621.796,86 y 62 de los 92 tienen un monto mayor. En EVAL-007, por una medición ad hoc no
  versionada, 57 de los 92 tienen un monto de exactamente 10.000.000. La curva de este grupo
  nunca pasa por la región que el caso b de la sección 4 muestra como decisiva.

---

## 4 · Casos locales

### El umbral

**0,42775088540908546** (−0,2910 en log-odds), el punto que elige recall con precisión ≥ 0,99
sobre las 5.787.030 filas de referencia, in-sample (ADR 0011). Marca 4.510 filas, 4.466 fraudes
y 44 legítimas: precisión 0,990244, recall 0,997543.

| Grupo | Filas en referencia | Marcadas por el modelo |
|---|---|---|
| Fraudes que la regla cubre | 4.385 | 4.385 |
| Fraudes que la regla no cubre | 92 | 81 |
| Legítimas | 5.782.553 | 44 (1 en la muestra) |

### Los cuatro casos

Elegidos con el criterio declarado antes de medir, en el docstring de
`src/fraud_detection/explain/local_explain.py`, entre las filas de la muestra. El identificador
es la posición de la fila en `data/processed/reference.parquet`.

| Caso | Fila | Entradas | Probabilidad | Log-odds | Las tres contribuciones SHAP mayores |
|---|---|---|---|---|---|
| a · la regla y el modelo | 5186345 | step 361, `CASH_OUT`, amount 140.807,76, oldbalanceOrg 140.807,76, oldbalanceDest 137.903,30 | 1,000000 | 25,1288 | bandera +34,8795 · `oldbalanceOrg` −0,3686 · `type_CASH_OUT` +0,1782 |
| b · el modelo, no la regla | 3163674 | step 237, `TRANSFER`, amount 10.000.000,00, oldbalanceOrg 13.924.102,18, oldbalanceDest 0 | 0,999992 | 11,7405 | `amount` +15,4322 · `oldbalanceOrg` +11,5932 · `type_TRANSFER` −3,4204 |
| c · falso positivo | 2958457 | step 230, `CASH_OUT`, amount 581.769,38, oldbalanceOrg 0, oldbalanceDest 0 | 0,546954 | 0,1884 | `oldbalanceDest` +4,0299 · `oldbalanceOrg` +2,7955 · `amount` +1,4905 |
| d · legítima cerca del umbral | 472816 | step 19, `CASH_OUT`, amount 433.504,72, oldbalanceOrg 0, oldbalanceDest 0 | 0,265983 | −1,0151 | `oldbalanceDest` +3,9424 · `oldbalanceOrg` +2,5714 · `amount` +1,2069 |

**Figuras:** `waterfall_a.png` a `waterfall_d.png`.

### Qué significa cada uno

- **a.** El modelo marca esta transacción por una sola razón: se lleva el saldo completo de la
  cuenta. La bandera aporta +34,88 log-odds, y las otras diez features juntas suman −0,17. En
  este caso el modelo reproduce la regla.
- **b.** Esta transferencia no vacía la cuenta —mueve 10.000.000 de 13.924.102,18— y por eso la
  regla no la ve. El modelo la marca por la combinación de un monto de exactamente 10.000.000
  (+15,43) con un saldo de origen mayor que ese monto (+11,59). Que 10.000.000 sea un tope del
  simulador por transacción es una inferencia, no una medición: lo medido es que es el monto
  máximo de los fraudes de referencia (EVAL-007, ad hoc).
- **c.** Un retiro de 581.769,38 desde una cuenta con saldo de origen 0 hacia un destino con
  saldo 0. Sin la firma, el modelo lo marca (0,547 contra 0,428) porque los dos saldos en cero y
  el monto alto empujan hacia fraude. La etiqueta dice que es legítimo: es la única legítima de
  la muestra que el modelo marca, de 44 en toda la referencia.
- **d.** El mismo perfil que el falso positivo —`CASH_OUT` de 433.504,72 con los dos saldos en
  cero— y las mismas tres features empujando hacia fraude, con menos fuerza: queda en 0,266,
  debajo del umbral. Junto con el caso c, dice que para las legítimas el borde de decisión pasa
  por esa combinación de saldos en cero, `CASH_OUT` y monto alto; lo que separa a una de la
  otra es la magnitud de cada empuje, no un rasgo distinto.

---

## 5 · Costo

| Cálculo | Primera corrida | Segunda corrida |
|---|---|---|
| Carga del periodo de referencia | 0,6 s | 0,7 s |
| SHAP de la muestra (24.477 filas) | 2,0 s | 3,0 s |
| Features del modelo sobre la referencia | 1,6 s | 1,9 s |
| Puntajes sobre la referencia | 2,3 s | 3,7 s |
| Permutación (11 features × 10 semillas sobre 5.787.030 filas) | 361,9 s | 490,8 s |
| ICE de `oldbalanceOrg` / `oldbalanceDest` / `amount` | 2,0 / 1,6 / 2,7 s | 1,9 / 1,7 / 2,8 s |

La permutación es el único cálculo que pesa. Los tiempos dependen de la carga de la máquina en
cada corrida y no son costos de referencia.

---

## 6 · Qué no permite concluir

- **Nada sobre la generalización.** Todo es in-sample, y producción (`step >= 401`) no se tocó:
  es la fase 4.
- **La importancia por permutación de `amount` y de la bandera** incluye filas fuera del
  dominio del modelo; cuánto pesan en la caída no está medido.
- **El PDP de toda la muestra** promedia sobre 18% de positivos, no sobre la población.
- **El ICE de `amount`** no cubre la región de los 10.000.000, donde están 62 de los 92 fraudes
  que la regla no ve.
- **Los 11 fraudes que la regla no cubre y el modelo tampoco marca** no se analizaron.

---

**Nota de estado (2026-10-21):** la generalización que esta medición no cubría se midió en la
fase 4, con el mismo artefacto y el mismo umbral: sobre producción, recall 0,9976 y precisión
0,9997 (`docs/analysis/monitoring.md`, EVAL-011). Las cifras de este documento no cambian.
