# Monitoreo del artefacto sobre producción

> Medición de registro, reproducible con `uv run python scripts/run_monitoring.py`. Medida el
> 2026-10-21 sobre el artefacto `artifacts/fraud_detection_pipeline.pkl` (sha256
> `41869ee0…d4ebd`, `fraud-detection-pipeline` versión 1, ADR 0010), con el umbral fijo
> **0,42775088540908546** (ADR 0011). El diseño y sus razones están en el ADR 0012; las tablas
> completas, en `reports/monitoring/`, que no está versionado; el reporte de Evidently, en
> `reports/deliverables/monitoring_report.html`. Corrida de MLflow `b7eabc74ebc64ae68b05472ff437bb3f`.
>
> **Es la primera lectura de producción** (`step >= 401`: 575.590 filas, steps 401 a 743). El
> modelo, el preprocesador y el umbral se ajustaron sin ver una fila de producción, y nada se
> reajustó para medir. Lo que sigue es desempeño fuera de muestra, no una autoevaluación.

---

## 1 · La regla sobre producción

| | Marcadas | Fraudes cubiertos | Falsos positivos | Fraudes no cubiertos | Precisión | Recall |
|---|---|---|---|---|---|---|
| **Producción** (3.736 fraudes) | 3.633 | 3.633 | **0** | 103 | **1,000** | **0,9724** |
| Referencia (4.477 fraudes) | 4.385 | 4.385 | 0 | 92 | 1,000 | 0,9795 |

El recall de la regla cae 0,0070 respecto de referencia, menos que su propia dispersión entre los
folds temporales (0,9788 ± 0,0101, EVAL-005). Por ventana completa va de 0,944 a 1,000, con
0,9736 ± 0,0173: la dispersión es mayor que entre folds porque una ventana tiene ~260 fraudes y un
fold ~450.

| Tipo | Filas legítimas | Fraudes | Fraudes con la firma |
|---|---|---|---|
| `CASH_IN` | 127.917 | 0 | 0 |
| `CASH_OUT` | 187.536 | 1.868 | 1.856 |
| `DEBIT` | 4.316 | 0 | 0 |
| `PAYMENT` | 202.484 | 0 | 0 |
| `TRANSFER` | 49.601 | 1.868 | 1.777 |

**La firma se sostiene.** Ninguna de las 571.854 filas legítimas se lleva el saldo completo, igual
que en referencia. La firma sigue apareciendo solo en `CASH_OUT` y `TRANSFER`, así que la
condición de tipo de la regla sigue sin cambiar ninguna cifra; se queda por la razón del ADR 0007.
Los 103 fraudes que la regla no ve son 12 `CASH_OUT` y 91 `TRANSFER`.

**Qué dice eso del generador.** La precisión 1,000 sobre un periodo que nadie miró al construir la
regla dice que la firma es una propiedad del código del agente de fraude, no del periodo de
referencia: el agente sigue vaciando cuentas, y nada legítimo lo hace. Los fraudes se reparten
exactamente por mitades entre `TRANSFER` y `CASH_OUT` (1.868 y 1.868). Que eso refleje el
patrón que describe la documentación del dataset, transferir a una cuenta y retirar desde ella,
es una inferencia, no una medición. Un atacante real se adaptaría a una regla que lo atrapa;
el del simulador no lo hace, y este monitoreo no puede decir nada sobre uno que sí lo haga.

---

## 2 · Desempeño por ventana

Ventanas de 24 steps ancladas en el 401 (ADR 0012). El modelo marca una fila cuando su puntaje es
al menos el umbral fijo. A la izquierda de cada métrica del modelo va la de la regla, sobre las
mismas filas. La ventana 15 es parcial: 7 steps.

| Ventana | Steps | Filas | Fraudes | Prevalencia | Regla P | Regla R | Modelo P | Modelo R | FN modelo | FP modelo | PR-AUC regla | PR-AUC modelo | Brier regla | Brier modelo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 401–424 | 209.359 | 308 | 0,0015 | 1,000 | 0,9675 | 1,0000 | 1,0000 | 0 | 0 | 0,9676 | 1,0000 | 0,00005 | 0,00000 |
| 2 | 425–448 | 17.118 | 250 | 0,0146 | 1,000 | 0,9440 | 1,0000 | **0,9920** | 2 | 0 | 0,9448 | 0,9999 | 0,00082 | 0,00011 |
| 3 | 449–472 | 16.601 | 248 | 0,0149 | 1,000 | 0,9879 | 1,0000 | 1,0000 | 0 | 0 | 0,9881 | 1,0000 | 0,00018 | 0,00000 |
| 4 | 473–496 | 25.549 | 232 | 0,0091 | 1,000 | 1,0000 | 1,0000 | 1,0000 | 0 | 0 | 1,0000 | 1,0000 | 0,00000 | 0,00000 |
| 5 | 497–520 | 26.350 | 266 | 0,0101 | 1,000 | 0,9699 | 1,0000 | 1,0000 | 0 | 0 | 0,9702 | 1,0000 | 0,00030 | 0,00000 |
| 6 | 521–544 | 59.283 | 244 | 0,0041 | 1,000 | 0,9918 | 1,0000 | 1,0000 | 0 | 0 | 0,9918 | 1,0000 | 0,00003 | 0,00000 |
| 7 | 545–568 | 45.497 | 262 | 0,0058 | 1,000 | 0,9847 | 1,0000 | 0,9962 | 1 | 0 | 0,9848 | 1,0000 | 0,00009 | 0,00002 |
| 8 | 569–592 | 45.713 | 258 | 0,0056 | 1,000 | 0,9845 | 0,9961 | 0,9961 | 1 | **1** | 0,9846 | 0,9997 | 0,00009 | 0,00003 |
| 9 | 593–616 | 33.795 | 256 | 0,0076 | 1,000 | 0,9961 | 1,0000 | 1,0000 | 0 | 0 | 0,9961 | 1,0000 | 0,00003 | 0,00000 |
| 10 | 617–640 | 11.729 | 268 | 0,0228 | 1,000 | 0,9701 | 1,0000 | 1,0000 | 0 | 0 | 0,9708 | 1,0000 | 0,00068 | 0,00000 |
| 11 | 641–664 | 14.324 | 268 | 0,0187 | 1,000 | 0,9590 | 1,0000 | 0,9963 | 1 | 0 | 0,9597 | 0,9963 | 0,00077 | 0,00007 |
| 12 | 665–688 | 36.519 | 234 | 0,0064 | 1,000 | 0,9530 | 1,0000 | 0,9957 | 1 | 0 | 0,9533 | 1,0000 | 0,00030 | 0,00002 |
| 13 | 689–712 | 29.199 | 278 | 0,0095 | 1,000 | 0,9640 | 1,0000 | 0,9964 | 1 | 0 | 0,9644 | 1,0000 | 0,00034 | 0,00003 |
| 14 | 713–736 | 4.474 | 284 | 0,0635 | 1,000 | 0,9577 | 1,0000 | 0,9965 | 1 | 0 | 0,9604 | 1,0000 | 0,00268 | 0,00022 |
| 15 (parcial) | 737–743 | 80 | 80 | 1,0000 | 1,000 | 0,9375 | 1,0000 | 0,9875 | 1 | 0 | — | — | 0,06250 | 0,01012 |
| **Producción** | 401–743 | 575.590 | 3.736 | 0,0065 | **1,000** | **0,9724** | **0,9997** | **0,9976** | 9 | 1 | 0,9726 | 0,9997 | 0,00018 | 0,00002 |

La ventana 15 tiene una sola clase, y el PR-AUC no está definido.

Sobre las 14 ventanas completas (media ± desviación estándar muestral, ddof = 1):

| Métrica | Modelo | Regla |
|---|---|---|
| Recall | 0,9978 ± 0,0025 (0,9920–1,0000) | 0,9736 ± 0,0173 (0,9440–1,0000) |
| Precisión | 0,9997 ± 0,0010 (0,9961–1,0000) | 1,0000 ± 0 |
| PR-AUC | 0,9997 ± 0,0010 | 0,9741 ± 0,0170 |
| ROC-AUC | 0,9997 ± 0,0010 | 0,9868 ± 0,0087 |
| Brier | 0,000036 ± 0,000060 | 0,000455 ± 0,000700 |
| Falsos positivos | 0,07 ± 0,27 por ventana | 0 |

### Qué significa cada cifra

- **Recall: el modelo no se degradó.** 0,9976 sobre producción (3.727 de 3.736) contra
  0,9970 ± 0,0036 en la validación temporal de referencia (EVAL-009) y 0,9975 in-sample
  (`docs/analysis/interpretability.md`, sección 4). Supera a la regla en 0,0252: 94 fraudes más,
  cinco veces la significancia práctica del proyecto. De los 103 fraudes que la regla no ve, el
  modelo marca 94; ninguno de los que la regla ve se le escapa. En referencia, in-sample,
  marcaba 81 de 92.
- **La precisión subió, y no porque el modelo mejorara.** 0,9997 contra 0,9902 in-sample en
  referencia. Al umbral fijo, los verdaderos positivos siguen a la cantidad de fraudes, que es
  casi constante por step (sección 4), y los falsos positivos siguen al volumen legítimo, que
  cayó 8,7 veces. Con la prevalencia multiplicada por 8,4, la precisión sube por aritmética. Las
  14 ventanas quedan sobre el piso de 0,99; la más baja es la 8, con 0,9961.
- **Los falsos positivos son horas de analista, y aquí casi no hay.** Cada ventana es un día
  simulado. La cola que el modelo manda a revisión tiene entre 232 y 308 transacciones por día,
  261 en promedio, y en 14 días solo una no era fraude: la de la ventana 8. Con *t* minutos por
  revisión, la cola cuesta unos 261·*t* minutos por día, y el trabajo desperdiciado en
  legítimas, *t* minutos en dos semanas. El proyecto no tiene medido *t*. En referencia,
  in-sample, eran 44 falsos positivos en 400 steps, 2,6 por día, sobre un volumen legítimo
  8,7 veces mayor.
- **Uno, donde se esperaban unos cuatro.** Con la tasa de falsos positivos por fila legítima de
  referencia (44 de 5.782.553), las 571.854 legítimas de producción darían 4,35 esperados.
  Observar 1 o menos tiene probabilidad 0,069 bajo Poisson. Es un cálculo sobre cifras medidas,
  no una medición, y con conteos tan chicos no distingue un cambio de tasa del azar.
- **PR-AUC y Brier al lado de la regla.** El PR-AUC del modelo (0,9997) queda arriba del de la
  regla (0,9726) en todas las ventanas: es el mismo hecho que el recall, porque la región de la
  firma lo satura (ADR 0009). El Brier del modelo es diez veces menor que el de la regla (0,00002
  contra 0,00018); los dos están dominados por la mayoría legítima y no miden calibración en la
  cola. Con `scale_pos_weight` 34,9 las probabilidades no están calibradas (ADR 0011), y el Brier
  sale chico porque casi todas las filas son legítimas con puntaje casi cero.

---

## 3 · El criterio de degradación

Declarado antes de medir (ADR 0012): el recall del modelo por ventana completa contra
**0,9934 = 0,9970 − 0,0036**; 2 ventanas consecutivas por debajo son una alerta y 3 son
degradación.

| Ventanas debajo de la línea | Rachas | Estado |
|---|---|---|
| Solo la 2 (248 de 250, 0,9920) | Una de largo 1 | **Sin alerta** |

**Lectura.** Una ventana sola por debajo es ruido, como declaraba el criterio: dos fraudes
perdidos en 250. El modelo pierde a lo sumo 2 fraudes en cualquier día de producción.

**Lo que el criterio podía y no podía ver.** Con los positivos que tuvo cada ventana, si el recall
verdadero fuera 0,9970 en todas, una ventana caería debajo de la línea por azar con probabilidad
entre 0,067 y 0,210; una alerta saldría por azar con probabilidad **0,302**, y una degradación con
**0,054**. Que no haya salido ninguna es compatible con un modelo estable, pero la alerta es un
instrumento ruidoso a esta resolución: con ~260 fraudes por ventana, un fraude perdido mueve el
recall 0,004, más que la dispersión que define la línea.

---

## 4 · Drift

### El objetivo: la prevalencia se multiplicó por 8,4, y es el denominador

| | Filas legítimas por step | Fraudes por step | Prevalencia |
|---|---|---|---|
| Referencia (400 steps) | 14.456 | 11,19 | 0,000774 |
| Producción (343 steps) | 1.667 | 10,89 | 0,006491 |
| Ventanas completas, rango | 175 (v14) a 8.710 (v1) | 9,67 a 12,83 | 0,0015 a 0,0635 |

**El fraude es un proceso de tasa constante, y el volumen legítimo no.** Los fraudes por step casi
no cambian entre periodos ni entre ventanas, mientras las legítimas caen 8,7 veces y varían 50
veces entre ventanas. Toda la subida de la prevalencia sale del denominador: menos legítimas por
el mismo fraude. La ventana 15 lo lleva al extremo: sus 7 steps tienen 80 filas y las 80 son
fraude.

| Hora del día | 9 | 10 | 11 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|
| Legítimas, referencia | 0,047 | 0,070 | 0,075 | 0,085 | 0,100 | 0,082 |
| Legítimas, producción | 0,016 | 0,036 | 0,021 | 0,157 | 0,125 | 0,134 |

Los fraudes se reparten casi uniformes sobre las 24 horas en los dos periodos (entre 0,032 y 0,050
por hora, contra 1/24 = 0,042). Las legítimas, no: en producción el volumen se corre de la mañana
a las 18–20 h. El agente de fraude ignora la hora; la población legítima, no.

### Las 11 features, producción entera contra referencia

El nivel aplica los cortes 0,1 (moderado) y 0,25 (significativo). **Son una convención**, no se
derivaron de estos datos, y tienen un punto ciego medido: la bandera de la firma se multiplica
por 8,3 y su PSI es 0,012, "estable" (al final de esta subsección).

| Feature | Tipo | Bins | PSI | Nivel | KS |
|---|---|---|---|---|---|
| `amount` | continua | 10 | 0,0034 | estable | 0,021 |
| `oldbalanceOrg` | continua | 8 | 0,0055 | estable | 0,032 |
| `oldbalanceDest` | continua | 7 | 0,0031 | estable | 0,023 |
| `type_CASH_IN` | discreta | 3 | 0,0000 | estable | 0,003 |
| `type_CASH_OUT` | discreta | 3 | 0,0027 | estable | 0,025 |
| `type_DEBIT` | discreta | 3 | 0,0002 | estable | 0,001 |
| `type_PAYMENT` | discreta | 3 | 0,0010 | estable | 0,015 |
| `type_TRANSFER` | discreta | 3 | 0,0005 | estable | 0,006 |
| `amount_equals_origin_balance` | discreta | 3 | 0,0118 | estable | 0,006 |
| `hour_sin` | discreta | 22 | **0,3182** | **significativo** | 0,162 |
| `hour_cos` | discreta | 23 | **0,3254** | **significativo** | 0,247 |

Las monetarias tienen menos de 10 bins porque los deciles de referencia repiten el cero, y los
bins duplicados se funden. Los de la hora son sus 21 y 22 valores distintos en punto flotante,
más el de lo no visto.

- **Solo driftan las horas, y es el corrimiento del volumen legítimo de la tabla anterior.**
  Su soporte es idéntico por construcción, así que su drift no es un valor que el modelo no vio:
  es otra forma de repartir las transacciones sobre el día. El modelo apenas usa la hora: su
  permutación cuesta 11,7 y 15,3 fraudes en referencia, por debajo de la significancia práctica
  (`docs/analysis/interpretability.md`, sección 2).
- **Montos, saldos y mezcla de operaciones no se movieron**, ni en PSI ni en KS.
- **El PSI no ve el drift de una categoría rara.** La bandera pasa del 0,076% al 0,631% de las
  filas, 8,3 veces más, y su PSI es 0,0118, "estable". El PSI pesa diferencias absolutas de
  proporción, y una categoría del 0,1% no tiene masa para moverlo. Por eso la prevalencia se mide
  directo, y no a través de esta feature.

### Ventana por ventana

PSI de cada ventana contra toda la referencia:

| Ventana | `amount` | `oldbalanceOrg` | `oldbalanceDest` | bandera | `hour_sin` | `hour_cos` | Tipos, máximo |
|---|---|---|---|---|---|---|---|
| 1 | 0,007 | 0,001 | 0,002 | 0,000 | 2,850 | 3,008 | 0,001 |
| 2 | 0,029 | 0,026 | 0,008 | 0,038 | 3,531 | 3,065 | 0,019 |
| 3 | 0,003 | 0,009 | 0,005 | 0,042 | 2,697 | 2,780 | 0,003 |
| 4 | 0,005 | 0,010 | 0,004 | 0,021 | 2,342 | 2,114 | 0,011 |
| 5 | 0,004 | 0,007 | 0,007 | 0,023 | 3,602 | 3,666 | 0,010 |
| 6 | 0,003 | 0,004 | 0,003 | 0,006 | 3,036 | 3,172 | 0,006 |
| 7 | 0,008 | 0,008 | 0,004 | 0,010 | 1,184 | 1,374 | 0,004 |
| 8 | 0,009 | 0,011 | 0,003 | 0,010 | 1,146 | 1,035 | 0,004 |
| 9 | 0,048 | 0,015 | 0,006 | 0,016 | 2,727 | 2,703 | 0,010 |
| 10 | 0,003 | 0,018 | 0,003 | 0,073 | 3,404 | 3,405 | 0,002 |
| 11 | 0,018 | 0,021 | 0,007 | 0,055 | 3,640 | 3,707 | 0,012 |
| 12 | 0,004 | 0,007 | 0,001 | 0,011 | 2,822 | 2,699 | 0,003 |
| 13 | 0,010 | 0,051 | 0,008 | 0,021 | 2,616 | 2,794 | 0,005 |
| **14** | 0,051 | **2,645** | **0,160** | **0,267** | 3,032 | 3,056 | 0,018 |
| 15 (parcial) | 2,320 | 2,900 | 0,290 | 9,267 | 4,668 | 4,655 | 2,873 |

**El nulo: qué PSI da el puro muestreo a cada tamaño.** Percentil 95 de 20 muestras sacadas de la
propia referencia (semillas 42 a 61):

| Tamaño | Peor monetaria | Bandera | Tipos, máximo | Hora |
|---|---|---|---|---|
| 4.474 filas (la ventana completa más chica) | 0,0032 | 0,0007 | 0,0010 | 0,0062 |
| 80 filas (la ventana parcial) | 0,2566 | 0,0055 | 0,0735 | 0,6788 |

En toda ventana completa el ruido queda al menos 16 veces por debajo del corte de 0,1: ahí la
premisa del ADR 0012 se cumple y el corte no depende del tamaño. **En la ventana parcial no:**
con 80 filas el puro muestreo ya cruza el corte moderado en las monetarias y el significativo en
la hora. Además las 80 filas son fraude, así que su PSI mezcla ruido con un cambio total de la
mezcla de clases. No se interpreta como drift de covariables, y por eso está fuera del criterio.

**La hora driftea en todas las ventanas, de 1,0 a 3,7.** Una ventana es un día, y se compara con
el perfil horario acumulado de 400 steps. El nulo a esos tamaños es ≤ 0,0062, así que no es ruido:
cada día de producción reparte su volumen sobre las horas de forma muy distinta al promedio de
referencia.

**La ventana 14 es el único cambio de covariables, y es de las legítimas, no de la mezcla.** Sus
4.474 filas tienen 6,3% de fraude, y una feature que distingue fraude de legítima driftea con
solo cambiar la mezcla. Para separar las dos cosas, el script compara las legítimas de cada
ventana con las legítimas de referencia:

| Ventana 14 | `amount` | `oldbalanceOrg` | `oldbalanceDest` | bandera |
|---|---|---|---|---|
| PSI, todas las filas | 0,051 | 2,645 | 0,160 | 0,267 |
| PSI, solo legítimas | 0,115 | **3,722** | 0,156 | 0,000 |

- La bandera es pura mezcla: ninguna legítima la tiene, en ningún periodo.
- `oldbalanceOrg` no: driftea entre las legítimas mismas. Lo que se movió está medido:

| Legítimas con saldo cero | Referencia | Ventanas 1–13 | Ventana 14 |
|---|---|---|---|
| `oldbalanceOrg == 0` | 0,334 | 0,245 a 0,334 | **0,000** |
| `oldbalanceDest == 0` | 0,423 | 0,426 a 0,460 | 0,582 |

En la ventana 14 el generador dejó de producir transacciones legítimas desde cuentas vacías, que
en referencia son un tercio, y el cambio es abrupto: la ventana 13 todavía tiene 0,245.

### ¿Coincide el drift con el momento en que el desempeño se mueve?

**No coincide.**

- La única ventana debajo de la línea, la **2**, no tiene ningún drift propio: la hora driftea
  como en todas, y sus monetarias, su bandera y sus tipos quedan debajo de 0,04.
- La ventana con el único drift de covariables, la **14**, no movió el desempeño: recall 0,9965,
  un fraude perdido de 284, y ningún falso positivo. Por qué el modelo no lo siente no está
  medido. La hipótesis: el cambio está en las legítimas, y a los fraudes el modelo los decide
  sobre todo por la bandera (`docs/analysis/interpretability.md`), que entre los fraudes no
  cambió de significado.
- **La coincidencia temporal no es causa.** Si una caída de recall y un drift hubieran caído en
  la misma ventana, eso no habría probado que el drift la causó: los dos pueden venir de un
  tercer factor, como el colapso del volumen legítimo que mueve a la vez la mezcla, el perfil
  horario y la prevalencia, o de la suerte, cuando una ventana cae debajo de la línea por azar
  con probabilidad de hasta 0,21. Separar causa de coincidencia exigiría una intervención, y un
  monitoreo solo observa.

---

## 5 · Evidently y el contraste de instrumentos

El reporte (`reports/deliverables/monitoring_report.html`, 4,2 MB) tiene el drift de las 11
features por PSI y la clasificación al umbral fijo. No carga nada de afuera: el JavaScript va
embebido. Su lado de referencia en clasificación es in-sample, porque el artefacto se ajustó con
esas filas: es contexto, no un baseline.

| | Propio | Evidently | Diferencia |
|---|---|---|---|
| PSI de las 8 features discretas | — | — | ≤ 1,1e-16 |
| PSI de `amount` / `oldbalanceOrg` / `oldbalanceDest` | 0,0034 / 0,0055 / 0,0031 | 0,0059 / 0,0123 / 0,0053 | hasta 0,0068 |
| Precisión y recall del modelo al umbral | 0,999732 / 0,997591 | 0,999732 / 0,997591 | 0 |

Donde los dos instrumentos miden lo mismo con los mismos bins, coinciden al último bit; el
script se detiene si no. En las monetarias difieren por diseño: Evidently arma bins de ancho
igual sobre referencia y producción juntas (ERR-019), y los propios salen solo de referencia. Las
dos lecturas coinciden en el nivel: estable.

---

## 6 · Costo

| Cálculo | Tiempo |
|---|---|
| Carga de los dos periodos | 0,6 s |
| Features y puntajes de referencia / de producción | 4,0 s / 0,4 s |
| Perfiles de referencia de las 11 features | 2,0 s |
| Drift de producción entera / ventana por ventana | 0,9 s / 9,0 s |
| Reporte de Evidently | 68,5 s |

Los tiempos dependen de la carga de la máquina y no son costos de referencia.

---

## 7 · Qué no permite concluir

- **Nada sobre datos reales.** La regla se sostiene porque el agente de fraude del simulador no
  cambia de comportamiento. Un atacante real se adapta.
- **Nada causal** sobre drift y desempeño (sección 4).
- **Las etiquetas llegaron con las filas.** Aquí el monitoreo de desempeño conoce el fraude de
  cada ventana al cerrarla. En una operación real la etiqueta llega tarde, y esta mitad del
  monitoreo iría atrasada; el drift no necesita etiquetas.
- **El costo en horas** de los falsos positivos: el tiempo por revisión no está medido.
- **Los 9 fraudes que el modelo pierde** no se analizaron.
- **El poder del criterio:** con 14 ventanas, una alerta sale por azar con probabilidad 0,30, y
  una degradación real de pocos fraudes por día no se distinguiría del ruido.
- **Más allá del step 743:** la simulación termina ahí.

---

**Nota del Ejecutor sobre las corridas (2026-10-21):** el script corrió tres veces en este turno.
Las corridas `aeee8e748a174686b2c9668d019f07b8` y `e9a7d78d74744e18bdffcb0e089a870d` son
versiones anteriores del mismo script: sus cifras de desempeño, del criterio y del drift
coinciden con las de este documento, y les faltan las métricas de contexto de la regla y las
mediciones de lectura de la sección 4 (las legítimas solas, los saldos en cero, el nulo, el
volumen y el perfil horario). La de registro es `b7eabc74ebc64ae68b05472ff437bb3f`.
