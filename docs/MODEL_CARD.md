# Model Card — `fraud-detection-pipeline`, versión 1

> Contrato: qué hace el modelo, con qué datos, para qué **no** debe usarse y cuáles son sus
> limitaciones (`docs/METHODOLOGY.md`, sección 4). Cada cifra va con su lectura, su baseline al
> lado y su consecuencia operativa, y donde no se puede interpretar se dice qué dato falta
> (`docs/METHODOLOGY.md`, sección 6.7). Las mediciones de registro viven en
> `docs/EVALUATION.md` y en `docs/analysis/`; este documento las cita y las interpreta.
>
> Escrito el 2026-10-21, al cierre de la fase 5.

---

## 1 · Identificación

| Campo | Valor |
|---|---|
| **Qué es** | Un solo `Pipeline` de scikit-learn: el preprocesamiento (ADR 0008) y un LightGBM (ADR 0010), serializados juntos. El notebook 03, el notebook 04 y los scripts de interpretabilidad y de monitoreo cargan este mismo objeto; ninguno reimplementa una transformación |
| **Registro** | `fraud-detection-pipeline`, **versión 1** del Model Registry de MLflow en DagsHub, registrada desde la corrida `e1fb3e9b12ed4dfc9a025a751eb9ff04`. La versión lleva adjuntas las figuras y tablas de su interpretabilidad (ADR 0011, nota de estado) |
| **Artefacto** | `artifacts/fraud_detection_pipeline.pkl`, versionado en git: 512.224 bytes, sha256 `41869ee0e5239ca66e0024a10bff4b727cf7ba0a6b6a8087a45da010de5d4ebd` (ADR 0010). Los scripts de monitoreo y el notebook 03 se detienen si el sha256 no es este |
| **Algoritmo** | LightGBM con 98 árboles, tasa de aprendizaje 0,0270, 56 hojas, profundidad 7, `min_child_samples` 249, `reg_lambda` 0,0100 y `scale_pos_weight` 34,89: el ensayo 5 de la búsqueda, elegido por la regla de desempate escrita antes de leer resultados (ADR 0010, EVAL-009) |
| **Datos de entrenamiento** | El periodo de referencia de PaySim, steps 1 a 400: 5.787.030 transacciones, 4.477 fraudes (0,0774%). Archivo `data/processed/reference.parquet`, sha256 `b6e1fc95444268334fee3172040333285f2e0a9f68bd18e3e51da6609ec1a5dc`. Ninguna fila de producción (`step >= 401`) intervino en el modelo, el preprocesador, el umbral ni ningún estadístico |
| **Entradas** | Cinco columnas crudas: `step`, `type`, `amount`, `oldbalanceOrg` y `oldbalanceDest`. Todas existen **antes** de la transacción (`docs/DATA_DICTIONARY.md`); el Pipeline deriva de ellas las 11 features del modelo |
| **Salida** | Un puntaje entre 0 y 1. **No es una probabilidad calibrada**: el ajuste pesa cada fraude 34,89 veces (`scale_pos_weight`, ADR 0011). Una transacción se marca con puntaje ≥ **0,42775088540908546**, el umbral fijo (sección 4) |
| **Fechas** | Ajustado y registrado el 2026-10-19 (ADR 0010). Explicado el 2026-10-20 (ADR 0011). Evaluado sobre producción el 2026-10-21 (EVAL-011) |
| **Entorno del que depende el pickle** | Ver abajo |

**El pickle no es portable: depende del entorno bloqueado.** Guarda referencias por nombre, no
código, a clases y funciones de cuatro lugares (leído en los bytes del archivo, con un recorrido
ad hoc no versionado):

- **scikit-learn**, por rutas privadas como `sklearn.compose._column_transformer` y
  `sklearn.preprocessing._encoders`, que cambian entre versiones;
- **LightGBM** (`lightgbm.sklearn`, `lightgbm.basic`);
- **numpy**, por `numpy._core`, que solo existe desde numpy 2;
- **el paquete del proyecto**, `fraud_detection.features.builder` y
  `fraud_detection.features.pipeline`: sin el paquete instalado, el archivo no carga.

Cargarlo exige `uv sync --locked`, que instala Python 3.11 y las versiones exactas de `uv.lock`
—scikit-learn 1.9.1, LightGBM 4.7.0, numpy 2.4.6, pandas 2.3.3— con el paquete en modo editable.
Con esas versiones lo carga un clon limpio del repositorio, sin advertencias, y reproduce a seis
decimales las probabilidades de los cuatro casos de `docs/analysis/interpretability.md` (verificado
el 2026-10-21; README, sección 7). Con otras versiones **no se probó**, y en macOS x86_64 el lock
resuelve numpy 2.3.5, donde tampoco se probó.

---

## 2 · Uso previsto

**Triaje en el momento de la decisión, como insumo de un analista.** Cuando llega una
transacción, el modelo la puntúa con información que existe antes de ejecutarla, y las que
superan el umbral van a una cola de revisión. En producción esa cola tuvo entre 232 y 308
transacciones por día simulado, 261 en promedio, y una sola legítima en 14 días (EVAL-011).

**No es una decisión final automática, y no debe usarse como tal.** Tres razones:

1. **El piso de precisión admite errores** (medido). El umbral se eligió para que, a lo sumo, una
   de cada cien alertas sea legítima. En referencia, *in-sample*, marcó 44 legítimas entre 4.510
   alertas; un bloqueo automático las habría bloqueado sin que nadie las mirara.
2. **El puntaje no es una probabilidad** (por construcción; su calibración no se midió). No dice
   "la chance de que sea fraude es 0,6"; dice que esta transacción está por encima o por debajo de
   la línea que separa la cola.
3. **Casi toda la cola se justifica con un patrón del simulador** (medido). De las 4.510 alertas de
   referencia, 4.385 llevan la firma "se lleva el saldo completo" (`docs/analysis/interpretability.md`,
   sección 4). Un analista puede ver por qué entró cada una: la explicación SHAP de cada fila está
   disponible con el mismo artefacto (notebook 03), y para el 97% de la cola cabe en esa frase.

---

## 3 · Fuera de alcance, medido

**Datos reales, sin revalidación.** Es la limitación que gobierna todas las demás, y su razón
está medida.

La regla `type ∈ {TRANSFER, CASH_OUT} ∧ amount > 0 ∧ amount == oldbalanceOrg` —"la transacción se
lleva el saldo completo de la cuenta"— marca 4.385 transacciones de referencia, **las 4.385 son
fraude**, y no marca ninguna de las 5.782.553 legítimas: precisión 1,000 sobre 5,78 millones de
filas, con cero falsos positivos (ADR 0007). En producción, que nadie miró al escribirla, marca
3.633, las 3.633 fraude, y ninguna de las 571.854 legítimas (EVAL-011). Esa limpieza no es del
fenómeno: es una **firma del simulador**, cuyo agente de fraude toma una cuenta e intenta
vaciarla, y cuya población legítima, en 6,35 millones de transacciones, no vacía una cuenta al
centavo ni una vez. En datos reales habría transferencias legítimas por el saldo completo —cerrar
una cuenta, mover los ahorros— y la regla perdería esa precisión.

**El modelo se apoya en esa firma, así que su desempeño tampoco transfiere.** La bandera de la
firma concentra el 79,8% del |SHAP| medio, y un solo corte sobre ella, en la raíz del primer árbol,
separa 34,92 log-odds (`docs/analysis/interpretability.md`, secciones 1 y 2). Cuando el ICE le
quita la firma a los fraudes que la tienen, el modelo los puntúa en promedio **por debajo del
umbral**: su PDP queda entre −10,56 y −5,72 log-odds, con el umbral en −0,29 (sección 3 de ese
documento). Y al revés: el primer árbol manda toda fila con la firma a una hoja de +27,75
log-odds sin mirar ninguna otra feature, así que, si en datos reales la firma también la
produjeran clientes legítimos, el modelo los mandaría a la cola con el mismo empuje con que hoy
manda los fraudes.

**Cuánto del desempeño es la firma, medido: casi todo.** La ablación del ADR 0007 reentrena el
mismo modelo —los hiperparámetros y la semilla del ensayo 5— sin la bandera ni `oldbalanceOrg`,
por el mismo arnés y los mismos folds (EVAL-013):

| Sobre los 5 folds temporales | Recall con precisión ≥ 0,99 | PR-AUC | Brier |
|---|---|---|---|
| El modelo completo (`full`) | 0,9970 ± 0,0036 | 0,9978 ± 0,0028 | 0,000011 ± 0,000005 |
| La regla | 0,9788 ± 0,0101 | 0,9788 ± 0,0101 | 0,000015 ± 0,000008 |
| **Sin la firma (`no_signature`)** | **0 en los 5 folds** | 0,0737 ± 0,0354 | 0,0049 ± 0,0011 |

- **Sin la firma, el modelo no puede operar.** Ningún umbral alcanza precisión 0,99 en ningún
  fold, y tampoco 0,95, 0,90, 0,80 ni 0,50. No lo alcanza ni *in-sample*, sobre las mismas filas
  con que se ajustó, así que no tiene punto de operación que llevar a producción: su recall ahí
  es 0 por definición.
- **Queda señal, pero no alcanza.** Su PR-AUC es de 69 a 169 veces la prevalencia de cada fold,
  muy por encima del máximo del test de permutación (0,0112, EVAL-004): ordena el fraude por encima
  de lo legítimo en promedio. Pero la cabeza de su ranking está mezclada: lo mejor que logra un
  umbral es atrapar cerca de la mitad de los fraudes revisando de 4 a 13 legítimas por cada uno.
  Su Brier es de 5,6 a 9,3 veces el de un predictor constante: el puntaje inflado por
  `scale_pos_weight` le cuesta más de lo que el ranking le da.
- **Lo que esto vuelve evidencia.** La sección deja de ser un razonamiento sobre la firma: con la
  información de la firma fuera, no queda un detector, queda un ranking que no sirve para una cola
  de revisión. La ablación se lleva también parte de lo que el modelo atrapa fuera de la regla: en
  el caso b y en los 57 fraudes de exactamente 10.000.000, lo que los separa de las 2.900
  legítimas con el mismo monto es el saldo de origen (`docs/analysis/interpretability.md`, nota de
  estado), y `no_signature` no lo tiene.

**Lo que la ablación no mide.** Quita la bandera y el saldo de origen juntos, así que no separa
una del otro. Usa hiperparámetros elegidos con la firma, así que un modelo ajustado sin ella podría
rendir algo mejor. Y no es un experimento con datos reales: allí la firma no desaparecería, dejaría
de ser exclusiva del fraude, y para medir eso hace falta lo que el simulador no tiene,
transacciones reales etiquetadas con transferencias legítimas por el saldo completo.

Otros usos fuera de alcance:

- **Un atacante que se adapta.** El agente de PaySim no cambia de comportamiento cuando lo
  atrapan: la regla mantuvo precisión 1,000 en un periodo nuevo. Un atacante real dejaría de
  vaciar cuentas al centavo, y el monitoreo de este proyecto no tiene con qué decir qué pasaría.
- **Usar el puntaje como riesgo calibrado**, para precios, límites o reservas: no está calibrado
  (sección 4, Brier).
- **Más allá del step 743.** La simulación termina ahí.

---

## 4 · Las métricas, y por qué cada una

Las métricas se fijaron en la fase 2, **antes de medir ningún modelo** (ADR 0009). La regla de la
sección 3 es el baseline de todas: va al lado de cada cifra.

### Recall con precisión ≥ 0,99 — la que decide

- **Qué mide.** Qué parte de los fraudes entra a una cola en la que, a lo sumo, una de cada cien
  alertas es legítima.
- **Por qué esta.** El negocio revisa una cola de capacidad limitada. El piso de precisión fija el
  costo de la cola —no más de 1% de revisiones inútiles— y el recall es lo que se compra con ese
  costo. Las alternativas fallan por los extremos: el recall solo se maximiza marcando todo, y la
  precisión sola la maximiza la regla, que marca poco y nunca se equivoca. Una diferencia menor que
  0,005 no cuenta como mejora: son unos 22 fraudes de referencia (ADR 0009).
- **En producción se lee al umbral fijo.** Elegir el punto de precisión 0,99 con las etiquetas de
  producción sería ajustar el umbral con las filas sobre las que se lo juzga. El umbral es el que
  esta métrica eligió sobre referencia (ADR 0011), y en producción se reportan el recall y la
  precisión que da, con la precisión leída contra el piso de 0,99.
- **Qué significaría que se moviera.** Un día de producción tiene unos 260 fraudes, así que **un
  fraude perdido mueve el recall del día unos 0,004**, y con dos perdidos el día ya cae debajo de
  la línea de degradación, 0,9934. Leído al lado de la regla: si cae el recall del modelo y el de
  la regla no, apareció fraude sin la firma que el modelo no conoce; si cae el de la regla, el
  agente de fraude dejó de vaciar cuentas, y el modelo, que se apoya en la firma, perdería con él
  (sección 3).
- **Consecuencia operativa.** En producción, el modelo pierde 9 de 3.736 fraudes en 14,3 días,
  0,6 por día; la regla pierde 103, 7,2 por día. Los 94 fraudes de diferencia cuestan una revisión
  inútil en dos semanas.

### El precio del siguiente fraude

Al modelo ya no le queda un punto entero de recall por ganar: pierde 9 fraudes en producción,
0,0024. Lo que cuesta cada uno se midió leyendo los puntajes de producción, para cada fraude
perdido, cuántas legítimas agregaría a la cola un umbral bajado hasta él (EVAL-012):

| Fraudes ganados, acumulados | Revisiones inútiles agregadas en 14,3 días | Por día | Precisión |
|---|---|---|---|
| 1 | 18 | 1,3 | 0,9949 |
| 4 | 31 | 2,2 | 0,9915 |
| 5 | 48 | 3,4 | 0,9870 |
| 6 | 103 | 7,2 | 0,9729 |
| 8 | 314 | 22,0 | 0,9222 |
| 9 | 571.831 | 40.012 | 0,0065 |

- **Los cuatro primeros cuestan 31 revisiones en dos semanas** y mantienen la precisión sobre el
  piso. El quinto la rompe, y del sexto en adelante cada fraude cuesta decenas o cientos de
  revisiones. **El noveno no se puede comprar:** su puntaje es 0,0000015, y atraparlo exige mandar a
  revisión casi todo.
- **No es una recomendación de bajar el umbral.** Elegirlo con esta tabla sería ajustarlo con las
  filas de producción. Y el margen es de la mezcla, no del modelo: sobre referencia, *in-sample*,
  el primer fraude de más ya cuesta 6 legítimas y rompe el piso (precisión 0,9889), porque allí hay
  8,7 veces más legítimas por hora.
- **El dato que falta para decidir** es el costo de una revisión frente al de un fraude perdido.
  Con él, esta tabla dice dónde está el umbral que conviene; sin él, solo dice cuánto cuesta cada
  paso.

### Precisión al umbral fijo

- **Qué mide.** Qué parte de las alertas es fraude: uno menos la fracción del trabajo del analista
  que se desperdicia.
- **Por qué se reporta aparte.** Es el costo de la cola, y no es una propiedad del modelo solo: al
  umbral fijo, los verdaderos positivos siguen a la cantidad de fraudes y los falsos positivos
  siguen al volumen legítimo. **Se mueve con la prevalencia aunque el modelo no cambie.**
- **Qué significaría que se moviera.** Por debajo de 0,99, la cola sería más cara de lo declarado.
  Para separar un cambio del modelo de un cambio de mezcla hay que mirar la tasa de falsos
  positivos por transacción legítima, no la precisión.
- **Consecuencia operativa.** En producción, 1 revisión inútil en 14 días. En referencia,
  *in-sample*, 44 en 16,7 días: 2,6 por día.

### PR-AUC — contexto, no decide

- **Qué mide.** La precisión promedio sobre todos los umbrales posibles: la calidad del ranking
  entero, no la de un punto de operación.
- **Por qué no decide.** La firma la satura: cualquier modelo que la aprenda llega a ~0,98, y la
  regla sola da 0,9788 ± 0,0101 en la validación temporal y 0,9726 en producción. Entre dos modelos
  que aprendieron la firma, el PR-AUC casi no distingue nada (ADR 0009).
- **Qué significaría que se moviera.** Una caída del PR-AUC con el recall al umbral intacto diría
  que el ranking empeora lejos del punto de operación: importa solo si alguien quisiera mover el
  umbral. Una caída hasta ~0,97 diría que el modelo ya no ordena mejor que la regla.
- **Consecuencia operativa.** Ninguna directa: no corresponde a ninguna cola.

### Brier — contexto, y no mide calibración aquí

- **Qué mide.** El error cuadrático medio del puntaje contra la etiqueta.
- **Por qué no se lee como calibración.** Está dominado por la mayoría legítima, con puntaje casi
  cero, y el puntaje no está calibrado. La regla, con su 0/1, lo hace transparente: **su Brier en
  producción, 0,000179, es exactamente 103 fraudes perdidos entre 575.590 filas.** El del modelo,
  0,000017, es diez veces menor por la misma razón: sus 9 fraudes perdidos, con puntajes entre
  0,0000015 y 0,14, ya aportan 0,0000138, el 81%, y el resto es el error de las demás filas,
  cuyos puntajes no son exactamente 0 ni 1 (cuenta sobre los puntajes de EVAL-012).
- **Qué significaría que se moviera.** Sube con fraudes perdidos y con legítimas puntuadas alto,
  pero a esta prevalencia también con la mezcla: la ventana parcial, 80 filas y las 80 fraude, da
  0,0101 sin que el modelo cambie.
- **El dato que falta** para hablar de calibración: una curva de confiabilidad en la cola, entre
  las transacciones con puntaje alto. No se midió.

### ROC-AUC — solo comparabilidad

Se reporta porque la literatura lo usa. Con 0,08% de positivos es optimista: una tasa de falsos
positivos de 0,001 sobre 5,78 millones de legítimas son 5.783 revisiones inútiles, y apenas mueve
el área.

### Exactitud — no se usa, y este es el argumento

Un clasificador que responde "nunca es fraude" acierta el **99,92%** de las transacciones de
referencia (5.782.553 de 5.787.030) y no detecta ninguno. En producción acierta el 99,35%. Con las
cifras de EVAL-011 (cuenta, no medición nueva):

| Clasificador, en producción | Exactitud | Fraudes detectados |
|---|---|---|
| "Nunca es fraude" | 99,351% | 0 de 3.736 |
| La regla | 99,982% | 3.633 |
| El modelo | 99,998% | 3.727 |

**No detectar ningún fraude cuesta 0,65 puntos de exactitud, y los 94 fraudes que el modelo detecta
de más que la regla, menos su única falsa alarma, caben en 0,016 puntos.** Una métrica que comprime
todo el problema en la cuarta cifra decimal no puede decidir nada. Por eso aparece aquí una sola
vez.

---

## 5 · Desempeño

Siempre con la regla al lado, sobre las mismas filas.

| Periodo | Qué se mide | Modelo | Regla |
|---|---|---|---|
| **Referencia, validación temporal** (5 folds, steps 201–400, 2.223 fraudes) | Recall con precisión ≥ 0,99, media ± desviación estándar | **0,9970 ± 0,0036** | 0,9788 ± 0,0101 |
| **Producción, agregada** (steps 401–743, 575.590 filas, 3.736 fraudes) | Recall al umbral fijo | **0,9976** (3.727) | 0,9724 (3.633) |
| | Precisión al umbral fijo | **0,9997** (1 legítima) | 1,000 (0) |
| **Producción, por día** (14 ventanas completas de 24 steps) | Recall, media ± desviación (rango) | **0,9978 ± 0,0025** (0,9920–1,0000) | 0,9736 ± 0,0173 (0,9440–1,0000) |
| | Precisión, media ± desviación (rango) | **0,9997 ± 0,0010** (0,9961–1,0000) | 1,0000 ± 0 |

Fuentes: EVAL-009 y EVAL-011; la tabla día por día está en `docs/analysis/monitoring.md`.

**El modelo no se degradó.** Su recall sobre un periodo que nunca vio es el de la validación
temporal: 0,9976 contra 0,9970 ± 0,0036. El peor día perdió 2 fraudes de 250, y ningún día cayó dos
veces seguidas debajo de la línea: el criterio de degradación declarado antes de medir no dio
alerta (sección 9). En la validación temporal perdía 7 de 2.223 fraudes, contra 48 de la regla
(cuenta sobre los recalls por fold de EVAL-009 y EVAL-005).

**La regla fluctúa más que el modelo**, de 0,944 a 1,000 por día contra 0,992 a 1,000: qué fraudes
escapan a la firma cambia de un día a otro, y el modelo los absorbe. Sobre producción, de los 103
fraudes que la regla no ve, el modelo marca 94, y no pierde ninguno de los que la regla ve.

**La precisión subió en producción, y no por mérito del modelo.** Pasó de 0,9902 en referencia
(*in-sample*) a 0,9997. Al umbral fijo, los verdaderos positivos siguen a la cantidad de fraudes,
que es casi constante por step (11,19 en referencia, 10,89 en producción), y los falsos positivos
siguen al volumen legítimo, que cayó 8,7 veces. La prevalencia se multiplicó por 8,4 y la precisión
subió por aritmética. Con la tasa de falsos positivos por legítima de referencia (44 de 5.782.553),
las 571.854 legítimas de producción darían 4,35 esperados; se observó 1, y eso tiene probabilidad
0,069 bajo Poisson: con conteos tan chicos no se distingue una mejora de la suerte
(`docs/analysis/monitoring.md`, sección 2).

---

## 6 · Desempeño por segmento

Sobre producción, por tipo de transacción, el modelo al umbral fijo y la regla sobre las mismas
filas (EVAL-012; la tabla completa, con los conteos, está en `docs/analysis/monitoring.md`, nota
de estado de la fase 5):

| Tipo | Filas | Fraudes | Modelo: precisión | Modelo: recall | Regla: precisión | Regla: recall |
|---|---|---|---|---|---|---|
| `CASH_IN` | 127.917 | 0 | indefinida | indefinido | indefinida | indefinido |
| `CASH_OUT` | 189.404 | 1.868 | 0,9995 | 0,9957 | 1,000 | 0,9936 |
| `DEBIT` | 4.316 | 0 | indefinida | indefinido | indefinida | indefinido |
| `PAYMENT` | 202.484 | 0 | indefinida | indefinido | indefinida | indefinido |
| `TRANSFER` | 51.469 | 1.868 | 1,000 | 0,9995 | 1,000 | 0,9513 |

**En tres de los cinco tipos el modelo nunca se equivoca porque nunca tiene nada que acertar.**
`CASH_IN`, `DEBIT` y `PAYMENT` son el 58% de producción, 334.717 filas sin un solo fraude, y ni el
modelo ni la regla marcan ninguna. Su recall no es 1: no está definido, porque no hay fraudes que
detectar; su precisión tampoco, porque no hay alertas. Una exactitud por tipo les daría 100%, que
es la misma trampa de la sección 4 en otro lugar. Lo único que dicen es que ahí no hubo falsas
alarmas, y que el modelo no sabe nada de cómo sería un fraude en esos tipos: nunca vio uno.

**Lo que el agregado promedia son dos tipos que no se parecen.** En `TRANSFER` la regla pierde 91
fraudes y el modelo recupera 90: el recall pasa de 0,9513 a 0,9995, +0,0482, unas diez veces la
significancia práctica. Que esos 90 tengan el perfil del caso b de la interpretabilidad —una
transferencia grande que no vacía la cuenta— es plausible y no está medido. En `CASH_OUT` la regla
pierde 12 y el modelo
recupera 4: +0,0021, por debajo de la significancia práctica y dentro de la dispersión entre días
(0,9961 ± 0,0051 contra 0,9940 ± 0,0061). **En `CASH_OUT`, el modelo y la regla son en la práctica
el mismo detector**, y ahí están 8 de los 9 fraudes que el modelo pierde y su única falsa alarma.

**Consecuencia operativa.** El valor que el modelo agrega sobre la regla está en las
transferencias: unos 6 fraudes por día que la regla no ve. El riesgo residual está en los retiros
fuera de la firma: 8 perdidos en 14 días. Por qué el modelo no ve esos retiros no está medido; el
dato que falta es el análisis de esos 8 fraudes, que no se hizo.

---

## 7 · Equidad

**No aplica un análisis de equidad, porque los datos no contienen las variables que lo harían
posible.** PaySim no tiene atributos protegidos: ni edad, ni género, ni geografía, ni ningún
otro rasgo de las personas. Sus 11 columnas describen la transacción —tipo, monto, saldos, hora— y
los dos identificadores de cuenta son códigos sintéticos (`docs/DATA_DICTIONARY.md`). No se omitió
el análisis: no hay contra qué hacerlo. La única segmentación posible, por tipo de transacción
(sección 6), es una propiedad de la operación y no de quien la hace.

**Un sistema real con esos datos sí lo requeriría.** Un detector de fraude decide a quién se le
frena un pago, y sus errores pueden caer de forma desigual sobre grupos de clientes. Además, aun
sin atributos protegidos en el modelo, una feature como el saldo puede correlacionar con el
ingreso y funcionar como sustituto de uno: sin los atributos para medirlo, eso tampoco se puede
descartar aquí.

---

## 8 · Límites, todos medidos

- **Los fraudes que ni la regla ni el modelo marcan.** En referencia, *in-sample*, la regla deja
  fuera 92 fraudes y el modelo marca 81: quedan **11** sin marcar
  (`docs/analysis/interpretability.md`, sección 4). En producción la regla deja fuera 103 y el
  modelo marca 94: quedan **9**, todos fuera de la regla (EVAL-011): 8 `CASH_OUT` y 1 `TRANSFER`
  (EVAL-012). Cuánto costaría atraparlos está medido (sección 4, "el precio del siguiente
  fraude"); quiénes son y por qué el modelo no los ve, no: no se analizaron.
- **El umbral se eligió *in-sample*.** Es el punto de precisión ≥ 0,99 sobre los puntajes del
  artefacto en las mismas 5.787.030 filas con que se ajustó (ADR 0011), así que las cifras de
  referencia en ese punto —precisión 0,990244, recall 0,997543— son optimistas. Las de producción
  no lo son: el umbral se aplicó fijo a un periodo que no participó en elegirlo.
- **Los dos puntos ciegos del PSI** (`docs/METHODOLOGY.md`, sección 7.7). En una ventana chica el
  PSI es ruido: con 80 filas, el muestreo puro llega a 0,68, más que el corte "significativo". Y no
  ve el drift de una categoría rara: la bandera de la firma se multiplicó por 8,3 en producción y
  su PSI fue 0,012, "estable". Por eso la prevalencia se mide directo
  (`docs/analysis/monitoring.md`, sección 4).
- **La búsqueda se cortó en 8 de 150 ensayos**, por presión de memoria. Cuatro configuraciones
  empataron dentro de 0,005, y el resultado es de una búsqueda incompleta: que una completa no
  encontrara algo mejor no está medido (EVAL-009, ADR 0010).
- **La importancia por permutación** de la bandera, `amount` y `oldbalanceOrg` incluye filas fuera
  del dominio del modelo, y cuánto pesan en la caída no está medido (ADR 0011).
- **Las etiquetas llegaron con las filas.** El monitoreo de desempeño conoce el fraude de cada día
  al cerrarlo; en una operación real la etiqueta llega tarde, y esa mitad del monitoreo iría
  atrasada.
- **El criterio de degradación tiene poco poder:** con estas ventanas, una alerta saldría por azar
  con probabilidad 0,30 (sección 9).

**Lo que no se midió**, y por eso no se afirma:

- **El tiempo por revisión.** Las alertas por día están medidas; las horas de analista, no.
- **El dinero de los fraudes no detectados.** El monto de cada transacción está en los datos, y la
  suma de los fraudes perdidos no se calculó. Aun calculada, el monto no es la pérdida: el
  simulador no registra recuperaciones ni contracargos.
- **La frecuencia de reentrenamiento.** Hay un criterio que lo dispara (sección 9), no un
  calendario.
- **La generalización a datos reales** (sección 3).

---

## 9 · Mantenimiento

**Qué dispara un reentrenamiento: el criterio de degradación de la fase 4**, declarado el
2026-10-21 antes de leer una sola etiqueta de producción (ADR 0012, implementado en
`monitoring/performance.py`):

- Se mide el recall del modelo al umbral fijo en cada ventana completa de 24 steps, un día.
- La línea es **0,9934 = 0,9970 − 0,0036**: la media y la dispersión de la métrica de decisión en
  la validación temporal (EVAL-009, EVAL-010). Una ventana está debajo cuando su recall es
  estrictamente menor, comparado en enteros.
- **2 ventanas consecutivas debajo son una alerta; 3 son degradación.** Una ventana sola es ruido.
- **La degradación dispara el reentrenamiento; la alerta es un aviso que se investiga**, y el
  drift no dispara nada (ADR 0013). La razón está medida: con un modelo estable, en 14 días una
  ventana sola caería debajo por azar con probabilidad 0,931, una alerta con 0,302 y una
  degradación con 0,054. La ventana 2 de producción cayó debajo sin ningún drift propio.

**Cómo leerlo.** En producción solo la ventana 2 cayó debajo (248 de 250): sin alerta. A esta
resolución el criterio es ruidoso: si el recall verdadero fuera 0,9970 todos los días, una ventana
caería debajo por azar con probabilidad de 0,067 a 0,210, una alerta saldría con 0,302 y una
degradación con 0,054 (`docs/analysis/monitoring.md`, sección 3). Con un modelo estable, una alerta
saldría por azar casi una vez de cada tres periodos como este: una alerta sola no alcanza para
afirmar que el modelo se degradó, y por eso es aviso y no disparador.

**El drift no es disparador.** En producción la hora del día tuvo drift "significativo", PSI 0,32
y 0,33, y el recall no se movió: el modelo apenas usa la hora (su permutación cuesta 11,7 y 15,3
fraudes, debajo de la significancia práctica). La ventana 14 tuvo el único cambio de covariables
entre las legítimas, y perdió un fraude de 284. Que drift y desempeño coincidan en el tiempo no
sería causa (ADR 0012).

**Qué vigilar al lado: la regla, por sus dos cifras.** Su recall dice si el agente de fraude sigue
vaciando cuentas: en producción fue de 0,944 a 1,000 por día. Su precisión dice si alguna
legítima empezó a hacerlo: fue 1,000 todos los días. Si cae la primera, el modelo pierde el apoyo
de la firma (sección 3); si cae la segunda, la firma dejó de ser exclusiva del fraude. En los dos
casos este documento pierde su premisa. Ninguna de las dos tiene un criterio declarado: es
vigilancia, no un disparador.

**Cómo se reentrena:** `uv run python scripts/register_model.py` ajusta la configuración elegida
sobre todo el periodo de referencia, mide su estabilidad con tres semillas, la serializa y registra
una versión nueva. Un reentrenamiento real tendría que redefinir antes la referencia, recalcular el
umbral sobre ella y no sobre las filas que se van a monitorear, y redeclarar la línea con la nueva
validación temporal.
