# Fraud Detection MLOps

Detección de fraude en transacciones móviles sobre el dataset sintético **PaySim**, con
optimización de hiperparámetros, interpretabilidad local y global, y monitoreo de desempeño y
de drift sobre un periodo de producción real. Proyecto final del Curso III de la
especialización en Machine Learning Engineering.

**Estado:** lo obligatorio de la rúbrica y el Reto 1 están hechos; el Reto 2, la app web de
monitoreo, no se hizo ([estado contra la rúbrica](#6--estado-contra-la-rúbrica)). Qué hace el
modelo, para qué no sirve y cuáles son sus límites está en el
[Model Card](docs/MODEL_CARD.md). Cómo se trabaja y cómo se verifica cada cifra, en
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## 1 · El problema

Una plataforma de pagos móviles procesa millones de transacciones y tiene que decidir, en el
momento, cuáles bloquear o mandar a revisión. Los dos errores no cuestan lo mismo: un **falso
positivo** cuesta el tiempo de un analista que revisa una transacción legítima, y un **falso
negativo** cuesta el monto de la transacción fraudulenta que pasó.

El fraude es rarísimo. En el periodo de referencia hay 4.477 fraudes entre 5.787.030
transacciones, el **0,0774%**: de cada 10.000 transacciones, unas 8 son fraude. Un modelo que
responda "nunca es fraude" acierta el 99,92% de las veces y no detecta ninguno. Por eso en este
proyecto **la exactitud no se usa para evaluar ningún modelo**: con esta prevalencia, un número
alto de exactitud no distingue un detector de uno que no hace nada.

---

## 2 · La estrategia

**El tiempo se respeta.** Las transacciones están ordenadas por hora (`step`). Las primeras 400
horas son la **referencia**: con ellas se entrena y se ajusta todo. De la hora 401 a la 743 es
**producción**, y ninguna fila de producción influye en nada: ni en el modelo, ni en el
preprocesamiento, ni en un umbral, ni en un estadístico. Producción se leyó una sola vez, en la
fase 4, con el modelo, el preprocesamiento y el umbral ya fijos, para medir cómo se comporta el
modelo y cuánto cambian los datos en un periodo que nunca vio.

**La regla obvia es la vara.** La regla

```
type ∈ {TRANSFER, CASH_OUT}  ∧  amount > 0  ∧  amount == oldbalanceOrg
```

("la transacción se lleva el saldo completo") marca 4.385 transacciones de referencia, **las
4.385 son fraude**, y no marca ninguna de las 5.782.553 legítimas. No es una fuga de
información: el monto y el saldo existen antes de la transacción. Es una firma del simulador,
cuyo agente de fraude toma una cuenta e intenta vaciarla; en datos reales habría transferencias
legítimas por el saldo completo. Como la regla ya resuelve casi todo, es el **baseline
obligatorio** de todo el proyecto, y el problema de machine learning es lo que deja fuera: 92
fraudes ([ADR 0007](docs/adr/0007-simulator-signature-baseline.md)).

**La métrica se eligió antes de medir.** La métrica de decisión es el **recall con precisión ≥
0,99**: cuántos fraudes se detectan si se exige que, de cada 100 alertas, a lo sumo una sea
falsa. También se fijó antes de medir que una diferencia menor que 0,005 de recall, unos 22
fraudes, no cuenta como mejora ([ADR 0009](docs/adr/0009-evaluation-design.md)). Cómo leer esta
y las demás cifras está en la [sección 4](#4--cómo-leer-estas-métricas).

---

## 3 · Resultados

### En producción: 14 días que el modelo nunca vio

El modelo elegido es un LightGBM ([ADR 0010](docs/adr/0010-chosen-configuration.md)), con el
umbral fijado sobre referencia ([ADR 0011](docs/adr/0011-interpretability-design.md)), aplicado
sin cambios a los steps 401 a 743: 575.590 transacciones, 3.736 fraudes
([EVAL-011](docs/EVALUATION.md)).

| Al umbral fijo | Modelo | La regla (baseline) |
|---|---|---|
| **Recall** | **0,9976**: 3.727 de 3.736 | 0,9724: 3.633 |
| **Precisión** | **0,9997**: una alerta legítima en 14 días | 1,000: ninguna |
| Recall por día, media ± desviación estándar (14 días) | 0,9978 ± 0,0025 | 0,9736 ± 0,0173 |
| Fraudes perdidos | 9 | 103 |
| Alertas por día | 261 en promedio (232 a 308) | — |
| Criterio de degradación, declarado antes de medir | Sin alerta | — |

**Qué significa:**

- **El modelo no se degradó.** Su recall fuera de muestra, 0,9976, es el de la validación
  temporal sobre referencia, 0,9970 ± 0,0036. El peor día perdió 2 fraudes de 250.
- **Detecta 94 de los 103 fraudes que la regla deja pasar**, y no pierde ninguno de los que la
  regla ve, a cambio de una sola revisión inútil en dos semanas.
- **La precisión subió respecto de referencia (0,9902, *in-sample*), y no porque el modelo
  mejorara.** La prevalencia de producción es 8,4 veces la de referencia: el fraude por hora es
  casi el mismo y el volumen legítimo cayó 8,7 veces. Al umbral fijo, menos legítimas son menos
  falsas alarmas.
- **La regla sigue con precisión 1,000 en un periodo que nadie miró al escribirla.** La firma es
  del agente de fraude del simulador, no del periodo, y por eso nada de esto transfiere a datos
  reales sin revalidación ([Model Card](docs/MODEL_CARD.md), sección 3).

El detalle día por día, el drift y su interpretación están en
[`docs/analysis/monitoring.md`](docs/analysis/monitoring.md) y en el notebook 04.

### Cómo se eligió: la validación temporal sobre referencia

El modelo entrena con toda la historia disponible y predice las 40 horas siguientes, cinco
veces, cubriendo las horas 201 a 400. Cifras como media ± desviación estándar entre esos cinco
folds.

| Modelo | Recall @ precisión ≥ 0,99 | Contra la regla, en los mismos folds¹ |
|---|---|---|
| `SignatureRule` (baseline) | 0,9788 ± 0,0101 | — |
| Regresión logística, por defecto | 0,9911 ± 0,0028 | +0,0123 ± 0,0090 (+28 fraudes) |
| Random Forest, por defecto | 0,9983 ± 0,0038 | +0,0195 ± 0,0098 (+44 fraudes) |
| LightGBM, por defecto | 0 | el entrenamiento diverge |
| **LightGBM, elegido por la búsqueda** | **0,9970 ± 0,0036** | +0,0182 ± 0,0082 (+41 fraudes) |

- **Cuatro configuraciones empataron** a menos de 0,005 del mejor punto, y la regla de desempate,
  escrita antes de leer resultados, eligió la más barata de ajustar: LightGBM, 59 s contra 625–960
  s de los Random Forest. La búsqueda se cortó en 8 de 150 ensayos por presión de memoria
  ([EVAL-009](docs/EVALUATION.md)).
- **LightGBM con sus valores por defecto** no es que rinda poco: el entrenamiento diverge y los
  puntajes crecen sin cota (medición ad hoc, sin versionar:
  [ERR-014](docs/ERRORS_AND_LEARNINGS.md)). Con los hiperparámetros de la búsqueda llega a 0,9970.
- **Ignorar el tiempo no resultó más optimista.** Con folds aleatorios estratificados, el recall
  de la regresión logística baja 0,0036 y el de Random Forest, 0,0052¹.

Todas las corridas, con sus métricas por fold, están en MLflow sobre DagsHub, en el experimento
`fraud-detection`:
**https://dagshub.com/diego-Ballesteros/fraud-detection-mlops.mlflow/#/experiments/0**
(que el enlace abra sin credenciales todavía no está verificado). Cada resultado tiene su entrada
completa en el [registro de evaluación](docs/EVALUATION.md).

¹ *Cálculo ad hoc.* La diferencia pareada, su conteo en fraudes y la diferencia entre
esquemas son restas por fold sobre métricas que produce el arnés, que sí tiene tests; la resta
misma no tiene función ni test propios. Los 41 fraudes del LightGBM elegido son 48 de la regla
menos 7 del modelo, contados sobre los recalls por fold de EVAL-005 y EVAL-009.

---

## 4 · Cómo leer estas métricas

**Una métrica decide: el recall con precisión ≥ 0,99.** Un equipo de analistas revisa una cola
de tamaño limitado. El piso de precisión fija cuánto trabajo se tira —a lo sumo una revisión de
cada cien— y el recall dice cuántos fraudes se compran con ese costo. Ni el recall solo ni la
precisión sola sirven: el primero se maximiza marcando todo, y la segunda la maximiza la regla,
que marca poco y nunca se equivoca.

**En producción se lee al umbral fijo.** El umbral se eligió sobre referencia. Elegirlo otra vez
con las etiquetas de producción sería ajustarlo con las filas sobre las que después se lo juzga.
Por eso en producción se reportan el recall y la precisión que da ese umbral, y la precisión se
lee contra el piso de 0,99.

**Un fraude perdido en un día mueve el recall unos 0,004.** Un día de producción tiene unos 260
fraudes. Dos perdidos el mismo día lo dejan debajo de la línea de degradación, 0,9934, y el
criterio pide dos días seguidos para una alerta y tres para declarar degradación.

**La precisión no es solo del modelo.** Al umbral fijo sube y baja con la mezcla: si el volumen
legítimo cae, caen las falsas alarmas aunque el modelo sea el mismo. Por eso la de producción,
0,9997, no es una mejora.

**El agregado promedia dos tipos que no se parecen.** El fraude solo ocurre en `TRANSFER` y
`CASH_OUT`. En `TRANSFER` el modelo recupera 90 de los 91 fraudes que la regla pierde; en
`CASH_OUT`, 4 de 12, una diferencia por debajo de la significancia práctica. En los otros tres
tipos, el 58% de producción, no hay ningún fraude: ahí el modelo no se equivoca porque no tiene
nada que acertar, y su recall no es perfecto sino indefinido ([EVAL-012](docs/EVALUATION.md)).

**Qué cuesta el siguiente punto de recall.** Ya no queda un punto entero: el modelo pierde 9
fraudes, 0,0024 de recall. Leído sobre los puntajes de producción, sin aplicarlo, bajar el umbral
para atrapar los cuatro primeros habría costado 31 revisiones inútiles en dos semanas; el quinto
lleva la precisión debajo de 0,99, el sexto cuesta 103 revisiones y el noveno exige revisar casi
todo. Ese margen es de la mezcla y no del modelo: sobre referencia, el primer fraude de más ya
rompe el piso. Para decidir si conviene pagar ese precio falta un dato que el proyecto no tiene:
cuánto cuesta una revisión frente a un fraude perdido.

**Las otras métricas son contexto.** El PR-AUC está saturado por la firma: la regla sola da 0,97.
El Brier está dominado por la mayoría legítima y no mide calibración: el de la regla es exactamente
sus fraudes perdidos divididos por las filas. Y la exactitud no distingue nada: responder "nunca es
fraude" da 99,35% en producción, y el modelo, 99,998%.

**Qué no se puede afirmar:**

- **Nada sobre datos reales.** El modelo se apoya en una firma del simulador que en datos reales
  también producirían clientes legítimos —cerrar una cuenta es vaciarla—, y cuánto del desempeño
  es esa firma no está medido: la ablación sin ella, declarada en el ADR 0007, no se corrió.
- **Nada sobre dinero ni horas.** El tiempo por revisión no se midió, y el monto de los fraudes
  perdidos no se sumó.
- **Nada causal entre drift y desempeño.** La hora del día driftea en todos los días de producción y
  el recall no se movió; aun si hubieran coincidido, coincidir no es causar.
- **Nada más allá del step 743**, donde termina la simulación.

El detalle de cada métrica —qué mide, por qué esa, qué significaría que se moviera— está en el
[Model Card](docs/MODEL_CARD.md), sección 4.

---

## 5 · Por qué estas cifras son creíbles

Un recall de 0,998 en fraude es el tipo de número que, casi siempre, delata una fuga: alguna
columna que lleva la respuesta. Para que estas cifras estuvieran mal, el modelo tendría que estar
viendo la etiqueta por algún camino. Se midió que no:

- **Test de permutación.** Se barajan las etiquetas de entrenamiento y se reentrena por el
  mismo camino: 20 veces, con semillas distintas, para la regresión logística, y una vez para
  Random Forest. Si hubiera fuga, el modelo seguiría acertando. La regresión logística cae de
  0,9967 a un máximo de 0,005 de PR-AUC, y Random Forest, de 0,9983 a 0,0008: lo que aprenden
  sale de las etiquetas reales.
- **El piso del azar.** Un ranking al azar sobre las mismas validaciones da como máximo 1,5 veces
  la prevalencia en 100 sorteos. Es la referencia contra la que se lee el test de permutación.
- **Sin solapamiento temporal.** Un test afirma que ninguna hora aparece a la vez en
  entrenamiento y en validación del mismo fold; el mismo test, aplicado a un divisor por filas,
  falla. Otro afirma que el predictor constante da exactamente la prevalencia de cada fold.
- **Un periodo que nadie miró.** El modelo rinde en producción lo mismo que en la validación
  temporal, con el umbral fijado antes.

El detalle está en [EVAL-001 a EVAL-004](docs/EVALUATION.md) y en el
[ADR 0009](docs/adr/0009-evaluation-design.md).

---

## 6 · Estado contra la rúbrica

| Componente | Peso | Estado |
|---|---|---|
| Repositorio | 10% | **Hecho.** Estructura, CI, pre-commit y los datos fuera de git. Los cuatro notebooks: [`01_preprocessing`](notebooks/01_preprocessing.ipynb), [`02_training`](notebooks/02_training.ipynb), [`03_interpretability`](notebooks/03_interpretability.ipynb) y [`04_monitoring`](notebooks/04_monitoring.ipynb). La carpeta `app/` existe y está vacía: es la del Reto 2 |
| Documentación | 10% | **Hecho.** Diccionario de datos, [Model Card](docs/MODEL_CARD.md), doce ADR, registros de evaluación y de errores, dos análisis de registro y este README |
| Técnicas avanzadas de ML | 50% | **Hecho.** Búsqueda de hiperparámetros con el registro en MLflow, sobre una búsqueda incompleta: 8 de 150 ensayos ([EVAL-009](docs/EVALUATION.md)). Modelo elegido serializado y registrado en el Model Registry ([ADR 0010](docs/adr/0010-chosen-configuration.md)). Interpretabilidad local y global sobre el artefacto ([análisis](docs/analysis/interpretability.md), [ADR 0011](docs/adr/0011-interpretability-design.md)), con sus figuras adjuntas a la versión 1 del Model Registry |
| Monitoreo | 30% | **Hecho.** Desempeño por día con el criterio de degradación, drift por PSI y KS, su interpretación, y el reporte HTML de Evidently en [`reports/deliverables/`](reports/deliverables/) ([análisis](docs/analysis/monitoring.md), [ADR 0012](docs/adr/0012-monitoring-design.md), [EVAL-011](docs/EVALUATION.md)) |
| Reto ML 1: AutoML | +10% | **Hecho, sobre una búsqueda incompleta.** El algoritmo es un hiperparámetro categórico entre regresión logística, Random Forest y LightGBM ([EVAL-009](docs/EVALUATION.md)). XGBoost, que el plan listaba, no entró |
| Reto ML 2: app web de monitoreo | +10% | **No hecho** |

---

## 7 · Cómo reproducir

Requisitos: [UV](https://docs.astral.sh/uv/), el único gestor de paquetes del proyecto, y
Python 3.11 (si no está instalado, `uv` lo descarga). Opcional:
[Task](https://taskfile.dev/), cuyas tareas son los mismos comandos `uv run`.

```bash
git clone git@github.com:diego-Ballesteros/fraud-detection-mlops.git
cd fraud-detection-mlops
uv sync --locked                    # las versiones exactas de uv.lock, como en CI
uv run pre-commit install           # los hooks de calidad
```

**Sin los datos** se puede instalar, correr la suite, cargar el modelo y leer todo lo producido.
Verificado el 2026-10-21 en un clon limpio, sin datos ni `.env`, siguiendo solo esta sección:

- `uv sync --locked` instala el entorno exacto del lock.
- `uv run pytest` pasa con 2 tests saltados, los únicos que necesitan el CSV o los parquet.
- El artefacto versionado, `artifacts/fraud_detection_pipeline.pkl`, carga con
  `fraud_detection.models.registry.load_pipeline` y reproduce las probabilidades de los cuatro
  casos de `docs/analysis/interpretability.md`.
- Los notebooks se leen con sus salidas guardadas, y el reporte de Evidently está versionado en
  `reports/deliverables/`.

Lo que no corre sin los datos: `prepare_data.py`, los scripts de resultados, interpretabilidad y
monitoreo, y la ejecución de los notebooks. Cada script sale con código 1 y un
`FileNotFoundError` que nombra el archivo que falta, sin escribir nada en MLflow.

**Los datos no están versionados.** El CSV es el de
[Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
en Kaggle. Se descarga a mano y se coloca como
`data/raw/PS_20174392719_1491204439457_log.csv`.

```bash
uv run python scripts/prepare_data.py   # valida el CSV, escribe los dos parquet y mide
uv run pytest                           # la suite completa, con cobertura
task notebooks                          # solo local: necesita data/, que CI no tiene
```

`task notebooks` ejecuta cada notebook desde un kernel limpio, con el modo UTF-8 de Python
activado ([ERR-010](docs/ERRORS_AND_LEARNINGS.md)).

Los resultados se regeneran con scripts. Registran en DagsHub si `.env` tiene las credenciales
(plantilla en `.env.example`), y si no, en un store local:

```bash
uv run python scripts/run_leakage_check.py    # controles: azar y test de permutación
uv run python scripts/run_default_models.py   # los modelos, por los dos esquemas
```

La interpretabilidad se mide a partir del artefacto versionado y del parquet de referencia, y
escribe figuras y tablas en `reports/interpretability/`. Su último paso las adjunta a la versión 1
del Model Registry en DagsHub, y necesita las credenciales de `.env`: sin ellas, o sin red, los
archivos quedan escritos y el script sale con código 1, porque un store local no tiene versión 1.

```bash
uv run python scripts/run_interpretability.py # la compuerta de SHAP, lo global, lo local y el registro
```

El monitoreo lee los dos parquet y el artefacto, sin reajustar nada. Escribe sus tablas en
`reports/monitoring/` y el reporte de Evidently en `reports/deliverables/`, y registra la corrida
en MLflow como los demás:

```bash
uv run python scripts/run_monitoring.py       # producción por día y por tipo, drift y el HTML
```

---

## 8 · Dónde está cada cosa

- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md): qué hace el modelo, con qué datos, para qué no
  debe usarse, sus límites medidos y qué dispara un reentrenamiento.
- Los cuatro notebooks, la narrativa de cada fase:
  [`01_preprocessing`](notebooks/01_preprocessing.ipynb) (los datos y la fuga),
  [`02_training`](notebooks/02_training.ipynb) (la evaluación y la búsqueda),
  [`03_interpretability`](notebooks/03_interpretability.ipynb) (qué aprendió el modelo) y
  [`04_monitoring`](notebooks/04_monitoring.ipynb) (producción y drift).
- [`docs/ROADMAP.md`](docs/ROADMAP.md): el plan, su estado por fase, la estructura del
  repositorio y la trazabilidad contra la rúbrica.
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): cómo se trabaja y cómo se verifica.
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md): las columnas, cuándo existe cada una,
  las features derivadas y las columnas prohibidas.
- [`docs/EVALUATION.md`](docs/EVALUATION.md): el contrato de evaluación y cada resultado con su
  baseline.
- [`docs/ERRORS_AND_LEARNINGS.md`](docs/ERRORS_AND_LEARNINGS.md): cada error real, con su
  mecanismo.
- [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md): las dependencias y la procedencia de cada
  piso de versión.
- [`docs/adr/`](docs/adr/): una entrada por decisión, con su razón.
- [`docs/analysis/`](docs/analysis/): mediciones de registro, cada una reproducible por un
  script: la interpretabilidad del artefacto y su monitoreo sobre producción.
