# Fraud Detection MLOps

Detección de fraude en transacciones móviles sobre el dataset sintético **PaySim**, con
optimización de hiperparámetros, interpretabilidad local y global, y monitoreo de desempeño y
de drift sobre un periodo de producción real. Proyecto final del Curso III de la
especialización en Machine Learning Engineering.

**Estado:** entrega parcial; lo que falta está en [Qué falta](#5--qué-falta). Cómo se trabaja y
cómo se verifica cada cifra está en [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

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
horas son la **referencia**: con ellas se entrena y se ajusta todo. De la hora 401 en adelante es
**producción**, y ninguna fila de producción influye en nada: ni en el modelo, ni en el
preprocesamiento, ni en un umbral, ni en un estadístico. Producción todavía no se miró, a
propósito: solo se validó que el registro esté bien formado. Se va a usar una sola vez, para
medir cómo se comporta el modelo y cuánto cambian los datos en un periodo que nunca vio.

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
falsa. El PR-AUC global, la métrica habitual en clases desbalanceadas, no servía aquí: cualquier
modelo que aprenda la firma llega a ~0,98, y entonces no separa a un modelo de otro. También se
fijó antes de medir que una diferencia menor que 0,005 de recall, unos 22 fraudes, no cuenta como
mejora ([ADR 0009](docs/adr/0009-evaluation-design.md)).

---

## 3 · Resultados

La validación sigue el uso real: el modelo entrena con toda la historia disponible y predice las
40 horas siguientes, cinco veces, cubriendo las horas 201 a 400. Cifras como media ± desviación
estándar entre esos cinco folds; todos los modelos con sus hiperparámetros por defecto.

| Modelo | Recall @ precisión ≥ 0,99 | Contra la regla, en los mismos folds¹ |
|---|---|---|
| `SignatureRule` (baseline) | 0,9788 ± 0,0101 | — |
| Regresión logística | 0,9911 ± 0,0028 | +0,0123 ± 0,0090 (+28 fraudes) |
| Random Forest | 0,9983 ± 0,0038 | +0,0195 ± 0,0098 (+44 fraudes) |
| LightGBM (por defecto) | 0 | el entrenamiento diverge |

**Qué significa cada cifra:**

- **La regla.** Sobre toda la referencia, su precisión es 1,000: **toda transacción que marca es
  fraude**, y ninguna alerta le hace perder tiempo a un analista. Su recall es 0,979: se le
  escapan 92 de los 4.477 fraudes. En los folds de validación deja pasar entre 3 y 14 por fold,
  48 en total ([EVAL-005](docs/EVALUATION.md)).
- **Random Forest** detecta 44 fraudes más que la regla, de los 48 que se le escapan, **sin bajar
  de precisión 0,99**: a lo sumo un falso positivo por cada cien alertas. Los 4 que no detecta
  caen todos en el fold 1, el que entrena con menos historia ([EVAL-007](docs/EVALUATION.md)).
  Eso es consistente con que el modelo mejore con la historia acumulada, que es justo lo que la
  validación temporal deja ver; con un único fold que falla, no alcanza para afirmarlo.
- **La regresión logística** detecta 28 fraudes más que la regla, con la misma exigencia de
  precisión. Las dos mejoras superan el umbral de 0,005 fijado antes de medir.
- **LightGBM** con sus valores por defecto no es que rinda poco: el entrenamiento diverge y los
  puntajes crecen sin cota hasta que el modelo deja de distinguir nada (medición ad hoc, sin
  versionar: [ERR-014](docs/ERRORS_AND_LEARNINGS.md)). Con los hiperparámetros que
  encontró la búsqueda llega a 0,9970, y es el modelo elegido
  ([ADR 0010](docs/adr/0010-chosen-configuration.md)).
- **Ignorar el tiempo no resultó más optimista.** Con folds aleatorios estratificados, el recall
  de la regresión logística baja 0,0036 y el de Random Forest, 0,0052¹. El esquema temporal no
  está castigando a los modelos. Por qué el aleatorio da algo menos no está determinado: los dos
  esquemas validan sobre filas distintas (horas 201–400 contra 1–400), y solo el último fold
  temporal entrena con más filas (5,19 M) que un fold aleatorio (4,63 M; la métrica
  `train_rows` de cada corrida en MLflow).

Todas las corridas, con sus métricas por fold, están en MLflow sobre DagsHub, en el experimento
`fraud-detection`:
**https://dagshub.com/diego-Ballesteros/fraud-detection-mlops.mlflow/#/experiments/0**
(que el enlace abra sin credenciales todavía no está verificado). Cada resultado tiene su entrada
completa en el [registro de evaluación](docs/EVALUATION.md).

¹ *Cálculo ad hoc.* La diferencia pareada, su conteo en fraudes y la diferencia entre
esquemas son restas por fold sobre métricas que produce el arnés, que sí tiene tests; la resta
misma no tiene función ni test propios.

---

## 4 · Por qué estas cifras son creíbles

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

El detalle está en [EVAL-001 a EVAL-004](docs/EVALUATION.md) y en el
[ADR 0009](docs/adr/0009-evaluation-design.md).

---

## 5 · Qué falta

| Componente | Peso | Estado |
|---|---|---|
| Repositorio | 10% | Estructura, CI, pre-commit y los datos fuera de git: hecho. Los cuatro notebooks, `01_preprocessing`, `02_training`, `03_interpretability` y `04_monitoring`: hechos. La carpeta `app/` existe, vacía |
| Documentación | 10% | Diccionario de datos, doce ADR, registros de evaluación y de errores, y este README: hecho para lo construido hasta ahora |
| Técnicas avanzadas de ML | 50% | Arnés de evaluación, baselines y registro en MLflow: hecho. Búsqueda de hiperparámetros: hecha, pero incompleta (8 de 150 ensayos, [EVAL-009](docs/EVALUATION.md)). Modelo elegido serializado y registrado en el Model Registry: hecho ([ADR 0010](docs/adr/0010-chosen-configuration.md)). Interpretabilidad local y global: medida sobre el artefacto ([análisis](docs/analysis/interpretability.md), [ADR 0011](docs/adr/0011-interpretability-design.md)), con su narrativa en `notebooks/03_interpretability.ipynb`, y sus figuras adjuntas a la versión 1 del Model Registry |
| Monitoreo | 30% | Desempeño por ventana, criterio de degradación y drift, medidos e interpretados sobre producción, y el reporte HTML de Evidently ([análisis](docs/analysis/monitoring.md), [ADR 0012](docs/adr/0012-monitoring-design.md), [EVAL-011](docs/EVALUATION.md)), y su narrativa en `notebooks/04_monitoring.ipynb` |
| Reto ML 1: AutoML | +10% | Hecho sobre una búsqueda incompleta: el algoritmo es un hiperparámetro categórico entre regresión logística, Random Forest y LightGBM ([EVAL-009](docs/EVALUATION.md)) |
| Reto ML 2: app web de monitoreo | +10% | Pendiente |

---

## 6 · Cómo reproducir

Requisitos: [UV](https://docs.astral.sh/uv/), el único gestor de paquetes del proyecto, y
Python 3.11 (si no está instalado, `uv` lo descarga). Opcional:
[Task](https://taskfile.dev/), cuyas tareas son los mismos comandos `uv run`.

```bash
git clone git@github.com:diego-Ballesteros/fraud-detection-mlops.git
cd fraud-detection-mlops
uv sync --locked                    # las versiones exactas de uv.lock, como en CI
uv run pre-commit install           # los hooks de calidad
```

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

Los resultados se regeneran con dos scripts. Registran en DagsHub si `.env` tiene las
credenciales (plantilla en `.env.example`), y si no, en un store local:

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
uv run python scripts/run_monitoring.py       # producción por ventanas, drift y el HTML
```

---

## 7 · Dónde está cada cosa

- [`docs/ROADMAP.md`](docs/ROADMAP.md): el plan, la estructura del repositorio y la trazabilidad
  contra la rúbrica.
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
  script; hoy, la interpretabilidad del artefacto.
