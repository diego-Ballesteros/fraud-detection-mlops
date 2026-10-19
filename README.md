# Fraud Detection MLOps

Detección de fraude en transacciones móviles sobre el dataset sintético **PaySim**, con
optimización de hiperparámetros, interpretabilidad local y global, y monitoreo de desempeño y
de drift sobre un periodo de producción real. Es el proyecto final del Curso III de la
especialización en Machine Learning Engineering.

El problema tiene un desbalance extremo: en el periodo de referencia hay 4.477 fraudes entre
5.787.030 transacciones, el 0,0774%. Con esa prevalencia, la accuracy de un clasificador que
nunca marca fraude supera el 99,9% y no dice nada. Por eso cada métrica de este repositorio se
reporta con su baseline al lado, con su dispersión entre particiones, y con la métrica
declarada **antes** de medir.

**Estado:** entrega parcial. Las fases 0 (fundación) y 1 (datos y EDA) están cerradas; la
fase 2 (modelado) tiene el arnés de evaluación y los baselines medidos, y le falta la búsqueda
de hiperparámetros. El detalle está en [Estado contra la rúbrica](#estado-contra-la-rúbrica).

---

## La regla dura del proyecto: el corte temporal

```
step <= 400   →  referencia: entrenamiento y ajuste de todo
step >= 401   →  producción
```

**Ninguna fila de producción influye en el ajuste de nada:** ni del modelo, ni del
preprocesador, ni de un umbral, ni de un estadístico. Los dos periodos se escriben en archivos
parquet separados, así que leer producción exige nombrarla y queda a la vista en el código.

Producción **todavía no se midió, a propósito.** Sobre ella solo se validaron propiedades del
registro (dominios, nulos, partición), con chequeos que responden verdadero o falso y nunca un
conteo, y se contaron sus filas desde el pie del archivo parquet, sin leer ninguna. La prevalencia, las distribuciones y el desempeño en producción se miden en la fase 4,
la de monitoreo.

---

## El hallazgo central: la firma del simulador

La regla

```
type ∈ {TRANSFER, CASH_OUT}  ∧  amount > 0  ∧  amount == oldbalanceOrg
```

dispara en **4.385** filas de referencia, y **las 4.385 son fraude**, con **cero falsos
positivos** en las 5.782.553 filas legítimas: precisión 1,000 y recall 0,979. Deja fuera 92
fraudes.

**No es fuga.** Las dos columnas existen antes de la transacción: el monto pedido y el saldo
propio de quien la pide. Es una **firma determinista del generador**: el agente de fraude de
PaySim toma una cuenta e intenta vaciarla. En datos reales habría transferencias legítimas
por el saldo completo, y la regla no tendría esta precisión.

Por eso **la regla es el baseline obligatorio de todo el proyecto**, y el problema de machine
learning son los 92 fraudes que no cubre. La decisión y sus consecuencias están en el
[ADR 0007](docs/adr/0007-simulator-signature-baseline.md); la medición, en el
[diccionario de datos](docs/DATA_DICTIONARY.md), sección 4.a.

---

## Resultados actuales

Recall con precisión ≥ 0,99, sobre los 5 folds temporales (cada fold valida 40 steps y entrena
con toda la historia anterior: steps 201–400 validados en total), media ± desviación estándar
muestral entre folds. Todos los modelos con hiperparámetros por defecto y la configuración de
features `full`.

| Modelo | Recall @ precisión ≥ 0,99 | Contra la regla, en los mismos folds¹ |
|---|---|---|
| `SignatureRule` (baseline) | 0,9788 ± 0,0101 | — |
| Regresión logística | 0,9911 ± 0,0028 | +0,0123 ± 0,0090 (+28 fraudes) |
| Random Forest | 0,9983 ± 0,0038 | +0,0195 ± 0,0098 (+44 fraudes) |
| LightGBM (por defecto) | 0 | diverge: ver [ERR-014](docs/ERRORS_AND_LEARNINGS.md) |

- **La métrica de decisión se declaró antes de medir.** El PR-AUC global está saturado por la
  firma: cualquier modelo que la aprenda llega a ~0,98. Lo que separa a los modelos es cuánto
  recuperan más allá de la regla con precisión alta ([ADR 0009](docs/adr/0009-evaluation-design.md)).
- **El umbral de significancia práctica, +0,005 de recall, también se declaró antes.**
  Regresión logística y Random Forest lo superan.
- **El contraste estratificado resultó más bajo que el temporal**¹, no más alto: −0,0036 en
  regresión logística y −0,0052 en Random Forest. Se esperaba que ignorar el tiempo fuera
  optimista, y en estos datos no lo fue.

Antes de comparar modelos, el arnés pasó dos controles: el predictor constante da exactamente
la prevalencia de cada fold, y un test de permutación con el objetivo barajado no encuentra
fuga ([EVAL-001 a EVAL-004](docs/EVALUATION.md)). Esa conclusión se apoya en la regresión
logística y en Random Forest: LightGBM por defecto no aprende ni con las etiquetas reales, así
que como detector no tenía poder ([ERR-014](docs/ERRORS_AND_LEARNINGS.md)). Cada resultado tiene su entrada completa en
el [registro de evaluación](docs/EVALUATION.md).

¹ *Cálculo ad hoc.* La diferencia pareada, su conteo en fraudes y la diferencia entre
splitters son restas por fold sobre métricas que produce el arnés, que sí tiene tests; la
resta misma no tiene función ni test propios.

---

## Experiment tracking

Todas las corridas están registradas en MLflow sobre DagsHub, en el experimento
`fraud-detection`:

**https://dagshub.com/diego-Ballesteros/fraud-detection-mlops.mlflow/#/experiments/0**

Cada corrida lleva el modelo, el splitter, la configuración de features, las métricas por fold
y sus agregados, el sha256 del parquet de referencia y el commit desde el que se corrió. Que el
enlace se abra sin credenciales todavía no está verificado: es un punto de la fase 5.

---

## Estado contra la rúbrica

| Componente | Peso | Estado |
|---|---|---|
| Repositorio | 10% | Estructura, CI, pre-commit y los datos fuera de git: hecho. Notebook `01_preprocessing`: hecho. Notebooks 02, 03 y 04: pendientes. La carpeta `app/` existe, vacía |
| Documentación | 10% | Diccionario de datos, nueve ADR, registros de evaluación y de errores, y este README: hecho para las fases 0 a 2A |
| Técnicas avanzadas de ML | 50% | Arnés de evaluación, baselines y registro en MLflow: hecho. **Búsqueda de hiperparámetros: pendiente.** Interpretabilidad local y global: pendiente (fase 3) |
| Monitoreo | 30% | **Sin empezar** (fase 4) |
| Reto ML 1: AutoML | +10% | Pendiente |
| Reto ML 2: app web de monitoreo | +10% | Pendiente |

---

## Reproducir

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

Los resultados de arriba se regeneran con los scripts de la fase 2. Registran en DagsHub si
`.env` tiene las credenciales (plantilla en `.env.example`), y si no, en un store local:

```bash
uv run python scripts/run_leakage_check.py    # controles: aleatorio y test de permutación
uv run python scripts/run_default_models.py   # los modelos, por los dos splitters
```

---

## Dónde está cada cosa

- [`docs/ROADMAP.md`](docs/ROADMAP.md): el plan por fases, la estructura del repositorio y la
  trazabilidad contra la rúbrica.
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
