# ADR 0010 — La configuración elegida

- **Estado:** aceptado
- **Fecha:** 2026-10-19
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Decisión

**Gana el ensayo 5 de la búsqueda (LightGBM)** por la regla de desempate escrita en
`docs/EVALUATION.md` a las 08:09 del 19/10, antes de leer ningún resultado. Cuatro
configuraciones quedaron dentro de 0,005 del mejor, y el costo decide: 59 s contra 625–960 s.

| Hiperparámetro | Valor |
|---|---|
| `n_estimators` | 98 |
| `learning_rate` | 0,0270 |
| `num_leaves` | 56 |
| `max_depth` | 7 |
| `min_child_samples` | 249 |
| `reg_lambda` | 0,0100 |
| `scale_pos_weight` | 34,89 |

Los valores con precisión completa están en `src/fraud_detection/models/registry.py`
(`CHOSEN_PARAMS`). Su recall con precisión ≥ 0,99 sobre los 5 folds temporales es
0,9970 ± 0,0036, contra 0,9788 ± 0,0101 de la regla (EVAL-009).

**El presupuesto se cortó en 8 de 150 ensayos por presión de memoria, y eso queda escrito:
el resultado es de una búsqueda incompleta.**

## Resultado de AutoML que sí es concluyente

LightGBM pasó de divergir (recall 0, ERR-014) a 0,9970. Qué parámetro contiene la divergencia
**no** está determinado: `min_child_samples` y la tasa de aprendizaje se movieron juntos. Y
`reg_lambda` grande no hizo falta, así que la hipótesis del Arquitecto se sostiene solo a
medias.

## Alternativa descartada

**Random Forest**, que tiene el punto estimado más alto (0,9983 sin acotar) y empata dentro
del umbral a 10 veces el costo. Acotar RF a una configuración desplegable cuesta 0,0009,
también por debajo del umbral.

## Riesgo registrado

La configuración elegida está cerca de la región donde esta familia diverge. Se midió con tres
semillas (EVAL-010): con las semillas 43, 44 y 45, el recall por fold es idéntico al del ensayo
5, 0,9970 ± 0,0036, con distancia 0,0000 y una tolerancia de ±0,0036. La medición tenía con qué
fallar: la semilla sí cambia el modelo, porque el PR-AUC del fold 1 da 0,99270, 0,99294 y
0,99315; solo la métrica de decisión no se movió.

## El artefacto

La configuración se reajustó sobre todo el periodo de referencia (5.787.030 filas, 4.477
positivos), dentro del mismo `Pipeline` que el arnés evalúa (`evaluation.compose`), y es el
artefacto único del proyecto. El notebook 03, el monitoreo y la app lo cargan; ninguno
reimplementa una transformación.

- Registrado en el Model Registry de MLflow como `fraud-detection-pipeline`, **versión 1**,
  desde la corrida `e1fb3e9b12ed4dfc9a025a751eb9ff04`, con firma de entrada, hiperparámetros y
  las etiquetas de identificación de datos.
- La copia local es `artifacts/fraud_detection_pipeline.pkl`: 512.224 bytes, sha256
  `41869ee0e5239ca66e0024a10bff4b727cf7ba0a6b6a8087a45da010de5d4ebd`.
- La versión que vuelve del registro predice exactamente lo mismo que la copia local, y esta,
  lo mismo que el Pipeline en memoria (`scripts/register_model.py`, paso 5).
- Se reproduce con `uv run python scripts/register_model.py`.

---

**Nota del Ejecutor sobre la procedencia (2026-10-19):** el ajuste final sobre toda la
referencia tardó 14,5 s y se midió con algo de swap en el ambiente (el archivo de paginación
pasó de 16% a 23%); ese tiempo no es un costo de referencia.

**Nota de estado (2026-10-21, fase 5):** la sección "El artefacto" dice que "el notebook 03, el
monitoreo y la app lo cargan". La tercera parte es falsa: la app del Reto 2 no se construyó, y
`app/` solo contiene `.gitkeep`. Lo cargan el notebook 03, `scripts/run_interpretability.py`,
`scripts/run_monitoring.py` y el notebook 04, y un clon limpio del repositorio lo carga con el
mismo sha256 (`docs/MODEL_CARD.md`, identificación). La decisión no cambia.
