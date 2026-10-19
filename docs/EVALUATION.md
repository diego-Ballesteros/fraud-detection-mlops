# Registro de evaluación

> Contrato: qué se midió, cómo, con qué baseline al lado y con qué resultado. Una métrica
> sin su baseline, su número de particiones y su dispersión no se registra.

---

## Contrato de evaluación

Declarado antes de la primera medición de la fase 2. Las razones de cada punto están en el
ADR 0009; la implementación, en `src/fraud_detection/models/evaluation.py`.

| Elemento | Valor |
|---|---|
| **Métrica de decisión** | Recall con precisión ≥ 0,99 (`recall_at_p0.99`) |
| **Secundaria** | Recall con precisión ≥ 0,95 (`recall_at_p0.95`), por si ningún modelo alcanza el piso de 0,99 |
| **Contexto, siempre reportado** | PR-AUC (precisión promedio), ROC-AUC y Brier |
| **Baseline** | `SignatureRule` (ADR 0007): 0,979 en la métrica de decisión, sobre todo el periodo de referencia |
| **Significancia práctica** | +0,005 de recall. Equivale a unos 22 de los 4.477 fraudes de referencia, unos 11 de los 2.223 de las validaciones temporales y unos 2 por fold |
| **Splitter que decide** | Temporal por rangos de step con ventana expansiva (`temporal_expanding`): 5 folds que validan 201–240, …, 361–400, cada uno entrenando con todos los steps anteriores |
| **Splitter de contraste** | `StratifiedKFold`, 5 folds mezclados con `settings.RANDOM_STATE` (`stratified_kfold`). No decide: mide el optimismo de ignorar el tiempo, y se reporta al lado del temporal para cada modelo |
| **Validación cruzada anidada** | No hay. Producción (`step >= 401`) es el holdout real, y no se toca hasta la fase 4 |
| **Dispersión** | Media y desviación estándar muestral (ddof = 1) entre los 5 folds |

**Cómo se calcula "recall con precisión ≥ p".** Política declarada por el Ejecutor en el
turno 2A, en el docstring de `recall_at_precision` y con tests de empates construidos a mano:

1. Un umbral marca toda fila con puntaje mayor o igual; las filas con el mismo puntaje se
   marcan juntas, porque ningún umbral puede separarlas.
2. Entre los umbrales que alcanzan el piso gana el de mayor recall. Dos umbrales con la
   misma precisión se resuelven así por el recall.
3. Entre umbrales con el mismo recall máximo se reporta el más alto, que marca menos filas.
4. Si ningún umbral alcanza el piso, el recall es 0 y el piso consta como no alcanzado
   (`attained_p0.99 = 0`).

La precisión se compara exacta, en enteros: un punto con precisión igual al piso lo alcanza.

**Dónde queda el baseline sobre la curva de cada modelo.** En cada fold se mide el punto de
la regla sobre las mismas filas (`baseline_precision`, `baseline_recall`) y, sobre la curva
del modelo, la precisión que alcanza con el recall de la regla
(`precision_at_baseline_recall`) y el recall que alcanza con su precisión
(`recall_at_baseline_precision`).

---

## Formato de entrada

Cada entrada se agrega al final de la sección **Entradas**, con un identificador
correlativo. Las entradas no se reescriben: si un resultado deja de ser válido, se le
agrega una nota de estado fechada.

```markdown
### EVAL-NNN · <qué se midió> · AAAA-MM-DD

| Campo | Valor |
|---|---|
| **Pregunta** | Qué se quería saber |
| **Periodo evaluado** (obligatorio) | Referencia (`step <= 400`) o producción (`step >= 401`), con el rango de steps exacto |
| **Tamaño de muestra** (obligatorio) | Filas evaluadas y cuántas son positivas (prevalencia) |
| **Método** | Esquema de validación, modelo y configuración |
| **Métrica** | Nombre y por qué es la adecuada para esta pregunta |
| **Resultado** | Valor ± dispersión |
| **Baseline** (obligatorio) | Contra qué se compara y su valor ± dispersión, sobre las mismas particiones |
| **Número de particiones** (obligatorio) | Sobre cuántas particiones o ventanas se calculó |
| **Dispersión** (obligatorio) | Qué estadístico (desviación estándar, intervalo) y cómo se calculó |
| **Run de MLflow** | ID o enlace del run |
| **Reproducción** | Comando exacto que regenera la cifra |
| **Interpretación** | Qué significa, y qué **no** permite concluir |
```

---

## Entradas

### EVAL-001 · Control del arnés: predictor constante de la clase mayoritaria · 2026-10-18

| Campo | Valor |
|---|---|
| **Pregunta** | ¿El arnés calcula el PR-AUC como se declaró? Con todos los puntajes empatados, tiene que dar exactamente la prevalencia de cada fold de validación |
| **Periodo evaluado** | Referencia. Temporal: valida 201–240, 241–280, 281–320, 321–360 y 361–400, y entrena con todos los steps anteriores. Estratificado: steps 1–400 |
| **Tamaño de muestra** | Temporal: validaciones de 808.499, 602.140, 595.967, 797.269 y 600.973 filas, con 465, 456, 434, 408 y 460 positivos (prevalencia de 0,000512 a 0,000765). Estratificado: 1.157.406 filas por fold, con 895 o 896 positivos (0,000773–0,000774) |
| **Método** | `majority_class` (`DummyClassifier(strategy="most_frequent")`) detrás del Pipeline `full`, por los dos splitters, cada uno en un proceso guardado |
| **Métrica** | PR-AUC (precisión promedio). Es un control del instrumento, no una métrica del modelo |
| **Resultado** | \|PR-AUC − prevalencia\| = 0,0 en los 10 folds, leído del servidor. Temporal: 0,000668 ± 0,000116. Estratificado: 0,000774 ± 0,0000005. ROC-AUC 0,5 y Brier igual a la prevalencia en todos los folds |
| **Baseline** | El punto de `SignatureRule` sobre las mismas filas: recall 0,9788 ± 0,0101 (temporal) y 0,9794 ± 0,0055 (estratificado), con precisión 1,0 en los 10 folds |
| **Número de particiones** | 5 temporales y 5 estratificadas |
| **Dispersión** | Desviación estándar muestral (ddof = 1) entre los 5 folds de cada splitter |
| **Run de MLflow** | `ec12c2c301de4468b2dd1584655c6f2a` (temporal) y `a8849c6081d24829b1118a9cb4865544` (estratificado), experimento `fraud-detection` en DagsHub |
| **Reproducción** | `uv run python scripts/run_default_models.py --models majority_class` |
| **Interpretación** | El arnés calcula el PR-AUC como se declaró, y el piso queda medido: la tasa base de cada fold. No dice nada sobre ningún modelo. La corrida `2c86f39c…` es un intento anterior del mismo control que quedó colgado por ERR-012, cerrado como FAILED |

### EVAL-002 · Test de objetivo barajado: no colapsa · 2026-10-18

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Filtra la etiqueta a través de la estructura de folds o del preprocesamiento? |
| **Periodo evaluado** | Referencia, splitter temporal: valida 201–400 en 5 ventanas de 40 steps |
| **Tamaño de muestra** | Las validaciones de EVAL-001, con sus etiquetas verdaderas. Las etiquetas de entrenamiento se permutaron con `settings.RANDOM_STATE`, y conservan los conteos: 2.254, 2.719, 3.175, 3.609 y 4.017 positivos |
| **Método** | Regresión logística, Random Forest y LightGBM con hiperparámetros por defecto, detrás del Pipeline `full`, por el mismo arnés que los modelos, con `shuffle_training_labels=True` |
| **Métrica** | PR-AUC / prevalencia de cada fold. Criterio declarado antes de medir (ADR 0009): colapsa si es ≤ 10 en cada fold de cada familia |
| **Resultado** | Regresión logística: 2,00 · 1,82 · 1,63 · 2,10 · 1,18 (1,75 ± 0,36). Random Forest: 1,06 · 1,09 · 1,07 · 1,01 · 1,00 (1,04 ± 0,03). LightGBM: 2,05 · **14,83** · 2,41 · 1,02 · 1,05 (4,27 ± 5,93). **No colapsa:** LightGBM falla en el fold 2, con PR-AUC 0,0112 |
| **Baseline** | La prevalencia de cada fold, que es el piso (EVAL-001), y un ranking al azar sobre las mismas etiquetas: 1,015–1,028 veces la prevalencia en promedio, percentil 99 de 1,13 a 1,25 (ADR 0009, nota del Ejecutor) |
| **Número de particiones** | 5 folds temporales × 3 familias |
| **Dispersión** | Por fold, arriba; entre paréntesis, media ± desviación estándar muestral (ddof = 1) |
| **Run de MLflow** | `62b675e957464670b916885e03fdfbad` (regresión logística), `4217ce9edf5446919f41fa0d1c0848a9` (Random Forest), `fd9bcbe7d54442d8b1257d1eb1a4f711` (LightGBM) |
| **Reproducción** | `uv run python scripts/run_leakage_check.py` (sale con código 1) |
| **Interpretación** | Es un bloqueante: la evaluación de los modelos no se corrió. En el fold que falla, el recall con precisión ≥ 0,95 es 0, el recall a la precisión de la regla es 0, y el ROC-AUC es 0,26: unos pocos fraudes quedan arriba del ranking y el resto por debajo de las filas legítimas. **No permite concluir** si es azar o fuga. El diagnóstico declarado tampoco lo separa: en ese fold cayeron 4 positivos barajados en filas con la firma, frente a 2,27 esperados, y el fold 1 tuvo 5 frente a 2,09 con un cociente de 2,05 |

**Nota de estado (2026-10-18):** el criterio de esta entrada quedó reemplazado (ADR 0009,
nota de estado), y el test se repitió con 20 permutaciones: pasa (EVAL-004). Esta permutación
es el sorteo 0 de EVAL-004, que reproduce sus 15 valores exactos, y el 14,83 resultó ser el
máximo de esa distribución. Dos afirmaciones de esta entrada dejaron de ser ciertas:

- Que "unos pocos fraudes quedan arriba del ranking": medido, el top 100 tiene cero positivos,
  y el PR-AUC sale de bloques de puntaje empatado en los puestos ~1.400–1.550 (ERR-013).
- Su comando de reproducción: `scripts/run_leakage_check.py` corre ahora el test de
  permutación y sale con código 0. La permutación de esta entrada es su sorteo 0.

### EVAL-003 · Clasificador aleatorio: el piso de inflación por escasez de positivos · 2026-10-18

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Cuánto PR-AUC por encima de la prevalencia da un ranking que no sabe nada, solo porque los positivos son pocos? |
| **Periodo evaluado** | Referencia, splitter temporal: valida 201–400 en 5 ventanas de 40 steps |
| **Tamaño de muestra** | Las validaciones de EVAL-001: entre 595.967 y 808.499 filas, con 408 a 465 positivos |
| **Método** | `UniformRandomScores`: un puntaje uniforme por fila, sin modelo, por el mismo arnés. 20 semillas, de `settings.RANDOM_STATE` a `settings.RANDOM_STATE + 19` |
| **Métrica** | PR-AUC y PR-AUC / prevalencia; ROC-AUC al lado |
| **Resultado** | Sobre los 100 valores (20 semillas × 5 folds): PR-AUC mínimo 0,000491, mediana 0,000736, p95 0,000825, **máximo 0,001135**. Cociente a la prevalencia: mínimo 0,92, mediana 1,02, p95 1,12, **máximo 1,50**. ROC-AUC: mínimo 0,470, mediana 0,501, p95 0,524, máximo 0,539, con el 54% por encima de 0,5 |
| **Baseline** | La prevalencia de cada fold, que es lo que da el constante (EVAL-001). La medición ad hoc del turno anterior (100 rankings por fold) dio un máximo de 1,51, igual que esta |
| **Número de particiones** | 5 folds temporales × 20 semillas |
| **Dispersión** | Mínimo, mediana, percentil 95 (interpolación lineal) y máximo de los 100 valores |
| **Run de MLflow** | Corrida padre `e1407831343443ffbe76729f66433652`, con 20 hijas |
| **Reproducción** | `uv run python scripts/run_leakage_check.py` (primera sección) |
| **Interpretación** | Mide el piso de inflación por escasez de positivos: ~1,02 veces la prevalencia, con un máximo de 1,50. Descarta la escasez como explicación de los 14,83 de EVAL-002 y señala a la estructura del ranking (ERR-013). **No** es el nulo del test de objetivo barajado: ese nulo es el mismo modelo permutado (EVAL-004) |

### EVAL-004 · Test de objetivo barajado como test de permutación: pasa · 2026-10-18

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Filtra la etiqueta a través de la estructura de folds o del preprocesamiento? |
| **Periodo evaluado** | Referencia, splitter temporal: valida 201–400 en 5 ventanas de 40 steps |
| **Tamaño de muestra** | Las validaciones de EVAL-001, con sus etiquetas verdaderas; en cada fold se permutan solo las etiquetas de entrenamiento, que conservan sus conteos |
| **Método** | El Pipeline `full` por el mismo arnés. LightGBM y regresión logística con 20 permutaciones cada uno (semillas `settings.RANDOM_STATE + i`, i de 0 a 19); Random Forest con 1 (la semilla 0), porque una fuga es propiedad del pipeline y de los folds, no de la familia |
| **Métrica** | El PR-AUC máximo sobre todas las permutaciones y folds, en valor absoluto, y la mediana del ROC-AUC contra el rango del clasificador aleatorio (ADR 0009, nota de estado) |
| **Resultado** | PR-AUC (mín / mediana / p95 / **máx**): LightGBM 0,000289 / 0,000780 / 0,003446 / **0,011227**; regresión logística 0,000403 / 0,001065 / 0,002075 / **0,004968**; Random Forest 0,000518 / 0,000769 / 0,000813 / **0,000822**. ROC-AUC, mediana (rango): LightGBM 0,242 (0,034–0,684); regresión logística 0,545 (0,310–0,753); Random Forest 0,504 (0,501–0,514). Los 20 PR-AUC de LightGBM en el fold 2, de la semilla 0 a la 19: 0,011227 · 0,000537 · 0,000847 · 0,000673 · 0,000562 · 0,003695 · 0,000658 · 0,000643 · 0,000553 · 0,001100 · 0,001456 · 0,004532 · 0,000532 · 0,001008 · 0,000975 · 0,000787 · 0,001198 · 0,000568 · 0,002283 · 0,000951 |
| **Baseline** | El clasificador aleatorio sobre los mismos folds (EVAL-003): PR-AUC máximo 0,001135, ROC-AUC entre 0,470 y 0,539. La firma con etiquetas reales: ~0,98 |
| **Número de particiones** | 5 folds × 20 permutaciones (LightGBM y regresión logística), 5 × 1 (Random Forest) |
| **Dispersión** | Mínimo, mediana, percentil 95 (interpolación lineal) y máximo de los valores de cada familia |
| **Run de MLflow** | Corridas padre `a6794aa2d2c04f41ab2ed06f08e58218` (LightGBM), `88c64aba14d747db9736d7f778af3d89` (regresión logística) y `d1c9d1e2b0fd4a3685b8dc83666a01c6` (Random Forest), cada una con sus hijas |
| **Reproducción** | `uv run python scripts/run_leakage_check.py` (sale con código 0) |
| **Interpretación** | **Pasa.** El máximo, 0,0112, está por debajo de 0,05 y unas 87 veces por debajo de la firma, y es la permutación original, reproducida exactamente. LightGBM queda sistemáticamente por debajo de 0,5 en ROC-AUC: anti-generaliza, no filtra. La regresión logística queda apenas por encima del rango del aleatorio (0,545 contra 0,539): se aceptó con el mecanismo **sin determinar** (ADR 0009). No permite concluir nada sobre el desempeño de ningún modelo con etiquetas reales |

**Nota de estado (2026-10-18):** medido después de esta entrada (EVAL-008), LightGBM con sus
hiperparámetros por defecto no aprende ni con las etiquetas reales: su PR-AUC es 0,034 en el
splitter temporal. Un detector que no aprende la etiqueta verdadera tampoco aprendería una
filtrada, así que **su parte de este test no tenía poder**. La conclusión de que el control
pasa se sostiene en los otros dos detectores, que con etiquetas reales sí aprenden: la
regresión logística, con 20 permutaciones (máximo 0,0050 contra 0,9967 con etiquetas reales,
EVAL-006), y Random Forest, con una (máximo 0,00082 contra 0,9983, EVAL-007).

### EVAL-005 · `SignatureRule`, el baseline, por el arnés · 2026-10-18

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Qué da el baseline del ADR 0007 en la métrica de decisión, por los dos splitters? |
| **Periodo evaluado** | Referencia. Temporal: valida 201–400 en 5 ventanas de 40 steps. Estratificado: 1–400 |
| **Tamaño de muestra** | Temporal: 3.404.848 filas validadas con 2.223 positivos. Estratificado: 5.787.030 con 4.477 (EVAL-001 da el detalle por fold) |
| **Método** | `SignatureRule` sobre las filas crudas (preprocesador passthrough), por el arnés guardado |
| **Métrica** | Recall con precisión ≥ 0,99, la de decisión (ADR 0009) |
| **Resultado** | Temporal: **0,9788 ± 0,0101** (0,9742 · 0,9693 · 0,9862 · 0,9926 · 0,9717), con precisión 1,0 en los 5 folds. Estratificado: 0,9794 ± 0,0055. Secundaria, igual. PR-AUC 0,9788 ± 0,0101 y ROC-AUC 0,9894 ± 0,0050 (temporal). Deja fuera 48 de los 2.223 fraudes temporales y 92 de los 4.477 estratificados |
| **Baseline** | El constante (EVAL-001): recall 0, PR-AUC igual a la prevalencia |
| **Número de particiones** | 5 temporales y 5 estratificadas |
| **Dispersión** | Desviación estándar muestral (ddof = 1) entre los 5 folds |
| **Run de MLflow** | `04156a1487754e738a60f595df24616c` (temporal), `81fc654b177f46e1af89c8a06065b73a` (estratificado) |
| **Reproducción** | `uv run python scripts/run_default_models.py --models signature_rule` |
| **Interpretación** | Es la línea contra la que se mide todo: el 0,979 del ADR 0007, medido ahora por el arnés y con su dispersión entre folds. En el temporal, la regla deja fuera entre 3 y 14 fraudes por fold |

### EVAL-006 · Regresión logística, hiperparámetros por defecto · 2026-10-18

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Cuánto recupera, por encima de la regla, un modelo lineal sin ajustar? |
| **Periodo evaluado** | Como en EVAL-005 |
| **Tamaño de muestra** | Como en EVAL-005 |
| **Método** | `LogisticRegression` con los valores por defecto de scikit-learn (lbfgs, C = 1, 100 iteraciones), detrás del Pipeline `full` |
| **Métrica** | Recall con precisión ≥ 0,99, comparado con la regla sobre los mismos folds (diferencia pareada) |
| **Resultado** | Temporal: **0,9911 ± 0,0028** (0,9871 · 0,9890 · 0,9931 · 0,9926 · 0,9935), alcanzado en los 5 folds. Estratificado: 0,9875 ± 0,0029. Secundaria: 0,9956 ± 0,0040 (temporal). PR-AUC 0,9967 ± 0,0036, ROC-AUC 0,9976 ± 0,0025 (temporal). Deja fuera 20 de los 2.223 fraudes temporales (la regla, 48) |
| **Baseline** | `SignatureRule` sobre los mismos folds: 0,9788 ± 0,0101 (temporal). **Diferencia pareada: +0,0123 ± 0,0090**, es decir, 28 fraudes más; por encima del umbral de 0,005 en 4 de los 5 folds (en el fold 4 empata). Estratificado: +0,0080 ± 0,0048 |
| **Número de particiones** | 5 temporales y 5 estratificadas |
| **Dispersión** | Desviación estándar muestral (ddof = 1) entre los 5 folds, de cada valor y de la diferencia pareada |
| **Run de MLflow** | `6a7b87abd3584ef3b7126174fb41700f` (temporal), `6cd8c5b8270c419eaf8650547966071c` (estratificado) |
| **Reproducción** | `uv run python scripts/run_default_models.py --models logistic_regression` |
| **Interpretación** | Supera a la regla por encima de la significancia práctica. El estratificado da **menos** que el temporal (−0,0036): en estos datos, ignorar el tiempo no resultó optimista. No permite concluir cuál es el mejor modelo: no hubo búsqueda de hiperparámetros |

### EVAL-007 · Random Forest, hiperparámetros por defecto · 2026-10-18

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Cuánto recupera, por encima de la regla, un ensamble de árboles sin ajustar? |
| **Periodo evaluado** | Como en EVAL-005 |
| **Tamaño de muestra** | Como en EVAL-005 |
| **Método** | `RandomForestClassifier` con los valores por defecto de scikit-learn (100 árboles, `max_depth=None`), `n_jobs=-1`, detrás del Pipeline `full` |
| **Métrica** | Recall con precisión ≥ 0,99, comparado con la regla sobre los mismos folds (diferencia pareada) |
| **Resultado** | Temporal: **0,9983 ± 0,0038** (0,9914 · 1,0 · 1,0 · 1,0 · 1,0): en los folds 2 a 5 recupera todos los fraudes con precisión ≥ 0,99. Estratificado: 0,9931 ± 0,0018. PR-AUC 0,9983 ± 0,0038, ROC-AUC 0,9991 ± 0,0019 (temporal). Deja fuera 4 de los 2.223 fraudes temporales, todos en el fold 1 |
| **Baseline** | `SignatureRule` sobre los mismos folds: 0,9788 ± 0,0101 (temporal). **Diferencia pareada: +0,0195 ± 0,0098**, 44 fraudes más, por encima del umbral en los 5 folds. Estratificado: +0,0136 ± 0,0055 |
| **Número de particiones** | 5 temporales y 5 estratificadas |
| **Dispersión** | Desviación estándar muestral (ddof = 1) entre los 5 folds, de cada valor y de la diferencia pareada |
| **Run de MLflow** | `6f48b95f699f4b80a7d36dc0a481993e` (temporal), `4a26082546b34e6ca9191192c79cf8a6` (estratificado) |
| **Reproducción** | `uv run python scripts/run_default_models.py --models random_forest` |
| **Interpretación** | Recupera casi todo lo que la regla deja fuera. Un resultado tan alto se lee con la prueba de permutación al lado (EVAL-004, con su nota de estado): no hay fuga de etiqueta. De los 92 fraudes de referencia que la regla no cubre, 57 tienen `amount` exactamente 10.000.000, el máximo de todos los fraudes del periodo (medición ad hoc del Ejecutor, no versionada). Que eso explique el recall de este modelo es una inferencia, no una medición. El estratificado da menos que el temporal (−0,0052) |

### EVAL-008 · LightGBM, hiperparámetros por defecto: no aprende · 2026-10-18

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Cuánto recupera, por encima de la regla, un ensamble de boosting sin ajustar? |
| **Periodo evaluado** | Como en EVAL-005 |
| **Tamaño de muestra** | Como en EVAL-005 |
| **Método** | `LGBMClassifier` con los valores por defecto de LightGBM (100 iteraciones, 31 hojas, tasa de aprendizaje 0,1), `n_jobs=-1`, detrás del Pipeline `full` |
| **Métrica** | Recall con precisión ≥ 0,99, comparado con la regla sobre los mismos folds |
| **Resultado** | **0 en los 10 folds**: ningún umbral alcanza precisión ≥ 0,95. PR-AUC 0,0340 ± 0,0718 (temporal) y 0,0196 ± 0,0130 (estratificado); ROC-AUC 0,4708 ± 0,0574 y 0,5062 ± 0,0566 |
| **Baseline** | `SignatureRule` sobre los mismos folds: 0,9788 ± 0,0101 (temporal). Diferencia pareada: −0,9788 |
| **Número de particiones** | 5 temporales y 5 estratificadas |
| **Dispersión** | Desviación estándar muestral (ddof = 1) entre los 5 folds |
| **Run de MLflow** | `8f65764af11d43af8adf82aa19f8c6fb` (temporal), `ef0d141550234b799e309c56182e94cb` (estratificado) |
| **Reproducción** | `uv run python scripts/run_default_models.py --models lightgbm` |
| **Interpretación** | Con sus valores por defecto, LightGBM no aprende en estos datos. **No** permite concluir que la familia no sirva: una medición ad hoc del Ejecutor, no versionada, mostró que el entrenamiento diverge (el PR-AUC es alto en las primeras iteraciones y colapsa al llegar a la 100, también en entrenamiento). Esas cifras quedan en el reporte del turno hasta que la medición se versione. Consecuencia para EVAL-004: como detector de fuga, LightGBM por defecto no tenía poder |
