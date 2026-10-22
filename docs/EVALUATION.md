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

**Regla de desempate entre configuraciones.** La declaró el Arquitecto el 2026-10-19 a las
08:09, con la búsqueda de hiperparámetros en curso y antes de leer ninguno de sus resultados.
Lo único que ya se había visto: un ensayo corto, en un almacenamiento descartable, corrió los
tres puntos de partida, que son las mismas configuraciones que los ensayos 0 a 2 de la
búsqueda.

- El rango usable de la métrica de decisión mide 0,021, de 0,9788 (el baseline) a 1,0. Con
  la significancia práctica en 0,005, se distinguen a lo sumo cuatro niveles, así que es
  esperable que varias configuraciones queden indistinguibles arriba.
- **Todo ensayo que quede dentro de 0,005 del mejor se reporta como empate, no como peor.**
- El desempate **no** es el tercer decimal. Es, en este orden:
  1. el costo de ajuste medido;
  2. la simplicidad de la configuración: menos profundidad, menos árboles, menos features;
  3. la dispersión entre folds: a menor dispersión, mayor confianza.
- El reporte dice cuántas configuraciones empataron, no solo cuál ganó. Coronar un ganador por
  el tercer decimal de una media de cinco folds es ruido presentado como resultado.

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

**Nota de estado (2026-10-21):** los 57 de los 92 fraudes fuera de la regla con monto de
exactamente 10.000.000, que esta entrada daba como medición ad hoc, quedaron medidos en
`notebooks/03_interpretability.ipynb` con `profiling.rule_outcome`: la cifra se sostiene. La misma
medición encontró 2.900 transacciones legítimas con ese monto, así que la cifra no es exclusiva del
fraude ni un tope del simulador (`docs/analysis/interpretability.md`, nota de estado). Que esos
fraudes expliquen el recall de este modelo sigue siendo una inferencia.

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

### EVAL-009 · Búsqueda de hiperparámetros con el algoritmo como hiperparámetro: 8 de 150 ensayos · 2026-10-19

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Qué configuración maximiza el recall con precisión ≥ 0,99, cuando el algoritmo es un hiperparámetro más? |
| **Periodo evaluado** | Referencia, splitter temporal: valida 201–400 en 5 ventanas de 40 steps |
| **Tamaño de muestra** | Como en EVAL-005: 3.404.848 filas validadas, con 2.223 positivos |
| **Método** | Optuna 5.0 con muestreador TPE (semilla `settings.RANDOM_STATE`) y podador de mediana sobre el valor de cada fold; estudio en SQLite. El algoritmo (regresión logística, Random Forest o LightGBM) es categórico, sobre la configuración `full`, con el espacio de `models/search_space.py`. Los ensayos 0 a 2 son los puntos de partida: los valores por defecto donde caben en el espacio y la cota más permisiva donde no. Presupuesto: 4 h o 150 ensayos. **Se cortó a los 8 ensayos terminados (~38 min) por presión de memoria**; el ensayo 8 quedó marcado como FAIL |
| **Métrica** | La media, sobre los 5 folds, del recall con precisión ≥ 0,99. Empates y desempate según la regla del contrato, escrita antes de leer los resultados |
| **Resultado** | Ver la tabla de abajo. **Empatan cuatro configuraciones** a menos de 0,005 del mejor (el ensayo 3, 0,9978): los ensayos 1, 3, 5 y 6. Por el primer criterio, el costo de ajuste, **gana el ensayo 5** (LightGBM, 59 s, contra 625–960 s de los RF) |
| **Baseline** | `SignatureRule` sobre los mismos folds: 0,9788 ± 0,0101 (EVAL-005). Los puntos de partida (ensayos 0 a 2) y los valores por defecto reales: regresión logística 0,9911, Random Forest 0,9983, LightGBM 0 (EVAL-006 a EVAL-008) |
| **Número de particiones** | 5 folds temporales por ensayo |
| **Dispersión** | Desviación estándar muestral (ddof = 1) entre los 5 folds de cada ensayo |
| **Run de MLflow** | Corrida padre `613513c493774f1e804f76cd02d4c63b` (cerrada como KILLED, con la etiqueta `cut_reason`), y una hija por ensayo: `1b8397e44f5f473aad223ca2a7bae02f` (0), `ce364f17fcec4322b4e4675c6eaef56f` (1), `560e866405d441608967fadb5bc5a29e` (2), `b86250c6d9f344bcb91a9581d0d2fde9` (3), `23d9192f1122457a8b8cb7aa7f884dd4` (4), `354bd9179fdf40aaa2c57cdfb7c34b9b` (5), `d25c81ad35c445398c535c5a62caad56` (6), `032e04b492004c03918d43ac23b277db` (7) |
| **Reproducción** | `uv run python scripts/run_training.py` continúa el estudio guardado en `optuna.db`, que no está versionado; en un clon limpio empieza un estudio nuevo con los mismos tres puntos de partida |
| **Interpretación** | Ver abajo |

| Ensayo | Familia y configuración | Recall @ P≥0,99 | `fit` total² | Lectura |
|---|---|---|---|---|
| 0 | Regresión logística, punto de partida (C 1, sin pesos) | 0,9911 ± 0,0028 | 23 s | completo |
| 1 | RF, punto de partida (100 árboles, profundidad 16, hoja 20, sqrt) | 0,9974 ± 0,0039 | 625 s | **empate** |
| 2 | LightGBM, punto de partida (100 árboles, lr 0,1, 31 hojas, profundidad 12, min_child 20, reg_lambda 0,001) | 0 | 42 s | diverge |
| 3 | RF (140 árboles, profundidad 6, hoja 32, log2, balanced_subsample) | 0,9978 ± 0,0037 | 960 s | **empate**, el punto más alto |
| 4 | Regresión logística (C 0,008, balanced) | 0,9859 ± 0,0091 | 26 s | completo |
| 5 | LightGBM (98 árboles, lr 0,027, 56 hojas, profundidad 7, min_child 249, reg_lambda 0,010, scale_pos_weight 34,9) | 0,9970 ± 0,0036 | **59 s** | **empate, elegido** |
| 6 | LightGBM (109 árboles, lr 0,0125, 122 hojas, profundidad 12, min_child 269, reg_lambda 0,033, scale_pos_weight 2,0) | 0,9970 ± 0,0036 | 72 s | **empate** |
| 7 | Regresión logística (C 0,30, balanced) | 0,9763 en el fold 1 | 4 s | podado |

² *Costos medidos con swap.* Durante la búsqueda, la memoria disponible bajó hasta 231 MB y el
archivo de paginación pasó de 16% a 25%. Los costos están inflados, sobre todo los de Random
Forest; que el orden entre familias (LightGBM unas 10 veces más barato) resista esa inflación
es una inferencia, no una medición.

**Interpretación.**

- **Resultado negativo, regresión logística:** la búsqueda no mejoró el punto de partida, que
  coincide con el valor por defecto real (0,9911).
- **Resultado negativo, Random Forest:** la búsqueda empató con el valor por defecto sin acotar
  (0,9983, EVAL-007) y con su propio punto de partida acotado (0,9974). Acotar RF a una
  configuración desplegable cuesta 0,0009 en el punto de partida, por debajo del umbral.
- **Resultado positivo, LightGBM:** pasó de divergir (0) a 0,9970. Qué parámetro contiene la
  divergencia no está determinado: los ensayos 5 y 6 subieron `min_child_samples` a ~250 **y**
  bajaron la tasa de aprendizaje a la vez, con `reg_lambda` todavía chico.
- Los ensayos 5 y 6 tienen recalls por fold idénticos, y los cuatro empatados comparten 0,9914
  en el fold 1. Es coherente con la resolución de la métrica: en un fold de ~456 positivos, el
  recall se mueve en pasos de ~0,0022.
- **No permite concluir** que una búsqueda completa no encontraría algo mejor: el presupuesto
  se cortó en 8 de 150 ensayos.

### EVAL-010 · Estabilidad de la configuración elegida ante la semilla · 2026-10-19

| Campo | Valor |
|---|---|
| **Pregunta** | La configuración elegida está cerca de la región donde LightGBM diverge (ERR-014). ¿Su métrica de decisión se mantiene si cambia la semilla? |
| **Periodo evaluado** | Referencia, splitter temporal: valida 201–400 en 5 ventanas de 40 steps |
| **Tamaño de muestra** | Como en EVAL-005 |
| **Método** | El ensayo 5 de EVAL-009 reajustado por el arnés con las semillas 43, 44 y 45 (`settings.RANDOM_STATE + 1, + 2, + 3`; el ensayo original usó 42). Criterio, fijado antes de medir: cada media dentro de la dispersión ya medida del ensayo 5 (±0,0036); si no, no se serializa |
| **Métrica** | Recall con precisión ≥ 0,99; PR-AUC y Brier al lado, para saber si la semilla cambió algo |
| **Resultado** | Las tres semillas dan el **mismo recall por fold** que el ensayo 5: 0,9914 · 0,9956 · 1,0 · 1,0 · 0,9978, es decir 0,9970 ± 0,0036, con distancia 0,0000. La semilla **sí** cambia el modelo: el PR-AUC del fold 1 da 0,99270, 0,99294 y 0,99315, y el Brier se mueve en la segunda cifra significativa |
| **Baseline** | El ensayo 5, semilla 42: 0,9970 ± 0,0036 |
| **Número de particiones** | 5 folds temporales × 3 semillas |
| **Dispersión** | Desviación estándar muestral (ddof = 1) entre los 5 folds de cada semilla |
| **Run de MLflow** | `407b9dc2f5e14fc69cf1e6bfe215ebd3` (43), `ff284846e2324d9383491803c1fe8d2c` (44), `275800d052f24f8fa84079ece18657ff` (45) |
| **Reproducción** | `uv run python scripts/register_model.py` (el paso 1) |
| **Interpretación** | La inquietud queda cerrada para la semilla, y se midió. La medición no era vacía: el PR-AUC muestra que el modelo cambió con la semilla. Con `subsample` y `colsample_bytree` en 1, la vía más probable es la muestra de filas con que LightGBM construye los bins, pero eso es hipótesis y no medición. Lo que la medición **no** cubre: la sensibilidad a los hiperparámetros vecinos, que es otra pregunta |

### EVAL-011 · El artefacto sobre producción, al umbral fijo, por ventanas de un día · 2026-10-21

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Cómo rinde el artefacto elegido (ADR 0010) sobre un periodo que no vio, con el umbral fijado en referencia (ADR 0011), y se degrada con el tiempo? |
| **Periodo evaluado** | Producción, steps 401–743, leída por primera vez: 14 ventanas completas de 24 steps (401–736) y una parcial de 7 (737–743) |
| **Tamaño de muestra** | 575.590 filas y 3.736 positivos (prevalencia 0,006491, 8,4 veces la de referencia). Ventanas completas: de 4.474 a 209.359 filas, con 232 a 308 positivos. La parcial: 80 filas, las 80 fraude |
| **Método** | El artefacto cargado, con su sha256 verificado contra el ADR 0010; nada se reajusta. El modelo marca una fila con puntaje ≥ 0,42775088540908546, y el script verifica que ese umbral es el que la métrica de decisión elige sobre referencia. Criterio de degradación declarado antes de medir (ADR 0012): recall por ventana completa contra 0,9934; 2 ventanas seguidas debajo son alerta, 3 son degradación |
| **Métrica** | Recall y precisión al umbral fijo: en producción el umbral es el punto de operación y no se puede elegir con sus etiquetas. PR-AUC, ROC-AUC y Brier como contexto |
| **Resultado** | Agregado: **recall 0,9976** (3.727 de 3.736) y **precisión 0,9997** (un falso positivo en 14 días). Por ventana completa: recall 0,9978 ± 0,0025 (0,9920–1,0000) y precisión 0,9997 ± 0,0010, siempre sobre el piso de 0,99. PR-AUC 0,9997, ROC-AUC 0,9997, Brier 0,000017. Criterio: solo la ventana 2 cae debajo de la línea (248 de 250): **sin alerta**. La tabla por ventana está en `docs/analysis/monitoring.md` |
| **Baseline** | `SignatureRule` sobre las mismas filas: **precisión 1,000 y recall 0,9724** (3.633 de 3.736); por ventana completa, recall 0,9736 ± 0,0173. PR-AUC 0,9726, ROC-AUC 0,9862, Brier 0,000179. El modelo supera a la regla en +0,0252 de recall: 94 de los 103 fraudes que la regla no ve. El modelo en la validación temporal de referencia: 0,9970 ± 0,0036 (EVAL-009) |
| **Número de particiones** | 14 ventanas completas; la parcial entra al agregado y queda fuera de la dispersión y del criterio |
| **Dispersión** | Desviación estándar muestral (ddof = 1) entre las 14 ventanas completas |
| **Run de MLflow** | `b7eabc74ebc64ae68b05472ff437bb3f` (`monitoring__production`), con la serie de cada ventana como pasos. `aeee8e748a174686b2c9668d019f07b8` y `e9a7d78d74744e18bdffcb0e089a870d` son corridas anteriores del mismo script en el mismo turno, con las mismas cifras de desempeño (`docs/analysis/monitoring.md`, nota) |
| **Reproducción** | `uv run python scripts/run_monitoring.py` |
| **Interpretación** | El modelo no se degradó: su recall fuera de muestra es el de la validación temporal. La regla se sostiene con precisión 1,000, así que la firma es del agente de fraude y no del periodo. La precisión del modelo subió porque la prevalencia se multiplicó por 8,4 a umbral fijo, no porque el modelo mejorara. **No permite concluir** nada sobre datos reales, donde un atacante se adapta, ni nada causal entre drift y desempeño. Tampoco tiene mucho poder: con estas ventanas, una alerta saldría por azar con probabilidad 0,30 |

### EVAL-012 · El artefacto sobre producción por tipo de transacción, y el precio de los fraudes que pierde · 2026-10-21

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Dónde está lo que el agregado de EVAL-011 promedia, y cuánto costaría, en revisiones, atrapar los fraudes que el modelo pierde? |
| **Periodo evaluado** | Producción, steps 401–743. El precio, también sobre referencia, steps 1–400, *in-sample* |
| **Tamaño de muestra** | Producción: 575.590 filas y 3.736 positivos. Por tipo: `CASH_IN` 127.917 y 0; `CASH_OUT` 189.404 y 1.868; `DEBIT` 4.316 y 0; `PAYMENT` 202.484 y 0; `TRANSFER` 51.469 y 1.868. Referencia: 5.787.030 y 4.477 |
| **Método** | Las puntuaciones de EVAL-011, al umbral fijo, separadas por `type` (`performance.segment_table`), con la regla sobre las mismas filas. El precio: para cada puntaje distinto de un fraude perdido, de mayor a menor, los fraudes ganados y las legítimas agregadas por un umbral bajado hasta él, acumulados (`performance.lowering_cost`). Se lee y no se aplica: el umbral no se mueve (ADR 0012) |
| **Métrica** | Precisión y recall por tipo: el agregado no dice en qué tipo está el valor del modelo, y en un tipo sin fraude el recall no está definido. El precio, en legítimas agregadas por fraude ganado: es lo que cuesta el siguiente punto de recall en la unidad que paga el negocio, revisiones |
| **Resultado** | Modelo contra regla, precisión / recall: `CASH_OUT` 0,9995 / 0,9957 contra 1,000 / 0,9936; `TRANSFER` 1,000 / 0,9995 contra 1,000 / 0,9513. `CASH_IN`, `DEBIT` y `PAYMENT`: ni el modelo ni la regla marcan ninguna fila; precisión y recall indefinidos. Los 9 fraudes perdidos son 8 `CASH_OUT` y 1 `TRANSFER`, todos fuera de la regla, y la única legítima marcada es `CASH_OUT`. Precio en producción: los 4 primeros fraudes, 31 legítimas, con precisión 0,9915; el 5.º, 48 en total (0,9870); el 6.º, 103; el 8.º, 314; el 9.º, con puntaje 0,0000015, 571.831. En referencia, *in-sample*: el 1.º, 6 legítimas y precisión 0,9889; los 11, 1.514 |
| **Baseline** | La regla, por tipo, sobre las mismas filas (arriba). Para el precio no hay baseline: la regla no tiene puntaje que bajar |
| **Número de particiones** | 5 tipos; en los dos con fraude, además, las 14 ventanas completas |
| **Dispersión** | Desviación estándar muestral (ddof = 1) entre las 14 ventanas: `CASH_OUT`, recall del modelo 0,9961 ± 0,0051 y de la regla 0,9940 ± 0,0061; `TRANSFER`, 0,9995 ± 0,0020 y 0,9532 ± 0,0304. El precio es un conteo sobre todo el periodo, sin dispersión |
| **Run de MLflow** | `0a146ff55e604e328fb7d0d251b4eff5` (`monitoring__production`), que reproduce además todas las cifras de EVAL-011 |
| **Reproducción** | `uv run python scripts/run_monitoring.py` (paso 3) |
| **Interpretación** | El valor del modelo sobre la regla está casi entero en `TRANSFER`: 90 de los 94 fraudes que agrega, +0,0482 de recall. En `CASH_OUT` agrega 4, +0,0021, por debajo de la significancia práctica y dentro de la dispersión entre días: ahí el modelo y la regla son, en la práctica, el mismo detector, y ahí están 8 de sus 9 fraudes perdidos y su única falsa alarma. El 0,9976 agregado promedia esos dos tipos; los otros tres no aportan nada al recall, porque no tienen fraude. El precio dice que el umbral no deja fraude barato fuera: en producción, 4 fraudes por 31 revisiones en dos semanas, y en referencia el primero de más ya rompe el piso de 0,99. La diferencia entre los dos periodos es el volumen legítimo, no el modelo. **No permite concluir** por qué el modelo pierde los `CASH_OUT` fuera de la firma, que no se analizó, ni que convenga bajar el umbral: elegirlo con esta curva sería ajustarlo con producción |

### EVAL-013 · La ablación del ADR 0007: sin la firma, el modelo no llega a ningún punto de operación · 2026-10-22

| Campo | Valor |
|---|---|
| **Pregunta** | ¿Cuánto del desempeño es la firma del simulador? |
| **Periodo evaluado** | Referencia, splitter temporal: valida 201–400 en 5 ventanas de 40 steps. Además, la referencia entera *in-sample* para el umbral, y producción, steps 401–743 |
| **Tamaño de muestra** | Validación: 3.404.848 filas, 2.223 positivos. Referencia: 5.787.030 y 4.477. Producción: 575.590 y 3.736 |
| **Método** | La configuración `no_signature` (sin `oldbalanceOrg` ni `amount_equals_origin_balance`) con los hiperparámetros y la semilla del ensayo 5, sin reajustarlos, por el mismo arnés guardado, con `full` y la regla al lado. Declarado antes de medir, en el docstring de `scripts/run_ablation.py`: control de que `full` reproduce el ensayo 5 fold por fold (lo reproduce: 0,9914 · 0,9956 · 1,0 · 1,0 · 0,9978); si no alcanza precisión 0,99 el recall es 0 y es el resultado; dónde sí llega, sobre los puntajes de cada fold reajustado por el mismo camino (coinciden con el arnés a 1e-12); y en producción, un umbral propio elegido *in-sample* sobre referencia con la política del ADR 0011, no el del modelo completo |
| **Métrica** | Recall con precisión ≥ 0,99, la de decisión; PR-AUC y Brier al lado. Para dónde sí llega: el recall en los pisos 0,99, 0,95, 0,90, 0,80 y 0,50, y la mejor precisión de cualquier umbral que marque al menos un fraude, con su recall |
| **Resultado** | **Recall con precisión ≥ 0,99: 0 en los 5 folds**, y 0 también en los pisos 0,95, 0,90, 0,80 y 0,50. PR-AUC 0,0737 ± 0,0354 (0,0478 · 0,1281 · 0,0503 · 0,0507 · 0,0917); Brier 0,0049 ± 0,0011; ROC-AUC 0,8985 ± 0,0089. Mejor precisión por fold: 0,086 · 0,206 · 0,071 · 0,095 · 0,163, con recall de 0,39 a 0,55. Referencia *in-sample*: ningún umbral llega a 0,99, así que no hay punto de operación, y la mejor precisión es 0,172 con recall 0,497. Producción: al no haber umbral, el recall en el punto de decisión es 0; PR-AUC 0,354, ROC-AUC 0,916, Brier 0,0082; leída y no aplicada, la mejor precisión es 0,491 con recall 0,528 |
| **Baseline** | Sobre los mismos folds, `full` (el ensayo 5): recall 0,9970 ± 0,0036, PR-AUC 0,9978 ± 0,0028, Brier 0,000011 ± 0,000005. La regla: recall 0,9788 ± 0,0101, PR-AUC 0,9788 ± 0,0101, Brier 0,000015 ± 0,000008. Sobre producción, `full` al umbral fijo: recall 0,9976, PR-AUC 0,9997, Brier 0,000017; la regla: 0,9724, 0,9726 y 0,000179. El nulo: el test de permutación llega a PR-AUC 0,0112 como máximo (EVAL-004), y el predictor constante da un Brier igual a la prevalencia (EVAL-001) |
| **Número de particiones** | 5 folds temporales; referencia y producción, un valor cada una |
| **Dispersión** | Desviación estándar muestral (ddof = 1) entre los 5 folds |
| **Run de MLflow** | Corrida padre `2c3673e3a13442e39d7da047e8788d38` (`ablation__no_signature`), con las métricas de producción, y tres hijas por el arnés: `1edd21e008154703ba34fcbe218bc894` (`no_signature`), `41fd986d057b4d1abe1ab6f3812a19e5` (`full`) y `1aac5018cdcb4addb65b774d4e4384b8` (la regla) |
| **Reproducción** | `uv run python scripts/run_ablation.py` |
| **Interpretación** | **Casi todo el desempeño operativo es la firma.** Sin ella y sin el saldo de origen del que sale, el mismo modelo no tiene un solo umbral con precisión 0,99, ni siquiera 0,50, en ningún fold, ni *in-sample*, ni en producción: pasa de 0,9970 a 0 en la métrica de decisión. **Queda señal, pero no alcanza para operar.** El PR-AUC es de 69 a 169 veces la prevalencia de cada fold y está muy por encima del máximo del test de permutación, así que el modelo ordena el fraude por encima de lo legítimo en promedio. Pero la cabeza de su ranking está mezclada: lo mejor que logra un umbral es atrapar cerca de la mitad de los fraudes revisando de 4 a 13 legítimas por cada uno. Su Brier es de 5,6 a 9,3 veces el de un predictor constante, porque `scale_pos_weight` 34,9 infla el puntaje de muchas legítimas. En producción la mejor precisión sube a 0,49 por la misma aritmética de prevalencia que subió la del modelo completo. **No permite concluir** cuánto rendiría un modelo sin la firma con hiperparámetros propios, porque los del ensayo 5 se eligieron con la firma; ni separar la bandera del saldo de origen, que la ablación quita juntos; ni nada sobre datos reales, donde la firma no desaparecería sino que dejaría de ser exclusiva del fraude |
