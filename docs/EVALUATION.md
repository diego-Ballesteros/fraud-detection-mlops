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
