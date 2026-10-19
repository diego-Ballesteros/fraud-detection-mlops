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
