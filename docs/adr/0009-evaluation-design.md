# ADR 0009 — El diseño de la evaluación

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Contexto

La métrica y el esquema de validación se declaran antes de ver un solo resultado. Una
métrica elegida después de medir es la métrica que favorece el resultado que salió.

## Decisión

### 1 · El splitter: rangos de step con ventana expansiva

Los folds se cortan por rangos de `step`, nunca dentro de uno, con ventana expansiva:

| Fold | Entrena | Valida |
|---|---|---|
| 1 | 1–200 | 201–240 |
| 2 | 1–240 | 241–280 |
| 3 | 1–280 | 281–320 |
| 4 | 1–320 | 321–360 |
| 5 | 1–360 | 361–400 |

La ventana expansiva es exactamente la situación de producción: entrenar con toda la
historia y predecir el periodo siguiente.

**Por qué no `TimeSeriesSplit`.** Parte por índice de fila, así que un borde de fold puede
caer dentro de un `step`: el modelo entrenaría con parte de una hora y validaría con el
resto. Es una fuga pequeña y silenciosa.

Es el splitter que decide. Vive en `src/fraud_detection/models/evaluation.py`
(`ExpandingStepSplit`).

### 2 · La métrica de decisión: recall con precisión ≥ 0,99

**Por qué el PR-AUC global no sirve aquí.** Está saturado: cualquier modelo que aprenda la
firma llega a ~0,98, así que no discrimina. Lo que discrimina son los 92 fraudes que la
regla no cubre. El baseline (ADR 0007) puntúa 0,979 en esta misma escala.

- **Secundaria:** recall con precisión ≥ 0,95, por si ningún modelo alcanza el piso de 0,99.
- **Contexto, siempre reportado:** PR-AUC, ROC-AUC y Brier.

### 3 · El umbral de significancia práctica: +0,005 de recall

Declarado antes de medir. Por debajo de eso, dos modelos son el mismo modelo. Son unos 22
fraudes.

### 4 · No hay validación cruzada anidada

Tenemos un periodo temporal realmente retenido, producción, que es mejor que cualquier
bucle interno. Lo que hace falta no es anidar: es no tocar producción hasta la fase 4.

## Alternativa descartada

**`StratifiedKFold` como esquema primario.** Se mide igual, con 5 folds mezclados con
`settings.RANDOM_STATE`, y se reporta al lado del temporal para cada modelo. No es una
alternativa: es la medida del optimismo que introduce ignorar el tiempo.

---

**Nota del Ejecutor sobre la procedencia (2026-10-18):** los "unos 22 fraudes" del umbral de
significancia son 0,005 × 4.477, los positivos de todo el periodo de referencia. Medido en
este turno, las cinco validaciones temporales juntas (steps 201–400) tienen 2.223
positivos, así que allí +0,005 de recall equivale a unos 11 fraudes, y en un fold, que
tiene entre 408 y 465, a unos 2.
