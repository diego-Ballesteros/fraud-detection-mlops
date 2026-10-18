# ADR 0007 — La firma del simulador como baseline

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Decisión

La regla

```
type ∈ {TRANSFER, CASH_OUT}  ∧  amount == oldbalanceOrg
```

es el **baseline obligatorio de todo el proyecto**: precisión 1,000 y recall 0,980 en
referencia. Toda métrica de la fase 2 se reporta contra ella.

La medición está en `docs/DATA_DICTIONARY.md`, sección 4.a.

## Consecuencias

**Sobre el diseño de la fase 2.** El PR-AUC global está saturado por la región que cubre la
regla, así que la métrica de decisión tiene que ser el punto de operación **más allá** de
la regla. Esa métrica se declara antes de ver resultados, en la fase 2.

**Sobre la fase 3.** Tenemos verdad conocida. Si SHAP no encuentra la regla, el problema es
el explicador. Es la única oportunidad del proyecto de validar el instrumento de
interpretabilidad contra una respuesta ya sabida.

**Sobre la fase 4.** La pregunta de drift se vuelve concreta: si la regla mantiene su
precisión y su recall en los steps 401+.

## Ablación declarada

Una configuración sin `oldbalanceOrg` ni la derivada de igualdad
(`amount_equals_origin_balance`), para cuantificar cuánto del desempeño es la firma.

- Se corre una vez por familia de modelo; no es una búsqueda.
- El Pipeline del turno siguiente tiene que ser parametrizable para permitirla, con **un
  solo objeto Pipeline configurable**, no dos implementaciones.

## Alternativa descartada

**Dejar que la firma domine y reportar ~0,98 sin más.** Es honesto solo si se documenta,
pero vacía las cuatro secciones de la rúbrica:

- los hiperparámetros no se separan,
- la interpretabilidad se vuelve trivial,
- y el monitoreo no tiene nada que detectar.
