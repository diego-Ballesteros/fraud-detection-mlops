# ADR 0005 — El costo medido del pin de pandas

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.
- **Supersede:** la afirmación de costo del ADR 0002. Su decisión se mantiene.

---

## Decisión

La decisión del ADR 0002 se mantiene: pandas sigue en `>=2.2,<3`.

La razón original no cambió: compatibilidad declarada no es compatibilidad en ejecución, y
el proyecto no necesita nada de pandas 3 que no dé el backend de pyarrow.

## Lo que cambia: el costo

Con `pandas<3`, la resolución universal elige **numba 0.53.1 y llvmlite 0.36.0**, y ninguna
de las dos tiene rueda cp311-win_amd64.

| Variante de la resolución en seco | numba | llvmlite |
|---|---|---|
| V0: pila completa, universal, pandas sin cota | 0.68.0 (0.53.1 solo en macOS x86_64) | 0.50.0 (0.36.0 solo en macOS x86_64) |
| V1: V0 + `pandas<3` | 0.53.1, todas las plataformas | 0.36.0, todas las plataformas |
| V3: con `pandas<3`, solo Windows (no universal) | 0.68.0 | 0.50.0 |

- **El disparador es el pin** (V0 contra V1).
- **Ocurre solo en resolución universal** (V3 sale bien), y `uv.lock` es universal.

El costo real es **una línea** de `constraint-dependencies` (`numba>=0.57`), medida con y
sin la constraint: con ella, numba 0.68.0 y llvmlite 0.50.0; sin ella, 0.53.1 y 0.36.0
(ADR 0003).

## Alternativa descartada

**Levantar el pin.** Resuelve el problema de numba y devuelve el riesgo que el pin existía
para evitar. Una línea de constraint contra un riesgo no medido en ejecución sigue siendo el
lado barato.

## Lo que quedó sin determinar

Por qué `pandas<3` inclina la resolución hacia numba 0.53.1. **El síntoma está medido; el
mecanismo no.**

## Error de método

El ADR 0002 afirmó un costo sin medirlo. Un cambio de cota es un cambio en el grafo de
resolución, y el grafo es medible.
