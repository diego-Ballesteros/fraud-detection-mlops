# ADR 0004 — Pin de Python a 3.11

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Decisión

`requires-python = ">=3.11,<3.12"`.

La justificación original, "mejor soporte de ruedas precompiladas en Windows para
LightGBM, XGBoost y SHAP", **no estaba medida cuando se escribió**. La medición de abajo
es la evidencia.

## Evidencia: ruedas para CPython 3.11 en win_amd64

**Cómo se midió:** para cada versión, la lista de archivos publicados se obtuvo de la API
JSON de PyPI (`https://pypi.org/pypi/<paquete>/<versión>/json`, campo `urls`), sin instalar
nada. Cada nombre de rueda se evaluó con `packaging.tags` contra el conjunto completo de
tags que acepta CPython 3.11 en win_amd64. Las versiones son las que se resuelven para
3.11 en la resolución en seco de la pila del proyecto.

| Paquete | Versión | Rueda utilizable | Tipo |
|---|---|---|---|
| shap | 0.50.0 | Sí | cp311-win_amd64 |
| xgboost | 3.2.0 | Sí | py3-none-win_amd64 |
| lightgbm | 4.7.0 | Sí | py3-none-win_amd64 |
| numba | 0.68.0 | Sí | cp311-win_amd64 |
| llvmlite | 0.50.0 | Sí | cp311-win_amd64 |
| scikit-learn | 1.9.1 | Sí | cp311-win_amd64 |
| scipy | 1.17.1 | Sí | cp311-win_amd64 |
| numpy | 2.4.6 | Sí | cp311-win_amd64 |
| optuna | 5.0.0 | Sí | py3-none-any |
| imbalanced-learn | 0.14.2 | Sí | py3-none-any |

**shap cambió de 0.51.0 a 0.50.0:** con la constraint `numba>=0.57` (ADR 0003), la
resolución para 3.11 elige shap 0.50.0. Las otras nueve versiones no cambiaron, y shap
0.50.0 tiene rueda cp311-win_amd64 (mismo verificador).

**Control negativo del verificador:** shap 0.54.1, que exige Python ≥3.12, y numba 0.56.4,
anterior al soporte de 3.11, dan "sin rueda utilizable". El verificador puede fallar.

## Alternativa descartada

**Python 3.12**, que da shap 0.54.1 y xgboost 3.4.1: las últimas versiones de las dos
exigen Python ≥3.12.

Descartada porque el proyecto no usa nada de esas versiones, y cambiar el intérprete es la
clase de cambio que abre problemas de compilación en Windows.

## Condición para moverse

- Que una medición de ruedas falle, o
- que una fase necesite una API que solo exista en Python ≥3.12.

## Riesgo medido: decidido y aplicado

Sin una cota sobre numba, la resolución universal de la pila completa elige **numba 0.53.1
y llvmlite 0.36.0** para Python 3.11, y **ninguna de las dos tiene rueda utilizable** para
cp311 en win_amd64 (mismo verificador).

**Decidido y aplicado** sin mover el intérprete: `[tool.uv] constraint-dependencies =
["numba>=0.57"]`. Con la constraint, la resolución elige numba 0.68.0 y llvmlite 0.50.0, las
de la tabla; sin ella (control) vuelve a 0.53.1 y 0.36.0. El mecanismo está en el ADR 0003,
y el costo del pin de pandas que lo dispara, en el ADR 0005.
