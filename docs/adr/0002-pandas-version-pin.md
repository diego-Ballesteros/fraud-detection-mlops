# ADR 0002 — Pin de pandas a `>=2.2,<3`

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Decisión

`pandas` se fija a `>=2.2,<3` en `pyproject.toml`. Versión instalada tras re-lockear:
**pandas 2.3.3**.

**Justificación de la medición:** el pin **no** quedó justificado por la resolución en
seco; es una decisión conservadora sin ganancia medida.

## Por qué

- El proyecto no gana nada de pandas 3 que no pueda obtener con `dtype_backend` de
  pyarrow, que ya es dependencia.
- El costo de descubrir una incompatibilidad con SHAP o LightGBM en la fase 2, con los
  notebooks 01 y 02 ya escritos contra la semántica de pandas 3, es un turno de
  corrección más una reverificación.
- El costo de fijarlo es cercano a cero.

## Medición previa: resolución en seco de la pila de ML

Se resolvió la pila de las fases siguientes sin instalar nada ni escribir
`pyproject.toml` ni `uv.lock`:

```bash
printf '%s\n' pandas pyarrow pydantic pydantic-settings scikit-learn lightgbm xgboost \
  shap optuna mlflow imbalanced-learn \
  | uv pip compile --universal --python-version 3.11 --annotation-style line -
```

Con todas las librerías presentes, el resolvedor eligió **pandas 3.0.6**. Restricciones
sobre pandas declaradas por cada versión resuelta, leídas de la metadata de PyPI con un
script ad hoc no versionado:

| Librería (versión resuelta) | Restricción declarada sobre pandas |
|---|---|
| scikit-learn 1.9.1 | Ninguna |
| lightgbm 4.7.0 | `pandas>=1.3.4`, solo en los extras `pandas` y `dask` |
| xgboost 3.2.0 | `pandas>=1.2` en el extra `pandas`; sin cota en el extra `dask` |
| shap 0.51.0 | `pandas`, sin cota |
| optuna 5.0.0 | `pandas` sin cota, solo en los extras `document` y `optional` |
| mlflow 3.17.1 | `pandas<4` |
| imbalanced-learn 0.14.2 | `pandas>=2.0.3,<3`, solo en los extras `docs` y `optional` |

Ninguna limita pandas por debajo de 3 en sus dependencias base. La medición cubre
restricciones **declaradas**, no compatibilidad en ejecución.

## Condición para levantar el pin

Se levanta cuando se cumplan las dos cosas:

1. **SHAP, LightGBM y XGBoost declaran soporte explícito para pandas 3** en su changelog
   o en sus release notes. La ausencia de una cota superior no cuenta: hoy ninguna de las
   tres la tiene.
2. Existe una medición de memoria que justifica el cambio.

---

**Nota de estado (2026-10-18):** la afirmación "el costo de fijarlo es cercano a cero" quedó
medida como falsa en esta misma fase: el pin hace que la resolución universal elija numba y
llvmlite sin rueda para Windows. La supersede el ADR 0005; la decisión del pin se mantiene.
