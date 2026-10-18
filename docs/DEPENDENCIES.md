# Dependencias

> Contrato: la tabla viva de las dependencias directas del proyecto y de sus pisos de
> versión, con la procedencia de cada piso. Es el dueño único de esa tabla.
>
> - La **política** de cotas es el ADR 0003 (`docs/adr/0003-dependency-bounds.md`). Los pisos
>   de la fase 0 se justificaron allí primero; la tabla de ese ADR es el registro histórico de
>   la fase 0 y no se reescribe. Esta es la tabla viva.
> - **Todo `uv add` actualiza este documento en el mismo turno**, con la cota revisada, no la
>   que escribe la herramienta.
> - Las versiones exactas viven en `uv.lock`; aquí solo van las cotas declaradas.

---

## Procedencia de un piso

| Etiqueta | Significado |
|---|---|
| **medido** | Verificado ejecutando: se resolvió en el piso, se instaló y se importó o se corrió, o se midió la metadata publicada (por ejemplo, qué ruedas existen) |
| **derivado** | Sale de un requisito que declara otra dependencia |
| **política** | Lo fija la regla del ADR 0003 (el inicio de la serie mayor en uso; en las series `0.x`, la minor en uso), sin medición propia del piso |

## Dependencias de ejecución

| Dependencia | Cota declarada | Procedencia | Justificación |
|---|---|---|---|
| `matplotlib` | `>=3.7` | medido | El piso 3.6 admitía una configuración que falla: en la máquina de desarrollo, sin el VC++ Redistributable (ADR 0003), `matplotlib.pyplot` 3.6.0 importado solo da `DLL load failed while importing _path`. Desde 3.7.0 importa solo (medido en 3.7.0, 3.8.0, 3.9.0 y 3.10.0). En el piso, 3.7.0, el notebook 01 corre sin errores, pero con matplotlib-inline 0.2.2 la primera figura de cada sesión de kernel sale como texto `<Figure ...>`, sin imagen; con la versión del lock salen todas |
| `pandas` | `>=2.2,<3` | política | Decisión del ADR 0002, con su costo medido en el ADR 0005; el piso 2.2.0 importa (ADR 0003) |
| `pyarrow` | `>=16` | medido | El piso 14 resuelve con numpy 2 y falla al importar; 16 importa (ADR 0003) |
| `pydantic` | `>=2.3` | derivado | Es el requisito que declara `pydantic-settings` 2.2.0 (ADR 0003) |
| `pydantic-settings` | `>=2.2` | medido | Primera versión cuyo código tiene `env_ignore_empty`, que usa `config.py` (ADR 0003) |
| `scikit-learn` | `>=1.2` | medido | `Pipeline.set_output` y `OneHotEncoder(sparse_output=...)`, que usa el Pipeline, aparecen en 1.2: en 1.1.3 faltan las dos. En los pisos de todas las dependencias (scikit-learn 1.2.0, pandas 2.2.0, numpy 1.26.4), la suite completa pasa (84 tests, 2026-10-18) |

## Dependencias de desarrollo

| Dependencia | Cota declarada | Procedencia | Justificación |
|---|---|---|---|
| `ipykernel` | `>=7.0` | política | Inicio de la serie mayor en uso (ADR 0003) |
| `mypy` | `>=2.0` | política | Inicio de la serie mayor en uso (ADR 0003) |
| `nbclient` | `>=0.11` | política | Serie `0.x`: la minor en uso. Ejecuta los notebooks desde un kernel limpio (`task notebooks`), porque la reproducibilidad por un tercero es requisito del proyecto, y un comando que solo existió en un turno no es reproducible por nadie. Medido en el piso: un notebook que debe pasar sale con 0 y uno que debe fallar, con 1. En Windows exige el modo UTF-8 de Python (ERR-010) |
| `nbformat` | `>=5.0` | política | Inicio de la serie mayor en uso. La resolución en los pisos elige 5.2.0, no 5.0, porque `nbclient` 0.11 lo exige (medido) |
| `pre-commit` | `>=4.0` | política | Inicio de la serie mayor en uso (ADR 0003) |
| `pytest` | `>=9.0` | política | Inicio de la serie mayor en uso (ADR 0003) |
| `pytest-cov` | `>=7.0` | política | Inicio de la serie mayor en uso (ADR 0003) |
| `ruff` | `>=0.17` | política | Serie `0.x`: la minor en uso (ADR 0003) |

## Restricciones sobre dependencias transitivas

No son dependencias directas: van en `[tool.uv] constraint-dependencies`.

| Dependencia | Cota | Procedencia | Justificación |
|---|---|---|---|
| `numba` | `>=0.57` | medido | Sin ella, la resolución universal retrocede a numba 0.53.1, sin rueda cp311 para Windows; 0.57 es la primera con soporte de Python 3.11 (ADR 0003, ADR 0004) |

## Cómo se verifica un piso

1. Resolver el conjunto en los pisos, para la plataforma del proyecto:
   `uv pip compile pyproject.toml --group dev --resolution lowest-direct --python-version 3.11 --python-platform x86_64-pc-windows-msvc -o <archivo>`.
2. Instalar ese conjunto exacto en un entorno aislado y efímero:
   `uv run --isolated --no-project --python 3.11 --with-requirements <archivo> --with-editable . <comando>`.
3. Importar el paquete y, donde la API importa, correr los tests que la usan.
4. Un piso que admite una configuración que falla sube; uno que admite una configuración que
   el proyecto no corre, pero que no falla, se queda (ADR 0003).
