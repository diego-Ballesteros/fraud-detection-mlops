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
>
> **Aclaración de alcance de la política del ADR 0003.** El piso cubre el comportamiento de
> la librería, no el de una combinación de frontend de notebook. `import matplotlib.pyplot`
> fallando en 3.6.0 es la librería rota, y por eso el piso subió a 3.7. Que el backend inline
> no registre el formateador PNG en la primera figura de la sesión es una interacción entre
> matplotlib y los hooks de display de IPython: la librería funciona y los tests pasan en el
> piso. Garantizar el renderizado exigiría acotar matplotlib-inline, ipykernel e IPython,
> congelando la pila entera para proteger una configuración que nadie corre. Lo reproducible
> es `uv sync --locked`, y en el lock la figura sale.

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
| `evidently` | `>=0.7.16,<0.8` | medido | Entró en la fase 4 por el reporte HTML de drift y desempeño (ADR 0012), con una medición previa: agrega 40 paquetes al lock (154 → 194), todos dependencias base, entre ellos un servidor web (litestar, uvicorn), nltk, statsmodels, plotly y un cliente de telemetría, y no cambia la versión de ninguno de los existentes. `uv add` escribió `>=0.7.23`, la última. La cota de mayor es la del ADR 0003 para la serie `0.x`: la API cambió entre parches de la 0.7. El piso se midió en el punto exacto, porque la política "la minor en uso" dejaba pasar configuraciones que fallan (ERR-020): 0.7.0 no resuelve (declara `numpy<2.1`, y shap 0.51 exige `numpy>=2`); los tests del reporte fallan en 0.7.1 y de 0.7.12 a 0.7.15, las versiones medidas, porque el `dict()` de un `Snapshot` no trae la clave `config` en sus métricas, y en 0.7.1 además `save_html` escribe un fragmento con iframe, no un documento; 0.7.16 pasa la suite. De 0.7.2 a 0.7.11 no se probaron una por una. Su telemetría la lee `DO_NOT_TRACK`, que `monitoring/report.py` fija antes de importarlo |
| `lightgbm` | `>=4.1` | medido | Inicio de la serie mayor en uso (ADR 0003), y subió a 4.1 por la política de ese ADR: con numpy 2.4.6, el que resuelve en los pisos, lightgbm 4.0.0 importa pero falla al ajustar sobre un DataFrame (`np.find_common_type was removed in the NumPy 2.0 release`); 4.1.0 ajusta. La verificación anterior no lo vio porque ningún test ajustaba LightGBM: lo encontró el test de `splits_on` (ERR-011, nota de estado). En la máquina de desarrollo, sin el VC++ Redistributable, `import lightgbm` solo, sin scikit-learn importado antes, falla con la versión del lock (`lib_lightgbm.dll (or one of its dependencies)`) y no con 4.0.0, que importa sola (medido en las dos). Por eso `models/estimators.py` lo importa dentro de su fábrica, y un test construye el modelo en un intérprete limpio. Desde el 2026-10-20 la máquina de desarrollo tiene el Redistributable x64 (14.51.36247, instalado por el humano), y lightgbm 4.7.0 importa solo; la importación diferida se queda, porque protege a un tercero que no lo tenga (ADR 0003, nota de estado de la fase 3) |
| `matplotlib` | `>=3.8.4` | medido | Subió dos veces por la política del ADR 0003. Primero de 3.6 a 3.7: sin el VC++ Redistributable, `matplotlib.pyplot` 3.6.0 importado solo da `DLL load failed while importing _path`. Después a 3.8.4: con numpy 2.4.6, el que resuelve en los pisos desde que entraron mlflow y lightgbm, `matplotlib.pyplot` 3.7.0 no importa (`numpy.core.multiarray failed to import`), y con `>=3.8` el resolvedor elige 3.8.4, que importa y renderiza una figura. La nota sobre el backend inline (aclaración de alcance del encabezado) se midió en 3.7.0; el notebook no se corrió en 3.8.4 |
| `mlflow` | `>=3.9,<4` | medido | La cota de mayor es la del ADR 0003: sin ella el resolvedor retrocede en silencio a mlflow 1.27.0. El piso subió de 3.0 a 3.9 por la política del ADR 0003 (un piso que admite una configuración que falla sube), con el punto exacto medido: con SQLAlchemy 2.1.5, la transitiva que se resuelve, mlflow 3.0.0 a 3.8.1 fallan al abrir el store sqlite (`cannot import name 'FallbackAsyncAdaptedQueuePool' from 'sqlalchemy.pool'`), y 3.9.0 pasa la suite. mlflow declara `pyarrow<26` |
| `optuna` | `>=5.0` | medido | Inicio de la serie mayor en uso (ADR 0003); es la versión que eligió la resolución en seco del ADR 0002. En el piso, 5.0.0, un estudio con `TPESampler` corre. Entró para la búsqueda del turno 2B |
| `pandas` | `>=2.2,<3` | política | Decisión del ADR 0002, con su costo medido en el ADR 0005; el piso 2.2.0 importa (ADR 0003). Con mlflow y lightgbm en el grafo, la resolución en los pisos elige 2.2.2: el conjunto resuelve numpy 2.4.6, y pandas 2.2.0 declara `numpy<2` |
| `pyarrow` | `>=16` | medido | El piso 14 resuelve con numpy 2 y falla al importar; 16 importa (ADR 0003) |
| `pydantic` | `>=2.3` | derivado | Es el requisito que declara `pydantic-settings` 2.2.0 (ADR 0003) |
| `pydantic-settings` | `>=2.2` | medido | Primera versión cuyo código tiene `env_ignore_empty`, que usa `config.py` (ADR 0003) |
| `scikit-learn` | `>=1.4.2` | medido | El piso de la fase 1 era 1.2, por la API: `Pipeline.set_output` y `OneHotEncoder(sparse_output=...)` aparecen en 1.2. Subió a 1.4.2 en la fase 2, por la política del ADR 0003, con el punto exacto medido: con mlflow o lightgbm en el grafo, cualquiera de las dos, la resolución en los pisos pasa de numpy 1.26.4 a numpy 2.4.6, y con numpy 2 la 1.2.0 falla al importar (`numpy.dtype size changed, may indicate binary incompatibility`) igual que la 1.3.0 (`cannot import name 'ComplexWarning'`). 1.4.2 importa. Ninguna de las dos dependencias declara `numpy>=2`; por qué la resolución se inclina hacia numpy 2 no está determinado (`docs/ERRORS_AND_LEARNINGS.md`, ERR-011) |
| `shap` | `>=0.51` | medido | Entró en la fase 3 por `TreeExplainer` en modo path-dependent (ADR 0011). `uv add` escribió `>=0.51.0`, la última; la cota revisada es la minor en uso, por ser serie `0.x` (ADR 0003). La resolución en seco sin cota del ADR 0003 había elegido 0.50.0 para Python 3.11; la cota que escribe la herramienta excluye 0.50.0, y el lock eligió 0.51.0. Esa versión declara `numba<0.63` y `llvmlite<0.46` solo para macOS x86_64, así que el lock se bifurca ahí (numba 0.62.1, llvmlite 0.45.1 y numpy 2.3.5); en Windows sigue numba 0.68.0, llvmlite 0.50.0 y numpy 2.4.6. La rueda instalada es `cp311-cp311-win_amd64`. Declara `numpy>=2` |

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
| `numba` | `>=0.57` | medido | Sin ella, la resolución universal retrocede a numba 0.53.1, sin rueda cp311 para Windows; 0.57 es la primera con soporte de Python 3.11 (ADR 0003, ADR 0004). Desde la fase 3 shap la trae al grafo, y la constraint dejó de ser inerte. El 2026-10-20, en un intérprete limpio, numba 0.57.0 y 0.68.0 importan solas con el VC++ Redistributable instalado; antes de instalarlo fallaban las dos (ADR 0003, nota de estado de la fase 3) |

## Última verificación en los pisos

2026-10-21, al agregar evidently. La resolución en los pisos fija 181 paquetes, entre ellos
evidently 0.7.16, junto a scikit-learn 1.4.2, pandas 2.2.2, lightgbm 4.1.0, mlflow 3.9.0, shap
0.51.0 y numpy 2.4.6. Instalado ese conjunto con el procedimiento de abajo, la suite completa
pasa (244 tests), y ejerce lo que la dependencia aporta: un `Report` con el drift por PSI y la
clasificación al umbral, cuyas cifras se comparan con las propias, y el HTML escrito a disco.
Con el piso en `>=0.7`, la misma verificación resolvía evidently 0.7.1 y fallaban 3 tests; el
punto exacto se buscó resolviendo los pisos con cada versión candidata (ERR-020).

La verificación anterior fue el 2026-10-20, al agregar shap. La resolución en los pisos fijó
140 paquetes, entre ellos shap 0.51.0, numba 0.68.0 y llvmlite 0.50.0, junto a scikit-learn
1.4.2, pandas 2.2.2, lightgbm 4.1.0, mlflow 3.9.0 y numpy 2.4.6. Instalado ese conjunto con el
procedimiento de abajo, la suite completa pasó (201 tests). Esa vez la suite sí ejercía lo que
la dependencia aportaba: `TreeExplainer` sobre un LightGBM ajustado, la aditividad de sus
valores con un control que debe fallar, y las figuras de shap.

La verificación anterior fue el 2026-10-18, al agregar optuna. La resolución en los pisos
fijó 136 paquetes, entre ellos scikit-learn 1.4.2, mlflow 3.9.0, lightgbm 4.1.0, optuna
5.0.0, matplotlib 3.8.4, pandas 2.2.2, numpy 2.4.6, pyarrow 16.0.0, pydantic 2.9.0,
pydantic-settings 2.2.0 y SQLAlchemy 2.1.5. Instalado ese conjunto, la suite completa pasó
(155 tests), y además se comprobó lo que la suite no ejercía: LightGBM ajusta sobre un
DataFrame, `matplotlib.pyplot` importa y renderiza una figura, y un estudio de optuna corre.

La anterior a esa, al agregar mlflow y lightgbm, había dado la suite en verde (142
tests) con lightgbm 4.0.0 y matplotlib 3.7.0 en los pisos, y los dos estaban rotos con
numpy 2: ningún test ejercía esas rutas. Una suite en verde en los pisos prueba lo que la
suite ejecuta, no la dependencia entera.

## Cómo se verifica un piso

1. Resolver el conjunto en los pisos, para la plataforma del proyecto:
   `uv pip compile pyproject.toml --group dev --resolution lowest-direct --python-version 3.11 --python-platform x86_64-pc-windows-msvc -o <archivo>`.
2. Instalar ese conjunto exacto en un entorno aislado y efímero:
   `uv run --isolated --no-project --python 3.11 --with-requirements <archivo> --with-editable . <comando>`.
3. Importar el paquete y, donde la API importa, correr los tests que la usan.
4. Un piso que admite una configuración que falla sube; uno que admite una configuración que
   el proyecto no corre, pero que no falla, se queda (ADR 0003).
