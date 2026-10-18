# ADR 0003 — Cotas de dependencias

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Decisión

- Las dependencias directas declaran **el mínimo que el proyecto realmente requiere**, no
  la última versión disponible el día que corrió `uv add`.
- Toda dependencia cuya API nos importa lleva **piso de versión mayor**, para que un
  retroceso silencioso se vuelva un error de resolución.

## Mecanismo medido

`uv add` escribió `pyarrow>=26.0.0`; mlflow 3.17.1 declara `pyarrow<26`. El resolvedor
**no falla**: retrocede hasta mlflow 1.27.0, publicado en 2022. Sale en verde.

Es la clase de error de `docs/METHODOLOGY.md`, sección 7.6: el instrumento tiene éxito y
produce evidencia falsa, indistinguible de evidencia buena.

| Entrada de la resolución en seco | mlflow elegido |
|---|---|
| `pyarrow>=26.0.0` + `pandas>=2.2,<3` + `mlflow` | **1.27.0**, sin error |
| `pyarrow` sin cota + `pandas>=2.2,<3` + `mlflow` | 3.17.1 |
| Cotas de esta decisión + `mlflow>=3,<4` + resto de la pila | 3.17.1 (con pyarrow 25.0.1) |

## Alternativa descartada

Dejar las cotas que escribe `uv add`. Es el comportamiento por defecto y produjo
exactamente este problema.

## Consecuencia

Todo `uv add` de las fases siguientes revisa la cota que la herramienta escribió antes de
darla por buena, y declara piso de mayor donde la API importa.

## Política de pisos

Un piso no debe admitir una configuración que **falle**. No tiene que excluir toda
configuración que el proyecto no corra.

**Razón:** esto es una aplicación, no una librería. Nadie instala el proyecto en sus pisos
salvo un comando de verificación; lo que se corre es el lock. El piso existe para excluir la
región que rechazaríamos —la región mlflow 1.27—, no para describir una configuración
soportada.

**El test del piso es empírico, no de metadata:** se resuelve a los pisos, se instala y se
importa. Si importa, el piso sirve.

**Mecanismo general:** uv ignora las cotas superiores de `requires_python`, así que fijar el
intérprete crea pisos sobre dependencias transitivas: numba 0.53.1 declara `>=3.6,<3.10` y
aun así es elegible en 3.11. Las transitivas se acotan con `constraint-dependencies`, no
declarándolas como directas.

### Consecuencias medidas, con su veredicto

| Piso | Qué pasa en su piso | Veredicto |
|---|---|---|
| pyarrow 14 | Resuelve con numpy 2.4.6 y falla al importar | El piso tenía que subir |
| pyarrow 15 | Resuelve arrastrando numpy a 1.26.4 e importa | Habría sido aceptable. El piso quedó en 16 porque ya estaba aplicado y el lock no se movió, no porque 16 sea el número correcto |
| pandas 2.2.0 (`numpy<2`) | Resuelve con numpy 1.26.4 e importa | **Ninguna acción** |
| numba 0.57 (`numpy<1.25`) | Resuelve con numpy 1.24.4 y llvmlite 0.40.1. Importa en las mismas condiciones que numba 0.68.0 (ver la condición de entorno más abajo) | **Ninguna acción** |

Los casos de pandas y numba quedan **cerrados sin cambio**: que un piso admita una
configuración que el proyecto no corre no es razón para subirlo, mientras esa configuración
no falle.

## Cotas aplicadas

| Dependencia | Antes | Después | De dónde sale el piso | ¿Verificado en ejecución? |
|---|---|---|---|---|
| pandas | `>=2.2,<3` | sin cambio | ADR 0002 | **Importable** |
| pyarrow | `>=26.0.0` | `>=16` | Decisión tras medir los pisos 14 y 15 | **Importable** |
| pydantic | `>=2.14.1` | `>=2.3` | Derivado: es el requisito que declara pydantic-settings 2.2.0 | **Importable** |
| pydantic-settings | `>=2.16.0` | `>=2.2` | Medido: primera versión cuyo código contiene `env_ignore_empty`, que usa `config.py` | **Importable** |
| ipykernel, mypy, pre-commit, pytest, pytest-cov | `>=<última>` | `>=<mayor>.0` | Política: el inicio de la serie mayor en uso | El módulo importa; la herramienta no se corrió |
| ruff | `>=0.17.1` | `>=0.17` | Política: en 0.x, cada minor puede romper compatibilidad | El módulo importa (es un envoltorio del binario); el binario no se corrió |
| numba (constraint) | — | `>=0.57` | Primera versión con soporte de Python 3.11 | **No verificado en ejecución:** numba no está en el grafo hasta que entre shap, en la fase 2 |

**Cómo se verificó "importable" (2026-10-18):**

1. Los pisos, directos y de dev, se resolvieron con
   `uv pip compile --resolution lowest-direct --python-version 3.11 --python-platform x86_64-pc-windows-msvc`,
   que pinea 59 paquetes.
2. Ese conjunto exacto se instaló en un entorno aislado y efímero con
   `uv run --isolated --no-project --python 3.11 --with-requirements <archivo pineado>`.
3. Se importaron pandas 2.2.0, pyarrow 16.0.0, pydantic 2.3.0 y pydantic-settings 2.2.0,
   con numpy 1.26.4. Todas importaron.

Las versiones del lock no cambiaron, porque bajar un piso no mueve el lock.

## Otras instancias del mismo mecanismo, medidas al aplicar esta decisión

### Los pisos 14 y 15 de pyarrow

Toda la serie 14.0.x declara `numpy>=1.16.6`, sin techo. El resolvedor acepta pyarrow
14.0.x junto a numpy 2.4.6, y esa combinación falla al importar
(`ImportError: numpy.core.multiarray failed to import`).

La serie 15.0.x declara `numpy<2`: un retroceso hasta ella arrastra numpy a 1.26.4, y
pyarrow 15.0.0 importa con numpy 1.26.4. pyarrow 16.0.0 importa con numpy 2.4.6.

El veredicto de cada piso está en la tabla de la política.

### numba y llvmlite retroceden a versiones sin rueda

En la resolución universal de la pila completa, `pandas<3` hace que el resolvedor elija
**numba 0.53.1 y llvmlite 0.36.0** para todo Python <3.12. Ninguna de las dos tiene rueda
utilizable para CPython 3.11 en win_amd64. Sin `pandas<3`, la misma resolución elige numba
0.68.0 y llvmlite 0.50.0 para Windows y Linux. Por qué es `pandas<3` lo que inclina la
resolución hacia esa versión no está determinado (ADR 0005).

**Corrección aplicada.** Como numba no es dependencia directa, el piso va en
`[tool.uv] constraint-dependencies = ["numba>=0.57"]`. numba 0.57.0 y 0.57.1, y los
llvmlite 0.40.x que arrastran, tienen rueda cp311-win_amd64 (medido con el verificador del
ADR 0004).

Resolución universal de la pila completa:

| | numba | llvmlite |
|---|---|---|
| Con la constraint | 0.68.0 | 0.50.0 |
| Sin la constraint (control) | 0.53.1 | 0.36.0 |

La constraint queda puesta aunque hoy sea inerte: shap todavía no está en el grafo, y la
guarda tiene que existir antes de que llegue la dependencia.

### Observación: una constraint se propaga de lado

**Inferida de la metadata, no medida.** Que shap para 3.11 baje de 0.51.0 a 0.50.0 con la
constraint puesta sí está medido. La causa probable es que shap 0.51.0 declara `numba<0.63`
y `llvmlite<0.46` solo para macOS x86_64.

Vale registrarla porque muestra que una constraint se propaga de lado, por marcadores de
plataforma, y no solo hacia abajo. shap 0.50.0 tiene rueda cp311-win_amd64, así que no es
bloqueante.

### Condición de entorno medida (no es un problema de cotas)

En la máquina de desarrollo no está instalado el VC++ Redistributable x64: no hay
`msvcp140.dll` en System32 ni su clave de registro. Las extensiones de numba y `llvmlite.dll`
declaran `MSVCP140.dll` en su tabla de importaciones.

Por eso numba, igual que lightgbm, **falla si se importa solo** e importa si antes se importó
scikit-learn, que trae y precarga una `msvcp140.dll` con ese nombre exacto. Pasa lo mismo en
el piso (0.57.0) y en el techo (0.68.0), así que no cambia el veredicto del piso. **Está
pendiente de decisión y no se corrigió.**
