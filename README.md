# Fraud Detection MLOps

Detección de fraude en transacciones móviles (dataset PaySim) con optimización de
hiperparámetros, interpretabilidad local y global, y monitoreo de desempeño y drift.

**Estado:** en desarrollo.

## Requisitos previos

- [UV](https://docs.astral.sh/uv/), el único gestor de paquetes del proyecto.
- Python 3.11. Si no está instalado, `uv sync` lo descarga.
- Opcional: [Task](https://taskfile.dev/) para los atajos de `Taskfile.yml`. Cada tarea es
  un comando `uv run` que también se puede ejecutar a mano.

## Setup local

```bash
git clone git@github.com:diego-Ballesteros/fraud-detection-mlops.git
cd fraud-detection-mlops
uv sync                        # crea .venv e instala el paquete en modo editable
uv run pre-commit install      # activa los hooks de calidad en cada commit
```

Copiar la plantilla de variables de entorno y completarla localmente (`.env` nunca se
versiona):

```bash
cp .env.example .env           # PowerShell: Copy-Item .env.example .env
```

El CSV de PaySim se coloca en `data/raw/`. Esa carpeta está gitignoreada: los datos no se
versionan.
