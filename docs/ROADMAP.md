# Proyecto Curso III — Fraud Detection MLOps

> Detección de fraude en transacciones móviles con optimización de hiperparámetros,
> interpretabilidad y monitoreo de drift sobre un periodo de producción real.
>
> **Especialización MLE 2026 — Curso III**
> Documento de diseño y plan de ejecución.

---

## Tabla de contenidos

1. [El problema](#1-el-problema)
2. [Qué se entrega y qué pesa](#2-qué-se-entrega-y-qué-pesa)
3. [Dataset y corte temporal](#3-dataset-y-corte-temporal)
4. [Arquitectura de la solución](#4-arquitectura-de-la-solución)
5. [Estrategia de evaluación](#5-estrategia-de-evaluación)
6. [Stack técnico](#6-stack-técnico)
7. [Estructura del repositorio](#7-estructura-del-repositorio)
8. [Estrategia de Git](#8-estrategia-de-git)
9. [Plan por fases](#9-plan-por-fases)
10. [Trazabilidad contra la rúbrica](#10-trazabilidad-contra-la-rúbrica)
11. [Riesgos y mitigaciones](#11-riesgos-y-mitigaciones)

---

## 1. El problema

### 1.1 Contexto

Una plataforma de pagos móviles procesa millones de transacciones. Una fracción diminuta es
fraudulenta, y hay que identificarla **en el momento en que ocurre**, no después.

El error tiene dos caras asimétricas:

| Error | Consecuencia |
|---|---|
| **Falso negativo** — dejar pasar un fraude | Pérdida directa del monto transferido |
| **Falso positivo** — bloquear una transacción legítima | Fricción con el cliente, costo operativo de revisión |

### 1.2 Definición formal

- **Tipo:** aprendizaje supervisado, clasificación binaria
- **Unidad de análisis:** una transacción
- **Target:** `isFraud` ∈ {0, 1}
- **Desbalance esperado:** extremo — del orden de 0,1% de clase positiva
- **Salida requerida:** una probabilidad ordenable, para poder priorizar revisiones

> **Lo que distingue a este problema del anterior.** Con 0,1% de positivos, la accuracy de
> un clasificador trivial supera el 99,8%. El desbalance deja de ser un detalle de método y
> pasa a ser la propiedad central del problema: las técnicas de remuestreo que el proyecto
> anterior midió y descartó sobre una prevalencia del 22% **vuelven a estar sobre la mesa y
> hay que remedirlas**, porque fueron diseñadas exactamente para este régimen.

### 1.3 El riesgo principal: la fuga

Antes de medir nada hay que resolver qué columnas existían **antes** de la transacción.
Dos candidatas a fuga, a verificar en la fase 1 y no asumir:

- **El indicador que produce el propio sistema de detección.** No es información previa a
  la transacción: es la salida de una regla del sistema. Si entra como feature, el modelo
  aprende la regla en vez del fenómeno.
- **Las columnas de saldo posteriores al movimiento.** Pueden guardar inconsistencias
  aritméticas que correlacionen casi perfectamente con el fraude por cómo se construyó la
  simulación.

La prueba del target barajado es la defensa y se corre antes que cualquier comparación de
modelos.

---

## 2. Qué se entrega y qué pesa

| Componente | Peso | Qué exige |
|---|---|---|
| Repositorio GitHub | 10% | Estructura definida, los cuatro notebooks, datos, carpeta de la app de monitoreo |
| **Técnicas avanzadas de ML** | **50%** | Optimización de hiperparámetros · registro en MLflow · **interpretación local y global** del algoritmo final |
| **Monitoreo de modelos** | **30%** | Solución que monitoree desempeño y drift de variables, **con su interpretación** |
| Documentación | 10% | Markdown, notebooks, docstrings, naming |
| **Reto ML 1** | +10% | Pipeline de **AutoML**: construye un espacio de búsqueda, optimiza y selecciona el mejor algoritmo entre varios |
| **Reto ML 2** | +10% | **Aplicación web de monitoreo** que calcule métricas en tiempo real |

### Los cuatro notebooks obligatorios

`01_preprocessing` · `02_training` · `03_interpretability` · `04_monitoring`

### Por qué los dos retos entran en el plan por defecto

No son proyectos extra: son la forma correcta de hacer lo obligatorio.

**Reto 1.** El 50% obligatorio ya pide optimización de hiperparámetros. Si el espacio de
búsqueda incluye **el algoritmo como un hiperparámetro más** en vez de tunear uno fijo, el
mismo trabajo cubre las dos cosas.

**Reto 2.** El 30% obligatorio ya pide una carpeta de app de monitoreo y un reporte HTML. Un
dashboard sobre los mismos artefactos cubre ambos.

---

## 3. Dataset y corte temporal

**PaySim** — simulación de transacciones móviles con etiqueta de fraude.

| Atributo | Valor |
|---|---|
| Registros | ~6,3 millones |
| Columnas | 11 |
| Clase positiva | ~0,13% (a verificar) |
| Ubicación | `data/raw/`, gitignoreado |

### El corte temporal es el regalo de este proyecto

El enunciado lo define y **no se negocia**:

```
step <= 400   →  referencia / entrenamiento
step >= 401   →  producción
```

Esto es exactamente lo que al proyecto anterior le faltó: allí el Model Card tuvo que
declarar *"no hay validación fuera de tiempo, y es bloqueante"*. Aquí la hay, y tiene dos
consecuencias:

1. **La validación es honesta por construcción.** El modelo se evalúa sobre un periodo que
   no vio.
2. **El drift que midamos es real**, no un control negativo. En el proyecto anterior el PSI
   salió ≈ 0 porque el flujo era submuestra de la referencia; aquí las dos poblaciones son
   genuinamente distintas y el detector tiene algo que detectar.

### Lo que hay que medir antes de modelar

- La prevalencia exacta, en referencia y en producción por separado
- Si el fraude ocurre solo en ciertos tipos de transacción — cambiaría el planteamiento
- Si las columnas de saldo son aritméticamente consistentes
- Cuántos pasos cubre cada periodo y cuántas transacciones hay en cada uno

---

## 4. Arquitectura de la solución

```
   FUENTE              PREPROCESAMIENTO           MODELADO
┌──────────┐        ┌──────────────────┐     ┌──────────────────┐
│ PaySim   │───────►│ Validación       │────►│ Baseline trivial │
│ (CSV)    │        │ Contrato de datos│     │ AutoML: espacio  │
└──────────┘        │ Feature eng.     │     │  de búsqueda     │
                    │ → parquet        │     │  multi-algoritmo │
                    └──────────────────┘     └────────┬─────────┘
                       step <= 400                    │
                                                      ▼
                                            ┌──────────────────┐
                                            │     MLflow       │
                                            │     Registry     │
                                            └────────┬─────────┘
                                 ┌───────────────────┼───────────────┐
                                 ▼                   ▼               ▼
                      ┌──────────────────┐  ┌────────────────┐  ┌─────────┐
                      │ INTERPRETABILIDAD│  │   MONITOREO    │  │  App    │
                      │ global: FI, SHAP │  │ drift: PSI/KS  │  │  web    │
                      │ PDP, ICE         │  │ desempeño en   │  │ (Reto 2)│
                      │ local: SHAP,LIME │  │ step >= 401    │  └─────────┘
                      └──────────────────┘  └────────────────┘
```

Los tres consumidores cargan **el mismo artefacto** del registry. Nunca se reimplementa una
transformación.

---

## 5. Estrategia de evaluación

### 5.1 Métricas del modelo

| Métrica | Por qué |
|---|---|
| **PR-AUC** | Métrica principal. Con 0,1% de positivos, ROC-AUC es engañosamente optimista |
| Recall a precisión fija | El negocio revisa una cola de capacidad limitada |
| Precision@k | "De las k transacciones que marqué, ¿cuántas eran fraude?" |
| ROC-AUC | Comparabilidad con la literatura |
| Matriz de confusión con costos | Traduce el modelo a pérdidas, no a porcentajes |

**Accuracy se reporta una sola vez**, al lado del clasificador trivial, para dejar
demostrado por qué no decide nada.

### 5.2 Validación

**Split temporal, no aleatorio.** `step <= 400` para entrenar y optimizar;
`step >= 401` para la evaluación de producción. Dentro del periodo de referencia, la
validación cruzada del AutoML respeta el orden temporal.

### 5.3 Métricas de monitoreo

| Qué | Cómo |
|---|---|
| **Drift de variables** | PSI por feature, con KS al lado. Umbral declarado antes de medir |
| **Drift del target** | Cómo cambia la prevalencia de fraude entre periodos |
| **Desempeño en producción** | PR-AUC y recall sobre `step >= 401`, contra la referencia |
| **Degradación en el tiempo** | La métrica por ventanas de steps, no un único número |

> **El detector se prueba con controles positivos.** Un detector de drift validado solo
> contra distribuciones idénticas no está validado. Los tests inyectan desplazamientos
> conocidos y verifican que el PSI sube de forma monótona.

---

## 6. Stack técnico

| Capa | Herramienta | Nota |
|---|---|---|
| Gestión de paquetes | **UV** | Exclusivamente |
| Datos | pandas, pyarrow | Parquet y dtypes explícitos desde el día uno |
| ML | scikit-learn, LightGBM, XGBoost | Candidatos del espacio de búsqueda |
| Desbalance | imbalanced-learn | Se mide, no se asume |
| **Optimización** | **Optuna** | El algoritmo entra como hiperparámetro (Reto 1) |
| **Interpretabilidad** | **SHAP, LIME, sklearn** | Global: feature importance, PDP, ICE. Local: SHAP, LIME |
| **Monitoreo** | **Evidently** | Reportes HTML de drift y desempeño |
| Tracking | **MLflow + DagsHub** | Registry incluido |
| **App** | **Streamlit** | Dashboard de monitoreo (Reto 2) |
| Testing | pytest, pytest-cov | Objetivo: ≥80% |
| Calidad | ruff, mypy | Pre-commit hooks |
| CI | GitHub Actions | Lint + tipos + tests |

**Fuera del alcance, y es deliberado:** contenedores y despliegue en nube. La rúbrica del
Curso III no los menciona en ningún punto, ni en lo obligatorio ni en los dos retos.

---

## 7. Estructura del repositorio

```
fraud-detection-mlops/
├── README.md                      ← el entregable principal
├── pyproject.toml  uv.lock  Taskfile.yml
├── .gitignore  .env.example  .pre-commit-config.yaml
│
├── docs/
│   ├── METHODOLOGY.md
│   ├── ROADMAP.md
│   ├── GIT_STRATEGY.md
│   ├── DATA_DICTIONARY.md         ← requisito: incluye cuándo existe cada columna
│   ├── MODEL_CARD.md
│   ├── EVALUATION.md
│   ├── ERRORS_AND_LEARNINGS.md
│   ├── adr/
│   └── analysis/
│
├── data/
│   ├── raw/.gitkeep               ← el CSV, gitignoreado
│   └── processed/.gitkeep
│
├── notebooks/                     ← los cuatro, requisito explícito
│   ├── 01_preprocessing.ipynb
│   ├── 02_training.ipynb
│   ├── 03_interpretability.ipynb
│   └── 04_monitoring.ipynb
│
├── src/fraud_detection/
│   ├── config.py
│   ├── data/          loader · schema · validator · preprocessor
│   ├── features/      builder
│   ├── models/        estimators · search_space · evaluation · tracking · registry
│   ├── explain/       global_explain · local_explain
│   └── monitoring/    drift · performance · report
│
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_training.py            ← el AutoML
│   ├── run_prediction.py
│   ├── run_leakage_check.py
│   ├── run_interpretability.py
│   └── run_monitoring.py          ← genera el reporte HTML
│
├── app/                           ← requisito explícito (Reto 2)
│   └── monitoring_app.py
│
├── reports/deliverables/          ← el Report HTML del monitoreo
├── tests/
└── .github/workflows/ci.yml
```

---

## 8. Estrategia de Git

**Modelo:** GitHub Flow con rama `develop` persistente.

```
main ──────●───────────────────────────●─────► v1.0.0
            \                         /
develop      ●──●──●──●──●──●──●──●──●
              \  /    \  /    \  /
        feature/*  feature/*  feature/*
```

- `main` protegida; solo recibe merges desde `develop` vía PR
- Una rama por fase: `feature/00-foundation`, `feature/01-data-and-eda`, …
- Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`
- Merge commits, no squash — el historial es evidencia evaluada
- Tag `v1.0.0` al final, con release notes

**Nota de equivalencia:** la rama de integración se llama `develop` siguiendo la convención
de git-flow. Si el enunciado menciona una rama "Development", es esta misma rama.

---

## 9. Plan por fases

> Cada fase termina con una rama mergeada a `develop` vía PR.
> El avance se mide por hitos cumplidos, no por días.

---

### FASE 0 — Fundación · `feature/00-foundation`

**Objetivo:** esqueleto completo antes de tocar una línea de modelado.

- [ ] Estructura de carpetas, `pyproject.toml`, `uv sync`
- [ ] `.gitignore`, `.env.example`, pre-commit, `Taskfile.yml`
- [ ] CI en GitHub Actions: lint + tipos + tests
- [ ] `config.py` con `RANDOM_STATE` y las rutas del proyecto
- [ ] `docs/GIT_STRATEGY.md`
- [ ] Registros vacíos: `EVALUATION.md`, `ERRORS_AND_LEARNINGS.md`
- [ ] Cuenta de DagsHub conectada, `.env` con las credenciales de MLflow

> ✅ **Hito 0:** repo profesional funcionando, CI en verde.

---

### FASE 1 — Datos y EDA · `feature/01-data-and-eda`

**Objetivo:** entender el dataset y cerrar el contrato de datos. Es la fase donde se decide
qué columnas pueden entrar al modelo.

- [ ] Carga con dtypes explícitos (6,3M filas; pandas en memoria duele si no)
- [ ] `docs/DATA_DICTIONARY.md` con las 11 columnas y, para cada una, **cuándo existe**
- [ ] Validador de esquema que falla ruidosamente ante lo que el contrato no conoce
- [ ] **Medición del corte temporal:** prevalencia, volumen y tipos de transacción en cada
      periodo por separado
- [ ] **Caza de fuga, medida:** consistencia aritmética de los saldos, y qué pasa con la
      columna que produce el propio sistema de detección
- [ ] Feature engineering sobre lo que sí existe antes de la transacción
- [ ] Pipeline de preprocesamiento serializable
- [ ] `notebooks/01_preprocessing.ipynb` — narrativa de lo medido, mínimo 12 visualizaciones
      con interpretación escrita

> ✅ **Hito 1:** dataset procesado y reproducible con un comando; las columnas prohibidas
> declaradas con su razón en un ADR.

---

### FASE 2 — Modelado y AutoML · `feature/02-training`

**Objetivo:** el 50% de la rúbrica, salvo la interpretabilidad.

- [ ] MLflow contra DagsHub; primer experimento
- [ ] **Baseline trivial** y clasificador aleatorio — el piso contra el que todo se compara
- [ ] **Prueba del target barajado** — si no colapsa, paramos y depuramos
- [ ] Modelo lineal regularizado como referencia simple
- [ ] **Pipeline de AutoML (Reto 1):** espacio de búsqueda con Optuna donde el algoritmo es
      un hiperparámetro. Candidatos: regresión logística, random forest, LightGBM, XGBoost
- [ ] Estrategias de desbalance medidas: sin tratamiento vs. pesos de clase vs. remuestreo
- [ ] Selección del umbral operativo con una matriz de costos explícita
- [ ] Modelo ganador registrado en MLflow Model Registry
- [ ] `notebooks/02_training.ipynb`

> ✅ **Hito 2:** modelo productivo registrado, con su baseline al lado y el AutoML
> documentado. **Reto ML 1 cubierto.**

---

### FASE 3 — Interpretabilidad · `feature/03-interpretability`

**Objetivo:** la mitad del 50% que casi nadie hace bien. El enunciado pide interpretación
**local y global**, no solo un gráfico de importancias.

- [ ] **Global:** feature importance (permutación y del modelo), SHAP summary, PDP, ICE
- [ ] **Local:** SHAP waterfall y LIME sobre casos elegidos con criterio — un fraude
      detectado, un fraude perdido, un falso positivo, un caso en el umbral
- [ ] Contraste entre importancia por permutación y la del modelo: cuando difieren, dicen
      algo
- [ ] **Interpretación escrita**, no solo figuras. Qué aprendió el modelo y si tiene sentido
      de negocio
- [ ] `notebooks/03_interpretability.ipynb`

> ✅ **Hito 3:** el modelo explicado en los dos niveles, con los nombres de features
> propagados correctamente desde el pipeline.

---

### FASE 4 — Monitoreo · `feature/04-monitoring`

**Objetivo:** el 30% de la rúbrica más el Reto 2.

- [ ] `monitoring/drift.py`: PSI y KS por feature, con umbrales declarados y **controles
      positivos** en los tests
- [ ] Drift de `step >= 401` contra la referencia, feature por feature
- [ ] Drift del target: cómo cambia la prevalencia entre periodos
- [ ] Desempeño del modelo en producción, **por ventanas de steps** y no como un número
- [ ] Reportes de Evidently → `reports/deliverables/`
- [ ] **Interpretación escrita del monitoreo:** qué features driftan, por qué, y si el
      desempeño se degrada con ellas
- [ ] **App de Streamlit (Reto 2):** métricas calculadas sobre los artefactos, con la
      ventana de análisis seleccionable
- [ ] `notebooks/04_monitoring.ipynb`

> ✅ **Hito 4:** drift y desempeño medidos e interpretados, reporte HTML generado, app
> funcionando. **Reto ML 2 cubierto.**

---

### FASE 5 — Cierre · `feature/05-closing`

- [ ] **README completo:** problema, diagrama de flujo, dataset y diccionario, Model Card,
      resultados con su baseline, interpretabilidad, monitoreo, conclusiones
- [ ] Model Card final: limitaciones, uso previsto, uso **no** previsto
- [ ] Cobertura de tests ≥80%
- [ ] **Verificación en clon limpio:** seguir solo lo que dice el README y confirmar que las
      cifras coinciden
- [ ] Enlace público de MLflow verificado sin credenciales
- [ ] Revisión de secretos y datos pesados en el historial
- [ ] PR final `develop` → `main`, tag `v1.0.0` con release notes
- [ ] **Congelar**

> ✅ **Hito 5 — Proyecto entregado.**

---

## 10. Trazabilidad contra la rúbrica

| Requisito | Peso | Dónde se cumple | Fase |
|---|---|---|---|
| Estructura del repositorio | 10% | Sección 7 | 0 |
| Notebook de preprocesamiento | — | `notebooks/01_preprocessing.ipynb` | 1 |
| Notebook de entrenamiento | — | `notebooks/02_training.ipynb` | 2 |
| Notebook de interpretabilidad | — | `notebooks/03_interpretability.ipynb` | 3 |
| Notebook de monitoreo | — | `notebooks/04_monitoring.ipynb` | 4 |
| Carpeta de datos | — | `data/` | 1 |
| Carpeta de la app de monitoreo | — | `app/` | 4 |
| Optimización de hiperparámetros | 50% | Optuna, fase 2 | 2 |
| Registro en MLflow | 50% | DagsHub, todas las fases | 2-4 |
| Interpretación global | 50% | FI, SHAP, PDP, ICE | 3 |
| Interpretación local | 50% | SHAP, LIME | 3 |
| Monitoreo de desempeño | 30% | `monitoring/performance.py` | 4 |
| Monitoreo de drift | 30% | `monitoring/drift.py` | 4 |
| Interpretación del monitoreo | 30% | Notebook 04 + README | 4 |
| Report HTML | 30% | `reports/deliverables/` | 4 |
| Documentación y docstrings | 10% | Todo el repositorio | Todas |
| **Reto ML 1** — AutoML | +10% | Espacio de búsqueda multi-algoritmo | 2 |
| **Reto ML 2** — app web | +10% | `app/monitoring_app.py` | 4 |

---

## 11. Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| **Fuga no detectada** | **Alta** | La prueba del target barajado antes de comparar modelos. Un PR-AUC sospechosamente alto es señal de fuga, no de éxito |
| 6,3M filas en memoria | Media | dtypes explícitos, parquet, y muestreo estratificado para el AutoML si hace falta. Se declara si se usa |
| El AutoML consume mucho tiempo | **Alta** | Pruning de Optuna, número de trials declarado de antemano, y presupuesto de tiempo fijado antes de arrancar |
| Prevalencia tan baja que ningún modelo ordena | Media | Es un resultado si pasa. Se documenta con el análisis de errores en vez de forzar una métrica |
| SHAP sobre millones de filas | Media | Muestra estratificada para el cálculo global, declarada con su tamaño |
| La app queda a medias | Media | Es el último entregable. Un dashboard simple y funcional antes que uno ambicioso e incompleto |

**Regla de corte:** si una fase se desborda, se entrega el hito mínimo viable y se
documenta lo que faltó. Un sistema modesto, completo y honesto supera a uno ambicioso e
incompleto.

---

*Documento vivo — actualizar al cerrar cada fase.*
