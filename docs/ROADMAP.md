# Proyecto Curso III — Fraud Detection MLOps

> Detección de fraude en transacciones móviles con optimización de hiperparámetros,
> interpretabilidad y monitoreo de drift sobre un periodo de producción real.
>
> **Especialización MLE 2026 — Curso III**
> Documento de diseño y plan de ejecución.
>
> **Estado al 2026-10-21:** las fases 0 a 4 están cerradas y mergeadas a `develop`; la 5 está en
> curso. El Reto 2 no se hizo. En la sección 9, cada casilla marcada dice dónde está su evidencia,
> y cada una sin marcar, qué falta.

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
| Clase positiva | 0,0774% en referencia: 4.477 de 5.787.030 (medido el 2026-10-18). 0,6491% en producción: 3.736 de 575.590 (medido el 2026-10-21, EVAL-011) |
| Ubicación | `data/raw/`, gitignoreado |

La tasa de producción no se midió hasta la fase 4, y tampoco se estimó: con la cifra de
referencia y un dato externo se podía despejar por resta, y una estimación influye las
decisiones igual que una medición. Medida, es 8,4 veces la de referencia, y la del periodo
completo, 0,1291% (8.213 de 6.362.620). El porqué está en `docs/analysis/monitoring.md`.

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
                      │ local: SHAP      │  │ step >= 401    │  └─────────┘
                      └──────────────────┘  └────────────────┘
```

Los tres consumidores cargan **el mismo artefacto** del registry. Nunca se reimplementa una
transformación.

---

## 5. Estrategia de evaluación

### 5.1 Métricas del modelo

La métrica de decisión, su secundaria y las de contexto se fijaron en la fase 2, antes de
medir: ADR 0009, con el contrato en `docs/EVALUATION.md`. La tabla de abajo es el plan
original; donde difiere, manda el ADR.

| Métrica | Por qué |
|---|---|
| **PR-AUC** | Plan original: métrica principal. Con 0,1% de positivos, ROC-AUC es engañosamente optimista. Quedó como contexto: la firma la satura (ADR 0009) |
| **Recall a precisión fija** | El negocio revisa una cola de capacidad limitada. Es la métrica de decisión, con precisión ≥ 0,99 (ADR 0009) |
| Precision@k | "De las k transacciones que marqué, ¿cuántas eran fraude?" |
| ROC-AUC | Comparabilidad con la literatura |
| Matriz de confusión con costos | Traduce el modelo a pérdidas, no a porcentajes |

**Accuracy se reporta una sola vez**, al lado del clasificador trivial, para dejar
demostrado por qué no decide nada (`docs/MODEL_CARD.md`, sección 4).

**Lo que no se hizo de esta tabla:** Precision@k y la matriz de confusión con costos. El punto
de operación lo fija la métrica de decisión (ADR 0011), y una matriz de costos necesita el costo
de una revisión, que no se midió. Lo que se midió en su lugar es cuántas revisiones cuesta cada
fraude adicional (EVAL-012).

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
| ML | scikit-learn, LightGBM, XGBoost | Candidatos del espacio de búsqueda. XGBoost no entró: el espacio quedó en regresión logística, Random Forest y LightGBM (EVAL-009) |
| Desbalance | imbalanced-learn | No entró. El desbalance se trató con pesos de clase dentro de la búsqueda; el remuestreo no se midió |
| **Optimización** | **Optuna** | El algoritmo entra como hiperparámetro (Reto 1) |
| **Interpretabilidad** | **SHAP, sklearn** | Global: feature importance, PDP, ICE. Local: SHAP. LIME quedó fuera (fase 3, alcance declinado) |
| **Monitoreo** | **Evidently** | Reportes HTML de drift y desempeño |
| Tracking | **MLflow + DagsHub** | Registry incluido |
| **App** | **Streamlit** | Dashboard de monitoreo (Reto 2). No se hizo, y no está en las dependencias |
| Testing | pytest, pytest-cov | Objetivo: ≥80%. Medido el 2026-10-21: 98% |
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
│   ├── DEPENDENCIES.md            ← dependencias y sus pisos; se lee antes de un uv add
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
├── src/fraud_detection/           ← sus módulos y su estado: tabla de abajo
│
├── scripts/                       ← sus scripts y su estado: tabla de abajo
│
├── app/                           ← requisito explícito (Reto 2); hoy solo .gitkeep
│   └── monitoring_app.py          ← no se construyó
│
├── reports/deliverables/          ← el Report HTML del monitoreo
├── tests/
└── .github/workflows/ci.yml
```

### Módulos de `src/fraud_detection/`

Un módulo es un archivo `.py` del paquete, sin contar los `__init__.py`. `existe` quiere decir
que el archivo está en el repositorio; `fase N`, que está planificado para esa fase y todavía
no existe. `tests/test_docs_consistency.py` verifica la tabla contra el código en las dos
direcciones: un módulo sin fila, una fila `existe` sin archivo o una fila planificada cuyo
archivo ya existe hacen fallar el test.

| Paquete | Módulo | Estado |
|---|---|---|
| (raíz) | config | existe |
| data | loader | existe |
| data | schema | existe |
| data | validator | existe |
| data | profiling | existe |
| features | builder | existe |
| features | pipeline | existe |
| models | baseline | existe |
| models | estimators | existe |
| models | search_space | existe |
| models | evaluation | existe |
| models | fit_guard | existe |
| models | tracking | existe |
| models | registry | existe |
| tracking | mlflow_setup | existe |
| explain | global_explain | existe |
| explain | local_explain | existe |
| monitoring | drift | existe |
| monitoring | performance | existe |
| monitoring | report | existe |

### Scripts de `scripts/`

Mismo vocabulario que la tabla de módulos: `existe` quiere decir que el archivo está en el
repositorio; `fase N`, que está planificado para esa fase y todavía no existe.
`tests/test_docs_consistency.py` verifica esta tabla contra la carpeta en las dos
direcciones, con las mismas reglas.

| Script | Qué hace | Estado |
|---|---|---|
| `prepare_data.py` | CSV → dos parquet, validación y mediciones | existe |
| `run_training.py` | El AutoML: la búsqueda con el algoritmo como hiperparámetro | existe |
| `register_model.py` | Mide la estabilidad de la configuración elegida con tres semillas, la ajusta sobre toda la referencia, la serializa y la registra en MLflow | existe |
| `run_default_models.py` | Los baselines y las tres familias con hiperparámetros por defecto, por los dos splitters | existe |
| `run_ablation.py` | La ablación del ADR 0007: el modelo elegido sin la firma, por el arnés temporal al lado de `full` y de la regla, y sobre producción con su propio umbral de referencia | existe |
| `run_leakage_check.py` | La prueba del objetivo barajado, como test de permutación con el clasificador aleatorio como control; es de la fase 2 porque necesita un modelo contra el que barajar | existe |
| `diagnose_head.py` | Qué tiene la cabeza del ranking de un modelo permutado (ERR-013) | existe |
| `diagnose_ap.py` | De qué bloques de puntaje sale el PR-AUC de un modelo permutado (ERR-013) | existe |
| `run_interpretability.py` | Explica el artefacto: la compuerta de SHAP sobre la firma, las importancias globales lado a lado, ICE y PDP —`amount` también sobre su rango completo—, los cuatro casos locales con las cuentas del caso b, y adjunta figuras y tablas a la versión 1 del Model Registry | existe |
| `run_monitoring.py` | Mide producción: el desempeño por ventanas de 24 steps al umbral fijo junto a la regla, por tipo de transacción, el precio de los fraudes que el modelo pierde, el criterio de degradación, el drift por PSI y KS, el reporte HTML de Evidently y su contraste con los instrumentos propios | existe |

`run_prediction.py` salió del plan: el monitoreo calcula sus propias predicciones sobre producción y la app del Reto 2 predice de forma interactiva cargando el artefacto, así que un script de predicción por lotes no tiene usuario. La fase 4 no lo necesitó: `run_monitoring.py` puntúa producción cargando el artefacto.

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

- [x] Estructura de carpetas, `pyproject.toml`, `uv sync`
- [x] `.gitignore`, `.env.example`, pre-commit, `Taskfile.yml`
- [x] CI en GitHub Actions: lint + tipos + tests (`.github/workflows/ci.yml`)
- [x] `config.py` con `RANDOM_STATE` y las rutas del proyecto (ADR 0001)
- [x] `docs/GIT_STRATEGY.md`
- [x] Registros vacíos: `EVALUATION.md`, `ERRORS_AND_LEARNINGS.md`
- [x] Cuenta de DagsHub conectada, `.env` con las credenciales de MLflow

> ✅ **Hito 0:** repo profesional funcionando, CI en verde. **Cumplido.**

---

### FASE 1 — Datos y EDA · `feature/01-data-and-eda`

**Objetivo:** entender el dataset y cerrar el contrato de datos. Es la fase donde se decide
qué columnas pueden entrar al modelo.

- [x] Carga con dtypes explícitos (6,3M filas; pandas en memoria duele si no)
- [x] `docs/DATA_DICTIONARY.md` con las 11 columnas y, para cada una, **cuándo existe**
- [x] Validador de esquema que falla ruidosamente ante lo que el contrato no conoce
- [x] **Medición del corte temporal:** prevalencia, volumen y tipos de transacción en cada
      periodo por separado. La de referencia, en esta fase; la de producción, a propósito, recién
      en la fase 4 (EVAL-011)
- [x] **Caza de fuga, medida:** consistencia aritmética de los saldos, y qué pasa con la
      columna que produce el propio sistema de detección (ADR 0006)
- [x] Feature engineering sobre lo que sí existe antes de la transacción
- [x] Pipeline de preprocesamiento serializable (ADR 0008)
- [x] `notebooks/01_preprocessing.ipynb` — narrativa de lo medido, mínimo 12 visualizaciones
      con interpretación escrita (tiene 13)

> ✅ **Hito 1:** dataset procesado y reproducible con un comando; las columnas prohibidas
> declaradas con su razón en un ADR. **Cumplido** (ADR 0006).

---

### FASE 2 — Modelado y AutoML · `feature/02-training`

**Objetivo:** el 50% de la rúbrica, salvo la interpretabilidad.

- [x] MLflow contra DagsHub; primer experimento
- [x] **Baseline trivial** y clasificador aleatorio — el piso contra el que todo se compara
      (EVAL-001, EVAL-003)
- [x] **Prueba del target barajado** — si no colapsa, paramos y depuramos. No colapsó con una
      permutación, se paró, y con 20 pasa (EVAL-002, EVAL-004, ERR-013)
- [x] Modelo lineal regularizado como referencia simple (EVAL-006)
- [x] **Pipeline de AutoML (Reto 1):** espacio de búsqueda con Optuna donde el algoritmo es
      un hiperparámetro. Candidatos: regresión logística, random forest y LightGBM; **XGBoost no
      entró**. La búsqueda se cortó en 8 de 150 ensayos (EVAL-009)
- [ ] Estrategias de desbalance medidas: sin tratamiento vs. pesos de clase vs. remuestreo.
      **Parcial:** los pesos de clase entraron al espacio de búsqueda como un hiperparámetro más,
      sin una comparación aislada contra el modelo sin tratamiento; **el remuestreo no se midió**
- [ ] Selección del umbral operativo con una matriz de costos explícita. **No se hizo:** el
      umbral lo eligió la métrica de decisión (ADR 0011), y una matriz de costos necesita el costo
      de una revisión, que no se midió
- [x] Modelo ganador registrado en MLflow Model Registry (ADR 0010)
- [x] `notebooks/02_training.ipynb`

> ✅ **Hito 2:** modelo productivo registrado, con su baseline al lado y el AutoML
> documentado. **Reto ML 1 cubierto.** **Cumplido**, sobre una búsqueda incompleta.

---

### FASE 3 — Interpretabilidad · `feature/03-interpretability`

**Objetivo:** la mitad del 50% que casi nadie hace bien. El enunciado pide interpretación
**local y global**, no solo un gráfico de importancias.

- [x] **Global:** feature importance (permutación y del modelo), SHAP summary, PDP, ICE
- [x] **Local:** SHAP waterfall sobre casos elegidos con criterio — un fraude que la regla
      y el modelo capturan, uno que la regla no captura y el modelo sí, un falso positivo, una
      legítima cerca del umbral (ADR 0011)
- [x] Contraste entre importancia por permutación y la del modelo: cuando difieren, dicen
      algo
- [x] **Interpretación escrita**, no solo figuras. Qué aprendió el modelo y si tiene sentido
      de negocio (`docs/analysis/interpretability.md`)
- [x] `notebooks/03_interpretability.ipynb`

> ✅ **Hito 3:** el modelo explicado en los dos niveles, con los nombres de features
> propagados correctamente desde el pipeline. **Cumplido.**

**Alcance declinado: LIME.** Medido contra el lock, agrega seis paquetes nuevos —lime y cinco
transitivos: scikit-image, imageio, lazy-loader, networkx y tifffile— para un contraste
sobre cuatro filas. La rúbrica del Curso III pide interpretación local y global, sin nombrar
LIME, y SHAP cubre las dos (ADR 0011).

---

### FASE 4 — Monitoreo · `feature/04-monitoring`

**Objetivo:** el 30% de la rúbrica más el Reto 2.

- [x] `monitoring/drift.py`: PSI y KS por feature, con umbrales declarados y **controles
      positivos** en los tests
- [x] Drift de `step >= 401` contra la referencia, feature por feature
- [x] Drift del target: cómo cambia la prevalencia entre periodos
- [x] Desempeño del modelo en producción, **por ventanas de steps** y no como un número
      (EVAL-011); y por tipo de transacción, desde la fase 5 (EVAL-012)
- [x] Reportes de Evidently → `reports/deliverables/`
- [x] **Interpretación escrita del monitoreo:** qué features driftan, por qué, y si el
      desempeño se degrada con ellas (`docs/analysis/monitoring.md`)
- [ ] **App de Streamlit (Reto 2):** métricas calculadas sobre los artefactos, con la
      ventana de análisis seleccionable. **No se hizo**; se decide después del cierre
- [x] `notebooks/04_monitoring.ipynb`

> ✅ **Hito 4:** drift y desempeño medidos e interpretados, reporte HTML generado, app
> funcionando. **Reto ML 2 cubierto.** **Cumplido sin la app:** el Reto 2 no está cubierto.

---

### FASE 5 — Cierre · `feature/05-closing`

- [ ] **README completo:** problema, diagrama de flujo, dataset y diccionario, Model Card,
      resultados con su baseline, interpretabilidad, monitoreo, conclusiones. **Parcial:** tiene
      el problema, el dataset con el enlace al diccionario, el Model Card, los resultados de
      producción con la regla al lado, el monitoreo y cómo leer las métricas. Falta el diagrama de
      flujo, que está en la sección 4 de este documento, y la interpretabilidad va enlazada, no
      resumida
- [x] Model Card final: limitaciones, uso previsto, uso **no** previsto (`docs/MODEL_CARD.md`)
- [x] Cobertura de tests ≥80%: 98%, medida el 2026-10-21
- [ ] **Verificación en clon limpio:** seguir solo lo que dice el README y confirmar que las
      cifras coinciden. **Parcial:** el 2026-10-21, un clon sin datos instaló el lock, pasó la
      suite y cargó el artefacto con sus probabilidades documentadas; que las cifras coincidan
      necesita el CSV en el clon, y no se hizo
- [ ] Enlace público de MLflow verificado sin credenciales
- [ ] Revisión de secretos y datos pesados en el historial
- [ ] PR final `develop` → `main`, tag `v1.0.0` con release notes. La PR #14 ya llevó `develop`
      a `main` antes del cierre; faltan la de la fase 5 y el tag
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
| Interpretación local | 50% | SHAP, waterfall de cuatro casos | 3 |
| Monitoreo de desempeño | 30% | `monitoring/performance.py` | 4 |
| Monitoreo de drift | 30% | `monitoring/drift.py` | 4 |
| Interpretación del monitoreo | 30% | Notebook 04 + README | 4 |
| Report HTML | 30% | `reports/deliverables/` | 4 |
| Documentación y docstrings | 10% | Todo el repositorio | Todas |
| **Reto ML 1** — AutoML | +10% | Espacio de búsqueda multi-algoritmo | 2 |
| **Reto ML 2** — app web | +10% | `app/monitoring_app.py`: **no se hizo** | 4 |

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
