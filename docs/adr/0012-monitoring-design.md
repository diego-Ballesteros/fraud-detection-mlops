# ADR 0012 — El diseño del monitoreo

- **Estado:** aceptado
- **Fecha:** 2026-10-21
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe. Las tres
  preguntas abiertas del turno (Evidently, el criterio de dos niveles, la ventana parcial) las
  decidió el humano antes de medir producción.

---

## Contexto

La fase 4 lee producción (`step >= 401`) por primera vez. Hasta acá solo se había contado. El
modelo, el preprocesador y el umbral se ajustaron sin ver una fila de producción, así que lo que
se mida es una estimación honesta y no una autoevaluación. La implementación está en
`src/fraud_detection/monitoring/` y se reproduce con `uv run python scripts/run_monitoring.py`;
las cifras, en `docs/analysis/monitoring.md`.

## Decisión

### 1 · Nada se reajusta y el umbral queda fijo

Todo sale de cargar `artifacts/fraud_detection_pipeline.pkl` (ADR 0010). El script verifica su
sha256 antes de puntuar y se detiene si no es el del ADR 0010.

El umbral queda en **0,42775088540908546**, el que eligió la métrica de decisión in-sample sobre
referencia (ADR 0011). **Reoptimizarlo sobre producción destruiría lo que el corte protege:**
el umbral pasaría a ser un estadístico ajustado con las filas sobre las que después se lo
juzga. El docstring de `performance` lo dice. El script recalcula el punto sobre referencia, y
solo sobre referencia, y se detiene si no coincide con la constante.

### 2 · Desempeño por ventana, no agregado

Producción se parte en **ventanas de 24 steps** ancladas en el step 401. Por ventana y en el
agregado se reportan filas, positivos y prevalencia; precisión y recall del modelo al umbral
fijo; **precisión y recall de la regla sobre las mismas filas**; y PR-AUC y Brier. Las dos
cifras de la regla sobre producción se reportan antes que ninguna del modelo.

**La ventana parcial (decisión del humano).** Producción va de 401 a 743: 14 ventanas completas
y 7 steps sobrantes (737–743). Los sobrantes son la ventana 15, marcada parcial, fuera del
criterio de degradación; sus filas sí entran al agregado.

### 3 · El criterio de degradación, declarado antes de medir

Una caída mayor que **±0,0036**, la dispersión entre folds de referencia, **sostenida en más de
una ventana**. Una ventana sola por debajo es ruido. No se cambia después de ver los números.

**Los dos niveles (decisión del humano).** El texto del Arquitecto admitía dos lecturas,
"sostenida en más de una ventana" y "tres seguidas son tendencia". Quedaron las dos, con nombre
propio: **2 ventanas completas consecutivas** por debajo de la línea son una **alerta**; **3**
son **degradación**.

### 4 · Drift: PSI decide, KS acompaña, y todo detector lleva su control positivo

- **PSI es la medida primaria**, porque su corte no se mueve con el tamaño de muestra.
- **KS se reporta como distancia. Su p-valor nunca es el disparador:** con 575.590 filas
  cualquier diferencia es significativa, y eso no dice nada.
- **PSI ≈ 0 es un control negativo, no evidencia de ausencia de drift.** Sin control positivo,
  un detector mudo y uno roto se leen igual. Todo test de drift lleva un desplazamiento
  sintético de magnitud conocida que el detector tiene que disparar, y el control va en los
  tests versionados (`tests/test_drift.py`), no en un script ad hoc.
- Se monitorean las 11 features transformadas, y se declara qué significa el drift en cada
  una. `hour_sin` y `hour_cos` tienen soporte idéntico por construcción, así que un drift ahí
  significa otra cosa que en las monetarias: solo puede ser un cambio en cómo se reparte el
  volumen sobre las horas del día, nunca un valor que el modelo no vio.

### 5 · Evidently, y su cota

El reporte HTML va a `reports/deliverables/`, que `.gitignore` desbloquea para eso. **Se
instala Evidently (decisión del humano)**, medido antes de instalar: agrega 40 paquetes al lock
(154 → 194) y no cambia la versión de ninguno de los existentes. Todos son dependencias base;
no hay instalación mínima. La cota la gobierna el ADR 0003, con verificación en los pisos; el
resultado está en `docs/DEPENDENCIES.md`.

### 6 · La interpretación

Cada cifra con su lectura, no solo la tabla. Como mínimo: qué significan operativamente los
falsos positivos por ventana; si la regla se sostiene, qué dice eso del generador; y si hay
drift, en qué feature y si coincide con el momento en que el desempeño se mueve. **La
coincidencia temporal no es causa, y se escribe.**

## Alternativas descartadas

- **Reoptimizar el umbral sobre producción:** la razón del punto 1.
- **Un número agregado como monitoreo:** es una evaluación sobre holdout, no un monitoreo.
- **El p-valor de KS como disparador:** la razón del punto 4.
- **Una sola lectura del criterio** (solo 2 o solo 3 ventanas): medido antes de producción, cada
  una tiene una tasa de falsa alarma muy distinta (nota del Ejecutor, abajo), y el texto
  admitía las dos.
- **Fundir los 7 steps sobrantes en la ventana 14, o descartarlos:** la ventana 14 dejaría de
  ser comparable con las demás, o se perderían filas del agregado.
- **No instalar Evidently y armar el HTML propio, o instalarlo en un grupo de dependencias
  aparte:** el ROADMAP nombra a Evidently en el stack, y el humano eligió instalarlo.

---

**Nota del Ejecutor: operacionalizaciones, declaradas en el código antes de leer una etiqueta
de producción.**

- *La línea.* El recall del modelo al umbral fijo, por ventana completa, contra
  **0,9934 = 0,9970 − 0,0036** (media y dispersión de la métrica de decisión en EVAL-009 y
  EVAL-010), estricto y comparado en enteros: `performance.DEGRADATION_LINE` es la fracción
  exacta 4967/5000. La precisión no entra al criterio, porque al umbral fijo se mueve con la
  prevalencia; se lee contra el piso de 0,99. Una ventana sin positivos no dice nada y corta la
  racha.
- *La tasa de falsa alarma del criterio, medida antes de producción.* Bajo el nulo de recall
  verdadero 0,997 en las 14 ventanas, con n positivos por ventana:

  | n por ventana | P(una ventana debajo) | P(alerta: ≥ 2 seguidas) | P(degradación: ≥ 3 seguidas) |
  |---|---|---|---|
  | 100 | 0,260 | 0,532 | 0,153 |
  | 150 | 0,363 | 0,768 | 0,346 |
  | 200 | 0,122 | 0,162 | 0,019 |
  | 250 | 0,173 | 0,294 | 0,052 |
  | 300 | 0,227 | 0,445 | 0,109 |
  | 400 | 0,120 | 0,158 | 0,018 |
  | 500 | 0,065 | 0,051 | 0,003 |

  No es monótona en n porque el recall de una ventana se mueve en saltos de 1/n. La calculé
  antes de medir con un script ad hoc; `performance.probability_below` y
  `performance.probability_of_run` la reproducen exacta, y el script la recalcula con los
  positivos que tuvo cada ventana de producción.
- *Los bins del PSI salen solo de referencia.* `FeatureProfile.from_reference` recibe valores de
  referencia y nada más. Las tres monetarias usan los deciles de referencia; las otras ocho, un
  bin por valor distinto de referencia más uno para lo no visto. La partición la decide el
  propio `ColumnTransformer` del artefacto (su rama monetaria), no un número mágico.
- *Los cortes del PSI*: 0,1 (moderado) y 0,25 (significativo). **Son una convención**, no se
  derivaron de estos datos. Su punto ciego, medido después: la bandera de la firma pasa del
  0,076% al 0,631% de las filas, 8,3 veces más, y su PSI es 0,012, "estable". Y la premisa del
  punto 4, que el corte no se mueve con el tamaño de muestra, vale solo con muestras grandes:
  con 4.474 filas el ruido de muestreo queda en 0,0062, y con 80 llega a 0,68
  (`docs/analysis/monitoring.md`, sección 4).
- *El piso de las proporciones nulas* es la regla de Evidently, y un bin vacío en los dos lados
  se descarta, para que los dos instrumentos coincidan donde sus bins coinciden.
- *Evidently con tipos declarados.* Las monetarias, numéricas; las otras ocho, categóricas. Para
  una numérica con más de 20 valores distintos, Evidently arma bins de ancho igual sobre
  referencia y producción juntas, aunque su docstring dice "reference quantiles"
  (ERR-019). `hour_sin` y `hour_cos` tienen 21 y 22 valores distintos en punto flotante, no 13,
  así que inferidas habrían tomado ese camino.
- *El contraste entre instrumentos es un chequeo, no una nota.* El script se detiene si el PSI de
  Evidently difiere del propio en una feature discreta, o si su precisión y su recall difieren
  de los propios, en más de 1e-9.
- *Telemetría.* `DO_NOT_TRACK` se fija antes de importar Evidently. En 0.7.23 la generación de
  reportes no importa su cliente de telemetría (leído en el código, no medido).
- *Para leer los números*, el script mide además el drift de las filas legítimas solas, la
  proporción de saldos en cero por ventana y el nulo del PSI al tamaño de cada ventana. No son
  disparadores: separan un cambio en las features de un cambio en la mezcla de clases, y el
  ruido del muestreo de la señal.

---

**Nota de estado (2026-10-21, notebook 04), decisiones del humano:**

- El reporte HTML pesa unos 4 MB, sobre el límite de 1.024 KB del hook `check-added-large-files`.
  La excepción es para ese archivo exacto, no un límite global más alto: es el entregable
  declarado del 30%, y quien clona el repositorio no puede regenerarlo sin el CSV, que no está
  versionado. La razón está escrita junto a la excepción, en `.pre-commit-config.yaml`.
- `notebooks/04_monitoring.ipynb` es el único notebook que lee producción, en una sola celda de
  código que dice por qué está autorizada. `tests/test_project_rules.py` lo hace cumplir: falla si
  otra celda u otro notebook nombra producción, y si no ve la lectura autorizada.
