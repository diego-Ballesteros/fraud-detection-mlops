# ADR 0011 — El diseño de la interpretabilidad

- **Estado:** aceptado
- **Fecha:** 2026-10-20
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Contexto

La fase 3 explica el artefacto único del proyecto (ADR 0010), cargado de
`artifacts/fraud_detection_pipeline.pkl`. No se reajusta nada ni se construye un modelo
equivalente: un modelo reconstruido para la explicación explicaría otro objeto. La
implementación está en `src/fraud_detection/explain/` y se reproduce con
`uv run python scripts/run_interpretability.py`; las cifras, en
`docs/analysis/interpretability.md`.

La firma del simulador es una respuesta conocida (ADR 0007): es la única oportunidad del
proyecto de validar el instrumento de interpretabilidad contra una verdad ya sabida.

## Decisión

### 1 · El explicador: `TreeExplainer` path-dependent

`shap.TreeExplainer` con `feature_perturbation="tree_path_dependent"`, sin datos de fondo.
Sus valores están en la salida cruda del modelo, los log-odds, y suman exactamente esa salida:
el código lo verifica en cada fila que explica.

**Por qué, en palabras del Arquitecto.** `amount_equals_origin_balance` es una función
determinista de `amount` y `oldbalanceOrg`, así que las tres son perfectamente dependientes.
El modo intervencional rompe la dependencia muestreando valores marginales, y construiría
filas donde la bandera afirma la igualdad y los valores la contradicen: el modelo nunca vio
eso y su comportamiento ahí no está definido por construcción. Path-dependent se queda sobre
la variedad de los datos. **No es la opción convencional: es la única que no evalúa el
modelo fuera de su dominio.**

**Alternativa descartada:** el modo intervencional, por la razón de arriba.

### 2 · La compuerta, corregida antes de medir

Antes de producir ninguna figura, SHAP tiene que encontrar la firma. La formulación quedó así:

- Las tres features que codifican la firma —`amount_equals_origin_balance`, `amount` y
  `oldbalanceOrg`— tienen que sumar la **mayoría** (estrictamente más de la mitad) del
  |SHAP| medio.
- La bandera tiene que estar **entre las tres primeras**.
- Si dominan `hour_sin`, `hour_cos` o las dummies de `type`, el turno para y se reporta: el
  problema sería el explicador, no el modelo.

**La formulación original pedía que la bandera quedara primera, y el Arquitecto la corrigió
antes de medir, no después.** Con tres features perfectamente dependientes el crédito se
reparte según la estructura de cada árbol, así que exigir el primer puesto podía fallar por
una razón que no es de SHAP. Es el problema de R1 y R2 del ADR 0009, evitado esta vez antes de
ver el resultado.

En la interpretación, el reparto del crédito entre las tres se explica como lo esperado de
features dependientes bajo path-dependent, **no como un hallazgo**.

### 3 · La muestra

Todas las filas positivas de referencia más 20.000 negativas tomadas con
`settings.RANDOM_STATE`: la misma muestra del gráfico 6 del notebook 01
(`profiling.sample_with_all_positives`). Una muestra uniforme sobre 0,077% de prevalencia no
tendría positivos suficientes para que SHAP dijera algo de ellos.

### 4 · El umbral operativo

El proyecto no tenía umbral declarado. Se toma el que elige la métrica de decisión:
`recall_at_precision` con precisión ≥ 0,99, sobre los puntajes del artefacto en las
5.787.030 filas de referencia, con la política de empates de la fase 2. El artefacto se
ajustó con esas mismas filas, así que **el umbral y todo conteo en él son in-sample, y
optimistas**. Producción no participa.

**Alternativas descartadas:**

- **0,5**, el corte de `predict`: con `scale_pos_weight` 34,9 las probabilidades no están
  calibradas, y ninguna métrica del proyecto se ancla a 0,5.
- **El mismo punto sobre la muestra:** con los negativos submuestreados unas 289 veces, la
  precisión sale inflada.

### 5 · La importancia por permutación

Sobre la métrica de decisión, no sobre exactitud, en las 5.787.030 filas de referencia y al
nivel de las features del modelo: el preprocesamiento del artefacto corre una vez y se
permuta cada una de sus 11 columnas de salida, 10 veces, con las semillas
`derived_seeds(10)`. Es in-sample, y **permutar un one-hot de `type` suelto crea filas con
dos tipos activos o ninguno**: la advertencia se declara.

**Alternativas descartadas:**

- **La muestra de SHAP:** con 20.000 negativos cada falso positivo pesa unas 289 veces
  menos, y la métrica pasa a medir otra cosa, mucho más laxa.
- **Las columnas crudas a través del Pipeline:** permutar `amount` rompe también la bandera
  derivada, y los nombres dejan de coincidir con los 11 de SHAP, así que se pierde la
  comparación lado a lado.

### 6 · ICE y PDP sobre columnas crudas, a través del artefacto completo

Se varían las columnas crudas `oldbalanceOrg`, `oldbalanceDest` y `amount`, y cada fila
modificada pasa por el artefacto entero. La derivación recalcula la bandera en cada punto de
la grilla, así que ninguna fila la contradice, y el eje queda en moneda.

**El límite, declarado:** la bandera no tiene curva propia, porque no es una entrada. Y en una
fila con la firma, donde la grilla no cae justo en su `amount`, la bandera vale 0: su ICE
muestra al modelo **sin** la firma.

**Alternativas descartadas:**

- **Las features transformadas:** para la bandera y `oldbalanceOrg` evalúa filas que
  contradicen la derivación, la misma objeción que excluyó al SHAP intervencional, y el eje
  queda en log1p escalado.
- **Solo features fuera del trío de la firma:** nunca sale del dominio, pero no dice nada
  sobre la firma, que tiene la mayor parte del |SHAP|.

### 7 · LIME queda fuera del turno y del proyecto

El Arquitecto revirtió su decisión anterior de incluirlo como segundo explicador sobre los
cuatro casos locales. La razón, medida: agrega seis paquetes nuevos al lock —lime y cinco
transitivos: scikit-image, imageio, lazy-loader, networkx y tifffile— para un contraste
sobre cuatro filas. Y la rúbrica del Curso III pide interpretación local y global, las dos
cubiertas por SHAP; LIME estaba en el ROADMAP porque lo había escrito el Arquitecto.

## Consecuencias

- Las tres ordenaciones globales —SHAP, permutación y la importancia interna de LightGBM
  (split y gain)— se reportan lado a lado, y donde discrepan se reporta sin resolver a favor
  de ninguna.
- Los cuatro casos locales se eligen con un criterio declarado antes de medir, en el
  docstring de `src/fraud_detection/explain/local_explain.py`.

---

**Nota del Ejecutor (2026-10-20):** la objeción del punto 1 alcanza también a la permutación
del punto 5, y no solo a sus one-hot. Permutar la bandera, `amount` u `oldbalanceOrg` por
separado construye filas donde la bandera y los valores se contradicen. Se declara y no se
evita. El Arquitecto aceptó la permutación con la advertencia del one-hot; esta extensión es
del Ejecutor.

**Nota del Ejecutor sobre la procedencia (2026-10-20):** en el turno en que se decidió el
punto 7, el Ejecutor había listado pillow entre los paquetes que lime arrastra. Medido contra
el lock, pillow ya estaba, porque lo trae matplotlib. Las seis del punto 7 son las nuevas.
