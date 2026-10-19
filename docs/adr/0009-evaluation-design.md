# ADR 0009 — El diseño de la evaluación

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Contexto

La métrica y el esquema de validación se declaran antes de ver un solo resultado. Una
métrica elegida después de medir es la métrica que favorece el resultado que salió.

## Decisión

### 1 · El splitter: rangos de step con ventana expansiva

Los folds se cortan por rangos de `step`, nunca dentro de uno, con ventana expansiva:

| Fold | Entrena | Valida |
|---|---|---|
| 1 | 1–200 | 201–240 |
| 2 | 1–240 | 241–280 |
| 3 | 1–280 | 281–320 |
| 4 | 1–320 | 321–360 |
| 5 | 1–360 | 361–400 |

La ventana expansiva es exactamente la situación de producción: entrenar con toda la
historia y predecir el periodo siguiente.

**Por qué no `TimeSeriesSplit`.** Parte por índice de fila, así que un borde de fold puede
caer dentro de un `step`: el modelo entrenaría con parte de una hora y validaría con el
resto. Es una fuga pequeña y silenciosa.

Es el splitter que decide. Vive en `src/fraud_detection/models/evaluation.py`
(`ExpandingStepSplit`).

### 2 · La métrica de decisión: recall con precisión ≥ 0,99

**Por qué el PR-AUC global no sirve aquí.** Está saturado: cualquier modelo que aprenda la
firma llega a ~0,98, así que no discrimina. Lo que discrimina son los 92 fraudes que la
regla no cubre. El baseline (ADR 0007) puntúa 0,979 en esta misma escala.

- **Secundaria:** recall con precisión ≥ 0,95, por si ningún modelo alcanza el piso de 0,99.
- **Contexto, siempre reportado:** PR-AUC, ROC-AUC y Brier.

### 3 · El umbral de significancia práctica: +0,005 de recall

Declarado antes de medir. Por debajo de eso, dos modelos son el mismo modelo. Son unos 22
fraudes.

### 4 · No hay validación cruzada anidada

Tenemos un periodo temporal realmente retenido, producción, que es mejor que cualquier
bucle interno. Lo que hace falta no es anidar: es no tocar producción hasta la fase 4.

### 5 · El test de objetivo barajado y su margen de colapso

Es el control positivo de todo el arnés. Se baraja la etiqueta **solo en la porción de
entrenamiento de cada fold**, preservando la proporción de clases, con
`settings.RANDOM_STATE`; la validación queda intacta. Después corre el pipeline completo por
el mismo arnés, no por un camino aparte, o se estaría validando otra cosa. Barajar la
etiqueta completa, validación incluida, no prueba nada: el PR-AUC daría la tasa base
trivialmente. Si no colapsa, algo filtra a través de la estructura de folds o del
preprocesamiento, y es un bloqueante.

**El margen:** colapsa si, para cada familia y en cada fold, el PR-AUC de validación es a lo
sumo **10 veces la prevalencia de ese fold**. Lo declaró el Ejecutor antes de medir, y el
Arquitecto lo aceptó. Se fijó midiendo un caso donde la fuga es imposible por construcción:
datos sintéticos con los positivos agrupados y una feature binaria rara que marca al 98% de
ellos, como la firma, ajustados con scikit-learn y LightGBM sin el arnés del proyecto. Con
la etiqueta barajada, el PR-AUC llegó a 4,6 veces la prevalencia en regresión logística y a
2,0 veces en LightGBM, en 12 permutaciones cada una. Un margen de 2 habría marcado como fuga
ese montaje sin fuga en 3 de las 12. Es por fold y no en promedio porque una fuga por la
estructura de folds puede afectar a un solo fold.

**Por qué un modelo barajado supera la prevalencia sin que haya fuga.** El PR-AUC está
sesgado hacia arriba cuando los positivos son pocos. Además, un modelo que sobreajusta el
ruido del fold de entrenamiento todavía puede ordenar la validación algo mejor que el azar,
por casualidad.

**Por qué el margen no es flojo donde importa.** Entre "colapsó" (≤ 0,0077) y "algo filtra"
(la firma da ~0,98) hay más de dos órdenes de magnitud.

## Alternativa descartada

**`StratifiedKFold` como esquema primario.** Se mide igual, con 5 folds mezclados con
`settings.RANDOM_STATE`, y se reporta al lado del temporal para cada modelo. No es una
alternativa: es la medida del optimismo que introduce ignorar el tiempo.

---

**Nota del Ejecutor sobre la procedencia (2026-10-18):** los "unos 22 fraudes" del umbral de
significancia son 0,005 × 4.477, los positivos de todo el periodo de referencia. Medido en
este turno, las cinco validaciones temporales juntas (steps 201–400) tienen 2.223
positivos, así que allí +0,005 de recall equivale a unos 11 fraudes, y en un fold, que
tiene entre 408 y 465, a unos 2.

**Nota del Ejecutor sobre el margen de colapso (2026-10-18):**

- *Las cifras de calibración* (4,6, 2,0 y 3 de 12) salen de un script ad hoc que no está
  versionado.
- *Cuánto pesa cada uno de los dos efectos.* Medido sobre las etiquetas verdaderas de cada
  fold de validación temporal, con 100 rankings al azar por fold, el sesgo por pocos
  positivos lleva el PR-AUC a entre 1,015 y 1,028 veces la prevalencia en promedio, con
  percentil 99 entre 1,13 y 1,25 y máximo de 1,51. Explica, a lo sumo, alrededor de 1,5
  veces. Los múltiplos de hasta 4,6 de la calibración vienen del segundo efecto: los
  positivos están agrupados en el espacio de features, y un modelo ajustado sobre ruido que
  por azar se inclina hacia esa región los ordena juntos.
- *La distancia.* De 0,0077 a 0,979 hay un factor de unos 127: algo más de dos órdenes de
  magnitud. Los tres órdenes son la distancia desde la prevalencia (0,00077) hasta el
  baseline.
- *Cuando un fold no colapsa*, `scripts/run_leakage_check.py` muestra además cuántos
  positivos barajados cayeron en filas de entrenamiento con la firma, frente a los
  esperados. Ahí se separa el azar de la fuga; el veredicto no cambia.

---

**Nota de estado (2026-10-18, fase 2): el criterio de colapso cambió dos veces después de
verlo fallar.** Reemplaza el margen de la sección 5, que no se reescribe.

**Primer cambio, del Arquitecto.** Con una sola permutación, el margen de la sección 5 falló:
LightGBM, fold 2, 14,83 veces la prevalencia (EVAL-002). En palabras del Arquitecto: *"lo
estoy cambiando después de ver que el anterior falló"*. Sus dos razones:

- La razón del cambio es la forma de la métrica —un cociente a prevalencia 7,6e-4 es
  hipersensible por construcción—, no el valor 14,83.
- La protección no es el umbral nuevo: es la distribución de permutaciones. Con veinte valores
  medidos en el orden de 1e-2 y etiquetas reales en ~0,98, dónde se traza la línea entre
  medio no sostiene la conclusión. El resultado se hace robusto al umbral en vez de defender
  el umbral.

El criterio nuevo: 20 permutaciones del mismo modelo; el PR-AUC máximo de las 20 por debajo de
**0,05 en valor absoluto**; y la distribución del ROC-AUC conteniendo 0,5 sin elevación
sistemática.

La comparación de escalas, del Arquitecto: una fuga que recupera la etiqueta deja el PR-AUC
en el rango 0,5–1,0; la firma con etiquetas reales da ~0,98; el máximo medido bajo
permutación es 0,0112 (EVAL-004). Entre el peor caso del ruido y el mejor caso de una fuga
hay más de un orden de magnitud de separación.

**Segundo cambio, del Arquitecto, sobre la operacionalización del Ejecutor.** El Ejecutor
operacionalizó la condición del ROC-AUC antes de medir: R1, 0,5 dentro de [mín, máx] de la
familia; R2, la mediana de la familia no supera el máximo del clasificador aleatorio. Medido,
la regresión logística no pasó R2 (0,545 contra 0,5395) y Random Forest no pasó R1 (sus cinco
valores, de 0,501 a 0,514). El Arquitecto retiró las dos, y **el defecto es formal, no del
resultado**: R2 compara una mediana contra un estadístico de orden extremo de otra
distribución, que no son cantidades comparables; R1 prueba el signo de una desviación con
n = 5 e ignora su magnitud, y cinco valores del mismo lado de 0,5 bajo un nulo simétrico es un
evento del 3%. Las dos son afirmaciones sobre la construcción del test, no sobre lo que
devolvió.

**Es la segunda vez en esta fase que el Arquitecto cambia un criterio después de verlo
fallar.** El correctivo es que la referencia sea el nulo empírico medido, y no un criterio
inventado.

La condición nueva del ROC-AUC: la mediana bajo permutación cae dentro del rango observado
del clasificador aleatorio (0,470–0,539) o por debajo. Por debajo no es fuga: es
anti-generalización. LightGBM pasa (0,242), Random Forest pasa (0,504), y la regresión
logística queda apenas afuera (0,545 contra 0,539). Se acepta, y el hueco queda escrito como
hueco:

- Por magnitud no es fuga: 0,045 de elevación con PR-AUC ≤ 0,005, contra una fuga que daría
  ROC-AUC ~1,0 y PR-AUC ~0,98.
- No existe camino en el diseño: los folds son disjuntos por step, y un test lo afirma.
- **El mecanismo queda sin determinar.** Hipótesis no medida: el barajado es solo del
  entrenamiento, la validación conserva sus etiquetas reales, y el ranking de un modelo
  lineal es una función suave de las features; si la dirección del vector de coeficientes
  ajustado sobre ruido no es de signo aleatorio entre permutaciones, el ranking correlaciona
  débilmente con la relación verdadera.

**El nulo correcto, del Arquitecto.** El nulo de un test de objetivo barajado es una
permutación del mismo modelo, nunca una línea base aleatoria. El clasificador aleatorio acota
la inflación por escasez de positivos; la permutación acota la inflación por estructura. Son
cotas de cosas distintas, y la que importaba era la segunda. Medido, el aleatorio es el piso
de inflación por escasez: mediana de 1,02 veces la prevalencia y máximo de 1,50 en 100
sorteos (EVAL-003). Eso descarta la escasez como explicación de 14,83 y señala a la
estructura del ranking (ERR-013).

**Decisiones del Ejecutor en este cambio**, a su nombre:

- Las semillas son `settings.RANDOM_STATE + i`, para i de 0 a 19 (`derived_seeds`). La 0 es
  la permutación original, y que la reproduzca exactamente (0,011227402) es el control de
  determinismo del test.
- Cada test de permutación es una corrida padre en MLflow con una corrida hija por sorteo; el
  padre lleva la distribución.
- El clasificador aleatorio (`UniformRandomScores`) vive en `estimators.py`, pero fuera de
  `MODEL_SPECS`: es un control, no uno de los modelos evaluados.
- R1 y R2, retiradas arriba.
