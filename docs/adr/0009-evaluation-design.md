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
