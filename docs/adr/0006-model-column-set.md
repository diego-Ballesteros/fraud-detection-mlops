# ADR 0006 — El conjunto de columnas del modelo

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Decisión

**Entran:** `amount`, `oldbalanceOrg`, `oldbalanceDest`, `type`, más tres derivadas:
`amount_equals_origin_balance`, `hour_sin` y `hour_cos`. Sus fórmulas están en
`docs/DATA_DICTIONARY.md`, sección 2; su codificación, en el ADR 0008.

**`day_of_week` no entra:** se descartó por medición (ver abajo).

**Quedan fuera:**

| Columna | Razón |
|---|---|
| `newbalanceOrig`, `newbalanceDest` | Son `POST_TRANSACTION`. La medición 12 es la evidencia (`docs/DATA_DICTIONARY.md`, sección 4.b) |
| `isFlaggedFraud` | Es la salida de otro sistema de decisión, y además es inútil: 4 de 4.477 fraudes, recall 0,0009 |
| `nameOrig`, `nameDest` | Son identificadores |
| `step` en crudo | Ver abajo: la razón no es fuga |

## La derivada de igualdad exige `amount > 0`

```
amount_equals_origin_balance = (amount > 0) & (amount == oldbalanceOrg)
```

La feature significa "la transacción se lleva el saldo completo", y un movimiento de cero
sobre un saldo de cero no se lleva nada. Una columna con dos significados es la sección 7.1
de `docs/METHODOLOGY.md` en forma de feature. Las 4 filas del caso degenerado, las 4 fraude
y las 4 `CASH_OUT`, pasan al conjunto que la regla no explica.

## Elegible contra seleccionada

- **Elegible:** puede usarse sin violar el contrato causal. Se deriva de la clasificación
  del esquema; es propiedad de los datos y vive en `schema.py`.
- **Seleccionada:** lo que una configuración concreta usa. Es propiedad del experimento,
  vive con el Pipeline, y es un **mapa** de configuración a features, no una constante,
  porque el ADR 0007 exige la ablación.
- `step` es elegible y no seleccionada: el Pipeline lo consume para derivar y no lo expone.

El mapa es `CONFIGURATIONS`, en `src/fraud_detection/features/pipeline.py`, con dos
entradas: `full` (la configuración completa) y `no_signature` (la ablación del ADR 0007).
Las claves van en inglés por la convención de identificadores de `docs/METHODOLOGY.md`.

## `step` en crudo: una feature cuyo dominio no se solapa entre entrenamiento y servicio

La razón para dejar fuera `step` no es fuga. Su soporte es {1..400} en referencia y
{401..744} en producción, disjuntos por la definición del corte. Cualquier corte que el
modelo aprenda sobre `step` lo violan todas las filas de producción.

Es un modo de falla distinto de la fuga, y conviene nombrarlo así: **una feature cuyo
dominio no se solapa entre entrenamiento y servicio.**

`step % 24` y `(step // 24) % 7` sí se solapan, y eso se sabe por aritmética, sin mirar
producción.

Evidencia adicional, no necesaria para la decisión: 137 de los 400 steps tienen todas sus
filas positivas. Es un artefacto de volumen, porque hay steps con 2 filas.

## `day_of_week`: descartada por medición

`day_of_week` se propuso razonando, no midiendo. Se midió sobre referencia antes de
construir nada, con un criterio fijado antes de medir: entra solo si la autocorrelación del
volumen diario en el rezago 7 es la mayor de los rezagos 1 a 8 **y** supera la banda
aproximada del 95%, 2/√n.

| | Valor |
|---|---|
| Días completos (n) | 15 |
| Autocorrelación en el rezago 7 | −0,035 |
| Rezago con la mayor autocorrelación | 1 (0,470) |
| Banda 2/√n | 0,516 |

El rezago 7 no es el mayor ni supera la banda: no hay estructura semanal que sostener, y
`day_of_week` no entra. La medición es `daily_volume_autocorrelation`, en
`src/fraud_detection/data/profiling.py`, y se reproduce con `scripts/prepare_data.py`.

## `oldbalanceDest` entra

Entra pese a que sus ceros de comercio son marcadores de "no sé", porque el prefijo que
los causa es equivalente a `type == PAYMENT`, que el modelo ya tiene
(`docs/DATA_DICTIONARY.md`, sección 4.c).

## Alcance declinado: agregados por `nameDest`

Agregados por `nameDest`, que se repite hasta 112 veces. Se declinan por tres razones:

- La rúbrica del Curso 3 no premia feature engineering.
- Cualquier agregado tendría que ajustarse dentro del Pipeline solo con referencia, o
  cruzaría el corte.
- El criterio del proyecto es hacer lo necesario.

Queda declinado, no olvidado.

---

**Nota del Ejecutor sobre la procedencia (2026-10-18):** el 744 de {401..744} sale de la
documentación pública del dataset ("Total steps 744"). El step máximo de producción no se
midió, y la decisión no depende de él: alcanza con que producción empiece en 401.
