# ADR 0006 — El conjunto de columnas del modelo

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Decisión

**Entran:** `amount`, `oldbalanceOrg`, `oldbalanceDest`, `type`, más tres derivadas:
`amount_equals_origin_balance`, `hour_of_day` y `day_of_week`. Sus fórmulas están en
`docs/DATA_DICTIONARY.md`, sección 2.

**Quedan fuera:**

| Columna | Razón |
|---|---|
| `newbalanceOrig`, `newbalanceDest` | Son `POST_TRANSACTION`. La medición 12 es la evidencia (`docs/DATA_DICTIONARY.md`, sección 4.b) |
| `isFlaggedFraud` | Es la salida de otro sistema de decisión, y además es inútil: 4 de 4.477 fraudes, recall 0,0009 |
| `nameOrig`, `nameDest` | Son identificadores |
| `step` en crudo | Ver abajo: la razón no es fuga |

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
