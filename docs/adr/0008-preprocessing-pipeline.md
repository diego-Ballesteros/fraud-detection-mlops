# ADR 0008 — El Pipeline de preprocesamiento

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Decisión

Un solo `Pipeline` por configuración, construido por una fábrica, en
`src/fraud_detection/features/pipeline.py`:

1. `derive`: conserva las columnas elegibles y agrega las derivadas
   (`src/fraud_detection/features/builder.py`).
2. `columns`: un solo `ColumnTransformer`, con ramas que **particionan** las columnas
   seleccionadas:

| Rama | Columnas | Transformación |
|---|---|---|
| Monetarias | `amount`, `oldbalanceOrg` si la configuración la usa, `oldbalanceDest` | `log1p` y después `RobustScaler` |
| Categórica | `type` | One-hot con `handle_unknown="error"` |
| Derivadas | `amount_equals_origin_balance` si aplica, `hour_sin`, `hour_cos` | Pasan tal cual |

Salida en pandas (`set_output(transform="pandas")`) y `remainder="drop"`. El drop es seguro
porque un test afirma la partición contra el esquema.

## Por qué una fábrica y no un objeto construido

De la metodología: un objeto construido invita a reutilizarlo entre folds, y un objeto
reutilizado es un objeto ya ajustado. La fábrica recibe el nombre de la configuración y
devuelve un `Pipeline` nuevo.

## La hora, con codificación cíclica

`hour_sin` y `hour_cos` se calculan desde `step % 24`.

- Preserva la adyacencia entre la hora 23 y la 0.
- Cuesta 2 columnas en vez de 24.
- Sirve para árboles y para modelos lineales.

**Alternativas descartadas:**

- **One-hot:** 24 columnas, y pierde la adyacencia.
- **Ordinal crudo:** implica que la hora 23 está lejos de la 0.

## Las columnas monetarias: `log1p` y `RobustScaler`

Los montos van de 0 a 92.445.516,64, con mediana 75.630,26. Un modelo lineal sobre el crudo
lo domina la cola. `log1p` es el tratamiento estándar de una cantidad no negativa de cola
pesada y está definido en 0, lo que importa porque `amount == 0` ocurre. Los árboles son
invariantes a esta transformación, así que no les cuesta nada.

**Alternativa descartada:** `StandardScaler`, que la cola domina.

## `type`: one-hot con `handle_unknown="error"`, no `"ignore"`

El dominio es cerrado y está validado en los dos periodos. Si en servicio aparece un tipo
nuevo queremos un error, no una fila de ceros en silencio.

## La garantía estructural de nivel 1

Cada rama recibe un marco que **no** contiene las columnas de las otras, de modo que una
violación levanta `KeyError` en vez de funcionar. Es la diferencia entre una regla escrita
y una regla imposible de romper (`docs/METHODOLOGY.md`, sección 6.5).

---

**Nota del Ejecutor sobre la procedencia (2026-10-18):** el mínimo, la mediana y el máximo
de los montos salen de `flag_profile`, medidos sobre las 5.787.026 filas de referencia **no
marcadas** por `isFlaggedFraud`, no sobre las 5.787.030. Las 4 filas marcadas tienen montos
entre 536.624,41 y 4.953.893,08.
