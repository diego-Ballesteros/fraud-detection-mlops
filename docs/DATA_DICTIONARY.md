# Diccionario de datos

> Contrato: columnas, tipos, rangos, features derivadas, y qué columnas están prohibidas y
> por qué. La versión ejecutable del contrato es `src/fraud_detection/data/schema.py`; este
> documento la explica y no la reemplaza.
>
> **Procedencia de las cifras.** Toda cifra de este documento sale de una función de
> `src/fraud_detection/data/profiling.py` cubierta por `tests/test_profiling.py`, medida
> sobre el periodo de referencia (`step <= 400`) el 2026-10-18, y se reproduce con
> `uv run python scripts/prepare_data.py`. Junto a cada cifra va la función que la produce.
> Las afirmaciones negativas ("cero filas", "ninguna") valen para esa fecha y ese periodo.

---

## 1. Columnas del CSV

"Cuándo existe" es la enumeración `Availability` del esquema:

| Valor | Significado |
|---|---|
| `PRE_TRANSACTION` | Existía antes de la transacción; es información disponible al decidir |
| `POST_TRANSACTION` | Solo existe después de que la transacción se ejecutó |
| `SYSTEM_OUTPUT` | Es la salida de otro sistema de decisión, no una entrada |
| `TARGET` | Es la etiqueta |
| `IDENTIFIER` | Identifica una entidad; no es una medición del fenómeno |

Una columna que no se pudo ubicar antes de la transacción a partir de su nombre y de la
documentación pública del dataset se declara `POST_TRANSACTION` por defecto. Ninguna de las
11 columnas admite nulos, y `amount`, `oldbalanceOrg` y `oldbalanceDest` se declaran no
negativas: el validador lo revisa en los dos periodos, y la guarda de `log1p` del Pipeline
es la segunda línea (sección 3).

| Columna | Descripción | dtype | Cuándo existe | Dominio | ¿Entra al modelo? |
|---|---|---|---|---|---|
| `step` | Hora de la simulación (1 step = 1 hora) | `int16` | `PRE_TRANSACTION` | — | **No.** Su soporte no se solapa entre entrenamiento y producción (ADR 0006); entra transformada como `hour_sin` y `hour_cos` |
| `type` | Tipo de operación solicitada | `category` | `PRE_TRANSACTION` | `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` | **Sí.** Es parte de la solicitud |
| `amount` | Monto que se pide mover | `float64` | `PRE_TRANSACTION` | ≥ 0 | **Sí.** Es parte de la solicitud |
| `nameOrig` | Cliente que inicia la transacción | `string[pyarrow]` | `IDENTIFIER` | — | **No.** Es un identificador |
| `oldbalanceOrg` | Saldo del origen antes de la transacción | `float64` | `PRE_TRANSACTION` | ≥ 0 | **Sí.** Es el saldo propio de quien solicita; se excluye en la ablación del ADR 0007 |
| `newbalanceOrig` | Saldo del origen después de la transacción | `float64` | `POST_TRANSACTION` | — | **No.** Solo existe una vez ejecutada (sección 4.b) |
| `nameDest` | Destinatario; el prefijo `M` marca un comercio | `string[pyarrow]` | `IDENTIFIER` | — | **No.** Es un identificador |
| `oldbalanceDest` | Saldo del destino antes de la transacción | `float64` | `PRE_TRANSACTION` | ≥ 0 | **Sí,** pese a que sus ceros de comercio son marcadores de "no sé" (sección 4.c, ADR 0006) |
| `newbalanceDest` | Saldo del destino después de la transacción | `float64` | `POST_TRANSACTION` | — | **No.** Solo existe una vez ejecutada (sección 4.b) |
| `isFraud` | Si la transacción la hizo un agente fraudulento de la simulación | `int8` | `TARGET` | `0`, `1` | — Es la etiqueta |
| `isFlaggedFraud` | Salida de la regla de negocio que marca transferencias grandes | `int8` | `SYSTEM_OUTPUT` | `0`, `1` | **No.** Es la salida de otro sistema de decisión, y marca solo 4 filas, las 4 fraude, de 4.477 fraudes (`flag_profile`, `class_balance`; ADR 0006) |

La documentación pública escribe `CASH-IN` y `CASH-OUT`; el CSV usa guion bajo, y el
contrato sigue al CSV.

### Elegible no es lo mismo que seleccionada

`MODEL_ELIGIBLE_COLUMNS`, en `schema.py`, es exactamente el conjunto `PRE_TRANSACTION`:
`step`, `type`, `amount`, `oldbalanceOrg` y `oldbalanceDest`. Ser elegible es una
propiedad de los datos; estar seleccionada es una propiedad del experimento, y vive con el
Pipeline como un mapa de configuración a features: `CONFIGURATIONS`, en
`src/fraud_detection/features/pipeline.py` (ADR 0006).

| Configuración | Columnas crudas que lee | Features que expone |
|---|---|---|
| `full` | `amount`, `oldbalanceOrg`, `oldbalanceDest`, `type`, `step` | `amount`, `oldbalanceOrg`, `oldbalanceDest`, las 5 columnas one-hot de `type`, `amount_equals_origin_balance`, `hour_sin`, `hour_cos` |
| `no_signature` | `amount`, `oldbalanceDest`, `type`, `step` | `amount`, `oldbalanceDest`, las 5 columnas one-hot de `type`, `hour_sin`, `hour_cos` |

`step` es elegible y nunca se expone: el Pipeline lo consume para derivar la hora.
`no_signature` es la ablación del ADR 0007.

---

## 2. Features derivadas

Se construyen **dentro del Pipeline** (`src/fraud_detection/features/builder.py`), solo a
partir de columnas `PRE_TRANSACTION`; por eso son también `PRE_TRANSACTION`. Su
codificación posterior está en el ADR 0008.

### `amount_equals_origin_balance`

```
amount_equals_origin_balance = (amount > 0) & (amount == oldbalanceOrg)
```

- Booleana: "la transacción se lleva el saldo completo". Es la firma del simulador
  (sección 4.a), y queda fuera de la ablación del ADR 0007.
- Exige `amount > 0`: un movimiento de cero sobre un saldo de cero no se lleva nada
  (ADR 0006). En referencia, ese caso son 4 filas, las 4 fraude y las 4 `CASH_OUT`
  (`signature_by_type`).
- Es una sola definición: la misma función la usan el Pipeline, el baseline y las
  mediciones de este documento.

### `hour_sin` y `hour_cos`

```
hour_sin = sin(2π · (step % 24) / 24)
hour_cos = cos(2π · (step % 24) / 24)
```

- La hora del día como punto del círculo unidad: la hora 23 queda junto a la 0 (ADR 0008).
- Se solapan entre periodos por aritmética, sin mirar producción: referencia tiene los 400
  steps (`step_distribution`: 400 distintos, ninguno faltante), así que contiene los 24
  restos de `step % 24`, y todo step de producción cae en alguno de ellos.

### `day_of_week`: no se construye

```
day_of_week = (step // 24) % 7
```

Se descartó por medición (ADR 0006). La autocorrelación del volumen diario sobre los 15
días completos de referencia es −0,035 en el rezago 7; el mayor es el rezago 1 (0,470), y la
banda 2/√15 es 0,516 (`daily_volume_autocorrelation`).

Los bordes, por si se reconsidera:

- Es la posición en un ciclo de 7 días contado desde el step 0, no un día del calendario:
  la documentación no dice qué día de la semana es el primer step.
- El step 24 es la hora 0 del día 1. Como el primer step observado es el 1
  (`step_distribution`), el día 0 tiene 23 horas.
- Los días 2, 3 y 4 de referencia tienen 6.749, 21.904 y 12.995 filas; los otros doce días
  completos, entre 349.800 y 452.761 (`daily_volume_autocorrelation`). Las tres posiciones
  del ciclo con la menor media de filas por step, 2, 3 y 4, son las que contienen esos días
  (`rows_per_step_by_day_of_cycle`).

---

## 3. Cómo se hace cumplir el contrato

`src/fraud_detection/data/validator.py` corre todos los chequeos, acumula los hallazgos y
falla al final con el listado completo. Un chequeo que no puede ejecutarse sale como
omitido, nunca como aprobado.

- **Sobre el archivo completo:** que las columnas sean exactamente las declaradas y en
  orden, que los dtypes declarados se puedan aplicar, y que el parser devuelva una fila por
  línea de datos.
- **Sobre referencia:** dominios, nulos y no negatividad, con conteos y valores en el
  detalle.
- **Sobre producción:** solo propiedades del registro (dominios, nulos, no negatividad,
  partición), y cada hallazgo dice `True` o `False`, nunca un conteo ni una proporción. Un
  conteo informa sobre el fenómeno, y eso cruza el corte.

La no negatividad tiene una segunda línea: la transformación `log1p` del Pipeline rechaza un
valor negativo. El validador atrapa el registro malformado; la guarda atrapa que alguien use
el Pipeline sobre datos que no pasaron por el validador.

El CSV se parsea con pyarrow y un tipo explícito por columna: `pandas.read_csv` con enteros
angostos desborda sin avisar (`docs/ERRORS_AND_LEARNINGS.md`, ERR-005).

---

## 4. Calidad de datos

### a. La firma del simulador

`amount_equals_origin_balance`, es decir `(amount > 0) & (amount == oldbalanceOrg)`, vale
`True` en **4.385** filas de referencia, y las **4.385 son fraude**: 2.219 en `CASH_OUT` y
2.166 en `TRANSFER` (`signature_by_type`). En las 5.782.553 filas legítimas vale `True`
**cero** veces, en todos los tipos. Las 4 filas `CASH_OUT` con `amount == 0` y
`oldbalanceOrg == 0`, las 4 fraude, quedan fuera por definición (sección 2).

Como regla contra la etiqueta (`signature_rule_outcome`):

| Regla | Marcadas | Fraudes cubiertos | Falsos positivos | Fraudes no cubiertos | Precisión | Recall |
|---|---|---|---|---|---|---|
| `amount_equals_origin_balance` | 4.385 | 4.385 | 0 | 92 | 1,000 | 0,979 |
| ídem, con `type` ∈ {`TRANSFER`, `CASH_OUT`} (el baseline) | 4.385 | 4.385 | 0 | 92 | 1,000 | 0,979 |

En referencia, la restricción de tipo no cambia ninguna cifra, porque la firma nunca ocurre
en `CASH_IN`, `DEBIT` ni `PAYMENT`. El baseline la conserva igual (ADR 0007). El estimador
`SignatureRule`, en `src/fraud_detection/models/baseline.py`, da las mismas cifras.

**No es fuga.** Las dos columnas son `PRE_TRANSACTION`: el monto pedido y el saldo propio
existen antes de la transacción. **Es una firma determinista del generador.** Según la
documentación del dataset, el agente de fraude de PaySim toma el control de una cuenta e
intenta vaciarla. En datos reales habría transferencias legítimas por el saldo completo, y
la regla no tendría esta precisión.

Es el baseline obligatorio del proyecto (ADR 0007).

### b. Los saldos posteriores

`newbalanceOrig` y `newbalanceDest` quedan fuera por ser `POST_TRANSACTION`: solo existen
una vez que la transacción se ejecutó. La medición 12 muestra además cuánto cargan la
etiqueta a través de la aritmética de la simulación.

El error de origen (`oldbalanceOrg - amount - newbalanceOrig`) es cero en el **99,26%** de
los fraudes y en el **19,16%** de las legítimas (`balance_consistency`).

La comparación hay que hacerla dentro de cada tipo. En `CASH_IN` el dinero entra al origen,
así que la fórmula da cero en el 0% de las filas por construcción, y el fraude solo ocurre
en `TRANSFER` y `CASH_OUT` (`balance_consistency_by_type`):

| Tipo | Error de origen cero, fraude | Error de origen cero, legítima |
|---|---|---|
| `TRANSFER` | 99,6% | 3,6% |
| `CASH_OUT` | 98,9% | 10,4% |

En el destino, el agregado esconde el efecto. Dentro de `TRANSFER`, el error de destino
(`oldbalanceDest + amount - newbalanceDest`) es cero en el **69,4%** de las legítimas y en
el **0,04%** de los fraudes (`balance_consistency_by_type`).

### c. Los ceros de saldo de destino

Para los comercios, un saldo de destino en cero es un marcador de "no sé", no un cero. La
documentación del dataset dice que los destinatarios con prefijo `M` no tienen información
de saldo, y el generador escribe 0 en su lugar. Es una imputación silenciosa
(`docs/METHODOLOGY.md`, sección 7.1).

El prefijo `M` equivale a `type == PAYMENT` (`destination_prefix_by_type`). Las 1.949.011
filas `PAYMENT` tienen destino `M`, y ninguna fila de los otros cuatro tipos lo tiene. En
`PAYMENT`, los dos saldos de destino son cero con monto positivo en el 100% de las filas
(`balance_consistency_by_type`).

Por eso **no se agrega un indicador de comercio**. En referencia sería idéntico, fila por
fila, a `type == PAYMENT`, que el modelo ya tiene. `oldbalanceDest` entra igual, porque la
causa de sus ceros de comercio ya está en `type` (ADR 0006).
