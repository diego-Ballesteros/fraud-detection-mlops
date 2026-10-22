# ADR 0007 — La firma del simulador como baseline

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Decisión

La regla

```
type ∈ {TRANSFER, CASH_OUT}  ∧  amount_equals_origin_balance
```

con `amount_equals_origin_balance = (amount > 0) & (amount == oldbalanceOrg)` (ADR 0006),
es el **baseline obligatorio de todo el proyecto**. En referencia dispara 4.385 filas, con
0 falsos positivos: precisión 1,000, recall 0,979, y 92 fraudes no cubiertos. Toda métrica
de la fase 2 se reporta contra ella.

La medición está en `docs/DATA_DICTIONARY.md`, sección 4.a. La regla vive como estimador
compatible con scikit-learn en `src/fraud_detection/models/baseline.py`, y llama a la misma
función que construye la feature: una sola definición, usada desde los dos lados.

## La condición de tipo se queda

Se queda aunque se midió como redundante en referencia. Que la igualdad no ocurra en
`CASH_IN`, `DEBIT` ni `PAYMENT` es un hecho del periodo de referencia. Quitar una
condición por redundante en entrenamiento es apoyarse en que el entrenamiento describe el
servicio, que es lo que el corte existe para impedir.

## El baseline es una regla, no un modelo

Su `fit` es un no-op real: no registra nada de los datos. Ajustarlo sobre dos conjuntos
distintos da predicciones idénticas, y un test lo afirma. Esa propiedad es lo que lo hace
un baseline y no un competidor. Ser un estimador permite que la fase 2 lo evalúe con el
mismo arnés que a los modelos, y que la comparación sea del mismo tipo.

## Consecuencias

**Sobre el diseño de la fase 2.** El PR-AUC global está saturado por la región que cubre la
regla, así que la métrica de decisión tiene que ser el punto de operación **más allá** de
la regla. Esa métrica se declara antes de ver resultados, en la fase 2.

**Sobre la fase 3.** Tenemos verdad conocida. Si SHAP no encuentra la regla, el problema es
el explicador. Es la única oportunidad del proyecto de validar el instrumento de
interpretabilidad contra una respuesta ya sabida.

**Sobre la fase 4.** La pregunta de drift se vuelve concreta: si la regla mantiene su
precisión y su recall en los steps 401+.

## Ablación declarada

Una configuración sin `oldbalanceOrg` ni la derivada de igualdad
(`amount_equals_origin_balance`), para cuantificar cuánto del desempeño es la firma.

- Se corre una vez por familia de modelo; no es una búsqueda.
- El Pipeline tiene que ser parametrizable para permitirla, con **un solo objeto Pipeline
  configurable**, no dos implementaciones. Es la configuración `no_signature` del mapa
  `CONFIGURATIONS` (ADR 0006), que construye la misma fábrica.

## Alternativa descartada

**Dejar que la firma domine y reportar ~0,98 sin más.** Es honesto solo si se documenta,
pero vacía las cuatro secciones de la rúbrica:

- los hiperparámetros no se separan,
- la interpretabilidad se vuelve trivial,
- y el monitoreo no tiene nada que detectar.

---

**Nota de estado (2026-10-22, fase 5):** la ablación declarada aquí no se corrió en la fase 2 y
se corrió en la de cierre, con dos diferencias respecto de lo declarado: **solo para LightGBM**,
el modelo elegido, con los hiperparámetros del ensayo 5 (ADR 0010), y no una vez por familia; y
además sobre producción, con un umbral propio elegido sobre referencia. Resultado (EVAL-013): sin
la bandera ni `oldbalanceOrg`, ningún umbral alcanza precisión 0,99 en ningún fold, y el recall de
decisión pasa de 0,9970 a 0. La consecuencia que este ADR anticipaba para la fase 2 —que el PR-AUC
global está saturado por la firma— queda medida desde el otro lado: sin la firma, el mismo modelo
da 0,074. La decisión no cambia.
