# ADR 0013 — Qué dispara un reentrenamiento

- **Estado:** aceptado
- **Fecha:** 2026-10-22
- **Origen:** decisión del humano en el turno de cierre, sobre la propuesta del Model Card de la
  fase 5; el Ejecutor la transcribe. La justificación es una medición del proyecto, no una
  preferencia. El texto dictado decía "la alerta como disparador habríamos reentrenado sobre
  ruido" con la ventana 2 como prueba; con los nombres del ADR 0012 la ventana 2 sola no es una
  alerta, y el humano eligió conservar esos nombres antes de que se escribiera este ADR.

---

## Contexto

El ADR 0012 declaró, antes de medir producción, un criterio de degradación con dos niveles sobre
el recall del modelo por ventana completa de 24 steps, contra la línea 0,9934: **2 ventanas
consecutivas** debajo son una **alerta**, y **3** son **degradación**. No dijo qué se hace con
cada nivel. El Model Card de la fase 5 propuso una respuesta y la marcó como pendiente de decisión.

## Decisión

| Lo que se observa | Qué se hace |
|---|---|
| Una ventana sola debajo de la línea | Nada: es ruido, y se registra |
| **Alerta:** 2 ventanas consecutivas debajo | **Aviso.** Se investiga; no se reentrena |
| **Degradación:** 3 ventanas consecutivas debajo | **Dispara el reentrenamiento** |
| Drift de una feature, sin caída de recall | Nada: el drift no es disparador |

## Por qué: está medido

- **Caer debajo de la línea pasa donde no pasa nada.** La ventana 2 de producción cayó debajo
  (248 de 250 fraudes, 0,9920) sin ningún drift propio: la hora driftea como en todas las
  ventanas, y sus monetarias, su bandera y sus tipos quedan debajo de 0,04 de PSI
  (`docs/analysis/monitoring.md`, sección 4). Fueron dos fraudes perdidos en un día.
- **El nulo del criterio, a los positivos que tuvo cada ventana de producción.** Si el recall
  verdadero fuera 0,9970 todos los días, en 14 ventanas al menos una caería debajo por azar con
  probabilidad **0,931**; una alerta saldría con **0,302**, y una degradación con **0,054**
  (`performance.probability_below` y `performance.probability_of_run`; las dos últimas cifras
  son las de EVAL-011, y la primera se calculó con las mismas funciones al escribir este ADR).
- **Cada reentrenamiento cuesta una validación entera.** Produce un modelo nuevo que hay que
  volver a evaluar por el arnés temporal, con un umbral nuevo elegido sobre referencia, y
  registrar y comparar antes de reemplazar al vigente. Un disparador que salta sobre ruido
  multiplica ese costo sin ninguna señal detrás.

Con la degradación como disparador, un modelo estable se reentrenaría por azar en uno de cada
~19 periodos como este; con la alerta, en casi uno de cada tres; con una ventana sola, en casi
todos.

## Alternativas descartadas

- **La alerta como disparador directo.** Con un modelo estable, saltaría por azar con
  probabilidad 0,302 en 14 días: reentrenar sobre ruido casi una vez de cada tres.
- **Una ventana sola como disparador.** Saltaría por azar con probabilidad 0,931: en producción
  habría reentrenado por la ventana 2, donde no pasaba nada.
- **El drift como disparador.** En producción la hora tuvo PSI 0,32 y 0,33, "significativo", en
  todas las ventanas, y el recall no se movió: el modelo apenas usa la hora (ADR 0012,
  `docs/analysis/monitoring.md`).

## Consecuencias

- **Se reacciona más tarde.** Una degradación real se detecta, como pronto, al cerrar su tercer
  día. Es el intercambio de bajar la tasa de falsa alarma, y se declara como tal.
- **La degradación tampoco es infalible:** con un modelo estable sale por azar con probabilidad
  0,054 en 14 días.
- **El criterio depende del tamaño de las ventanas.** Las probabilidades de arriba son las de los
  232 a 308 fraudes por día de esta producción; con otro volumen de fraude hay que recalcularlas
  con las mismas funciones.
- El Model Card (`docs/MODEL_CARD.md`, sección 9) cita esta decisión.
