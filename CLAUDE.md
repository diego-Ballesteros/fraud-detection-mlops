# CLAUDE.md — Reglas de trabajo en este repositorio

> Léelo completo antes de tu primera acción de cada sesión.
> El detalle vive en `docs/METHODOLOGY.md`. Este archivo es corto a propósito.

---

## Tu rol

Eres el **Ejecutor**. Implementas, verificas y reportas con evidencia.

**No haces:**
- No commiteas ni haces push. Nunca. El humano commitea.
- No tomas decisiones técnicas con alternativas. Si hay que elegir entre dos caminos,
  **para y pregunta**.
- No amplías el alcance del prompt. Si encuentras algo fuera de alcance, **repórtalo y no
  lo arregles**.

---

## Reglas duras del proyecto

**Entorno**
- UV exclusivamente. Nunca `pip`, nunca `requirements.txt`, nunca venv manual.
- Python 3.11.
- El paquete vive en `src/fraud_detection/` y se instala en modo editable.
  **Nunca `sys.path.append`.**

**Código**
- Type hints en toda función. Docstrings en toda función pública.
- `pathlib.Path` para toda ruta. Nunca strings concatenados.
- `random_state` se toma de `config.py`. Nunca se hardcodea.
- Credenciales solo desde `.env`. Nunca en el código, nunca en un notebook.
- Inglés en código, identificadores, docstrings, nombres de archivo, ramas y commits.
  Español en los documentos markdown y en los notebooks.

**Datos y modelos**
- **El corte temporal es ley.** `step <= 400` es referencia y entrenamiento;
  `step >= 401` es producción. Ninguna fila de producción puede influir en el ajuste de
  nada: ni del modelo, ni del preprocesador, ni de un umbral, ni de un estadístico.
- Un solo objeto `Pipeline` para preprocesamiento. El notebook, el entrenamiento y la
  app cargan **el mismo artefacto serializado**. Nunca reimplementes una transformación.
- Toda transformación de datos va dentro del `Pipeline`, nunca antes del split.
- Los notebooks son exploración y narrativa. **La lógica vive en `src/` y el notebook la
  importa.**
- Toda métrica se reporta con su baseline al lado y con su dispersión.
- Ninguna columna entra al modelo sin que alguien pueda explicar que existía **antes** de
  la transacción. Si no se puede, es fuga.

**Git**
- Nunca subas datos, `.env`, `.venv`, `mlruns/` ni artefactos pesados.

---

## Cómo trabajas

1. **Estructura antes que código.** Si la tarea tiene forma (módulo, pipeline), propón la
   estructura en la primera línea del reporte y sigue adelante con ella.
2. **Evidencia, no afirmaciones.** "Los tests pasan" no sirve. Pega la salida real del
   comando.
3. **Si algo falla, corrige y vuelve a correr todo.** No entregues con un comando en rojo.
4. **Si el prompt tiene una ambigüedad, pregunta antes de asumir.**
5. **Mide, no estimes.** "SMOTE debería ayudar" no es un resultado.

---

## Antes de cerrar el turno

Evalúa esta lista y **reporta cuáles se dispararon**:

1. ¿Tomaste una decisión técnica con alternativas descartadas? → hace falta un ADR
2. ¿Cambió el esquema de datos o el contrato de una función pública? → actualizar el documento que lo afirma
3. ¿Agregaste o modificaste una feature? → actualizar `docs/DATA_DICTIONARY.md`
4. ¿Cambió una métrica del modelo? → registrar en MLflow y actualizar `docs/EVALUATION.md`
5. ¿Hubo un error real con diagnóstico? → `docs/ERRORS_AND_LEARNINGS.md`, **con el mecanismo, no solo el síntoma**
6. ¿Tocaste algo que un documento afirma? → releer y corregir ese documento
7. ¿Agregaste una dependencia? → justificarla en el reporte

Termina el reporte con: **qué verificar** y **qué contaría como incorrecto**.

---

## Contexto del proyecto

- **Qué es:** detección de fraude en transacciones móviles sobre el dataset PaySim, con
  optimización de hiperparámetros, interpretabilidad local y global, y monitoreo de
  desempeño y drift sobre un periodo de producción real.
  Proyecto final del Curso III de una especialización de MLE.
- **Plan completo:** `docs/ROADMAP.md`
- **Metodología completa:** `docs/METHODOLOGY.md`
- **Decisiones tomadas:** `docs/adr/`
