# Metodología de trabajo — Fraud Detection MLOps

> Adaptación de la metodología Arquitecto / Ejecutor, con las lecciones del proyecto
> anterior ya incorporadas.
>
> Este documento es el **dueño del contenido**. `CLAUDE.md` es un puntero corto hacia aquí.
> Si algo se contradice entre los dos, este manda.

---

## 1. Los tres roles

| Rol | Quién | Qué hace | Qué NO hace |
|---|---|---|---|
| **Arquitecto** | Claude en el chat del proyecto | Diagnostica, debate, decide, enseña, redacta prompts | **Nunca escribe código.** No ve el repositorio |
| **Ejecutor** | Claude Code, dentro del repo | Implementa, verifica, reporta con evidencia | **Nunca commitea.** No toma decisiones técnicas |
| **Verificador** | Yo | Valido, decido, explico, commiteo | No implemento |

### Por qué el Arquitecto no ve el código

Parece una limitación y es el mecanismo de calidad del proyecto. **Obliga a que la
documentación sea buena**, porque es lo único que el Arquitecto tiene. Un documento
desactualizado se detecta de inmediato: el Arquitecto propone algo que ya existe, o razona
sobre una garantía que no está.

**Lección del proyecto anterior:** el Arquitecto no afirma hechos sobre el estado del
repositorio en el bloque de contexto de un prompt. Si un dato es verificable, el prompt
pide medirlo dentro del turno. Afirmar un conteo, un nombre de rama o un estado de archivo
"de memoria" produjo tres turnos de corrección innecesarios.

### Por qué solo yo commiteo

Un commit es una afirmación de que algo funciona. Además, en este proyecto **los commits
son evidencia evaluada**: la rúbrica mira el historial como prueba del proceso. Un commit
hecho por un agente es evidencia falsificada.

### La compuerta de aprendizaje

> **No se mergea nada que yo no pueda explicar en voz alta.**

Si no puedo explicar por qué una función existe, qué hace una transformación o por qué
elegimos una métrica, el paso no está terminado — está *escrito*, que no es lo mismo. En
ese caso el siguiente turno es una pregunta al Arquitecto, no un prompt al Ejecutor.

---

## 2. El ciclo

```
Objetivo del día (viene del roadmap)
   ↓
DIAGNÓSTICO / MEDICIÓN (solo lectura)   ← el paso que más se saltea
   ↓
DEBATE con el Arquitecto                ← decisiones, sin código de por medio
   ↓
PROMPT estructurado                     ← un solo paso, nunca dos
   ↓
EJECUCIÓN + REPORTE con evidencia       ← salida real de comandos, no afirmaciones
   ↓
VERIFICACIÓN + EXPLICACIÓN              ← ¿funciona? ¿lo entiendo?
   ↓
COMMIT
```

### Medir antes de modelar

- EDA antes de diseñar el pipeline
- Baseline trivial antes de cualquier modelo
- Medir la estrategia antes de adoptarla (¿SMOTE ayuda? se mide, no se asume)

En la práctica, esta medición casi siempre devuelve algo que invalida el plan.

### Un paso a la vez

**Nunca dos prompts de ejecución juntos.** Un paso puede destapar algo que hay que resolver
antes del siguiente, y con los dos entregados esa información llega tarde.

Cuando el reporte del Ejecutor pide verificación, el Arquitecto la traduce a un guion
ejecutable: qué comando correr, qué debería salir, y qué contaría como incorrecto.

---

## 3. Anatomía de un prompt al Ejecutor

```
## CONTEXT
Qué existe, dónde estamos, y POR QUÉ las decisiones ya tomadas son las que son

## TASK
Una frase — qué tiene que ser cierto al terminar

## STRUCTURE TO CREATE (si aplica)
Árbol exacto de archivos

## RULES (do not deviate)
Decisiones técnicas no negociables, incluidos los límites de qué NO tocar

## FILE SPECS
Especificación de cada archivo: qué hace, no cómo se escribe

## AFTER CREATING ALL FILES
Comandos de verificación a correr, con la salida pegada en el reporte
```

### Las razones viajan con las instrucciones

Un prompt que dice *"usá `random_state=42`"* produce código correcto. Uno que dice *"usá
`random_state=42` en todo, y tomalo de `config.py` en vez de hardcodearlo, porque
necesitamos que un tercero reproduzca la métrica exacta desde cero"* produce código
correcto **y un Ejecutor capaz de decidir bien en el caso que no previste**.

### Los límites explícitos

Decir qué **no** tocar es tan importante como decir qué hacer:

> *"Si encontrás X, reportalo y NO lo arregles."*

### Estructura antes que código

Para cualquier trabajo con forma, el prompt pide que el Ejecutor **proponga la estructura
al principio del reporte y siga adelante con ella**. Si la estructura está mal, se ve en la
primera línea en vez de después de 400 líneas.

### Autoría de los ADRs

El razonamiento de un ADR lo produce el **Arquitecto** en el chat, a partir de una decisión
discutida. El Ejecutor **transcribe y formatea**; nunca redacta la justificación desde
cero.

La razón: un ADR con justificación inventada se lee bien y no registra nada, y es peor que
no tenerlo, porque el autor lo leerá semanas después creyendo que es su propio
razonamiento. **Un ADR cuyo contenido no salió de una decisión discutida no se escribe.**

### Enrutamiento de modelo

| Modelo | Cuándo |
|---|---|
| El más liviano | Tareas mecánicas: renombrar, mover, formatear, docstrings |
| El intermedio | La mayoría del trabajo: módulos, tests, scripts, transcripción de ADRs |
| El más capaz | Pipeline de preprocesamiento, espacio de búsqueda del AutoML, diseño del monitoreo, cualquier cosa donde un error se propaga en silencio a todas las métricas |

Criterio: **no es la dificultad, es el costo del error.**

### Higiene de sesiones

- **Misma sesión mientras el dominio no cambie.** Turnos consecutivos sobre el mismo tema
  comparten contexto útil y abrir una sesión nueva lo desperdicia.
- **Sesión nueva cuando el dominio cambia** — de datos a modelado, de modelado a
  interpretabilidad, de interpretabilidad a monitoreo.

---

## 4. Arquitectura de documentos

| Documento | Contrato |
|---|---|
| `README.md` | El entregable. Qué es el proyecto, para el evaluador y para un tercero. También cumple el contrato de comportamiento del sistema: qué hace y qué lo restringe |
| `docs/DATA_DICTIONARY.md` | Contrato de datos: columnas, tipos, rangos, features derivadas, y qué columnas están prohibidas y por qué |
| `docs/MODEL_CARD.md` | Qué hace el modelo, con qué datos, qué NO debe usarse para, y sus limitaciones |
| `docs/adr/` | Una entrada por decisión, con su razón. **Nunca se corrige hacia atrás** |
| `docs/analysis/` | Mediciones de registro, cada una **reproducible por un script del repositorio**. Sus cifras se citan desde los ADR. No se reescriben hacia atrás: se les agrega una nota de estado fechada |
| `docs/ERRORS_AND_LEARNINGS.md` | Una entrada por error real, con **el mecanismo** — no solo el síntoma |
| `docs/EVALUATION.md` | Qué se midió, cómo, con qué baseline al lado, y con qué resultado |
| `docs/GIT_STRATEGY.md` | Ramas, commits, pull requests, releases y convención de idioma |
| `docs/METHODOLOGY.md` | Este documento. Dueño de la metodología de trabajo |
| `CLAUDE.md` | Puntero corto a este documento. **Corto**, o deja de leerse |

### Reglas duras

- **Cero números de línea** en cualquier documento. Nombrar un módulo está permitido;
  `loader.py:42` está prohibido. No sobrevive a un solo refactor.
- **Los ADRs no se reescriben.** Cuando una decisión deja de ser válida se marca
  `superseded` con la razón.
- **Las afirmaciones negativas se reverifican siempre.** *"El dataset no tiene valores
  faltantes"* es cierta el día que se escribe y puede ser falsa después. Son las que más
  daño hacen.
- **Nada se duplica entre documentos.** El mismo contenido en dos lados diverge, porque
  nadie actualiza dos veces.

### Estándares de código

Estas reglas están forzadas por `ruff` y `mypy`, y se escriben aquí igualmente: `CLAUDE.md`
es un puntero y no puede ser la fuente de una regla; una regla sin documento dueño no se
puede citar ni discutir.

- Type hints en toda función; docstrings en toda función pública.
- `pathlib.Path` para toda ruta; nunca strings concatenados.
- `random_state` se toma de `config.py`; nunca se hardcodea.
- Credenciales solo desde `.env`.

### Convención de idioma

| Elemento | Idioma |
|---|---|
| Identificadores, funciones, clases, variables | Inglés |
| Docstrings y comentarios en código | Inglés |
| Nombres de archivos, carpetas, ramas, commits | Inglés |
| Todos los documentos markdown y los notebooks | Español |

Los nombres de columnas del dataset se conservan como los entrega la fuente; lo que va en
español es su descripción.

---

## 5. Protocolo de auto-documentación

Al final de cada turno, el Ejecutor **evalúa esta lista y reporta cuáles se dispararon**:

1. ¿Se tomó una decisión técnica con alternativas descartadas? → ADR
2. ¿Cambió el esquema de datos o el contrato de una función pública? → actualizar el documento que lo afirma
3. ¿Se agregó o modificó una feature? → actualizar el diccionario de datos
4. ¿Cambió una métrica del modelo? → registrar el run en MLflow y actualizar `EVALUATION.md`
5. ¿Hubo un error real, con diagnóstico? → `ERRORS_AND_LEARNINGS.md`, **con el mecanismo**
6. ¿Se tocó algo que un documento afirma? → releer ese documento y corregirlo
7. ¿Se agregó una dependencia? → justificarla en el reporte

El criterio 6 es el más importante y el más difícil. Formulación ejecutable: *al cerrar un
tema, buscarlo por su nombre en los documentos*. Eso es un `grep`, no un acto de memoria.
Y un `grep` que devuelve vacío **necesita su control positivo**: sin él, un patrón mal
escrito se lee igual que un problema resuelto.

---

## 6. Disciplina de verificación

### 6.1 · Una métrica que no puede fallar no prueba nada

Este dataset tiene una prevalencia de fraude muy baja. Un modelo que siempre predice
"legítimo" saca una accuracy altísima. **Ese número no prueba nada.**

- **Baseline trivial primero.** Antes de cualquier modelo, medir el piso.
- **Prueba del target barajado.** Entrenar el pipeline completo con el target permutado
  aleatoriamente. El desempeño debe colapsar al nivel del azar. **Si no colapsa, hay
  leakage.**
- **Cada métrica reportada viene con su baseline al lado**, con su dispersión, y con el
  número de particiones sobre el que se calculó.

### 6.2 · Medir, no estimar

*"SMOTE debería ayudar"* no es un resultado. Y cuando algo no se puede medir, decirlo: *"no
puedo determinar si el modelo generaliza más allá del step 718 porque la simulación
termina ahí"* es infinitamente más útil que una conjetura presentada como hallazgo.

### 6.3 · Un solo pipeline, varios consumidores

El modo de falla más traicionero: el preprocesamiento del notebook, el del script de
entrenamiento y el de la app de monitoreo divergen. Todo verde en el notebook, números
distintos en el dashboard.

- **Un solo objeto `Pipeline`**, serializado y versionado en MLflow. Todos cargan **el
  mismo artefacto**.
- **Serialización real** en los tests: `dump`/`load` de verdad.
- **Test de ida y vuelta**: fila cruda → pipeline cargado desde disco → predicción.

### 6.4 · Una superficie nueva no se cierra sin un uso real

La app de monitoreo no está lista porque los tests pasan. Está lista cuando se abre y
muestra los números correctos. El proyecto anterior encontró dos defectos reales en la API
que los tests dejaban pasar, **porque los tests asertaban sobre el mensaje y no sobre el
campo**.

### 6.5 · El leakage se hace imposible, no se recuerda

Jerarquía, de más fuerte a más débil:

1. **La herramienta lo impide** — una función que recibe solo las columnas del periodo de
   referencia no puede leer las de producción
2. **Un test lo detecta** — la prueba del target barajado
3. **Está escrito en un documento** — se cumple mientras alguien lo recuerde

**El nivel 3 falla.** Antes de escribir una regla, preguntarse si puede subir de nivel.

### 6.6 · Reproducibilidad verificada, no supuesta

`random_state=42` en todo, tomado de `config.py`. Y al menos una vez antes de entregar:
clonar el repo en un directorio limpio, seguir **solo lo que dice el README**, y confirmar
que las métricas coinciden. En el proyecto anterior ese ejercicio encontró una cifra
documentada que ya no reproducía — no un fallo de instalación.

---

## 7. Modos de falla propios de ML, y sus defensas

### 7.1 · La imputación silenciosa

Un `NaN` convertido en `0` deja de ser *"no sé"* y pasa a ser un hecho de negocio. El
modelo aprende sobre una mentira bien formada, y ningún chequeo de error la detecta.

**Defensa:** toda imputación se declara en el diccionario de datos y en el Model Card, con
su columna indicadora cuando aplica. La ausencia de dato suele ser informativa por sí sola.

### 7.2 · La columna que no existía cuando había que decidir

Es el modo de falla central de este dataset. Una columna puede estar bien formada, ser
numérica, correlacionar fuerte con el target, y **no existir en el momento en que la
decisión se toma**. Un indicador producido por el propio sistema de detección, o un saldo
calculado después de que la transacción se resolvió, entran así.

**Defensa:** el diccionario de datos declara, para cada columna, **cuándo existe**. Una
columna sin ese dato no entra al pipeline. El validador falla ruidosamente si aparece una
columna que el contrato no conoce.

### 7.3 · La métrica sin contexto

Un PR-AUC no dice nada sin saber contra qué se compara, con cuántas particiones, con qué
dispersión y sobre qué prevalencia.

**Defensa:** toda métrica se reporta con baseline, número de particiones, dispersión y
tamaño de muestra. Un número solo es una opinión.

### 7.4 · El notebook como fuente de verdad

Un notebook con estado acumulado produce resultados que nadie puede reproducir.

**Defensa:** los notebooks son **exploración y narrativa**, nunca implementación. Toda
lógica vive en `src/` y el notebook la importa. La verificación no es "lo corrí": es
ejecutarlo con un kernel limpio y comprobar que los contadores de ejecución son
consecutivos desde uno. Un `.ipynb` sintácticamente roto sigue siendo un JSON válido, así
que ninguna herramienta de archivos lo detecta.

### 7.5 · El proceso de dos pasos donde el segundo se olvida

Entrenar y registrar. Cambiar la feature y actualizar el diccionario. Mejorar el modelo y
actualizar el Model Card. **Nada avisa cuando falta el segundo.**

**Defensa:** el protocolo de la sección 5, ejecutado al final de cada turno.

### 7.6 · El instrumento de medición sin tests

Un script de evaluación es código sin tests hasta que alguien se los escribe, y sus fallos
producen **evidencia falsa que se lee igual que evidencia buena**. En el proyecto anterior,
un detector comparaba cifras a una precisión que él mismo imponía y reportó tres hallazgos
falsos antes de que alguien lo mirara.

**Defensa:** todo script que produce una cifra que va a un documento tiene tests con casos
donde el valor esperado se calcula a mano.

---

## 8. Señales de que la metodología funciona

- El Ejecutor **corrige premisas del Arquitecto con evidencia** en vez de aceptarlas
- Los reportes distinguen **lo que se midió de lo que se infirió**
- Aparecen hallazgos que nadie buscaba
- Se declaran **intercambios** en vez de venderlos como mejoras puras
- Puedo explicar cualquier archivo del repo sin volver a abrirlo

## 9. Señales de que se está degradando

- Los prompts empiezan a mezclar dos temas
- Los reportes dicen "todo verde" sin pegar la salida de los comandos
- Se saltea el EDA o el baseline porque "ya sabemos qué va a dar"
- Aparece lógica de negocio dentro de un notebook
- Se acepta una métrica sin su baseline al lado
- Mergeo algo que no podría explicar

---

*Documento vivo. Actualizar cuando un error real enseñe algo que no está aquí.*