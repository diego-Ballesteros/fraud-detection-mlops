# Errores y aprendizajes

> Contrato: el registro guarda mecanismos que habrían costado un turno, hayan fallado o no,
> y documenta el **mecanismo** —por qué ocurrió— y no solo el síntoma. Un registro que solo
> admite errores que alcanzaron a fallar selecciona contra las buenas atrapadas.

---

## Formato de entrada

Cada entrada se agrega al final de la sección **Entradas**, con un identificador
correlativo.

```markdown
### ERR-NNN · <título corto> · AAAA-MM-DD

- **Síntoma:** qué se observó, con la salida real si la hay.
- **Mecanismo:** por qué ocurrió. La causa, no la descripción del síntoma.
- **Cómo se detectó:** qué comando, test o revisión lo hizo visible.
- **Corrección:** qué se cambió.
- **Defensa:** qué impide que vuelva a ocurrir, y en qué nivel de la jerarquía de
  `docs/METHODOLOGY.md`, sección 6.5: la herramienta lo impide, un test lo detecta, o
  está escrito en un documento.
- **Aprendizaje:** qué haríamos distinto la próxima vez.
- **Referencias:** commit, PR o ADR relacionados.
```

---

## Entradas

### ERR-001 · El CSV del dataset quedó commiteado · 2026-10-18

- **Síntoma:** la verificación de que git no ve el archivo de datos salió en rojo.
  `git check-ignore -v data/raw/PS_20174392719_1491204439457_log.csv` terminó con código 1
  sin imprimir nada, y `git ls-files data/` listó el CSV.
- **Mecanismo:** las reglas de `.gitignore` no se aplican a archivos que git ya rastrea, y
  `git check-ignore` no informa sobre ellos aunque una regla coincida. El CSV entró en el
  commit local `366df40` cuando el repositorio todavía no tenía `.gitignore`; escribir la
  regla después no lo saca del índice. Por eso una regla correcta y un archivo rastreado
  producen la misma salida roja que una regla mal escrita.
- **Evidencia medida:**
  - `366df40` agregó el CSV como 6.362.621 líneas; en ese commit no existía `.gitignore`.
  - `main` y `develop` locales estaban un commit por delante de `origin`, que seguía en el
    commit inicial: ninguna rama remota contenía el CSV.
  - El blob en el commit pesaba 493.534.783 bytes.
- **Cómo se detectó:** `git ls-files` al inicio del turno de andamiaje, antes de escribir
  ningún archivo.
- **Cómo se separaron las dos causas posibles:** `git check-ignore -v --no-index` evalúa
  las reglas sin mirar el índice. Con controles positivos y negativos, el CSV coincidió
  con `/data/**`, los `.gitkeep` y `.env.example` aparecieron re-incluidos por sus reglas
  de negación, y `config.py` y `uv.lock` no coincidieron con ninguna. La regla era
  correcta; la causa era el rastreo.
- **Corrección:** `366df40` fue reemplazado por `0387288`, que no contiene el CSV. Estado
  medido después: ninguna ref alcanza el blob (`git rev-list --all --objects` no lo
  lista) y `main` coincide con `origin/main`.
- **Defensa:** `.gitignore` excluye todo `data/` salvo los `.gitkeep`, y git no agrega
  archivos ignorados (nivel 1: la herramienta lo impide). El hook `check-added-large-files`
  detiene un archivo de más de 1024 KB añadido fuera de `data/` (nivel 1, mientras los
  hooks estén instalados).
- **Aprendizaje:** el `.gitignore` se escribe antes de que cualquier dato entre al árbol
  de trabajo. Una salida roja de `check-ignore` necesita su control con `--no-index` antes
  de tocar la regla: sin él, "regla mal escrita" y "archivo ya rastreado" se leen igual.
- **Referencias:** `366df40` (ya no alcanzable desde ninguna rama), `0387288`.

### ERR-002 · Un dato derivado presentado como medido sostuvo un requisito · 2026-10-18

- **Síntoma:** dos afirmaciones sobre el estado del repositorio se reportaron como hechos
  medidos sin serlo, y una de ellas llegó a un prompt como criterio de aceptación.
  - **Instancia 1, del Ejecutor:** el reporte afirmó que `ruff format` revisaba "4 `.py` +
    `pyproject.toml` + 6 `.md`". En realidad eran 4 `.py` y 7 `.md`; `ruff format` nunca
    revisa `pyproject.toml`. El dato equivocado entró en el prompt siguiente como criterio
    de aceptación: "debe reportar solo archivos Python y pyproject.toml".
  - **Instancia 2, del Arquitecto:** el prompt siguiente dijo que la medición del Ejecutor
    "confirmó" que la copia en disco del CSV tiene finales CRLF. No lo confirmó: la
    diferencia de exactamente un byte por línea entre la copia en disco y el blob es
    evidencia fuerte pero indirecta, y el archivo nunca se abrió.
- **Mecanismo:** un dato derivado se presentó como medido y después sostuvo un requisito.
  En la instancia 1, la lista de archivos de `ruff format` se dedujo de
  `ruff check --show-files`, que incluye `pyproject.toml` por las reglas RUF, en vez de
  contar la salida del propio `format`. En la instancia 2, una inferencia que el reporte
  marcaba como tal se citó como medición.
- **Cómo se detectó:** la instancia 1, al contar las líneas `format_path` de
  `ruff format --check --verbose`. La instancia 2, al contrastar el prompt con el reporte
  original, que decía "inferencia, no medido".
- **Corrección:** las dos se corrigieron como premisas en el turno siguiente, y el
  Arquitecto las aceptó.
- **Defensa:** nivel 3 de la jerarquía de `docs/METHODOLOGY.md`, sección 6.5: está escrito
  en este registro. Ninguna herramienta ni test lo detecta.
- **Aprendizaje:** un dato que se derivó de otra salida se reporta como derivado,
  nombrando de dónde salió. Y el Arquitecto no asciende una inferencia del Ejecutor a
  medición al citarla.
- **Referencias:** ERR-001, por la medición de tamaños del CSV que originó la inferencia
  de la instancia 2.
- **Nota de estado (2026-10-18):** la instancia 2 quedó medida en la fase 1 con
  `scripts/prepare_data.py`. El CSV en disco tiene 6.362.621 bytes `\n`, los 6.362.621
  precedidos por `\r`, y ningún `\r` suelto. La inferencia era correcta, y sigue siendo
  cierto que no estaba medida cuando se citó.

### ERR-003 · La resolución elige en silencio un paquete sin rueda para nuestra plataforma · 2026-10-18

- **Síntoma:** ninguno visible, y eso es lo que importa. La resolución universal de la pila
  completa salió en verde y eligió numba 0.53.1 y llvmlite 0.36.0 para todo Python <3.12.
  Ninguna de las dos tiene rueda cp311-win_amd64.
- **Mecanismo:** tres piezas, y ninguna avisa.
  - uv ignora las cotas superiores de `requires_python` de las dependencias. Por eso numba
    0.53.1, que declara `>=3.6,<3.10`, es elegible en 3.11.
  - El pin `pandas<3` inclina la resolución universal hacia esa versión. El síntoma está
    medido (V0 contra V1); el porqué no.
  - Todo sale en verde: ninguna herramienta compara lo resuelto con las ruedas que existen
    para la plataforma.
- **Cómo se detectó:** **no falló**. La resolución en seco que cerraba otro hallazgo (las
  cotas de pyarrow y mlflow) mostró líneas de numba distintas de las de la primera
  resolución. El verificador de ruedas, que se había construido para medir el pin de Python,
  confirmó que esas versiones no tienen rueda. Ninguna de las dos mediciones buscaba esto.
- **Corrección:** `[tool.uv] constraint-dependencies = ["numba>=0.57"]`. Con la constraint,
  la resolución elige numba 0.68.0 y llvmlite 0.50.0; sin ella (control), 0.53.1 y 0.36.0.
- **Defensa:** para numba, nivel 1 de `docs/METHODOLOGY.md`, sección 6.5: la constraint lo
  impide. Para la clase general, cualquier transitiva sin rueda, nivel 3: ningún test compara
  el lock con las ruedas disponibles.
- **Aprendizaje:** una resolución en verde no prueba que lo resuelto se pueda instalar en
  nuestra plataforma. Eso se mide aparte.
- **Referencias:** ADR 0003, ADR 0004, ADR 0005.

### ERR-004 · Un prompt del Arquitecto llegó truncado a mitad de frase · 2026-10-18

- **Síntoma:** el prompt terminaba en "…así que fijar el intérprete crea pisos", sin cerrar
  la frase y sin las secciones que cierran un prompt.
- **Mecanismo:** una instrucción incompleta se lee igual que una completa, salvo en el punto
  exacto del corte.
- **Defensa que funcionó:** no completar la frase e informar el corte antes de cualquier
  otra cosa. Es la regla de autoría de los ADR (`docs/METHODOLOGY.md`, sección 3) aplicada
  a la instrucción misma.
- **Corrección:** el Arquitecto envió la continuación, con el texto exacto de la frase.
- **Referencias:** ADR 0003, por la frase que llegó cortada.

### ERR-005 · pandas tipa en silencio: enteros que desbordan y categorías que se vuelven nulos · 2026-10-18

- **Síntoma:** ninguno visible, y por eso cuenta. Medido con pandas 2.3.3 antes de escribir
  el loader: `pd.read_csv(..., dtype={"a": "int8"})` leyó `300` como `44`, y
  `dtype={"c": "int16"}` leyó `40000` como `-25536`, sin error ni advertencia, con el motor
  C y con el motor pyarrow. Con `dtype=CategoricalDtype(categories=[...])`, un valor fuera
  de las categorías quedó como `NaN`, también sin aviso.
- **Mecanismo:** el valor resultante es el original módulo 2^n (300 − 256 = 44;
  40000 − 65536 = −25536): una truncación en complemento a dos, sin verificación de rango.
  En qué punto de pandas ocurre no quedó determinado. En la categórica, todo valor ajeno a
  las categorías fijadas se convierte en nulo: es una imputación silenciosa
  (`docs/METHODOLOGY.md`, sección 7.1).
- **Cómo se detectó:** un sondeo previo al código, que buscaba específicamente las
  conversiones capaces de corromper un dato sin fallar.
- **Corrección:** el CSV se parsea con `pyarrow.csv` y un tipo explícito por columna, que
  rechaza `300` en `int8` con un `ArrowInvalid` que nombra la columna. `type` se lee con
  categorías abiertas, y las declaradas se aplican después de validar el dominio, con una
  guarda que se niega a crear nulos.
- **Defensa:** nivel 2 de `docs/METHODOLOGY.md`, sección 6.5: dos tests de
  `tests/test_loader.py`, uno para el entero fuera de rango y otro para el tipo fuera de
  dominio.
- **Aprendizaje:** un dtype explícito no es un dtype verificado. Un entero angosto afirma un
  rango, y alguien tiene que comprobarlo.
- **Referencias:** `src/fraud_detection/data/loader.py`, en el docstring del módulo.

### ERR-006 · `astype` a una categórica no ordenada no reordena, y el síntoma se atribuyó al parquet · 2026-10-18

- **Síntoma:** el test de dtypes del parquet falló: las categorías de `type` volvían en
  orden de aparición (`PAYMENT`, `TRANSFER`, …) y no en el declarado (`CASH_IN`,
  `CASH_OUT`, …). La comparación de dtypes con `==` había pasado.
- **Mecanismo:** para pandas, dos categóricas no ordenadas con las mismas categorías son
  iguales aunque el orden difiera, y `Series.astype(dtype)` no hace nada cuando el dtype
  destino es igual al actual. Medido con pandas 2.3.3: `astype` deja
  `['TRANSFER', 'PAYMENT', 'CASH_IN']` intacto, y `cat.set_categories` lo lleva a
  `['CASH_IN', 'PAYMENT', 'TRANSFER']` conservando los valores.
- **Error de diagnóstico previo:** en el sondeo inicial, el mismo síntoma se atribuyó a la
  ida y vuelta por parquet, y así se informó durante el turno. Era falso: el sondeo hacía
  un `astype` antes de escribir, y ese `astype` nunca había reordenado. Un segundo sondeo,
  que construía la categórica desde cero, mostró que el parquet sí conserva el orden; la
  contradicción entre los dos no se investigó en el momento.
- **Cómo se detectó:** el test, que compara el orden de las categorías por separado
  precisamente porque `==` no lo compara.
- **Corrección:** `apply_declared_dtypes` recodifica con `cat.set_categories`.
- **Defensa:** nivel 2: la aserción de orden en el test de dtypes de `tests/test_loader.py`.
- **Aprendizaje:** dos sondeos que se contradicen son un hallazgo, no ruido. Y la igualdad
  de dos dtypes no prueba que sean el mismo dtype.
- **Referencias:** ERR-005, por la guarda contra nulos que comparte la misma función.

### ERR-007 · Una dependencia opcional de pandas falla solo al llamarse · 2026-10-18

- **Síntoma:** `scripts/prepare_data.py` completó trece mediciones y falló en la catorce con
  `ModuleNotFoundError: No module named 'scipy'`, desde
  `DataFrame.corrwith(..., method="spearman")`.
- **Mecanismo:** pandas importa scipy dentro del método, en el momento de la llamada, no al
  importar pandas. scipy no es dependencia del proyecto. ruff y mypy no pueden verlo, y
  ningún test ejecuta las funciones de medición.
- **Cómo se detectó:** la corrida completa del script.
- **Corrección:** Spearman se calcula por su definición, como Pearson sobre rangos
  promedio, sin scipy. Se verificó contra un caso con empates calculado a mano (3/√18).
- **Defensa:** ninguna automática: nivel 3. Las funciones de medición no tienen tests en
  esta fase, y es la defensa que falta (`docs/METHODOLOGY.md`, sección 7.6). Además, cuando
  entre scikit-learn en la fase 2, scipy llegará como dependencia transitiva y el síntoma
  desaparecerá sin que nadie lo haya decidido.
- **Aprendizaje:** que pandas importe no garantiza que todos sus métodos corran.
- **Referencias:** `src/fraud_detection/data/profiling.py`, en el docstring de la función
  de correlación.
- **Nota de estado (2026-10-18):** la defensa subió a nivel 2 en la misma fase.
  `tests/test_profiling.py` tiene un test por función de medición. El de correlación
  verifica Spearman con empates contra un valor calculado a mano, que se distingue del de
  Pearson, así que volver a `method="spearman"` falla en CI mientras scipy no esté en el
  grafo.
- **Nota de estado (2026-10-18), segunda:** scipy entró al grafo en la fase 1, no en la 2:
  llegó con scikit-learn (scipy 1.17.1 en el lock). Desde entonces, volver a
  `method="spearman"` ya no falla: calcula el mismo valor y el test pasa. El test sigue
  protegiendo la cifra, pero ya no detecta la dependencia oculta, y la nota anterior dejó
  de ser cierta en esa parte.

### ERR-008 · Una lista de inclusión vacía significa "todas las columnas" · 2026-10-18

- **Síntoma:** ninguno visible. Medido antes de corregir: `read_raw_csv` con una lista
  vacía de columnas, sobre un CSV con encabezado `foo,bar`, devolvió las dos columnas, con
  dtypes inferidos `int64` y `string`.
- **Mecanismo:** en `pyarrow.csv.ConvertOptions`, un `include_columns` vacío significa
  "todas las columnas", no "ninguna". El validador lee las columnas declaradas que aparecen
  en el encabezado. Si no aparece ninguna, la lista vacía invierte el significado, y el
  lector infiere tipos sobre columnas que nadie declaró: lo que el loader dice no hacer
  nunca.
- **Cómo se detectó:** al escribir los tests del validador, revisando el camino en que no
  hay nada que leer. Se confirmó con la medición antes de corregir.
- **Corrección:** el validador no lee nada cuando ninguna columna declarada está presente,
  y los chequeos que dependen de la lectura salen omitidos.
- **Defensa:** nivel 2: el test del validador que exige que un chequeo que no puede
  ejecutarse salga omitido cubre un archivo sin columnas declaradas. `loader.read_raw_csv`
  conserva el comportamiento con una lista vacía; quedó fuera del alcance del turno y se
  reportó.
- **Aprendizaje:** el caso vacío de un parámetro se prueba; "vacío" no siempre quiere
  decir "nada".
- **Referencias:** `src/fraud_detection/data/validator.py`, en la lectura con dtypes
  declarados.
- **Nota de estado (2026-10-18):** corregido también del lado del loader, en la misma
  fase. `loader.read_raw_csv` levanta `ValueError` ante una lista vacía, con test en
  `tests/test_loader.py`. Una lista vacía que significa "todas" reabría ERR-005 por el
  costado: columnas no declaradas con tipos inferidos.
- **Nota de estado (2026-10-18), segunda instancia:** `pre-commit run --files` con una lista
  vacía cae a su modo por defecto: guarda los cambios no preparados en un stash y los
  restaura. Ocurrió al pasarle `$(git ls-files --others)` cuando todos los archivos ya
  estaban preparados. Es el mismo mecanismo que `include_columns=[]`: una colección vacía
  significa "nada" para quien llama y "todo", o "el default", para la herramienta. La
  defensa es pasar la lista explícita y verificar que no esté vacía antes de invocar.

### ERR-009 · Un arnés de mutación leyó "no corrió ningún test" como "el mutante sobrevivió" · 2026-10-18

- **Síntoma:** el arnés de mutación del Ejecutor (ad hoc, no versionado) reportó 0 de 12
  mutantes detectados sobre el código de features, baseline y loader. Todos "sobrevivían".
- **Mecanismo:** dos piezas, y ninguna avisa.
  - La plantilla del plugin de pytest era una cadena no cruda con `"\r\n"` adentro. Al
    formatearse, el plugin generado quedó con un salto de línea real dentro de una cadena:
    un `SyntaxError`. pytest falló al cargar el plugin y no ejecutó ningún test.
  - El arnés decidía "detectado" buscando líneas `FAILED` en la salida. Sin tests
    ejecutados no hay líneas `FAILED`, y la ausencia se leyó como supervivencia.
- **Instancia relacionada, del mismo turno:** al generar un archivo desde un comando de
  shell, un `\\n` llegó como `\n` y rompió `scripts/prepare_data.py` con saltos de línea
  dentro de cadenas. Ruff lo detectó antes de cualquier corrida.
- **Cómo se detectó:** 0 de 12 era inverosímil frente a una suite que en el turno anterior
  detectó 17 de 17. Se leyó la plantilla en bytes antes de tocar ningún test.
- **Corrección:** el arnés se reescribió con plantilla cruda. Ahora corre primero un
  control sin mutación, que tiene que ejecutar los tests y pasarlos, y trata como "error
  del arnés" toda corrida en la que pytest no ejecutó tests. Con eso: 12 de 12 detectados.
- **Defensa:** nivel 3: está escrito aquí. El arnés no está versionado.
- **Aprendizaje:** un instrumento que puede fallar sin producir la señal que busca convierte
  su propio fallo en un resultado. Es `docs/METHODOLOGY.md`, sección 7.6, otra vez: la
  ausencia de evidencia de fallo se lee igual que la evidencia de éxito, salvo que haya un
  control que pruebe que el instrumento corrió.
- **Referencias:** ERR-007, por la otra vez que un fallo se manifestó solo al ejecutar.

### ERR-010 · El ejecutor de notebooks corrompió los acentos y todas las verificaciones dieron verde · 2026-10-18

- **Síntoma:** `uv run jupyter execute --inplace notebooks/*.ipynb` salió con código 0, con
  contadores consecutivos, cero errores, cero salidas a stderr y cero rutas absolutas. Pero
  las figuras decían "LegÃtima" y "AutocorrelaciÃ³n": el archivo guardado tenía 214 marcas
  de mojibake y ninguna palabra acentuada correcta.
- **Mecanismo:** `nbclient/cli.py` (0.11.0) abre el notebook con `input_path.open()`, sin
  `encoding`. Sin el modo UTF-8, Python en Windows usa la codificación local, que en esta
  máquina es cp1252. Los bytes UTF-8 se decodifican como cp1252, el kernel ejecuta las
  cadenas corruptas, y `nbformat.write` guarda la corrupción en UTF-8. Ninguna celda falla:
  una cadena mal decodificada sigue siendo una cadena válida.
- **Cómo se detectó:** mirando las figuras; ninguna verificación automática lo vio. Los
  controles del ejecutor (un notebook que debe pasar y otro que debe fallar) eran solo ASCII.
- **Corrección:** la tarea `notebooks` del Taskfile fija `PYTHONUTF8=1`. Un tercer control,
  un notebook que afirma el valor de una cadena acentuada, falla sin el modo UTF-8
  (`AssertionError: 'Leg\xc3\xadtima'`) y pasa con él, y el archivo guardado conserva el
  carácter.
- **Defensa:** nivel 1 para la tarea, porque su entorno fija la variable. Nivel 3 para quien
  ejecute `jupyter execute` a mano sin ella: está escrito aquí y en el Taskfile, y ningún
  test lo detecta.
- **Aprendizaje:** los controles de un arnés tienen que cubrir la clase de dato que el arnés
  procesa; un control solo ASCII no prueba nada sobre un texto en español. Es un sexto caso
  del verde en falso de `docs/METHODOLOGY.md`, sección 7.6.
- **Referencias:** `Taskfile.yml`, tarea `notebooks`; `docs/DEPENDENCIES.md`, fila de
  `nbclient`; ERR-009.
