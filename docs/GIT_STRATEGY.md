# Estrategia de Git

> Contrato: cómo se nombran las ramas, cómo se escriben los commits, cómo se integra el
> trabajo y cómo se publica una versión. En este proyecto el historial es evidencia
> evaluada: la rúbrica lo mira como prueba del proceso.

---

## 1. Modelo

**GitHub Flow con una rama de integración persistente, `develop`.** Cada fase del roadmap
vive en su propia rama `feature/*`, que sale de `develop` y vuelve a `develop` vía pull
request. `main` solo recibe a `develop`, también vía pull request, cuando hay una versión
que publicar.

```
main     ●─────────────────────────────────────────────────────●  ← tag v1.0.0
          \                                                   /
develop    ●────────────●────────────────●───── ··· ─────────●
            \          /  \             /
             ●───●────●    ●───●───●───●
           feature/00-     feature/01-
           foundation      data-and-eda
```

Cada `●` de `develop` donde termina una rama es un **merge commit**: conserva los commits
de la fase y deja registrado el punto de integración.

---

## 2. Rol de cada rama

| Rama | Rol | Quién escribe en ella |
|---|---|---|
| `main` | Rama de release. **Protegida.** Solo recibe merges desde `develop` vía PR. Cada merge a `main` corresponde a una versión con tag | Nadie hace push directo |
| `develop` | Rama de integración. Recibe las ramas de fase vía PR. Es la base de la que sale cada `feature/*` | Solo merges de PR |
| `feature/NN-*` | Una rama por fase. Aquí se commitea el trabajo del día a día | El Verificador |

Los commits y los merges los hace únicamente el Verificador; el Ejecutor nunca commitea
(ver `docs/METHODOLOGY.md`, sección 1).

### Protección de `main`

Configurada en GitHub, no confiada a la memoria:

- Exigir pull request antes de mergear; sin push directo.
- Exigir el check de CI **Lint, types and tests** en verde.

El workflow de CI se dispara en `pull_request` **sin filtro de rama** a propósito: un check
requerido solo aparece en la PR si corrió contra el commit de la PR, y un filtro de rama mal
escrito descarta el evento en silencio, sin ningún error visible.

---

## 3. Nombres de rama

Formato: `feature/NN-nombre-en-ingles`

- `NN` es el número de fase del roadmap, con dos dígitos.
- El nombre va en inglés, en minúsculas y separado por guiones.

Nombres canónicos, tomados de las fases de `docs/ROADMAP.md`:

| Fase | Rama |
|---|---|
| 0 — Fundación | `feature/00-foundation` |
| 1 — Datos y EDA | `feature/01-data-and-eda` |
| 2 — Modelado y AutoML | `feature/02-training` |
| 3 — Interpretabilidad | `feature/03-interpretability` |
| 4 — Monitoreo | `feature/04-monitoring` |
| 5 — Cierre | `feature/05-closing` |

---

## 4. Commits: Conventional Commits

Formato: `<prefijo>: <descripción>`

- La descripción va en inglés, en modo imperativo, en minúscula y sin punto final.
- Un commit es un cambio lógico. Si el mensaje necesita un "and" para describir dos cosas
  sin relación, son dos commits.
- El cuerpo, cuando hace falta, explica **por qué**; el diff ya muestra qué.

| Prefijo | Cuándo | Ejemplo |
|---|---|---|
| `feat:` | Funcionalidad nueva | `feat: add temporal split for reference and production periods` |
| `fix:` | Corrección de un defecto | `fix: fit the scaler on reference rows only` |
| `docs:` | Solo documentación | `docs: add git strategy` |
| `test:` | Tests nuevos o corregidos, sin cambiar código de producción | `test: add positive controls for the PSI detector` |
| `refactor:` | Cambio de estructura sin cambiar comportamiento | `refactor: move threshold selection into the evaluation module` |
| `chore:` | Configuración, dependencias, CI, herramientas | `chore: add CI workflow with lint, types and tests` |

---

## 5. Pull requests

- **Una PR por fase**, de `feature/NN-*` hacia `develop`. Al cerrar el proyecto, una PR de
  `develop` hacia `main`.
- **CI en verde** antes de mergear.
- La descripción dice qué cambia, por qué, y con qué evidencia: la salida real de los
  comandos de verificación, no una afirmación.
- **No se mergea nada que el Verificador no pueda explicar en voz alta** (la compuerta de
  aprendizaje de `docs/METHODOLOGY.md`, sección 1).

### Merge commit, nunca squash

Se mergea con **merge commit**. No se usa squash ni rebase merge.

**Por qué:** la evaluación mira el historial como evidencia del proceso. Un squash aplasta
todos los commits de una fase en uno solo y borra precisamente esa evidencia; un rebase
merge reescribe los hashes y pierde el punto de integración.

Para que la regla no dependa de recordarla, en la configuración del repositorio en GitHub
se deja habilitado únicamente **"Allow merge commits"**.

---

## 6. Releases

- Las versiones siguen **SemVer**: `vMAJOR.MINOR.PATCH`.
- El tag es **anotado** y se crea sobre `main`, justo después de mergear la PR de
  `develop`.
- La versión de `pyproject.toml` coincide con el tag.
- Cada tag lleva un **GitHub Release** con sus release notes.
- `v1.0.0` es la entrega final del proyecto (fase 5 del roadmap).

---

## 7. Nota de equivalencia: `develop`

La rama de integración se llama `develop`, siguiendo la convención de git-flow. **Si el
enunciado menciona una rama "Development", es esta misma rama.**

---

## 8. Qué nunca entra al repositorio

Datos, `.env`, `.venv/`, `mlruns/`, `mlflow.db` y artefactos pesados. La defensa está en
dos capas: `.gitignore` y el hook `check-added-large-files` de pre-commit, que detiene un
archivo grande fuera de las carpetas ignoradas.

Un archivo commiteado por error **no se arregla con un commit posterior**: el commit que
lo borra deja el archivo intacto en el historial. Hay que corregir el historial antes del
push.

---

## 9. Convención de idioma

| Elemento | Idioma |
|---|---|
| Identificadores, funciones, clases, variables | Inglés |
| Docstrings y comentarios en código | Inglés |
| Nombres de archivos, carpetas, ramas, commits | Inglés |
| Todos los documentos markdown y los notebooks | Español |

Los nombres de columnas del dataset se conservan como los entrega la fuente; lo que va en
español es su descripción.
