# ADR 0001 — Configuración y reproducibilidad

- **Estado:** aceptado
- **Fecha:** 2026-10-18
- **Origen:** decisión discutida con el Arquitecto; el Ejecutor la transcribe.

---

## Contexto

Todo módulo del proyecto toma la semilla, las rutas y el corte temporal del módulo
`src/fraud_detection/config.py`, a través de la instancia `settings`.

## Decisión

- Las constantes del proyecto —`RANDOM_STATE`, las rutas y el corte temporal— son
  inmutables y **no se pueden sobrescribir desde el entorno**.
- Las rutas se resuelven desde la ubicación del módulo, no desde el directorio de trabajo.
- El primer step de producción se deriva del último de referencia; no se escribe como
  literal.

## Por qué

La reproducibilidad exacta desde cero es un requisito que se verifica, no se supone.

- Una semilla que una variable de entorno puede mover convierte "no reproduce" en un
  problema sin diagnóstico, porque el código es idéntico en las dos corridas.
- Una ruta resuelta desde el directorio de trabajo hace que el mismo script lea un `.env`
  distinto según desde dónde se lo invoque, y en Windows eso pasa sin aviso.
- Dos literales para un corte temporal pueden desincronizarse y dejar un rango de steps
  que ningún periodo evalúa.

## Alternativas descartadas

### `Final` en lugar de `ClassVar`

Con pydantic 2.14.1, un atributo `Final` con valor por defecto se trata hoy como variable
de clase, pero la definición emite `PydanticDeprecatedSince211`: en V3 pasará a ser un
**campo normal**, es decir, sobrescribible desde el entorno. El propio aviso recomienda
anotarlo como `ClassVar`.

### Campos normales para las rutas

Medido con `RAW_DATA_DIR=C:\elsewhere` definida en el entorno:

| Declaración de la ruta | Valor resultante |
|---|---|
| Campo normal | `C:\elsewhere`: el entorno la redirige |
| `ClassVar` (la elegida) | `data\raw`: sin cambio |

Con el `Settings` real del proyecto y la misma variable definida, ejecutando desde otra
carpeta, la ruta siguió anclada a la raíz del repositorio.

Las mediciones de esta sección se hicieron con scripts ad hoc que no están versionados.

## Hueco que queda

Sobre la instancia, `settings.RANDOM_STATE = x` falla con `AttributeError`. Sobre la
clase, `Settings.RANDOM_STATE = x` **no lo detecta ni Python ni mypy**.

El test de humo lo atrapa solo si la asignación ocurre al importar el paquete; una
asignación en tiempo de ejecución dentro de un notebook no la atrapa nada.

Es una garantía de nivel 2 —un test lo detecta, según la jerarquía de
`docs/METHODOLOGY.md`, sección 6.5— con un borde descubierto.
