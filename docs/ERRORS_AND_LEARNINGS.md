# Errores y aprendizajes

> Contrato: una entrada por error real, que documenta el **mecanismo** —por qué ocurrió—
> y no solo el síntoma.

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
