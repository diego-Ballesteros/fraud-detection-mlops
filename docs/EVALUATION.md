# Registro de evaluación

> Contrato: qué se midió, cómo, con qué baseline al lado y con qué resultado. Una métrica
> sin su baseline, su número de particiones y su dispersión no se registra.

---

## Formato de entrada

Cada entrada se agrega al final de la sección **Entradas**, con un identificador
correlativo. Las entradas no se reescriben: si un resultado deja de ser válido, se le
agrega una nota de estado fechada.

```markdown
### EVAL-NNN · <qué se midió> · AAAA-MM-DD

| Campo | Valor |
|---|---|
| **Pregunta** | Qué se quería saber |
| **Periodo evaluado** (obligatorio) | Referencia (`step <= 400`) o producción (`step >= 401`), con el rango de steps exacto |
| **Tamaño de muestra** (obligatorio) | Filas evaluadas y cuántas son positivas (prevalencia) |
| **Método** | Esquema de validación, modelo y configuración |
| **Métrica** | Nombre y por qué es la adecuada para esta pregunta |
| **Resultado** | Valor ± dispersión |
| **Baseline** (obligatorio) | Contra qué se compara y su valor ± dispersión, sobre las mismas particiones |
| **Número de particiones** (obligatorio) | Sobre cuántas particiones o ventanas se calculó |
| **Dispersión** (obligatorio) | Qué estadístico (desviación estándar, intervalo) y cómo se calculó |
| **Run de MLflow** | ID o enlace del run |
| **Reproducción** | Comando exacto que regenera la cifra |
| **Interpretación** | Qué significa, y qué **no** permite concluir |
```

---

## Entradas
