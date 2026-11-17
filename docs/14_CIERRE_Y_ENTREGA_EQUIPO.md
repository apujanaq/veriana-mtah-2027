# veriana: cierre de refinamientos y entrega revisable

> **Decisión final del equipo (17-nov):** la entrega es `juliana_control_pares`. Cumple la regla fijada antes de
> medirla sobre la base final (mejora con ruido en los dos partidos externos, sin cambios en limpio, reparto y número
> de posesiones plausibles) y su prueba C3 completa es robusta (mínimo 0,921). El criterio de precisión de
> micro-posesiones de abajo no se cumple: medía la inserción de micro-posesiones, y este refinamiento sobre todo
> retira retornos breves no verificados. El texto siguiente se conserva tal como se escribió antes de esa decisión.

Se conserva **alejo_v2_robusta como referencia conservadora**. La mejor alternativa externa es
**juliana_control_pares**, que combina control H/A multiventana y verificación de retornos breves.
Mejora spellF1 con ruido en ambos samples sin cambiar las etiquetas en limpio. Sigue sin cumplir
la precisión mínima de microposesiones del empalme: se entrega como **experimental**, separada.

## Comparación externa reproducible

Todos los números corresponden a Metrica Sample 1 y 2 públicos. Ruido: AR(1), desviación 0,8 m,
memoria 0,4 s, semilla igual al número del sample. Referencia densa por eventos, tagging emulado.
Metrica ya es desarrollo conocido del equipo, no test ciego. No es una puntuación del concurso.

| Aproximación | spellF1 S1 limpio | S1 ruido | S2 limpio | S2 ruido |
| --- | ---: | ---: | ---: | ---: |
| Referencia v2 | 0,7263 | 0,4765 | 0,6851 | 0,4929 |
| Toques breves v1 | 0,7263 | 0,4765 | 0,6851 | 0,4929 |
| Parejas verificadas | 0,7263 | 0,4878 | 0,6851 | 0,5114 |
| Control multiventana | 0,7263 | 0,4953 | 0,6851 | 0,5018 |
| Control + parejas | 0,7263 | **0,5114** | 0,6851 | **0,5284** |

La combinación gana **3,50 y 3,55 puntos porcentuales** de spellF1 con ruido. MacroF1 pasa de
0,7550 a 0,7636 en S1 y de 0,7992 a 0,8093 en S2. La precisión de microposesiones pasa de
0,1000 a 0,1707 y de 0,1026 a 0,1750: mejora, pero queda lejos del 0,5 exigido. La definición
heredada cuenta todos los spells de menos de 2 s, incluidos los D cortos; no se cambia para
facilitar la aceptación. El criterio de +0,02 de spellF1 se cumple; el de precisión, no.

El criterio no se rebaja tras ver los resultados. La decisión automática conserva v2. Se puede
generar la alternativa solo con la opción explícita de candidata experimental, cuyo documento
y recibo identifican ese estado. No se sustituye el modelo global v4 ni los paquetes anteriores.

## Qué se implementó

- Variables de 0,4/0,8/1 s, dos vecinos por equipo, velocidades relativas y reacción colectiva.
- Clasificador de toques breves con referencias débiles DFL; calibración de parejas completas.
- Verificación de cambios H→A→H/A→H→A breves en tracking ruidoso. No cruza ausencias o periodos.
- Control denso aprendido con muestras simétricas H/A, 200 stumps, temperatura elegida en DFL dev.
- Viterbi binario condicionado a regiones vivas de la base; la D heredada queda bloqueada.
- Composición de modelos congelados, sin reajustar umbrales con Metrica.
- Cachés en memoria ligadas al contenido del modelo y guardas para la reconstrucción de C3.
- Generador de entrega con autorización explícita, destinos nuevos, validador oficial y relectura.
- Publicación por lista positiva sobre base oficial, sin incorporar historia o diagnósticos privados.

Se respetó el reparto: no se modificó src/mtah/eventos.py ni sus parámetros de D. Las nuevas
aproximaciones delegan las probabilidades base a FusionV2 sin pasar sus parámetros propios al
registro global. El control v4 sí utiliza geometría del balón; la pausa también usa movimiento
colectivo. No se activó ningún detector visual ni homografía adicional para corregir el balón.

## Ajuste permitido y límites

Entrenamiento nuevo: DFL J03WMX/J03WN1/J03WOH. Desarrollo: J03WOY. Se verificaron los manifiestos
publicados de esas particiones; los antiguos tests no se abrieron para ajustar o seleccionar.
BallStatus/BallPossession y eventos externos no son variables de los clasificadores.

El detector v1 localiza cambios cortos de posesión, no contactos físicos anotados. El primer
umbral de precisión de eventos resultó demasiado conservador. La segunda fase calibra el efecto
sobre spells en DFL dev. DFL carece de Smart Tagging: se usa v4 autónoma sin eventos, sin inventar
ventanas desde las etiquetas. La calibración temporal del control es condicional a D conocida
solo para evaluar H/A en desarrollo; en Metrica y en inferencia D se predice y queda bloqueada.
Ese cambio de dominio limita la evidencia, y se declara en vez de tratarlo como test independiente.

La composición es una ablación de dos refinamientos congelados. Todos los resultados se publican,
incluidos los que no pasan los criterios. No se usaron etiquetas del partido del concurso, sus
cinco minutos, anotaciones manuales ni diagnósticos reales para seleccionar modelos.

## Robustez y comprobaciones

Las pruebas sintéticas/regresión cubren simetría de equipos, persistencia temporal, ausencias,
periodos, D/X bloqueadas, entradas inmutables, reconstrucción con parámetros perturbados,
selección conservadora, autorización antes de leer y saneamiento de textos al exportar.

La **C3 parcial externa** perturba ±20% únicamente los parámetros nuevos H/A de la composición,
en ambos samples y condiciones. El mínimo spellF1 frente a su versión base es **0,944**, superior
al umbral 0,85. Las D quedan iguales. Esto **no es la C3 completa del bloque D ni una evaluación
del partido restringido**. La reproducción desde la rama saneada se comprueba por separado.

## Artefactos y reproducción

[Resultados públicos y selección](revision_pares_juliana_01/) contienen las comparaciones completas,
calibraciones DFL y robustez. Modelos congelados: modelos/toques_micro_dfl_v1.json,
modelo_pares_micro.json y control_multiventana.json. La referencia v4 sigue en modelos/modelo.json.

```powershell
python -m unittest discover -s tests -v
python scripts/validar_microposesiones.py --aprox juliana_control_pares --data-dir data --json outputs/ensayo_nuevo.json
python scripts/robustez_refinamiento.py --cache-base outputs/cache_externo_nuevo --json outputs/robustez_nueva.json
```

Para crear CSV locales se usa scripts/preparar_entrega_equipo.py, documentado en el README.
Cada salida tiene veriana.csv y veriana.md. Registros y recibos permanecen privados. La validación
de formato y reproducción no demuestra exactitud de las etiquetas. No hay una puntuación oficial.

## Publicación segura

La rama para el equipo parte de f8f2629, anterior al trabajo privado del equipo. No incluye sus
commits históricos, docs/10, docs/12, notebooks personales, CSV nuevos, vídeos, datos ni registros
reales. Se copian código, modelos de otros partidos y reportes externos por lista positiva.
Los textos heredados se sustituyen por documentación genérica; el AST verifica las operaciones
exportadas, salvo documentación y metadatos descriptivos de origen. La reproducción funcional
y las pruebas se ejecutan otra vez en esa rama. No se reescribe ninguna rama existente.
