# Empalme de Juliana: detector supervisado de posesiones breves

Se integró `origin/alejo/v1-fusion` hasta `6394d72`, conservando la rama y los artefactos anteriores.
El trabajo está en `jdmm/empalme-microposesiones`. Se respeta el reparto: Alejandro mantiene
`src/mtah/eventos.py`; Juliana desarrolla el canal H/A y la inserción posterior al HMM.

## Método y límites

La aproximación registrada `juliana_v2_micro` conserva las probabilidades y el HMM de
`alejo_v2_robusta`. Después busca un toque confiado del rival y una recuperación del dueño
en un intervalo breve. La inserción conserva todos los frames D y no cruza periodos,
ausencias de soporte ni tramos no evaluados. Dos toques simultáneos se consideran ambiguos.
La nueva clase conserva su detector cuando C3 la reconstruye con parámetros perturbados.

El clasificador usa 60 variables simétricas: los dos vecinos de cada equipo, ventaja de
distancia, apoyos, quiebres de la trayectoria, velocidades relativas al mismo slot y reacción
colectiva. Las ventanas de 0,4, 0,8 y 1 s reducen la dependencia de una diferencia entre frames
consecutivos. Una mediana corta del balón se calcula dentro de cada bloque observable;
no interpola posiciones ausentes. El modelo es boosting logístico de stumps, NumPy/pandas,
serializado en JSON; no requiere APIs ni servicios externos.

Los objetivos son cambios cortos H→A→H o A→H→A de BallPossession en DFL/IDSSE y su recuperación.
Son **referencias débiles de posesión**, no contactos físicos anotados. El score aprendido corresponde
a una pequeña ventana alrededor de esos cambios; su umbral se calibra en desarrollo. No debe
interpretarse como una probabilidad perfectamente calibrada de contacto en un frame exacto.
La supresión de máximos evita contar varias veces una misma respuesta temporal.

Una precisión sobre la documentación anterior: **el control v4 sí depende de la geometría del
balón** (`matriz_control_v4` → `matriz_v3` → `matriz_control`: distancias, velocidad relativa,
continuidad e intercepción). El movimiento colectivo sin balón corresponde al detector de pausa.
Por tanto, la fusión v2 también debe evaluarse con ruido en el balón.

## Protocolo congelado

Las condiciones están en [`config/micro_dfl_v1.json`](../config/micro_dfl_v1.json).
Entrenamiento: J03WMX/J03WN1/J03WOH. Desarrollo: J03WOY. No se abren los antiguos tests
J03WPY/J03WQQ/J03WR9. Se verifican tamaño y MD5 publicados de los archivos de train/dev.
Se usa tracking nativo a 25 Hz, sin introducir BallStatus, BallPossession ni eventos como variables.

El aumento AR(1) usa desviaciones de 0, 0,4, 0,8 y 1 m y memoria de 0,4 s. Se reinicia por periodo
y conserva las ausencias. Se conservan los positivos observables; los negativos submuestreados
reciben peso inverso a su probabilidad de muestreo. El umbral se elige exclusivamente en DFL dev.
No se usan los datos del concurso, sus cinco minutos ni sus predicciones para ajustar o elegir.

Después se mide una sola candidata congelada en Metrica Sample 1 y 2, limpio y con ruido de 0,8 m.
La referencia densa viene de eventos externos; el tagging está emulado, lo que es optimista para D.
Metrica ya es un banco de desarrollo del equipo: **no es test ciego ni puntuación del concurso**.
No se vuelve a calibrar al ver sus resultados. La aceptación requiere, en ambos partidos,
ganar al menos 0,02 de spellF1 con ruido, perder como máximo 0,01 en limpio y obtener precisión
de micro-posesiones ≥0,5 en cada escenario. Si falla, se conserva `alejo_v2_robusta`.

## Reproducción

Desde la raíz, con NumPy y pandas instalados y los corpus públicos descargados:

```powershell
python scripts/entrenar_toques_micro.py --dfl data/external/idsse/figshare_28196177 --salida outputs/micro_dfl_nuevo
# El modelo congelado ya está versionado en modelos/toques_micro_dfl_v1.json.
python scripts/validar_microposesiones.py --aprox juliana_v2_micro --data-dir data --json outputs/comparacion_micro_nueva.json
python -m unittest discover -s tests -v
git diff --check
```

La salida de entrenamiento debe ser nueva. El validador también rechaza sobrescribir su JSON.
`--cache-base outputs/cache_metrica_nueva` permite reutilizar la ruta v2; verifica la procedencia
de los samples y vincula la caché al código, al modelo v4 y al escenario de ruido.
`--solo-base` prepara esa ruta sin cargar el detector. Esta caché solo contiene partidos externos.

Las pruebas sintéticas cubren simetría H/A, separación de periodos, ausencias, no uso del campo
de velocidad suministrado del balón, preservación de originales y de D, ambigüedades, NMS,
exclusión de tests reservados y reconstrucción de C3. No se escriben anotaciones manuales.

## Resultados y decisión

**La candidata no se acepta.** Con el umbral DFL de 0,25 no insertó ninguna micro-posesión en
los cuatro escenarios externos. Sus etiquetas son idénticas a la v2 y no hay ganancia de métricas.
Se conserva `alejo_v2_robusta` como referencia del empalme; el detector queda experimental.

| Metrica externo | spellF1 v2 | spellF1 + detector | macroF1, ambas | Precisión micro, ambas | Inserciones |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sample 1 limpio | 0,7263 | 0,7263 | 0,8219 | 0,6491 | 0 |
| Sample 1 ruido | 0,4765 | 0,4765 | 0,7550 | 0,1000 | 0 |
| Sample 2 limpio | 0,6851 | 0,6851 | 0,8488 | 0,6098 | 0 |
| Sample 2 ruido | 0,4929 | 0,4929 | 0,7992 | 0,1026 | 0 |

La tabla usa exactamente `metricas_micro` del empalme: spells de duración menor a 2 s y
emparejamiento oficial IoU 0,5; esa implementación incluye los spells D cortos. Son métricas
de **partidos públicos distintos**, no del partido del concurso. La mejora exigida de +0,02
con ruido falla en ambos samples y la precisión micro mínima también falla con ruido.
No se rebaja el umbral al ver Metrica.

En DFL dev, el umbral obtuvo 7 eventos emparejados de 14 predichos y 528 referencias acumuladas
entre cuatro condiciones: precisión 0,50 y recall 0,0133. Las cuatro condiciones son aumentos
del mismo partido, no observaciones independientes. Además, emparejar eventos a ±12 frames
no demuestra un contacto exacto ni garantiza un spell con IoU suficiente. El detector dio
solo 2/1/2/4 toques confiados en los cuatro escenarios de la tabla y ninguno formó una pareja
oponente→recuperación insertable. **Pasar el mínimo de precisión de eventos fue insuficiente.**

La siguiente investigación debería calibrar parejas completas y su efecto sobre spells
directamente en DFL dev, y distinguir una referencia débil de posesión de un contacto físico.
Esto requeriría un experimento nuevo: aquí se conserva el resultado negativo y no se presenta
como una mejora del rendimiento.

Los JSON publicables están en [comparación Metrica](revision_micro_juliana_01/comparacion_metrica.json)
y [calibración DFL](revision_micro_juliana_01/calibracion_dfl.json). El modelo congelado contiene
160 stumps y 60 variables; el SHA-256 de su JSON canonizado es
`bd051d0fd10319a48db7e4a8ad7c41873761c36121041429af9d9392435ee7e6`.
La canonización usa `json.dumps(model, sort_keys=True, separators=(',', ':'), ensure_ascii=False)`
codificado en UTF-8, para que los saltos de línea de Windows/Linux no alteren la comprobación.

Validación: **154 pruebas sintéticas/de regresión correctas** y `git diff --check` sin errores.
La comparación reutiliza una ruta v2 idéntica para ambas aproximaciones. La auditoría confirma
cero frames modificados en los cuatro escenarios; las nuevas guardas de soporte también se
comprueban con fixtures sintéticos. No se accedió al bundle restringido para esta iteración.
Los CSV y paquetes anteriores se conservan. Esta primera iteración quedó local, sin push.
La publicación posterior y los refinamientos adicionales se describen en el documento 14.

Antes de publicar la rama integrada debe revisarse la documentación histórica entrante:
contiene diagnósticos derivados del partido del concurso. Este informe solo incorpora métricas
de corpus públicos externos y no reproduce esas cifras. `docs/10` y `docs/12` se dejan intactos;
la tabla de esta revisión se mantiene aquí para poder compartirla por separado.

Fuentes: [DFL/IDSSE, figshare 28196177](https://doi.org/10.6084/m9.figshare.28196177), con licencia
CC BY según el manifiesto fijado en `config/idsse_v4.json`, y
[Metrica Sports Sample Data](https://github.com/metrica-sports/sample-data), fijado al commit del
manifiesto local. Los datos externos se mantienen fuera del repositorio; el modelo y sus métricas
pueden reproducirse con las fuentes públicas.
