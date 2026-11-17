# 2. Las dos métricas y lo que exigen al modelo

El concurso se ordena por **spellF1** y usa **macroF1** como primer desempate. El contrato y la
implementación autoritativa están en [EVALUATION.md](../EVALUATION.md),
[possession_metrics.py](../evaluation/possession_metrics.py) y
[run_evaluation.py](../evaluation/run_evaluation.py). Algunos docstrings antiguos aún dicen que
macroF1 ordena el tablero; el código activo del scorer fija `RANK_METRIC = "spell_f1_mean"`.

## macroF1: acertar la etiqueta en cada frame

Solo se evalúan frames cuya referencia es `H`, `A` o `D`. Los frames con referencia `X` se
ignoran completamente: no producen verdaderos positivos, falsos positivos ni falsos negativos.

Para cada clase `k` entre `H/A/D`:

```text
TP_k = frames de clase k correctamente predichos como k
FP_k = frames de otra clase evaluada predichos como k
FN_k = frames de clase k predichos como otra clase o como X

precision_k = TP_k / (TP_k + FP_k)
recall_k    = TP_k / (TP_k + FN_k)
F1_k        = 2 TP_k / (2 TP_k + FP_k + FN_k)
macroF1     = (F1_H + F1_A + F1_D) / 3
```

El scorer devuelve cero en divisiones indefinidas, incluidas clases ausentes. Esto importa al
interpretar ejemplos cortos: si no tienen `D`, el F1 de D es cero, aunque H/A sean perfectos.
`X` no es una cuarta clase que se promedia; predecirlo sobre una clase evaluada cuenta como fallo.

No es accuracy: los F1 de las tres clases pesan igual. Y sigue siendo una métrica por frames:
una posesión larga aporta más observaciones que una corta. Un error de un frame suele cambiar
muy poco el resultado global.

## spellF1: recuperar posesiones como intervalos

Una posesión es un tramo máximo continuo de **H o A con el balón vivo**. `D` es una interrupción,
no una posesión. `H H D H` contiene dos posesiones H. `H H A H` contiene tres posesiones.

El scorer aplica exactamente esta secuencia:

1. Obtiene los bloques contiguos donde el GT no es X. No empareja a través de un descanso.
2. Dentro de cada bloque rellena los X predichos con la última etiqueta conocida. En la cabecera
   sin etiqueta anterior usa la primera conocida del propio bloque. Si todo el bloque es X,
   permanece sin respuesta. Las métricas por frame siguen usando los X originales.
3. Extrae los tramos máximos H/A de la referencia y de la predicción rellenada.
4. Considera parejas de intervalos **del mismo equipo** que solapen y calcula su IoU temporal:

```text
interseccion = max(0, min(fin_real, fin_predicho) - max(inicio_real, inicio_predicho))
union        = max(fin_real, fin_predicho) - min(inicio_real, inicio_predicho)
IoU          = interseccion / union
```

Los extremos son `[inicio, fin)`. Un intervalo real `[0,100)` y uno predicho `[20,120)` tienen
intersección 80, unión 120 e IoU = 2/3. Cambiar el equipo impide emparejar incluso con IoU 1.

5. Ordena las parejas por IoU descendente. Recorre de forma **voraz**, aceptando una pareja
   solo si no ha utilizado ninguno de los dos intervalos y supera el umbral. Es uno a uno;
   no es un algoritmo húngaro ni una asignación global que maximice el número de aciertos.
   En empates el orden de los índices hace el resultado determinista.
6. Para cada umbral `t` de `0.3, 0.4, 0.5, 0.6, 0.7` cuenta:

```text
TP_t = parejas aceptadas
FP_t = posesiones predichas sin pareja
FN_t = posesiones reales sin pareja
F1_t = 2 TP_t / (2 TP_t + FP_t + FN_t)
spellF1 = promedio de los cinco F1_t
```

No hay curva precision-recall ni confianza por posesión: por eso no se denomina mAP. Cada posesión
cuenta una vez, independientemente de su duración. Fragmentar una posesión crea intervalos extra;
fusionar dos elimina una oportunidad de emparejar. Desplazar los bordes reduce IoU, especialmente
en posesiones cortas. D importa indirectamente al separar o fusionar posesiones, aunque D no se
empareje como objeto.

## Un ejemplo que cambia la decisión de diseño

El módulo `mtah.metricas_demo` inventa una referencia de 400 frames:
`H×100, A×100, D×100, H×100`. No usa muestras ni etiquetas del partido del concurso.
Estos resultados se calcularon con el evaluador oficial:

| Error sobre esa referencia inventada | macroF1 | spellF1 |
|---|---:|---:|
| Ninguno | 1,0000 | 1,0000 |
| Un único frame A espurio dentro del primer H | 0,9975 | 0,6500 |
| Diez frames de retraso | 0,9167 | 1,0000 |
| X dentro de una posesión H | 0,9730 | 1,0000 |
| X que oculta gran parte del cambio H→A | 0,7692 | 0,5333 |
| Sustituir la parada por H | 0,6000 | 0,8667 |
| Siempre H | 0,2222 | 0,0000 |

Son ejemplos, no rendimientos esperados. El retraso de diez frames conserva suficiente IoU en
estas posesiones; no significa que diez frames sean gratis en cualquier partido. Tampoco es un
teorema que toda predicción constante tenga spellF1 cero en cualquier secuencia imaginable.

Para ver las secuencias, la matriz de confusión, el F1 de cada clase y los resultados por IoU:

```powershell
$env:PYTHONPATH = "src"
python -m mtah.metricas_demo --salida outputs/demo_metricas.html
```

El comando crea un destino nuevo. El ejemplo enseña por qué no basta con acertar muchos frames:
el modelo debe controlar las fragmentaciones, sin borrar cambios auténticos de corta duración.

## Qué significa abstenerse

`H X X H` se convierte en `H H H H` para spellF1, pero los dos X siguen siendo errores para
macroF1. `H X X A` mantiene H durante el hueco antes de cambiar a A: puede desplazar la transición.
Rellenar no descubre el estado verdadero; solo determina cómo el scorer trata la abstención.

Por tanto, una ausencia de detección no debe convertirse por defecto en D. El modelo temporal
puede conservar posesión o inferirla con observaciones anteriores/posteriores; cuando no hay
evidencia suficiente, X expresa incertidumbre y tiene un coste explícito. Una posterior del HMM
no es una prueba de acierto, sobre todo con emisiones sin calibrar.

## Protocolo de evaluación

- Las pruebas sintéticas verifican código y comportamientos concretos. No estiman rendimiento
  sobre fútbol real ni sirven para afirmar que se supera el baseline oficial.
- Los otros partidos permiten desarrollo y ajuste. Separar por partido evita que frames vecinos
  del mismo encuentro aparezcan a ambos lados de la partición.
- Los eventos de los samples públicos solo proporcionan referencias parciales de H/A. Nuestro
  informe externo calcula acuerdo y macroF1 **de esas referencias H/A**: no son macroF1 oficial
  de tres clases ni spellF1 de posesiones completas. No rellenamos esos puntos para fabricar un GT.
- No se usan los cinco minutos oficiales para entrenar, ajustar, elegir variables, comparar
  versiones o elegir duraciones. Su eventual uso es un chequeo de integración con la versión
  congelada. El código nuevo no los lee automáticamente.
- Si en el futuro existe un conjunto externo con posesiones completas y licencia apropiada,
  usar el mismo scorer, con evaluación por partido y análisis por posesiones, no una división
  aleatoria de frames. El bootstrap debe remuestrear unidades temporales, no tratar cada frame
  como independiente. Una diferencia pequeña en una muestra diminuta no establece superioridad.
