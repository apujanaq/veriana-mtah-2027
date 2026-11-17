# {{EQUIPO}} — Predicción de posesión por frame

More Than a Hack 2027, Fase 1. Implementación reproducible basada en evidencia geométrica,
movimiento colectivo y un modelo temporal offline. Esta documentación no contiene datos,
extractos, gráficos ni diagnósticos del partido restringido.

## Enfoque

La versión de entrega separa tres problemas: acceso/control del balón, juego vivo o detenido,
y continuidad de posesión. Produce H/A/D y utiliza X como abstención cuando la evidencia no
sustenta una etiqueta. No equipara automáticamente al jugador más cercano con el poseedor.

Se extraen distancias métricas al equipo local y visitante, diferencia de distancias,
velocidades relativas balón–slot, continuidad local, proximidad de otros jugadores y movimiento
colectivo derivado de posiciones. Los slots son tracklets, no identidades persistentes.
Discontinuidades físicamente incompatibles invalidan velocidades; no se corrigen por anotación.

Un clasificador logístico H/A sin intercepto utiliza variables antisimétricas: intercambiar los
equipos invierte sus logits. La candidata v2 añade un clasificador de pausa sobre movimiento
colectivo, sin usar «balón ausente» como predictor directo de D. Ese clasificador se combina
con señales independientes: balón lento observado, salida del terreno y contexto de saque.

Viterbi obtiene la secuencia H/A/D conjunta. Forward-backward permite combinar observaciones
anteriores y posteriores y expresa concentración del modelo, no una probabilidad calibrada de
acierto. Las mitades se procesan por separado. Las transiciones se penalizan, pero no se
impone una duración mínima que elimine cambios reales rápidos.

## Assets utilizados

- **ATD táctico:** entrada geométrica principal. Se valida el índice `1..N`, se convierte a
  vídeo `0..N-1` y se escala a metros. No se rellenan ni recortan registros para ocultar errores.
- **Smart Tagging automático:** prior débil de fases y contexto de saque. N/A y nombres
  desconocidos no se asignan por descarte; evidencias contradictorias se neutralizan. Una fase
  no se copia como posesión ni toda una ventana de saque se etiqueta D.
- **Homografías:** auditoría de geometría y calidad en el EDA local. No se usan sus errores como
  una probabilidad de que el ATD sea correcto, ni como entradas del clasificador final.
- **Vídeos y broadcast:** auditoría estructural de los contenedores y fuentes. No se emplean
  píxeles en esta versión y no se fusiona el broadcast sin una correspondencia temporal.

## Datos de entrenamiento y referencias externas

Se utilizan exclusivamente otros partidos de [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data),
commit `e706dd506b360d69d9d123d5b8026e7294b13996`. Los CSV se verifican por tamaño y hash Git.
Game 1 se usa para ajustar los clasificadores; Game 2 es validación de desarrollo, no un test
final ciego tras iteraciones. Se reconoce la autoría y provisión de los datos a **Metrica Sports**.
No creamos anotaciones manuales. La fuente no identifica a cada anotador individual.

El control usa referencias puntuales de PASS/RECOVERY/SHOT. La pausa usa referencias débiles:
interiores de pases completados y de intervalos salida–reanudación, con bordes censurados y
rechazo de acciones vivas intermedias, contradicciones y cruces de periodo. No se propagan
esos puntos para fabricar una verdad completa de posesión. En el ajuste externo v2 se usaron
1.623 referencias de control H/A y 4.257 frames de apoyo vivo/muerto muestreados cada diez frames.
La procedencia, desglose y coeficientes exactos quedan en `modelos/modelo.json`.

Las definiciones se consultaron en la [documentación pública de eventos](https://github.com/metrica-sports/sample-data/blob/e706dd506b360d69d9d123d5b8026e7294b13996/documentation/events-definitions.pdf).
Sus convenciones temporales pueden diferir de las del concurso; las referencias se presentan
como aproximaciones externas, no como etiquetas oficiales.

No se utiliza GT privado ni datos humanos sobre el partido del concurso. Los cinco minutos
oficiales no se usan para entrenamiento, ajuste, elección de variables, duraciones o versiones.
La candidata se congela con la evidencia externa antes de ejecutar el bundle restringido.

## Reglas de posesión implementadas

- Una proximidad aislada o un rebote no transfiere automáticamente la posesión. Se combina
  acceso geométrico, movimiento compatible y evidencia temporal.
- Durante vuelos u oclusiones se conserva la incertidumbre y se utiliza contexto anterior y
  posterior. Solo huecos internos de hasta 0,6 s, acotados y físicamente plausibles, pueden
  interpolarse; mantienen una marca de imputación y menor peso que una medición. No se
  interpolan huecos largos, fronteras de mitad o trayectorias incompatibles.
- D exige evidencia compatible de interrupción. La ausencia del balón por sí sola es neutra.
  El contexto de parada se reinicia ante evidencia de balón viajando o movimiento incompatible.
- Fuera de las regiones públicas de evaluación se escribe X. Dentro, la abstención depende
  de concentración y distancia a evidencia sustentada; no se corrigen etiquetas a mano.

## Evaluación y límites

Los tests incluyen rebotes, transiciones breves, periodos, huecos, saltos, formato CSV y
verificación de Viterbi/marginales por enumeración exhaustiva en un caso pequeño. El protocolo
externo compara v1 y v2 en observaciones originales y con oclusiones sintéticas reproducibles.

En el experimento externo inicial, macroF1 de referencias débiles H/A/D pasó de 0,553 a 0,719;
con oclusiones, de 0,517 a 0,645. El acuerdo en 1.957 referencias puntuales H/A se mantuvo
aproximadamente en 95,5% y 80,1%, respectivamente. **Estas cifras no son macroF1 oficial ni
spellF1, y no estiman precisión sobre todo el partido del concurso.** Las referencias de D
siguen siendo débiles y su cobertura no valida todas las paradas.

Los riesgos principales son cambio de dominio del tracking, identity switches, oclusiones
dependientes de la acción, posesión estática confundible con pausa, ausencia de evidencia en
huecos largos y fases tácticas con límites imprecisos. El modelo temporal tiene duraciones
implícitas geométricas. No se afirma una mejora de spellF1 oficial.

## Reproducción

Python 3.12; NumPy 2.3.5 y pandas 3.0.1 fijados en `requirements-pipeline.txt`. No se requieren
servicios remotos para inferencia. Desde el código público extraído:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pipeline.txt
.\.venv\Scripts\python.exe reproducir_entrega.py --datos "D:\bundle-local" --salida "{{SLUG}}.csv" --autorizar-datos-restringidos
```

En macOS/Linux usar `.venv/bin/python`. El bundle debe conservar `tactical/` y `broadcast/`
con los nombres oficiales. El programa valida el CSV, crea un destino nuevo y no imprime datos
ni diagnósticos restringidos. El paquete de entrega comprueba la igualdad byte a byte entre
la ejecución de trabajo y otra desde el código público extraído, sin acceso al repositorio original.

El CSV y los registros permanecen privados; el ZIP de código contiene solo fuentes aprobadas,
pruebas sintéticas, el modelo ajustado con datos públicos y esta documentación. La publicación
del código y el envío al formulario corresponden a una fase posterior de revisión del equipo.
