# {{EQUIPO}} — Candidata v4, control y juego detenido

More Than a Hack 2027, Fase 1. Código reproducible y parámetros públicos; originales y resultados
del partido restringido permanecen locales. No se entrenó, ajustó ni seleccionó con etiquetas del
partido suministrado o los cinco minutos oficiales. No se hicieron anotaciones manuales.

## Sistema

El preprocesamiento valida orden, número de frames, afiliación y unidades. El ATD táctico es la
geometría principal. Se conservan NaN y máscaras de observación; saltos incompatibles invalidan
velocidades. Las mitades se procesan por separado. La ausencia del balón no es una clase D.

La base v3 combina nueve variables antisimétricas H/A, evidencia física, pausa y HMM. V4 compara
un control ampliado con tiempo de acceso al balón: posición proyectada tras una reacción de 0,7 s
y distancia restante a velocidad de referencia 5 m/s. Son supuestos de ingeniería contrastados
solo externamente; no representan una trayectoria aérea exacta. Intercambiar equipos invierte
el control y conserva las variables de parada. Se evita identificar control con cercanía aislada.

El nuevo clasificador de D utiliza boosting logístico de árboles de una división sobre 13 señales
de movimiento colectivo, variación temporal, dispersión y separación espacial. No recibe
visibilidad del balón, BallStatus, BallPossession ni eventos humanos como features. Se exige
movimiento utilizable de al menos seis slots; si falta, se conserva la evidencia base. La lentitud
no actúa como una regla suficiente: el clasificador se ajusta con ejemplos vivos y detenidos.
Las velocidades de pausa y acceso usan desplazamientos a 0,2 s para mantener la escala temporal
entre desarrollo a 5 Hz e inferencia a 25 Hz; se rechazan huecos y saltos dentro de esa ventana.
La rama de 5 Hz coincide exactamente con la definición utilizada en el experimento congelado.

Los log-odds de pausa se combinan con los potenciales H/A y el HMM. Se compara mezcla parcial y
completa frente a v3. La temperatura del clasificador se elige por log-loss únicamente en
desarrollo. Las marginales HMM siguen expresando concentración, no probabilidad calibrada de acierto.

## Reanudaciones y posesión

Se conservan tipos de saque, equipo conocido/desconocido y conflictos. Para banda sin equipo se
requiere apoyo observado: balón lento junto a la lateral y jugador claramente próximo de un lado.
También se generan candidatos geométricos sin tag, sin usar etiquetas o eventos de referencia.
El prior de equipo se aplica débilmente al observar reanudación hacia dentro, nunca a toda la
espera. No se invierte automáticamente el último poseedor: un desvío rival puede mantener el saque
para ese equipo. Ambigüedad conserva equipo desconocido. Los falsos candidatos cerca de la lateral
se incluyen en las pruebas sintéticas y el prior se contrasta como ablación.

Durante vuelos y oclusiones se combina contexto anterior/posterior; se permite interpolación
acotada de huecos breves con menor peso. El control rival puede producir un cambio rápido: no se
impone mínimo duro. D expresa juego detenido, independientemente del equipo que reanudará. X
expresa abstención cuando falta evidencia. El HMM procesa por separado cada mitad.

Las opciones efectivamente activas —control nuevo, mezcla de pausa y peso de reanudación— están
en `modelos/modelo.json`, apartado `opciones`. Los componentes se implementaron y compararon;
solo la variante seleccionada externamente define esta candidata congelada.

## Datos externos y protocolo

Todos son encuentros distintos al partido del concurso, accesibles a los organizadores:

- **Metrica Sports sample-data**, commit `e706dd506b360d69d9d123d5b8026e7294b13996`:
  Game 1 para ajustes heredados; Game 2 como regresión de desarrollo ya utilizada, con referencias
  débiles. Se reconoce a Metrica Sports como proveedor.
- **SkillCorner Open Data**, commit `4340d274572876239c154c90bc507a9b3250a656`:
  1874553/1886347/1899585 para control. Etiquetas automáticas H/A; posiciones extrapoladas excluidas
  como observaciones. Los anteriores tests 1927964/1953632 no se reutilizan como test nuevo.
- **DFL/IDSSE**, Figshare `28196177`, versión 1: siete partidos Bundesliga 2022/23, tracking y
  referencias publicadas de posesión y balón activo/inactivo. Entrenamiento J03WMX/J03WN1/J03WOH;
  desarrollo J03WOY; evaluación reservada J03WPY/J03WQQ/J03WR9. Se reconoce a Deutsche Fußball Liga
  y Bassek, Rein, Weber y Memmert (2025), DOI `10.1038/s41597-025-04505-y`, licencia CC BY 4.0.

Los XML DFL se verifican por tamaño/MD5 publicado. Las fuentes anteriores usan hashes Git/LFS.
DFL se remuestrea a 5 Hz para desarrollo, sin interpolar etiquetas; se convierten metros centrados
a origen en esquina y km/h a m/s. Los relojes se validan. BallStatus/BallPossession permanecen
separados del ATD. Los eventos DFL solo delimitan análisis externos alrededor de saques, nunca
son entradas. Sus límites temporales pueden ser imprecisos. La cantidad de referencias se conserva
en el modelo y los informes públicos externos. No creamos anotaciones adicionales.

La partición, variantes y criterios de promoción se escriben antes del ajuste. El modelo y su hash
se congelan antes de evaluar los tres partidos reservados. No se reajusta tras conocerlos. El
estrés añade huecos dependientes de aglomeración/velocidad, saltos y cambios de slots, con semilla
fija y sin leer respuestas. Se comparan macroF1 y spellF1 con el algoritmo oficial sobre estas
referencias externas, a 5 Hz. No son puntuaciones del concurso y no justifican significancia por sí solas.

## Assets y límites

Smart Tagging es automático y aporta priors débiles, no GT. Los 26 Throw-in y Team=N/A mencionados
en el análisis proceden de DATA.md público, no de estadísticas privadas. Las ventanas no son
posesiones ni etiquetas D completas.

Homografía/vídeo mantienen auditoría local. Se añaden utilidades verificadas para proyectar campo,
invertir pies al césped y asociar ejecutantes con tracking, rechazando ambigüedad y degeneración.
Un detector visual no está activado en esta candidata: no se ha validado una cadena de píxeles a
posesión. La homografía no recupera altura de balón en manos/vuelo ni sincroniza broadcast.

Quedan riesgos de cambio de dominio: DFL es tracking revisado y más limpio, etiquetas de posesión
de otros proveedores, oclusiones prolongadas, errores de slot, faltas cerca de banda y pérdidas
del ejecutante. Los supuestos de acceso al balón son aproximados. No se afirma mejora oficial.

## Reproducción

Python 3.12, NumPy 2.3.5 y pandas 3.0.1. No se necesitan servicios remotos, GPU, pickle ni modelos
descargados para inferir. Desde el paquete de código:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pipeline.txt
.\.venv\Scripts\python.exe reproducir_entrega.py --datos "D:\bundle-local" --salida "{{SLUG}}.csv" --autorizar-datos-restringidos
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

El programa exige destino nuevo, genera 199.319 filas, valida sin GT y conserva el log privado.
La preparación comprueba igualdad byte a byte desde una extracción limpia. En Linux/macOS usar
`.venv/bin/python`. El flag de acceso no sustituye la autorización al agente para leer datos.

Para repetir el experimento externo: descargar los tres corpus mediante los scripts incluidos;
con `PYTHONPATH=src`, ejecutar `python -m mtah.experimento_v4 --modelo-v3 modelos/modelo.json --salida outputs/experimento_v4_nuevo`.
El argumento acepta v3 o su base incrustada en v4. Repetirlo reproduce una comparación; sus
partidos reservados ya utilizados no se convierten en un nuevo test ciego.

CSV, vídeo, tracking, cachés y diagnósticos del concurso quedan fuera del código publicable.
La preparación no envía el formulario ni publica automáticamente el repositorio.
