# {{EQUIPO}} — Posesión por frame, candidata multifuente v3

More Than a Hack 2027, Fase 1. El sistema combina tracking táctico, Smart Tagging automático,
control geométrico y continuidad temporal. Este documento solo describe código, datos públicos
de otros partidos y pruebas sintéticas. No contiene extractos ni estadísticas del bundle restringido.

## Método

La entrada ATD se valida por número de registros, orden de frames, equipos y unidades. Los slots
son tracklets. Las posiciones ausentes conservan NaN; saltos incompatibles invalidan velocidades.
Cada mitad se procesa por separado. El balón ausente no produce una etiqueta D.

El control H/A usa nueve variables antisimétricas: margen de distancia, diferencia de velocidades
relativas, continuidad del slot cercano, apoyo local, aproximación, diferencia de proximidad,
intercepción cinemática a 0,4 y 0,8 segundos y margen temporal centrado. Al intercambiar equipos
se invierte el signo. Una regresión logística sin intercepto se ajusta con igual peso por partido
y equilibrio H/A dentro de cada partido. La proyección es una aproximación de acceso al balón,
no un simulador de trayectorias aéreas. No asume control por cercanía aislada.

La evidencia aprendida se combina con geometría física, calidad de observación y prior débil de
Smart Tagging. Las fases contradictorias se neutralizan. Las ventanas de saque aportan contexto,
no etiquetas de parada ni fronteras exactas. Solo huecos internos plausibles de hasta 0,6 s
pueden interpolarse, con peso reducido y marca explícita de imputación.

Para juego detenido se conserva el clasificador v2 de movimiento colectivo, respaldado por
balón lento observado, salida del terreno o contexto de saque. Ni ausencia de balón ni lentitud
colectiva aislada prueban una parada. Un balón viajando o movimiento incompatible termina ese
contexto. La extensión de anticipación de paradas se evaluó como ablación y no quedó activa.

Viterbi y forward-backward resuelven conjuntamente H/A/D. X expresa abstención según concentración
del modelo y distancia a evidencia. Las marginales del modelo temporal no son probabilidades de
acierto calibradas. Se evaluó un HSMM de edad explícita con cola geométrica y salida permitida
en todas las edades, sin duración mínima dura. La selección externa mantuvo el HMM: añadir
duraciones no mejoró el promedio de desarrollo frente al control nuevo con HMM.

## Assets y fuentes

- ATD táctico: geometría y movimiento de entrada, con alineación al vídeo táctico.
- Smart Tagging: fases automáticas y contexto de reanudación, siempre con peso débil.
- Homografía y vídeos: auditoría estructural y geométrica local. Los píxeles y el broadcast no
  entran al predictor. Fusionar broadcast exige resolver cortes, repeticiones y otra línea temporal.
- [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data), commit
  `e706dd506b360d69d9d123d5b8026e7294b13996`: Game 1 para ajuste de control y pausa; Game 2
  como guardia de regresión de desarrollo. Se reconoce a Metrica Sports como proveedor.
- [SkillCorner Open Data](https://github.com/SkillCorner/opendata), commit
  `4340d274572876239c154c90bc507a9b3250a656`: seis partidos de A-League 2024/25. Se reconoce
  a SkillCorner y su iniciativa con PySport. El repositorio publica licencia MIT y solicita crédito.

Entrenamiento SkillCorner: 1874553, 1886347, 1899585. Desarrollo: 1925299. Evaluación reservada:
1927964 y 1953632. La asignación quedó fijada antes del entrenamiento. El modelo, su variante
y su hash se escribieron antes de evaluar los partidos reservados. No se reajusta tras ese test.

Los archivos públicos se verifican por hashes Git; para LFS se comprueba SHA-256 reconstruyendo
el puntero fijado. SkillCorner se procesa a sus 10 fps nativos, convirtiendo metros centrados a
coordenadas desde la esquina. Se descartan coordenadas extrapoladas del balón y jugadores como
observaciones. `possession`, las fases y los eventos del proveedor nunca entran como features.

Las etiquetas H/A de SkillCorner son automáticas, no revisión humana independiente. Un valor
null permanece X: no se convierte en D por ausencia de fase o de poseedor. Metrica aporta anclas
PASS/RECOVERY/SHOT y referencias débiles de pases completados y salida–reanudación, con bordes
censurados. No creamos anotaciones manuales. Los modelos y el informe externo contienen el
número de referencias y la procedencia exacta. No se utilizan etiquetas del partido del concurso,
ni los cinco minutos oficiales, para ajustar, seleccionar variables, modelos o versiones.

## Evaluación y límites

Se comparan v2, control nuevo, HSMM y anticipación de paradas. El control utiliza 17.215 referencias
externas de entrenamiento. La temperatura binaria se elige solo en desarrollo por log-loss entre
0,5, 1, 2 y 4. El valor elegido fue 1. La calibración se refiere a H/A con balón detectado y acceso
plausible, no a la corrección H/A/D de toda la secuencia.

Además de observaciones originales, se evalúa estrés reproducible con oclusiones condicionadas
por aglomeración y velocidad, huecos largos, saltos de balón, pérdida de tracklets e intercambios
de slots. Es un escenario de estrés diseñado, no una simulación validada del error del concurso.

SkillCorner permite medir acuerdo H/A y el algoritmo oficial de spellF1 sobre regiones H/A
conocidas. Se denomina **spellF1 condicional**: los X de referencia recortan el dominio y pueden
recortar posesiones. No evalúa D completo ni equivale al spellF1 oficial. Metrica permite medir
macroF1 sobre referencias débiles H/A/D y acuerdo sobre anclas. Se informan los resultados por
partido, sin inferir significancia estadística con solo dos partidos reservados. Consultar el
informe de revisión para cifras y decisión de promoción.

Riesgos: cambio de proveedor y liga, etiquetas automáticas correlacionadas con el tracking,
definiciones de posesión distintas, oclusiones largas, rebotes, control estático y fronteras de
parada sin referencia completa. La evaluación externa no demuestra mejora en el score oficial.
La inferencia no realiza consultas remotas ni correcciones humanas.

## Reproducción

Python 3.12, NumPy 2.3.5, pandas 3.0.1. Desde el código publicable extraído:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pipeline.txt
.\.venv\Scripts\python.exe reproducir_entrega.py --datos "D:\bundle-local" --salida "{{SLUG}}.csv" --autorizar-datos-restringidos
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

En macOS/Linux usar `.venv/bin/python`. Los originales conservan sus nombres bajo `tactical/`
y `broadcast/`. El CSV debe tener 199.319 filas y cabecera `frame,label`. El proceso valida el
formato sin leer GT, conserva registros privados y exige un destino nuevo. El paquete comprueba
reproducción idéntica byte a byte desde una extracción limpia del código.

Para repetir el desarrollo externo: `scripts/descargar_metrica.py`,
`scripts/descargar_skillcorner.py` y `python -m mtah.experimento_v3 --salida outputs/experimento_nuevo`
con `PYTHONPATH=src` y `--modelo-v2 modelos/modelo.json`. Ese argumento acepta el modelo v2
o su copia incrustada en el modelo v3, con procedencia verificada, para reproducir el experimento.
Los tests son sintéticos e incluyen enumeración exhaustiva de Viterbi y marginales HSMM.

CSV, datos y diagnósticos del concurso permanecen locales. El ZIP de código y el informe público
no contienen esos materiales. La preparación local no publica el repositorio ni envía la entrega.
