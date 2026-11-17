# Veriana · ¿Quién tiene el balón? Documento técnico de la entrega

More Than a Hack 2027, Fase 1 · FC Barcelona – Athletic Club · equipo Veriana (Alejandro Pujana, Juliana De Mier,
Juan Felipe Daza). Código: `juliana_control_pares` (`src/mtah/aproximaciones/juliana_control_pares.py`), que
refina el canal H/A de `alejo_v2_robusta` (`src/mtah/aproximaciones/alejo_v1_fusion.py`, clase `FusionV2`).
Ninguna etiqueta del partido suministrado (ni la de los 5 minutos públicos) se usó para entrenar, ajustar o elegir;
no hubo anotación manual.

## 1. Enfoque

Separamos el problema en dos canales y una decodificación temporal:

1. **Canal D (balón muerto), por reglas físicas ancladas a eventos.** Cada reanudación del Smart Tagging marca una
   ventana donde buscamos el instante del saque: el primer arranque del balón que se separa del jugador y queda dentro
   del campo (regla 9 y H3: el juego vuelve con el primer contacto). El inicio de la parada es el cruce de la línea si
   se ve; si no, el fin del último viaje libre del balón dentro del campo; en último recurso, el frenazo colectivo.
   Las paradas sin etiqueta del tagging (sobre todo faltas) se detectan como núcleos de quietud colectiva sin balón
   visible. Tras un gol, la parada va del gol al saque de centro sin cortes.
2. **Canal H/A (quién tiene el balón).** Medimos en el propio partido, sin etiquetas, cuánto se puede fiar uno de la
   posición del balón: una mezcla Rayleigh + fondo estimada por EM da el radio a partir del cual un "toque" deja de ser
   distinguible del ruido. En BAR-ATH ese radio es de 2,66 m (en los partidos de muestra de Metrica, 0,46 m): el
   tracking del balón es demasiado ruidoso para asignar la posesión al último toque. En ese régimen, H/A sale de
   controles **supervisados fuera del partido** y robustos a ese ruido:
   - el control de la v4 (movimiento y forma de los equipos), decodificado con el HMM común;
   - refinado por un **control multiventana** (ventanas de 0,4, 0,8 y 1 s, dos vecinos por equipo, velocidades
     relativas al balón y reacción colectiva; 200 árboles de una división) con un Viterbi binario de penalización
     fija elegida en desarrollo, que **no toca la D**;
   - y por **parejas verificadas**: un retorno breve H→A→H (o A→H→A) solo se conserva si un detector de toques
     supervisado lo respalda; si no, se retira.
   Los dos refinamientos (Juliana De Mier) se entrenaron con ruido simulado en el balón (AR(1), 0 a 1 m). Con
   tracking limpio se usaría el último toque combinado con el control v4, sin refinamiento.
3. **Decodificación.** Un HMM de tres estados con transiciones estimadas por EM, sin corregir las probabilidades por
   su promedio en el partido (esa corrección restaba ventaja al equipo que más tiene el balón en los frames dudosos).

**Por qué así y no un clasificador único.** El balón falta en el 41% de los frames calificados y, cuando está, en
BAR-ATH es ruidoso; un clasificador entrenado con un tracking más limpio hereda supuestos que aquí no se cumplen. Los
eventos (cruce de línea, saque) dan bordes de parada precisos donde el balón se ve, y el control sin balón aguanta el
ruido. Lo comprobamos ensuciando el balón de los partidos públicos de Metrica hasta que nuestro estimador da el mismo
radio que en BAR-ATH: el último toque cae de spellF1 0,73 a 0,26, la v2 se queda en 0,49 / 0,48 y la entrega
(v2 + refinamientos H/A) llega a 0,53 / 0,53 en los dos partidos, sin cambiar nada con el tracking limpio.

## 2. Recursos usados y descartados

| recurso | uso |
|---|---|
| ATD táctico (jugadores y balón) | base de todo: viajes del balón, separación, quietud colectiva, forma de los equipos |
| Smart Tagging, `SET PIECES` y `GOALS` | ventanas donde buscar cada reanudación y cada gol (nunca se copian como D) |
| Límites publicados de los saques iniciales | respaldo del saque de centro del 2.º tiempo cuando no se ve arrancar desde el punto central |
| Smart Tagging, fases con equipo | **probado y descartado** en H/A: casi no movía el reparto y bajaba la concordancia con los controles supervisados |
| Homografía y vídeos | no usados (la homografía no recupera la altura del balón; el broadcast no está sincronizado) |
| Lugar de reanudación para tipar paradas sin etiqueta | **probado y descartado**: 91% de acierto en Metrica, 36% en BAR-ATH (el balón en las manos se proyecta dentro del campo) |
| Último toque como dueño (con tracking limpio) | se mantiene en el código para tracking limpio; en BAR-ATH no se usa (régimen ruidoso) |

## 3. Datos de entrenamiento y validación

Ninguno procede del partido suministrado.

| conjunto | uso | quién etiquetó | cantidad |
|---|---|---|---|
| DFL/IDSSE (Bassek et al., 2025; Figshare 28196177, CC BY 4.0), Bundesliga 2022/23 | control H/A y pausa de la v4 (`modelos/modelo.json`); control multiventana, detector de toques breves y parejas (`modelos/control_multiventana.json`, `toques_micro_dfl_v1.json`, `modelo_pares_micro.json`), de Juliana De Mier | proveedor oficial de datos de la Bundesliga (`BallPossession`, `BallStatus`) | 7 partidos; entrenamiento J03WMX/J03WN1/J03WOH, desarrollo J03WOY; 169.888 referencias D/vivo. Los refinamientos se entrenaron con ruido AR(1) de 0, 0,4, 0,8 y 1 m en el balón |
| SkillCorner Open Data (MIT) | control H/A (base v3) | SkillCorner (`team_in_possession`) | 6 partidos; 105.938 referencias de control junto con DFL |
| Metrica Sample Games 1 y 2 (públicos) | **solo validación** de bordes de parada, posesión densa por último evento y robustez al ruido | eventos de Metrica | 2 partidos |

## 4. Reglas de posesión implementadas

- **Balón muerto:** empieza cuando el balón deja de disputarse (cruza la línea, o termina su último viaje libre dentro
  del campo antes de una falta); termina con el primer arranque del balón que se separa de quien lo tiene y queda
  dentro del campo. Tras un gol no hay juego hasta el saque de centro. Los frames 110.808–113.741 son D (FAQ).
- **Reanudaciones:** el primer dueño del tramo vivo es el equipo que reanuda (con tracking limpio, el jugador más
  cercano al balón en el arranque).
- **Balones sueltos y disputas:** la posesión sigue con el último equipo con evidencia de control; un roce o un duelo
  no la cambia salvo que la evidencia del rival se sostenga (persistencia del HMM y del Viterbi del control). Un
  cambio breve que vuelve al equipo anterior solo se conserva si un detector de toques lo verifica.
- **Sin X con el balón vivo:** una abstención en una transición se paga en spellF1.

## 5. Limitaciones

- **Faltas sin etiqueta del tagging.** El tagging marca 5 tiros libres; nuestras 50 paradas sin etiqueta tienen una
  mediana de 11,7 s, probablemente cortas. Con más tiempo: un detector de tipo de parada que no dependa del lugar del
  balón (firmas colectivas por tipo, ya prototipadas en `mtah/firmas.py`).
- **Micro-posesiones.** Las posesiones de menos de 2 s (rebotes, malos controles) son el 25% de los spells en las
  referencias de Metrica. Con el balón ruidoso casi no se detectan: las parejas verificadas sobre todo retiran
  retornos breves no respaldados (su precisión de micro-posesiones en Metrica con ruido es 0,24 y 0,21).
- **El ruido simulado es una hipótesis.** La mejora de los refinamientos H/A se midió con la misma familia de ruido
  con la que se entrenaron (en partidos distintos); si el ruido real de BAR-ATH tiene otra estructura, puede ser menor.
- **Sin medida de acierto en BAR-ATH.** Solo plausibilidad: 66 min en juego (LaLiga: 53–60, aunque la ventana pública
  sugiere una definición de D más corta), 100 paradas, 333 posesiones, Barça 62,1% (las fases del tagging dan 62,6%).
- **El régimen ruidoso** se infiere con una sola cifra (el radio de toque); el ruido simulado para validarlo es una
  hipótesis.

## 6. Reproducción

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-pipeline.txt     # Python 3.12, numpy y pandas
# bundle del concurso en data/ (o MTAH_DATA_DIR apuntando a él)
PYTHONPATH=src:.:evaluation .venv/bin/python scripts/generar_entrega.py --salida Veriana.csv --autorizar-datos-restringidos
```

El script valida el archivo con el validador oficial e imprime su SHA-256; dos ejecuciones dan el mismo archivo.
