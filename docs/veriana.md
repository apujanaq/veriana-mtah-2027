# veriana · Predicción de posesión por frame

**Aproximación de este CSV:** `{{APROXIMACION}}`.

**Estado:** {{ESTADO}}

## Fuentes y preprocesamiento

Se utilizan las posiciones de balón y jugadores del tracking táctico y las ventanas de Smart
Tagging. Las coordenadas están en metros, alineadas al frame del vídeo a 25 Hz. Los equipos se
normalizan H/A. Las ausencias son NaN, sin convertirlas automáticamente en D. Se separan periodos
y tramos evaluables. La homografía y geometría visual están disponibles, pero no hay detector de
vídeo ni mejora visual activada en esta entrega.

## Referencia común: alejo_v2_robusta

El bloque A infiere D desde eventos automáticos, viajes del balón, límites del campo y movimiento
de jugadores. La pausa aprendida de v4 puede adelantar determinados inicios de parada. Los saques
sin equipo etiquetado no reciben un equipo anotado a mano.

El canal H/A distingue la calidad observable del tracking. Cuando permite resolver contactos,
fusiona continuidad del último toque y control v4; con tracking ruidoso usa el control v4.
Este control sí usa geometría del balón: distancias, velocidad relativa, continuidad, apoyos,
aproximación, intercepción y tiempo de llegada. La cercanía sola no implica control. Un HMM común
estima persistencia sin etiquetas y decodifica por periodo. Fuera de los intervalos calificados
se conserva X. La ausencia del balón no constituye por sí misma una interrupción.

## Refinamientos experimentales

juliana_v2_micro aprende eventos breves de posesión con 60 variables en ventanas de 0,4–1 s,
dos vecinos por equipo, velocidades relativas y reacción colectiva. Los objetivos son cambios
cortos y recuperaciones de BallPossession de otros partidos: referencias débiles de contacto,
no anotaciones exactas de toque físico.

juliana_pares_micro calibra parejas completas y verifica retornos H→A→H/A→H→A breves con tracking
ruidoso. juliana_control_multiventana aprende control denso con boosting de stumps, combina
simétricamente evidencias y usa Viterbi H/A dentro de regiones vivas de la referencia.
juliana_control_pares compone ambos sin volver a ajustar parámetros. Todos conservan las D de
la referencia, periodos y tramos no calificados. Sin observación fiable del control nuevo se usa
evidencia temporal de la base. La verificación de parejas no atraviesa ausencias del balón.

## Entrenamiento y selección

Control y pausa v4 se entrenaron con corpus públicos fijados de Metrica, SkillCorner y DFL/IDSSE.
Los refinamientos usan DFL train J03WMX/J03WN1/J03WOH y dev J03WOY a 25 Hz. No se reutilizan los
antiguos tests J03WPY/J03WQQ/J03WR9 para ajustar o seleccionar. BallStatus/BallPossession y eventos
externos nunca entran como variables de los clasificadores.

El aumento AR(1) usa memoria 0,4 s y desviaciones 0/0,4/0,8/1 m; conserva ausencias y se reinicia
por periodo. Las muestras de control incluyen ambos equipos por frame, sin identidad de club.
Modelos, semillas, particiones y parámetros son inspeccionables en JSON.

DFL no aporta Smart Tagging: las parejas se calibran sobre v4 autónoma sin eventos. La persistencia
del control se calibra condicionalmente a intervalos H/A de la referencia externa. D solo se
conoce para evaluar H/A en desarrollo; en inferencia se bloquea a las D predichas por la base.
Estas condiciones optimistas se contrastan después en Metrica con D predicha.

Metrica Sample 1 y 2, limpio y con ruido, es desarrollo externo conocido. La referencia densa
deriva de eventos y el tagging se emula: no equivale al GT del concurso. Las cifras, alternativas
y decisión conservadora están en docs/14_CIERRE_Y_ENTREGA_EQUIPO.md.

## Ejecución y límites

La ejecución es local con NumPy/pandas, sin enviar datos a servicios externos. Cada CSV se crea
en un destino nuevo y pasa el validador oficial de estructura y una relectura. Esto no demuestra
precisión de sus etiquetas. No se entrenó, ajustó ni seleccionó con las etiquetas de los cinco
minutos oficiales; no se crearon anotaciones manuales del partido.

La referencia y alternativa experimental se conservan por separado. La mejora externa de la
alternativa todavía debe cumplir todos los criterios de precisión del equipo para aceptarse.
No se conoce una puntuación de esta solución sobre el GT privado del concurso.

Fuentes: [Metrica](https://github.com/metrica-sports/sample-data),
[SkillCorner](https://github.com/SkillCorner/opendata) y
[DFL/IDSSE](https://doi.org/10.6084/m9.figshare.28196177).
Licencias y atribuciones: THIRD_PARTY_NOTICES.md y manifiestos de config/.
