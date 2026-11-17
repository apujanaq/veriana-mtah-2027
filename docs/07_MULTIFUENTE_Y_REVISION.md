# Desarrollo multifuente y revisión de veriana

La candidata v3 mejora el control H/A usando otros partidos públicos. Se conservan v1, v2 y sus
outputs. No se ha publicado código ni enviado la entrega. El equipo debe revisar esta versión
conociendo que sus métricas externas no sustituyen el resultado del concurso.

## Fuentes adicionales y protocolo

Se descargaron seis partidos de [SkillCorner Open Data](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656),
con licencia MIT publicada, crédito a SkillCorner/PySport y verificación de blobs Git y SHA-256 LFS.
Se suman a los dos samples externos de Metrica. El manifiesto fijo es
`config/skillcorner_v3.json`; los datos quedan bajo `data/external/`, ignorados por Git.

La partición se fijó antes del entrenamiento: 1874553/1886347/1899585 entrenan,
1925299 permite elegir variante y temperatura, 1927964/1953632 se reservan para evaluación.
Metrica Game 1 mantiene el ajuste de pausa y aporta anclas de control; Game 2 sigue siendo una
guardia de regresión de desarrollo, pues ya se había utilizado antes.

SkillCorner se procesa a 10 fps nativos. Los metros centrados se trasladan a origen en la esquina;
las coordenadas extrapoladas se descartan como observaciones. Sus etiquetas de posesión son
automáticas y pueden compartir errores con el tracking. La posesión del proveedor, sus fases y
eventos nunca entran a las features. Null sigue siendo X: no hay un D completo e independiente.

También se consultaron [StatsBomb Open Data](https://github.com/statsbomb/open-data) y
[SoccerNet GSR](https://www.soccer-net.org/tasks/game-state-reconstruction). Los eventos y
freeze-frames de StatsBomb no aportan tracking continuo; GSR ofrece reconstrucción visual de
jugadores y exigiría otro desarrollo. No se descargaron datos humanos de esas fuentes sobre
el partido del concurso. Una integración visual propia y sincronización broadcast quedan pendientes.

## Componentes implementados y comparación

`mtah.candidata_v3` añade proyecciones cinemáticas de acceso al balón a 0,4 y 0,8 s, aproximación,
proximidad y contexto a las cuatro variables de control anteriores. La logística antisimétrica
usa nueve variables y 17.215 referencias externas. Cada partido pesa lo mismo; se equilibran
los equipos dentro de cada partido. Una búsqueda de temperatura por log-loss en desarrollo
eligió 1 entre 0,5/1/2/4. Su calibración solo se refiere a control H/A observable y plausible.

`mtah.duracion` implementa HSMM con edad explícita y cola geométrica. Todas las edades permiten
salida, sin mínimo duro que prohíba transiciones rápidas. Su inferencia Viterbi y forward-backward
se verifica contra enumeración exhaustiva. El riesgo H/A usa cambios conocidos; X y mitades
censuran. Los riesgos D conservan el prior v2 por falta de referencia completa independiente.
La rejilla de inferencia es 5 Hz, con precisión temporal limitada a 0,2 s.

Se compararon control nuevo, duración y anticipación de parada. En desarrollo, la media de
spellF1 condicional original/estrés fue 0,3052 en v2, 0,3846 en control, 0,3823 en duración,
0,3810 en duración+paradas y 0,3826 en HMM+paradas. Se congeló **control con HMM**, conservando
la pausa v2. La mayor complejidad no aportó una mejora media en este experimento.

El estrés usa oclusiones dependientes de aglomeración y velocidad, huecos largos, saltos del
balón, pérdidas de tracklets e intercambios de slots. Es determinista, no usa etiquetas para
generar ruido y no pretende reproducir exactamente los errores del concurso.

## Evaluación reservada y límites de las cifras

| Partido | Escenario | spellF1 cond. v2 | spellF1 cond. v3 | Acuerdo H/A v2 | Acuerdo H/A v3 |
|---|---|---:|---:|---:|---:|
| 1927964 | Original | 0,3299 | 0,4194 | 0,8116 | 0,8230 |
| 1927964 | Estrés | 0,3761 | 0,4477 | 0,7184 | 0,7326 |
| 1953632 | Original | 0,3148 | 0,3817 | 0,7739 | 0,7905 |
| 1953632 | Estrés | 0,3390 | 0,3956 | 0,6506 | 0,6659 |

La media de mejora es +7,12 puntos porcentuales de spellF1 condicional y +1,44 de acuerdo H/A.
SpellF1 condicional aplica el algoritmo oficial dentro de regiones H/A conocidas: sus X de
referencia recortan el dominio y pueden recortar posesiones. No puntúa D completo ni representa
el spellF1 oficial. Más estrés puede reducir fragmentación y subir esta métrica aunque baje
el acuerdo por frame; no debe interpretarse como que perder observaciones mejora la precisión.
Solo dos partidos reservados de una liga no permiten afirmar significancia o generalización.

Metrica Game 2, original: macroF1 de apoyo débil H/A/D pasó 0,7187→0,7554. Con el nuevo estrés,
0,6361→0,6631. En ambas condiciones el acuerdo en anclas H/A cayó **0,46 puntos porcentuales**
(original 95,45%→94,99%; estrés 79,20%→78,74%). D conserva exactamente el rendimiento v2:
en original su precisión es 99,05%, recall 35,48% y F1 52,24% sobre apoyo débil. No se afirma
que todos los indicadores mejoren ni que esté resuelto el juego detenido.

La candidata pasó los criterios de promoción fijados antes del test. Se conserva la elección
sin reajustes posteriores al test. Para otra ronda de selección habrá que reservar otros partidos,
sin reutilizar estos como un test ciego nuevo. `outputs/experimento_v3_01/` guarda el protocolo,
selección previa, modelo congelado e informe público externo.

## Entrega y revisión local

`reproducir_entrega.py` admite modelos v1/v2/v3. `mtah.entrega` separa el CSV y registros privados
del código publicable, usa listas positivas de archivos e incluye créditos y procedencia.
La candidata nueva se crea en un destino nuevo, conservando `outputs/entrega_veriana_v2_01/`.

El documento técnico v3 es `docs/ENTREGA_TECNICA_V3.md`. `scripts/informe_equipo.py` genera una
revisión HTML/Markdown offline y un ZIP para el equipo exclusivamente desde resultados de
otros partidos públicos y código, sin CSV o diagnósticos del concurso.

No se entrenó ni ajustó con los cinco minutos oficiales, no se crearon anotaciones manuales
y no se enviaron datos restringidos a servicios externos. Validar el CSV y reproducirlo demuestra
que la ejecución funciona; la calidad oficial de sus etiquetas sigue sin medirse.
