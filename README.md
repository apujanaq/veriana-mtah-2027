# veriana · More Than a Hack 2027

Código de posesión por frame y experimentos externos reproducibles. Esta rama contiene una
publicación saneada basada en la revisión oficial `f8f2629`. No incorpora el historial de trabajo
del equipo, sus notebooks personales, datos restringidos ni predicciones del partido.

**Entrega del equipo: `juliana_control_pares`.** La D sale del bloque A (paradas ancladas a eventos del tracking
y del Smart Tagging). El canal H/A depende de la calidad del tracking del balón: en régimen ruidoso, como el del
partido del concurso, parte del control v4 decodificado con el HMM común (`alejo_v2_robusta`) y se refina con un
control multiventana y parejas verificadas, entrenados con datos externos. Documento técnico:
[docs/entrega/Veriana_entrega.md](docs/entrega/Veriana_entrega.md).

- [Documento técnico de la entrega](docs/entrega/Veriana_entrega.md).
- [Cierre, comparación y decisión](docs/14_CIERRE_Y_ENTREGA_EQUIPO.md).
- [Documento técnico de veriana](docs/veriana.md).
- [Primer detector de microposesiones](docs/13_MICROPOSESIONES_JULIANA.md).
- [Modelos y resultados externos](docs/revision_pares_juliana_01/).
- [Reglas oficiales](RULES.md), [entrega](SUBMISSION.md) y [datos](DATA.md).

## Instalación y comprobación

```powershell
python -m pip install -r requirements-pipeline.txt
python -m unittest discover -s tests -v
git diff --check
```

La extracción numérica, modelos y decodificadores usan NumPy/pandas. No hay API de IA en
la inferencia. La geometría visual está preparada; no hay detector de vídeo activado.

## Reproducir el ensayo externo

Los scripts descargar_metrica.py y descargar_idsse.py de scripts/ descargan los corpus públicos
fijados (consultar --help). El ajuste nuevo solo utiliza tres partidos DFL train y uno dev;
los antiguos tests no se vuelven a usar.

```powershell
python scripts/validar_microposesiones.py --aprox juliana_control_pares --data-dir data --json outputs/comparacion_nueva.json
python scripts/entrenar_toques_micro.py --dfl data/external/idsse/figshare_28196177 --salida outputs/toques_nuevo
python scripts/calibrar_pares_micro.py --dfl data/external/idsse/figshare_28196177 --salida outputs/pares_nuevo
python scripts/entrenar_control_multiventana.py --dfl data/external/idsse/figshare_28196177 --salida outputs/control_nuevo
python scripts/calibrar_control_temporal.py --entrenamiento outputs/control_nuevo --salida outputs/temporal_nuevo
```

Los destinos deben ser nuevos. Los modelos congelados están en modelos/. Los parámetros se
seleccionan en DFL dev. Metrica es desarrollo externo conocido, no test ciego.

## Generar la entrega

Con el bundle del concurso en `data/` (o con `MTAH_DATA_DIR` apuntando a él), fuera de Git:

```bash
PYTHONPATH=src:.:evaluation python scripts/generar_entrega.py --salida Veriana.csv --autorizar-datos-restringidos
```

El script nunca sobrescribe un archivo, valida el CSV con el validador oficial e imprime su SHA-256: dos ejecuciones
producen el mismo archivo. El CSV se envía únicamente por el canal oficial; **no se sube a este repositorio**. Los
cinco minutos oficiales no intervienen en el entrenamiento, el ajuste ni la selección.
