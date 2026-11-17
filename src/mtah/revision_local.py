"""Documentación pública en README.md."""
import argparse
import json
from pathlib import Path
import numpy as np
from .carga import games
from .features_control import bloques
from .features import mascara_calificada
from .preprocesamiento import autorizar
from .validacion import pm, validar_csv

def resumir_prediccion(labels, mask=None, fps=25):
    labels = np.asarray(labels)
    mask = mascara_calificada(len(labels)) if mask is None else np.asarray(mask, bool)
    segments = bloques(mask)
    codes = pm.label_codes(labels)
    filled = pm.fill_abstentions(codes, segments)
    own_spells = pm.possession_spells(codes, segments)
    filled_spells = pm.possession_spells(filled, segments)
    length = np.array([stop - first for first, stop, _ in filled_spells], float) / fps
    return dict(conteos_calificados={label: int(((labels == label) & mask).sum()) for label in 'HADX'}, pct_X_calificado=float(100 * ((labels == 'X') & mask).sum() / mask.sum()) if mask.any() else 0.0, posesiones_con_X_como_hueco=len(own_spells), posesiones_con_X_propagado=len(filled_spells), mediana_posesion_predicha_s=float(np.median(length)) if len(length) else None, posesiones_predichas_menores_1s=int((length < 1.0).sum()), transiciones_predichas=len(pm.transitions(filled, segments)))

def texto_revision(eda, prediction, run):
    lines = ['# Revisión local del primer abordaje', '', '**RESTRINGIDO.** No enviar este archivo ni sus cifras a GitHub o servicios externos de IA.', '', 'Este resumen describe los inputs y la predicción. No contiene una evaluación de acierto: no se ha abierto ground truth del concurso, ni público ni privado.', '', '## 1. Tracking y cobertura', '']
    context = run.get('configuracion', {}).get('temporal', {}).get('contexto_max_s', 3.0)
    for name, source in eda.get('fuentes', {}).items():
        tracking = source.get('tracking')
        if not tracking:
            lines.append('- {}: tracking no disponible.'.format(name))
            continue
        lines.append('- **{}:** balón observado en {:.2f}% de la región analizada; {} slots. La región usada es: {}.'.format(name, tracking['pct_balon_visible_analizado'], tracking['slots'], source.get('region_analizada', 'indicada en el EDA')))
        gaps = tracking['huecos_s']
        if gaps.get('n'):
            lines.append('  Huecos sin observación: mediana {:.2f} s, p90 {:.2f} s, máximo {:.2f} s.'.format(gaps['p50'], gaps['p90'], gaps['max']))
            if gaps['p90'] > context:
                lines.append('  Parte de los huecos supera la ventana de contexto de esta versión. No deben resolverse interpolando posiciones indiscriminadamente; revisar la cobertura de tags y la abstención sin convertir ausencia en D.')
        lines.append('  Los huecos mezclan posibles oclusiones y paradas; esta estadística por sí sola no permite distinguirlas. Los saltos y duplicaciones son señales de calidad, no etiquetas.')
    lines += ['', '## 2. Smart Tagging', '']
    for name, source in eda.get('fuentes', {}).items():
        tags = source.get('smart_tagging')
        if not tags:
            continue
        lines.append('- {}: {} instancias; cobertura de prior {:.2f}%; conflictos {:.2f}%; contexto de saque {:.2f}%.'.format(name, tags['instancias'], tags['pct_cobertura_prior'], tags['pct_conflicto'], tags['pct_contexto_saque']))
        lines.append('  El prior no equivale a posesión. Los conflictos y equipos desconocidos se neutralizan. Las ventanas de saques no fijan por sí solas los frames de balón muerto.')
    lines += ['', '## 3. Homografías y vídeo', '']
    for name, source in eda.get('fuentes', {}).items():
        h, video = (source.get('homografia', {}), source.get('video', {}))
        if 'pct_utilizable' in h:
            lines.append('- {}: homografías utilizables {:.2f}%; {} filas sin medida de calidad disponible. La calidad ausente no se interpreta como error cero.'.format(name, h['pct_utilizable'], h.get('sin_medida_calidad', 0)))
        if 'muestras_codificadas' in video:
            lines.append('  Vídeo: {:.0f}×{:.0f}; {} muestras codificadas; coincidencia con el contrato publicado: {}.'.format(video['ancho_px'], video['alto_px'], video['muestras_codificadas'], 'sí' if video.get('coincide_con_frames_publicados') else 'no'))
        else:
            lines.append('  No se obtuvo una auditoría estructural completa del vídeo en esta fuente.')
    lines.append('El vídeo se ha auditado por metadatos. No se han medido cortes, replays o apariencia visual. Táctico y broadcast no se sincronizan ni se fusionan por número de fila.')
    lines += ['', '## 4. Predicción: diagnóstico sin verdad de referencia', '', '- Conteos en las regiones calificadas: ' + ', '.join(('{}={}'.format(k, v) for k, v in prediction['conteos_calificados'].items())) + '.', '- Abstención X en regiones calificadas: {:.2f}%.'.format(prediction['pct_X_calificado']), '- Posesiones predichas: {} contando X como hueco, {} propagando X dentro de cada mitad.'.format(prediction['posesiones_con_X_como_hueco'], prediction['posesiones_con_X_propagado']), '- Posesiones predichas menores de un segundo: {}.'.format(prediction['posesiones_predichas_menores_1s']), '- Transiciones predichas, tras propagación de X: {}.'.format(prediction['transiciones_predichas']), '', 'Estos conteos no indican si hay demasiadas o pocas posesiones reales: no conocemos sus etiquetas. Sirven para auditar fragmentación, continuidad y el coste de abstenerse. La propagación de X reproduce solo el tratamiento temporal del scorer sobre las mitades públicas; no descubre el dueño real.', '', '## 5. Integridad y próximos pasos', '', '- CSV validado sin GT: {}.'.format('sí' if run.get('csv_valido') else 'no confirmado'), '- Reproducción idéntica byte a byte: {}.'.format('sí' if run.get('reproduccion_identica') else 'no confirmada'), '- Modelo de control fijado con otro partido público; no se entrenó ni ajustó con este partido.', '- Detector D heurístico y posteriores sin calibración: no se afirma precisión ni mejora de spellF1.', '- Revisar los gráficos y diagnósticos localmente. No corregir etiquetas por frame ni crear anotaciones manuales.', '- Las mejoras se justifican con pruebas sintéticas, aprendizaje sin etiquetas o otros partidos permitidos; los cinco minutos oficiales no son un conjunto para ajustar o seleccionar versiones.', '']
    return '\n'.join(lines)

def crear_revision(folder, *, autorizar_datos=False):
    autorizar(autorizar_datos)
    folder = Path(folder)
    target = folder / 'REVISION_LOCAL.md'
    if target.exists():
        raise FileExistsError('La revisión ya existe')
    with (folder / 'estado_ejecucion.json').open(encoding='utf-8') as handle:
        run = json.load(handle)
    if not run.get('completado'):
        raise ValueError('La ejecución no está completa')
    with (folder / 'eda' / 'resumen.json').open(encoding='utf-8') as handle:
        eda = json.load(handle)
    csv = folder / 'equipo.csv'
    validar_csv(csv)
    labels = pm.read_label_csv(csv)['label'].to_numpy()
    prediction = resumir_prediccion(labels, fps=games.GAMES[games.GAME]['fps'])
    with target.open('x', encoding='utf-8') as handle:
        handle.write(texto_revision(eda, prediction, run))
    return target

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ejecucion', required=True)
    parser.add_argument('--autorizar-datos-restringidos', action='store_true')
    args = parser.parse_args(argv)
    if not args.autorizar_datos_restringidos:
        parser.error('Se requiere autorización para procesar los resultados locales')
    try:
        crear_revision(args.ejecucion, autorizar_datos=True)
    except (ValueError, OSError):
        print('No se creó la revisión; comprobar el estado de ejecución localmente')
        return 1
    print('Revisión interpretativa guardada localmente; no se imprime contenido del partido')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
