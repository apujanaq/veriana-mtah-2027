"""Documentación pública en README.md."""
import argparse
from collections import Counter
from dataclasses import replace
from html import escape
import json
from pathlib import Path
import numpy as np
from . import ROOT, carga
from .features import mascara_calificada
from .features_control import ConfigFeatures, bloques, extraer
from .fuentes import inspeccionar_mp4, leer_homografia
from .preprocesamiento import autorizar

def distribucion(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(n=0, ausentes=int(values.size))
    quantiles = np.percentile(finite, [0, 10, 50, 90, 99, 100])
    return dict(n=int(len(finite)), ausentes=int(values.size - len(finite)), **dict(zip(['min', 'p10', 'p50', 'p90', 'p99', 'max'], map(float, quantiles))))

def histograma(values, edges):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=edges)
    labels = ['{}–{}'.format(a, '∞' if np.isinf(b) else b) for a, b in zip(edges[:-1], edges[1:])]
    return dict(etiquetas=labels, conteos=counts.tolist())

def longitudes_huecos(visible, mask, fps=25, grupos=None):
    from .etapa_a import tramos
    return [(b - a) / fps for first, stop in bloques(mask, grupos) for a, b in tramos(~np.asarray(visible)[first:stop])]

def analizar_tracking(atd, frame, mask, fps=25, grupos=None):
    mask = np.asarray(mask, bool)
    visible = frame.visible.to_numpy()
    gaps = longitudes_huecos(visible, mask, fps, grupos)
    minute = max(1, round(60 * fps))
    coverage = [float(visible[a:min(a + minute, len(frame))].mean()) for a in range(0, len(frame), minute)]
    columns = ['d_H', 'd_A', 'margen_distancia', 'rel_H', 'rel_A', 'v_balon', 'v_colectiva', 'continuidad_H', 'continuidad_A', 'apoyo_H', 'apoyo_A', 'giro_balon_grados']
    stats = {name: distribucion(frame.loc[mask, name]) for name in columns}
    correlation = frame.loc[mask, columns[:7]].corr(min_periods=25).replace([np.inf, -np.inf], np.nan)
    return dict(frames=len(frame), slots=len(atd['equipo_slot']), slots_por_equipo={side: int((atd['equipo_slot'] == k).sum()) for k, side in enumerate(['H', 'A'])}, pct_balon_visible_total=float(100 * visible.mean()), pct_balon_visible_analizado=float(100 * visible[mask].mean()) if mask.any() else None, frames_analizados=int(mask.sum()), jugadores=distribucion(frame.loc[mask, 'jugadores_observados']), pct_frames_mas_de_22=float(100 * (frame.loc[mask, 'jugadores_observados'] > 22).mean()) if mask.any() else None, huecos_s=distribucion(gaps), hist_huecos=histograma(gaps, [0, 0.2, 1, 3, 5, 10, 30, 60, 120, np.inf]), pct_saltos_balon=float(100 * frame.loc[mask, 'salto_balon'].mean()) if mask.any() else None, cobertura_por_minuto=coverage, variables=stats, hist_distancias=histograma(frame.loc[mask, 'd_min'], [0, 0.5, 1, 1.5, 2, 3, 5, 10, 20, np.inf]), correlaciones={row: {col: float(value) if np.isfinite(value) else None for col, value in values.items()} for row, values in correlation.to_dict('index').items()}, notas=['Ausencias y saltos no se convierten en D.', 'Más de 22 slots observados es una señal de auditoría, no prueba de duplicación.', 'Correlación entre features no prueba control ni valida una etiqueta.', 'El lector sanea pares XY incompletos y velocidades negativas como ausentes.'])

def analizar_tags(events, frame, mask, fps=25):
    codes = Counter((e['code'] for e in events))
    teams = Counter(('H' if e.get('equipo') == 0 else 'A' if e.get('equipo') == 1 else 'sin_equipo' for e in events))
    types = Counter((e['tags'].get('Type', 'sin_tipo') for e in events if e['code'].casefold() == 'set pieces'))
    mask = np.asarray(mask, bool)
    return dict(instancias=len(events), codigos=dict(codes), equipos=dict(teams), tipos_saque=dict(types), duraciones_s=distribucion([(e['end_frame'] - e['start_frame']) / fps for e in events]), pct_cobertura_prior=float(100 * (frame.loc[mask, 'st_equipo'] >= 0).mean()) if mask.any() else None, pct_conflicto=float(100 * frame.loc[mask, 'st_conflicto'].mean()) if mask.any() else None, pct_contexto_saque=float(100 * frame.loc[mask, 'st_saque'].mean()) if mask.any() else None, notas=['Intervalos aproximados, no posesiones ni instantes de reanudación.', 'N/A y equipos desconocidos no se asignan por descarte.', 'Fases de defensa apoyan al rival; conflictos producen abstención.'])

def analizar_homografia(homography, fps=25, video=None):
    valid = homography['valid']
    result = dict(frames=len(valid), pct_declarada_valida=float(100 * homography['declarada_valida'].mean()), pct_utilizable=float(100 * valid.mean()), error=distribucion(homography['error'][valid]), condicion=distribucion(homography['condicion'][valid]), huecos_s=distribucion(longitudes_huecos(valid, np.ones(len(valid), bool), fps)), inconsistencias=int((homography['declarada_valida'] & ~valid).sum()))
    result['sin_medida_calidad'] = int((~homography.get('calidad_disponible', np.isfinite(homography['error']))).sum())
    if video and valid.any():
        matrices = homography['matrices'][valid]
        point = matrices @ np.array([0.5, 0.5, 1.0])
        with np.errstate(divide='ignore', invalid='ignore'):
            xy = point[:, :2] / point[:, 2:]
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < video['ancho_px']) & (xy[:, 1] >= 0) & (xy[:, 1] < video['alto_px'])
        result['pct_centro_campo_en_imagen'] = float(100 * inside.mean())
    result['notas'] = ['Transforma coordenadas normalizadas del césped a píxeles, no metros a píxeles.', 'El error de homografía no es una medida calibrada del error del ATD.', 'Un mapa válido no identifica el control del balón ni la sincronía entre cámaras.']
    return result

def _bar_chart(title, labels, counts):
    maximum = max([1, *counts])
    height = max(90, 28 * len(counts))
    rows = []
    for i, (label, count) in enumerate(zip(labels, counts)):
        y = 25 * i + 18
        width = max(0, 300 * count / maximum)
        rows.append(f'<text x="0" y="{y}">{escape(str(label))}</text><rect x="280" y="{y - 13}" width="{width:.1f}" height="17" fill="#2563a0"/><text x="{290 + width:.1f}" y="{y}">{count:.3g}</text>')
    return f'<h3>{escape(title)}</h3><svg role="img" aria-label="{escape(title)}" viewBox="0 0 720 {height}">' + ''.join(rows) + '</svg>'

def escribir_reporte(report, folder):
    """Documentación pública en README.md."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / 'resumen.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    sections = []
    for name, source in report.get('fuentes', {}).items():
        sections.append('<h2>' + escape(name) + '</h2>')
        tracking = source.get('tracking', {})
        for key, title in [('hist_huecos', 'Duración de huecos sin balón (segundos)'), ('hist_distancias', 'Distancia al jugador más cercano (metros)')]:
            if key in tracking:
                hist = tracking[key]
                sections.append(_bar_chart(title, hist['etiquetas'], hist['conteos']))
        if 'cobertura_por_minuto' in tracking:
            values = tracking['cobertura_por_minuto']
            points = ' '.join(('{:.1f},{:.1f}'.format(45 + 530 * i / max(1, len(values) - 1), 155 - 130 * value) for i, value in enumerate(values)))
            sections.append(f'<h3>Balón visible por minuto</h3><svg role="img" aria-label="Cobertura temporal del balón" viewBox="0 0 620 195"><text x="0" y="30">100%</text><text x="12" y="157">0%</text><line x1="45" y1="155" x2="575" y2="155" stroke="#8799a6"/><polyline points="{points}" fill="none" stroke="#2563a0" stroke-width="2"/><text x="45" y="185">0 min</text>' + f'<text x="520" y="185">{len(values)} min</text></svg>')
        tags = source.get('smart_tagging', {})
        if 'codigos' in tags:
            sections.append(_bar_chart('Instancias por código', list(tags['codigos']), list(tags['codigos'].values())))
        for modality in ('tracking', 'smart_tagging', 'homografia', 'video'):
            sections.append('<details open><summary>' + escape(modality) + '</summary><pre>' + escape(json.dumps(source.get(modality, {'estado': 'no disponible'}), indent=2, ensure_ascii=False, allow_nan=False)) + '</pre></details>')
    html = '<!doctype html><html lang="es"><head><meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">\n<title>EDA local — More Than a Hack</title><style>\nbody{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#172c3c;background:#fff}\nh1,h2,h3{line-height:1.25}h2{margin-top:48px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;padding:16px}\nsvg{width:100%;max-height:1200px}svg text{font:13px system-ui,sans-serif}summary{font-weight:600;cursor:pointer}\n.notice{background:#fff0cb;padding:16px;border-left:4px solid #946200}details{margin:16px 0}</style></head><body>\n<h1>EDA local: fuentes, calidad y oportunidades</h1><p class="notice">RESTRINGIDO: no subir este informe, sus cifras ni sus gráficos a GitHub o servicios externos de IA.</p>\n<p>Este informe no lee ground truth. Los resultados describen los inputs; no miden precisión de posesión.\nEl vídeo se audita por metadatos MP4: todavía no se han medido cortes, replays ni apariencia visual.\nTáctico y broadcast tienen relojes distintos y no se fusionan.</p>'
    html += ''.join(sections) + '</body></html>'
    with (folder / 'informe.html').open('x', encoding='utf-8') as handle:
        handle.write(html)

def correr(data_dir=None, sources=('tactical', 'broadcast'), *, autorizar_datos=False):
    autorizar(autorizar_datos)
    report = dict(version=1, acceso='solo_local', gt_leido=False, fuentes={})
    for source in sources:
        paths = carga.games.asset_paths(source=source, data_dir=data_dir)
        n = carga.games.source_frames(source)
        entry = dict(estado_archivos={key: Path(path).is_file() for key, path in paths.items()})
        report['fuentes'][source] = entry
        video = None
        if entry['estado_archivos']['video']:
            try:
                video = inspeccionar_mp4(paths['video'])
                video['coincide_con_frames_publicados'] = video['muestras_codificadas'] == n
                entry['video'] = video
            except (ValueError, OSError, struct_error):
                entry['video'] = dict(estado='metadatos no compatibles; requiere lector local adicional')
        config = ConfigFeatures()
        if source == 'broadcast' and video and video['fps_medio']:
            config = replace(config, fps=video['fps_medio'])
        mask = mascara_calificada(n) if source == 'tactical' else np.ones(n, bool)
        events = []
        if source == 'tactical' and entry['estado_archivos'].get('smart_tagging'):
            events = carga.cargar_smart_tagging(data_dir=data_dir, autorizar_datos=True)
        if entry['estado_archivos']['atd_meta'] and entry['estado_archivos']['atd_raw']:
            atd = carga.cargar_atd(source=source, data_dir=data_dir, usar_cache=False, autorizar_datos=True)
            frame = extraer(atd, events, mask, config=config)
            entry['tracking'] = analizar_tracking(atd, frame, mask, config.fps)
            entry['region_analizada'] = 'mitades tácticas publicadas' if source == 'tactical' else 'grabación broadcast completa; sin máscara de evaluación'
            if source == 'tactical' and entry['estado_archivos'].get('smart_tagging'):
                entry['smart_tagging'] = analizar_tags(events, frame, mask, config.fps)
            del atd, frame
        if entry['estado_archivos']['homography']:
            h = leer_homografia(paths['homography'], n)
            entry['homografia'] = analizar_homografia(h, config.fps, video)
    return report
from struct import error as struct_error

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--salida', required=True, help='Directorio nuevo para el informe local')
    parser.add_argument('--sources', nargs='+', choices=['tactical', 'broadcast'], default=['tactical', 'broadcast'])
    parser.add_argument('--autorizar-datos-restringidos', action='store_true')
    args = parser.parse_args(argv)
    if not args.autorizar_datos_restringidos:
        parser.error('Se requiere autorización explícita para el EDA local')
    if Path(args.salida).exists():
        parser.error('El destino ya existe')
    try:
        report = correr(args.data_dir, args.sources, autorizar_datos=True)
        if not any((any(source['estado_archivos'].values()) for source in report['fuentes'].values())):
            print('No se encontró el bundle; consultar DATA.md para obtenerlo')
            return 1
        escribir_reporte(report, args.salida)
    except (ValueError, OSError):
        print('FAIL: revisar los archivos localmente; no se imprime contenido restringido')
        return 1
    print('EDA guardado localmente; abrir informe.html en el equipo, sin compartirlo con servicios de IA')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
