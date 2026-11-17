"""Documentación pública en README.md."""
from dataclasses import asdict
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from .boosting import log_odds_stumps
from .candidata_v3 import evidencia_v3, extraer_v3, matriz_v3
from .features import mascara_calificada
from .features_control import ConfigFeatures, _velocidad, bloques
from .idsse import SPLITS_DFL
from .mejora import _distancia_contexto
from .reanudaciones import evidencia_reanudacion, features_reanudacion
from .temporal_tasas_fijas import ConfigTemporal, _lse, decodificar
D_VARIABLES = ['v50', 'v75', 'fraccion_movil', 'v10', 'v90', 'desviacion_velocidad', 'v50_contexto_2s', 'variacion_v50', 'extension_x', 'extension_y', 'extension_min', 'extension_max', 'separacion_centros']

def velocidad_ventana(xy, segments, fps, max_speed):
    """Documentación pública en README.md."""
    raw, bad = _velocidad(xy, segments, fps, max_speed)
    lag = max(1, round(fps * 0.2))
    if lag == 1:
        return raw
    result = np.full_like(xy, np.nan, dtype=float)
    known = np.isfinite(xy).all(axis=-1) & ~bad
    for first, stop in segments:
        if stop - first <= lag:
            continue
        cumulative = np.concatenate([np.zeros((1, xy.shape[1]), int), np.cumsum(known[first:stop], axis=0)])
        contiguous = cumulative[lag + 1:] - cumulative[:-(lag + 1)] == lag + 1
        delta = (xy[first + lag:stop] - xy[first:stop - lag]) * fps / lag
        usable = contiguous & (np.linalg.norm(delta, axis=-1) <= max_speed)
        delta[~usable] = np.nan
        result[first + lag:stop] = delta
    return result

def extraer_v4(atd, events=(), mask=None, groups=None, cf=None):
    cf = cf or ConfigFeatures()
    mask = np.ones(len(atd['balon']), bool) if mask is None else np.asarray(mask, bool)
    segments = bloques(mask, groups)
    frame = extraer_v3(atd, events, mask, groups, cf)
    xy = atd['jugadores'][:, :, :2]
    pv = velocidad_ventana(xy, segments, cf.fps, cf.max_jugador_ms)
    speed = np.linalg.norm(pv, axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        frame['v10_slots'] = np.nanpercentile(speed, 10, axis=1)
        frame['v90_slots'] = np.nanpercentile(speed, 90, axis=1)
        frame['std_slots'] = np.nanstd(speed, axis=1)
        frame['v50_pausa'] = np.nanmedian(speed, axis=1)
        frame['v75_pausa'] = np.nanpercentile(speed, 75, axis=1)
        spread = np.nanstd(xy, axis=1)
        frame['extension_x'], frame['extension_y'] = (spread[:, 0] / cf.largo, spread[:, 1] / cf.ancho)
        for team, side in [(0, 'H'), (1, 'A')]:
            keep = atd['equipo_slot'] == team
            extension = np.nanstd(xy[:, keep], axis=1) if keep.any() else np.full((len(frame), 2), np.nan)
            frame['extension_' + side] = np.linalg.norm(extension / [cf.largo, cf.ancho], axis=1)
    frame['separacion_centros'] = np.hypot(frame.centro_x_H - frame.centro_x_A, frame.centro_y_H - frame.centro_y_A) / cf.largo
    count = np.isfinite(speed).sum(1)
    frame['fraccion_pausa'] = np.divide(((speed > 1.4) & np.isfinite(speed)).sum(1), count, out=np.full(len(frame), np.nan), where=count > 0)
    for name in ['v50_pausa', 'v75_pausa', 'fraccion_pausa']:
        for first, stop in segments:
            frame.loc[first:stop - 1, name + '_suave'] = pd.Series(frame[name].to_numpy()[first:stop]).rolling(max(1, round(cf.fps * 0.6)), center=True, min_periods=1).median().to_numpy()
    for name, values in [('v50_contexto_2s', frame.v50_pausa_suave.to_numpy()), ('variacion_v50', np.r_[np.nan, abs(np.diff(frame.v50_pausa_suave.to_numpy()))])]:
        smooth = np.full(len(frame), np.nan)
        for first, stop in segments:
            raw = values[first:stop].copy()
            if name == 'variacion_v50':
                raw[0] = np.nan
            smooth[first:stop] = pd.Series(raw).rolling(max(1, round(cf.fps * 2)), center=True, min_periods=1).mean()
        frame[name] = smooth
    future = xy + pv * 0.7
    tti = []
    for team in [0, 1]:
        distance = np.linalg.norm(future[:, atd['equipo_slot'] == team] - atd['balon'][:, None, :2], axis=2)
        distance[~np.isfinite(distance)] = np.inf
        time = 0.7 + distance.min(axis=1) / 5.0
        time[~np.isfinite(time)] = np.nan
        tti.append(time)
    frame['tti_ventaja'] = np.nan_to_num(np.tanh((tti[1] - tti[0]) / 1.5))
    frame['tti_proximidad'] = np.nan_to_num(np.exp(-tti[0]) - np.exp(-tti[1]))
    frame = pd.concat([frame, features_reanudacion(events, len(frame))], axis=1)
    from .etapa_a import tramos
    seeds = frame.observada_original.to_numpy() & ~frame.salto_balon.to_numpy() & (frame.v_balon.to_numpy() < 0.7) & (frame.d_min.to_numpy() < 2.0) & ((frame.by.to_numpy() < 1.5) | (frame.by.to_numpy() > cf.ancho - 1.5))
    geometric = np.zeros(len(frame), bool)
    for first, stop in segments:
        for a, b in tramos(seeds[first:stop]):
            geometric[max(first, first + a - round(cf.fps)):min(stop, first + b + round(cf.fps * 2))] = True
    frame['saque_banda'] = frame.saque_banda.to_numpy() | geometric
    frame['saque_contexto'] = frame.st_saque.to_numpy() | geometric
    return frame

def matriz_control_v4(frame):
    return np.column_stack([matriz_v3(frame), frame.tti_ventaja, frame.tti_proximidad])

def matriz_pausa_v4(frame):
    return np.column_stack([frame.v50_pausa_suave / 3, frame.v75_pausa_suave / 5, frame.fraccion_pausa_suave, frame.v10_slots / 1.4, frame.v90_slots / 6, frame.std_slots / 3, frame.v50_contexto_2s / 3, frame.variacion_v50 / 2, frame.extension_x, frame.extension_y, np.fmin(frame.extension_H, frame.extension_A), np.fmax(frame.extension_H, frame.extension_A), frame.separacion_centros])

def evidencia_v4(frame, atd, segments, model, cf):
    base = evidencia_v3(frame, segments, model['base_v3'], cf)
    logits = base['logits'].copy()
    options = model['opciones']
    if options['control_nuevo']:
        old = np.clip(matriz_v3(frame) @ np.asarray(model['base_v3']['control_v3']['coeficientes']) / model['base_v3']['temperatura_control'], -4, 4)
        new = np.clip(matriz_control_v4(frame) @ np.asarray(model['control_v4']['coeficientes']), -4, 4)
        gate = np.nan_to_num(np.exp(-np.square(frame.d_min.to_numpy() / 3.6))) * frame.calidad_geometria.to_numpy()
        gate[frame.fuera.to_numpy()] = 0
        logits[:, 0] += gate * (new - old)
        logits[:, 1] -= gate * (new - old)
    x = matriz_pausa_v4(frame)
    reliable = np.isfinite(x).all(1) & (frame.n_velocidades_validas.to_numpy() >= 6)
    pause = np.full(len(frame), np.nan)
    pause[reliable] = log_odds_stumps(x[reliable], model['pausa_v4']) / model['temperatura_pausa']
    proposed = _lse(logits[:, :2], axis=1) + pause
    amount = options['mezcla_pausa']
    logits[reliable, 2] = (1 - amount) * logits[reliable, 2] + amount * proposed[reliable]
    restart_owner = evidencia_reanudacion(frame, atd, segments, cf)
    for side in [0, 1]:
        logits[:, side] += options['peso_reanudacion'] * (restart_owner == side)
    support = base['soporte'] | reliable & (pause > 0) | (restart_owner >= 0)
    logits[~support] = 0
    return dict(logits=logits, soporte=support, pausa_log_odds=pause, movimiento_fiable=reliable, equipo_reanudacion=restart_owner)

def predecir_v4(atd, eventos=(), calif=None, grupos=None, model=None, config_features=None):
    if model is None or model.get('version') != 4:
        raise ValueError('Se requiere modelo externo v4')
    cf = config_features or ConfigFeatures()
    mask = mascara_calificada(len(atd['balon'])) if calif is None else np.asarray(calif, bool)
    segments = bloques(mask, grupos)
    frame = extraer_v4(atd, eventos, mask, grupos, cf)
    evidence = evidencia_v4(frame, atd, segments, model, cf)
    context = _distancia_contexto(evidence['soporte'], segments, cf.fps)
    result = decodificar(evidence['logits'], segments, evidence['soporte'], context, ConfigTemporal(fps=cf.fps))
    return dict(**result, feat=frame, evidencia=evidence, configuracion=dict(features=asdict(cf), opciones=model['opciones'], temperatura_pausa=model['temperatura_pausa'], modelo_temporal='HMM'))

def cargar_v4(path):
    model = json.loads(Path(path).read_text(encoding='utf-8'))
    origin = model.get('procedencia', {})
    if model.get('version') != 4 or origin.get('gt_concurso_usado') is not False or origin.get('idsse_article') != 28196177 or (origin.get('splits_dfl') != SPLITS_DFL):
        raise ValueError('Procedencia v4 no permitida')
    from .skillcorner import COMMIT_SC, SPLITS
    from .externos import COMMIT
    base = model.get('base_v3', {})
    base_origin = base.get('procedencia', {})
    if base.get('version') != 3 or base_origin.get('gt_concurso_usado') is not False or base_origin.get('skillcorner_commit') != COMMIT_SC or (base_origin.get('splits') != SPLITS):
        raise ValueError('Base v3 sin procedencia externa fijada')
    base_weights = np.asarray(base.get('control_v3', {}).get('coeficientes'), float)
    base2 = base.get('base_v2', {})
    if base_weights.shape != (9,) or not np.isfinite(base_weights).all() or base2.get('procedencia', {}).get('commit') != COMMIT or (base2.get('procedencia', {}).get('gt_concurso_usado') is not False):
        raise ValueError('Parámetros de base v3 incompatibles')
    for part in ['control', 'pausa']:
        v = np.asarray(base2.get(part, {}).get('coeficientes'), float)
        if v.shape != (4,) or not np.isfinite(v).all():
            raise ValueError('Base v2 inválida')
    x = np.asarray(model.get('control_v4', {}).get('coeficientes'), float)
    if x.shape != (11,) or not np.isfinite(x).all():
        raise ValueError('Control v4 inválido')
    pause = model.get('pausa_v4', {})
    if pause.get('n_variables') != len(D_VARIABLES) or not np.isfinite(pause.get('intercepto', np.nan)):
        raise ValueError('Pausa v4 inválida')
    if not isinstance(pause.get('arboles'), list) or len(pause['arboles']) > 1000:
        raise ValueError('Ensamble de pausa inválido')
    for tree in pause.get('arboles', []):
        if not 0 <= tree['feature'] < len(D_VARIABLES) or not all((np.isfinite(tree[k]) for k in ['threshold', 'left', 'right'])):
            raise ValueError('Árbol inválido')
    if model.get('temperatura_pausa') not in [0.5, 1.0, 2.0]:
        raise ValueError('Temperatura v4 inválida')
    options = model.get('opciones', {})
    if type(options.get('control_nuevo')) is not bool or options.get('mezcla_pausa') not in [0.5, 1.0] or options.get('peso_reanudacion') not in [0.0, 0.5]:
        raise ValueError('Opciones v4 fuera de protocolo')
    return model
