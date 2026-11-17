"""Documentación pública en README.md."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .control import matriz_control, potenciales
from .features import mascara_calificada
from .features_control import ConfigFeatures, bloques, extraer
from .temporal_tasas_fijas import ConfigTemporal, decodificar

@dataclass(frozen=True)
class ConfigMejora:
    hueco_interpolable_s: float = 0.6
    peso_interpolado: float = 0.35
    suavizado_movimiento_s: float = 0.6
    min_velocidades: int = 6
    peso_pausa: float = 2.0
    borde_m: float = 1.5

def rellenar_huecos_cortos(atd, segments, config_features, config):
    ball = np.asarray(atd['balon']).copy()
    observed = np.isfinite(ball[:, :2]).all(1)
    filled = np.zeros(len(ball), bool)
    from .etapa_a import tramos
    for first, stop in segments:
        for a, b in tramos(~observed[first:stop]):
            a, b = (first + a, first + b)
            if a == first or b == stop or b - a > round(config.hueco_interpolable_s * config_features.fps):
                continue
            left, right = (ball[a - 1, :2], ball[b, :2])
            endpoints = np.stack([left, right])
            if ((endpoints < [0, 0]) | (endpoints > [config_features.largo, config_features.ancho])).any():
                continue
            speed = np.linalg.norm(right - left) * config_features.fps / (b - a + 1)
            if speed > config_features.max_balon_ms:
                continue
            alpha = np.arange(1, b - a + 1)[:, None] / (b - a + 1)
            ball[a:b, :2] = left + alpha * (right - left)
            ball[a:b, 2] = np.nan
            filled[a:b] = True
    return (dict(atd, balon=ball), observed, filled)

def extraer_mejora(atd, eventos=(), mask=None, grupos=None, config_features=None, config=None):
    cf, c = (config_features or ConfigFeatures(), config or ConfigMejora())
    n = len(atd['balon'])
    mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    segments = bloques(mask, grupos)
    prepared, observed, interpolated = rellenar_huecos_cortos(atd, segments, cf, c)
    frame = extraer(prepared, eventos, mask, grupos, cf)
    frame['observada_original'] = observed
    frame['interpolada'] = interpolated
    frame['calidad_geometria'] = np.where(observed, 1.0, np.where(interpolated, c.peso_interpolado, 0.0))
    frame.loc[frame.salto_balon, 'calidad_geometria'] = 0.0
    frame['visible'] = observed
    frame['balon_util'] = observed & ~frame.salto_balon.to_numpy()
    window = max(1, round(c.suavizado_movimiento_s * cf.fps))
    for name in ['v50_jugadores', 'v75_jugadores', 'fraccion_jugadores_moviles']:
        raw = frame[name].to_numpy()
        smooth = np.full(n, np.nan)
        for first, stop in segments:
            smooth[first:stop] = pd.Series(raw[first:stop]).rolling(window, center=True, min_periods=1).median()
        frame[name + '_suave'] = smooth
    return frame

def matriz_pausa(frame):
    return np.column_stack([np.ones(len(frame)), np.clip(frame.v50_jugadores_suave.to_numpy() / 3, 0, 3), frame.fraccion_jugadores_moviles_suave.to_numpy(), np.clip(frame.v75_jugadores_suave.to_numpy() / 5, 0, 3)])

def ajustar_pausa(frame, refs, stride=10, regularizacion=0.02, iteraciones=1600):
    x = matriz_pausa(frame)
    valid = (refs != 'X') & np.isfinite(x).all(1) & (frame.n_velocidades_validas.to_numpy() >= 6)
    valid &= np.arange(len(frame)) % stride == 0
    x, y = (x[valid], (refs[valid] == 'D').astype(float))
    if not len(y) or len(np.unique(y)) != 2:
        raise ValueError('Faltan referencias externas de ambas condiciones vivo/muerto')
    sample_weight = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    weights = np.zeros(4)
    for _ in range(iteraciones):
        probability = 1 / (1 + np.exp(-np.clip(x @ weights, -30, 30)))
        gradient = x.T @ ((probability - y) * sample_weight) / len(y)
        gradient[1:] += regularizacion * weights[1:]
        weights -= 0.2 * gradient
    return dict(coeficientes=weights.tolist(), variables=['intercepto', 'v50_div3', 'fraccion_movil', 'v75_div5'], referencias=int(len(y)), referencias_D=int(y.sum()), stride=stride, regularizacion=regularizacion, iteraciones=iteraciones)

def evidencia_mejora(frame, segments, model, config_features=None, config=None):
    cf, c = (config_features or ConfigFeatures(), config or ConfigMejora())
    geometry = frame.assign(balon_util=frame.balon_util.to_numpy() | frame.interpolada.to_numpy() & ~frame.salto_balon.to_numpy())
    base = potenciales(geometry, model=model['control'])
    logits = base['logits'].copy()
    quality = frame.calidad_geometria.to_numpy()
    for side in (0, 1):
        tag = 0.8 * (frame.st_equipo.to_numpy() == side)
        logits[:, side] = (logits[:, side] - tag) * quality + tag
    logits[:, 2] *= quality
    x = matriz_pausa(frame)
    reliable_motion = np.isfinite(x).all(1) & (frame.n_velocidades_validas.to_numpy() >= c.min_velocidades)
    pause = np.zeros(len(frame))
    pause[reliable_motion] = np.clip(x[reliable_motion] @ np.asarray(model['pausa']['coeficientes']), -3, 3)
    ball_speed = frame.v_balon.to_numpy()
    observed = frame.observada_original.to_numpy() & ~frame.salto_balon.to_numpy()
    inside = ~frame.fuera.to_numpy()
    travelling = observed & inside & np.isfinite(ball_speed) & (ball_speed > 3.0)
    still = observed & np.isfinite(ball_speed) & (ball_speed < 0.7)
    edge = (frame.bx.to_numpy() <= c.borde_m) | (frame.bx.to_numpy() >= cf.largo - c.borde_m) | (frame.by.to_numpy() <= c.borde_m) | (frame.by.to_numpy() >= cf.ancho - c.borde_m)
    seed = frame.st_saque.to_numpy() | frame.fuera.to_numpy() | edge & still
    contextual = np.zeros(len(frame), bool)
    for first, stop in segments:
        active = False
        for i in range(first, stop):
            if travelling[i] or (reliable_motion[i] and pause[i] < 0):
                active = False
            if seed[i] and reliable_motion[i] and (pause[i] > 0) and (not travelling[i]):
                active = True
            contextual[i] = active
    stop_support = reliable_motion & (contextual | still) & ~travelling
    logits[stop_support, 2] += c.peso_pausa * pause[stop_support]
    support = observed | frame.interpolada.to_numpy() | (frame.st_equipo.to_numpy() >= 0) | stop_support & (pause > 0)
    logits[~support] = 0
    return dict(logits=logits, soporte=support, pausa_log_odds=pause, contexto_parada=contextual, movimiento_fiable=reliable_motion, control=base['control'])

def _distancia_contexto(support, segments, fps):
    distance = np.full(len(support), np.inf)
    for first, stop in segments:
        index = np.arange(stop - first)
        valid = support[first:stop]
        previous = np.maximum.accumulate(np.where(valid, index, -np.inf))
        following = np.minimum.accumulate(np.where(valid, index, np.inf)[::-1])[::-1]
        distance[first:stop] = np.minimum(index - previous, following - index) / fps
    return distance

def predecir_mejora(atd, eventos=(), calif=None, grupos=None, model=None, config_features=None, config=None, config_temporal=None):
    if model is None or model.get('version') != 2:
        raise ValueError('Se requiere un modelo externo v2')
    cf, c, ct = (config_features or ConfigFeatures(), config or ConfigMejora(), config_temporal or ConfigTemporal())
    if not np.isclose(cf.fps, ct.fps):
        raise ValueError('Las etapas deben compartir fps')
    n = len(atd['balon'])
    mask = mascara_calificada(n) if calif is None else np.asarray(calif, bool)
    segments = bloques(mask, grupos)
    frame = extraer_mejora(atd, eventos, mask, grupos, cf, c)
    evidence = evidencia_mejora(frame, segments, model, cf, c)
    context = _distancia_contexto(evidence['soporte'], segments, cf.fps)
    decoded = decodificar(evidence['logits'], segments, evidence['soporte'], context, ct)
    return dict(**decoded, feat=frame, evidencia=evidence, modelo_control=model, configuracion=dict(features=asdict(cf), mejora=asdict(c), temporal=asdict(ct)))

def cargar_mejora(path):
    with Path(path).open(encoding='utf-8') as handle:
        model = json.load(handle)
    origin = model.get('procedencia', {})
    if model.get('version') != 2 or origin.get('fuente') != 'metrica-sports/sample-data' or origin.get('gt_concurso_usado') is not False or (origin.get('partido') != 'Sample_Game_1'):
        raise ValueError('Modelo v2 sin procedencia externa permitida')
    for part in ['control', 'pausa']:
        weights = np.asarray(model.get(part, {}).get('coeficientes'), float)
        if weights.shape != (4,) or not np.isfinite(weights).all():
            raise ValueError('Coeficientes v2 incompatibles')
    return model
