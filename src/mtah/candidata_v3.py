"""Documentación pública en README.md."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .control import matriz_control
from .duracion import decodificar_duracion
from .features import mascara_calificada
from .features_control import ConfigFeatures, _velocidad, bloques
from .mejora import ConfigMejora, _distancia_contexto, evidencia_mejora, extraer_mejora
from .skillcorner import COMMIT_SC, SPLITS
VARIABLES = ['margen', 'velocidad_relativa', 'continuidad', 'apoyo', 'aproximacion', 'proximidad', 'intercepcion_04', 'intercepcion_08', 'margen_contexto']

def extraer_v3(atd, events=(), mask=None, groups=None, cf=None):
    cf = cf or ConfigFeatures()
    mask = np.ones(len(atd['balon']), bool) if mask is None else np.asarray(mask, bool)
    segments = bloques(mask, groups)
    frame = extraer_mejora(atd, events, mask, groups, cf)
    ball = atd['balon'][:, :2]
    players = atd['jugadores'][:, :, :2]
    bv, bj = _velocidad(ball, segments, cf.fps, cf.max_balon_ms)
    pv, pj = _velocidad(players, segments, cf.fps, cf.max_jugador_ms)
    for horizon in [0.4, 0.8]:
        future_ball = ball + bv * horizon
        future_player = players + pv * horizon
        distance = np.linalg.norm(future_player - future_ball[:, None], axis=2)
        distance[~np.isfinite(distance) | pj | bj[:, None]] = np.inf
        nearest = []
        for team in [0, 1]:
            keep = atd['equipo_slot'] == team
            nearest.append(np.min(distance[:, keep], axis=1) if keep.any() else np.full(len(ball), np.inf))
        valid = np.isfinite(nearest[0]) & np.isfinite(nearest[1])
        difference = np.zeros(len(ball))
        difference[valid] = np.tanh((nearest[1][valid] - nearest[0][valid]) / 5.0)
        frame['intercepcion_' + str(horizon)] = difference
    smooth = np.zeros(len(frame))
    raw = np.tanh(frame.margen_distancia.to_numpy() / 3.0)
    for first, stop in segments:
        smooth[first:stop] = pd.Series(raw[first:stop]).rolling(max(1, round(cf.fps * 0.6)), center=True, min_periods=1).mean().fillna(0)
    frame['margen_contexto'] = smooth
    return frame

def matriz_v3(frame):
    approach = np.tanh((frame.aproximacion_H.to_numpy() - frame.aproximacion_A.to_numpy()) / 10.0)
    proximity = np.exp(-np.square(frame.d_H.to_numpy() / 2.0)) - np.exp(-np.square(frame.d_A.to_numpy() / 2.0))
    values = np.column_stack([matriz_control(frame), approach, proximity, frame['intercepcion_0.4'], frame['intercepcion_0.8'], frame.margen_contexto])
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

def ajustar_logistica(x, y, weights=None, regularizacion=0.02):
    """Documentación pública en README.md."""
    x, y = (np.asarray(x, float), np.asarray(y, float))
    if x.ndim != 2 or y.shape != (len(x),) or (not len(x)) or (not np.isfinite(x).all()) or (not np.isin(y, [0, 1]).all()) or (len(np.unique(y)) != 2):
        raise ValueError('Referencias de control inválidas')
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if w.shape != y.shape or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Pesos de entrenamiento inválidos')
    w = w / w.sum()
    beta = np.zeros(x.shape[1])

    def loss(b):
        score = x @ b
        return np.sum(w * (np.logaddexp(0, score) - y * score)) + regularizacion / 2 * (b @ b)
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        gradient = x.T @ (w * (p - y)) + regularizacion * beta
        hessian = x.T @ ((w * p * (1 - p))[:, None] * x) + regularizacion * np.eye(x.shape[1])
        direction = np.linalg.solve(hessian, gradient)
        step = 1.0
        current = loss(beta)
        while step > 1e-06 and loss(beta - step * direction) > current:
            step /= 2
        beta -= step * direction
        if np.linalg.norm(step * direction) < 1e-07:
            break
    return dict(coeficientes=beta.tolist(), variables=VARIABLES, regularizacion=regularizacion, referencias=len(y), intercepto=0.0)

def calibracion_binaria(scores, refs, temperature):
    scores, refs = (np.asarray(scores), np.asarray(refs, bool))
    p = 1 / (1 + np.exp(-np.clip(scores / temperature, -30, 30)))
    nll = np.mean(np.logaddexp(0, scores / temperature) - refs * scores / temperature)
    ece = 0.0
    for a, b in zip(np.arange(0, 1, 0.1), np.arange(0.1, 1.1, 0.1)):
        mask = (p >= a) & (p < b)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - refs[mask].mean())
    return dict(n=int(len(p)), nll=float(nll), brier=float(np.mean((p - refs) ** 2)), ece_10_bins=float(ece))

def evidencia_v3(frame, segments, model, cf):
    base = dict(model['base_v2'], control=dict(coeficientes=[0.0] * 4))
    evidence = evidencia_mejora(frame, segments, base, cf)
    logits = evidence['logits'].copy()
    odds = np.clip(matriz_v3(frame) @ np.asarray(model['control_v3']['coeficientes']) / model['temperatura_control'], -4, 4)
    quality = frame.calidad_geometria.to_numpy()
    gate = np.nan_to_num(np.exp(-np.square(frame.d_min.to_numpy() / 3.6))) * quality
    gate[frame.fuera.to_numpy()] = 0
    logits[:, 0] += gate * odds
    logits[:, 1] -= gate * odds
    positive = evidence['movimiento_fiable'] & (evidence['pausa_log_odds'] > 0)
    seeds = (frame.st_saque.to_numpy() | frame.fuera.to_numpy() | frame.observada_original.to_numpy() & (frame.v_balon.to_numpy() < 0.7)) & positive
    near_seed = _distancia_contexto(seeds, segments, cf.fps) <= model['opciones']['retro_parada_s']
    travelling = frame.observada_original.to_numpy() & ~frame.fuera.to_numpy() & (frame.v_balon.to_numpy() > 3)
    pause_support = positive & (near_seed | evidence['contexto_parada']) & ~travelling
    logits[pause_support, 2] += model['opciones']['peso_parada_extra'] * evidence['pausa_log_odds'][pause_support]
    support = evidence['soporte'] | pause_support
    logits[~support] = 0
    return dict(**{k: v for k, v in evidence.items() if k not in ['logits', 'soporte']}, logits=logits, soporte=support)

def predecir_v3(atd, eventos=(), calif=None, grupos=None, model=None, config_features=None):
    if model is None or model.get('version') != 3:
        raise ValueError('Se requiere modelo externo v3')
    cf = config_features or ConfigFeatures()
    mask = mascara_calificada(len(atd['balon'])) if calif is None else np.asarray(calif, bool)
    segments = bloques(mask, grupos)
    frame = extraer_v3(atd, eventos, mask, grupos, cf)
    evidence = evidencia_v3(frame, segments, model, cf)
    context = _distancia_contexto(evidence['soporte'], segments, cf.fps)
    if model['opciones']['duracion']:
        result = decodificar_duracion(evidence['logits'], segments, evidence['soporte'], context, cf.fps, model['duracion'], threshold=0.55)
    else:
        from .temporal_tasas_fijas import ConfigTemporal, decodificar
        result = decodificar(evidence['logits'], segments, evidence['soporte'], context, ConfigTemporal(fps=cf.fps))
    return dict(**result, feat=frame, evidencia=evidence, configuracion=dict(features=asdict(cf), opciones=model['opciones'], temperatura_control=model['temperatura_control'], duracion=model['duracion'] if model['opciones']['duracion'] else 'HMM_v2'))

def cargar_v3(path):
    model = json.loads(Path(path).read_text(encoding='utf-8'))
    origin = model.get('procedencia', {})
    if model.get('version') != 3 or origin.get('gt_concurso_usado') is not False or origin.get('skillcorner_commit') != COMMIT_SC or (origin.get('splits') != SPLITS):
        raise ValueError('Modelo v3 sin procedencia externa fijada')
    coefficients = np.asarray(model.get('control_v3', {}).get('coeficientes'), float)
    if coefficients.shape != (len(VARIABLES),) or not np.isfinite(coefficients).all():
        raise ValueError('Modelo de control v3 inválido')
    temperature = model.get('temperatura_control', 0)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperatura inválida')
    from .externos import COMMIT
    base_origin = model.get('base_v2', {}).get('procedencia', {})
    if base_origin.get('commit') != COMMIT or base_origin.get('gt_concurso_usado') is not False:
        raise ValueError('Base v2 sin procedencia fijada')
    for part in ['control', 'pausa']:
        v = np.asarray(model['base_v2'].get(part, {}).get('coeficientes'), float)
        if v.shape != (4,) or not np.isfinite(v).all():
            raise ValueError('Coeficientes base inválidos')
    duration = model.get('duracion', {})
    risks = np.asarray(duration.get('transferencia_por_s'), float)
    if duration.get('bordes_s') != [0.0, 0.5, 1.0, 2.0, 4.0, 8.0] or risks.shape != (6,) or (not np.isfinite(risks).all()) or (risks <= 0).any() or (duration.get('hz') != 5.0) or (duration.get('cola_s') != 8.0) or (duration.get('parada_por_s') != 0.1) or (duration.get('reanudacion_por_s') != 0.25):
        raise ValueError('Modelo de duraciones inválido')
    options = model.get('opciones', {})
    if type(options.get('duracion')) is not bool or options.get('peso_parada_extra') not in [0.0, 1.5] or options.get('retro_parada_s') not in [0.0, 0.8]:
        raise ValueError('Opciones fuera del protocolo externo')
    return model
