"""Documentación pública en README.md."""
import hashlib
import json
import numpy as np
from .boosting import log_odds_stumps
from .toques_supervisados import extraer_variables, bloques_validos

def decodificar_ha(z, runs, pena=4.0, escala=10.0, preferencia=None):
    """Documentación pública en README.md."""
    z = np.asarray(z, float)
    if not np.isfinite(z).all() or pena < 0 or escala <= 0:
        raise ValueError('Potenciales H/A inválidos')
    lab = np.full(len(z), 'X')
    for a, b in runs:
        if b <= a:
            continue
        delta = np.array([z[a], -z[a]]) / (2 * escala)
        if preferencia is not None:
            delta[int(preferencia[a] == 'A')] += 1e-09
        back = np.zeros((b - a, 2), np.int8)
        for t in range(a + 1, b):
            cand = np.array([[delta[0], delta[1] - pena], [delta[0] - pena, delta[1]]])
            back[t - a] = cand.argmax(axis=1)
            delta = cand.max(axis=1) + np.array([z[t], -z[t]]) / (2 * escala)
        state = int(delta.argmax())
        for t in range(b - 1, a - 1, -1):
            lab[t] = 'HA'[state]
            state = int(back[t - a, state])
    return lab

def potencial_control(datos, model):
    key = 'control_mv_' + hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
    if key in datos.extra:
        return datos.extra[key]
    x, valid, names = extraer_variables(datos.atd, datos.calif, datos.grupos)
    if names != model['variables'] or model['train'] != ['J03WMX', 'J03WN1', 'J03WOH'] or model['test_usado'] is not False:
        raise ValueError('Control multiventana fuera de protocolo')
    scores = log_odds_stumps(x.reshape(-1, x.shape[2]), model['clasificador']).reshape(len(x), 2)
    z = (scores[:, 0] - scores[:, 1]) / (2 * model['temperatura'])
    result = (z, valid.any(axis=1))
    datos.extra[key] = result
    return result

def refinar_control(base, z, valid, calif, grupos=None, pena=4.0, escala=10.0):
    base = np.asarray(base)
    out = base.copy()
    runs = bloques_validos(np.asarray(calif) & np.isin(base, ['H', 'A']), grupos)
    fallback = np.where(base == 'H', np.log(9.0), -np.log(9.0))
    evidence = np.where(valid, np.clip(z, -8, 8), fallback)
    decoded = decodificar_ha(evidence, runs, pena, escala, base)
    for a, b in runs:
        out[a:b] = decoded[a:b]
    return out
