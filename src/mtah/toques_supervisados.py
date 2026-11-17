"""Documentación pública en README.md."""
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from . import ROOT
from .boosting import log_odds_stumps, sigmoid
MODELO_TOQUES = ROOT / 'modelos' / 'toques_micro_dfl_v1.json'
VERSION_VARIABLES = 'micro-multiventana-1'

def bloques_validos(mascara, grupos=None):
    m = np.asarray(mascara, bool)
    if m.ndim != 1:
        raise ValueError('Máscara de toques inválida')
    if not len(m):
        return []
    g = np.zeros(len(m), int) if grupos is None else np.asarray(grupos)
    if g.shape != m.shape:
        raise ValueError('Periodos de toques inválidos')
    edges = np.r_[True, (g[1:] != g[:-1]) | (m[1:] != m[:-1]), True]
    cuts = np.flatnonzero(edges)
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if a < b and m[a]]

def _desplazar(x, lag, bloques):
    """Documentación pública en README.md."""
    out = np.full_like(x, np.nan, dtype=float)
    for a, b in bloques:
        if abs(lag) >= b - a:
            continue
        if lag > 0:
            out[a:b - lag] = x[a + lag:b]
        elif lag < 0:
            out[a - lag:b] = x[a:b + lag]
        else:
            out[a:b] = x[a:b]
    return out

def _norm(x):
    return np.linalg.norm(x, axis=-1)

def extraer_variables(atd, calif, grupos=None, fps=25):
    """Documentación pública en README.md."""
    if fps != 25:
        raise ValueError('Este detector requiere tracking nativo a 25 Hz')
    ball = np.asarray(atd['balon'], float)[:, :2]
    players = np.asarray(atd['jugadores'], float)[:, :, :2]
    teams = np.asarray(atd['equipo_slot'])
    n = len(ball)
    if players.shape[0] != n or teams.shape != (players.shape[1],):
        raise ValueError('Tracking de toques incompatible')
    if any(((teams == k).sum() < 2 for k in (0, 1))):
        raise ValueError('Se necesitan dos slots por equipo')
    runs = bloques_validos(np.asarray(calif, bool) & np.isfinite(ball).all(axis=1), grupos)
    smooth = np.full_like(ball, np.nan)
    for a, b in runs:
        smooth[a:b] = pd.DataFrame(ball[a:b]).rolling(5, center=True, min_periods=1).median()
    distance = _norm(players - smooth[:, None])
    d, ids = ([], [])
    for k in (0, 1):
        ds = np.where((teams == k)[None] & np.isfinite(distance), distance, np.inf)
        order = np.argsort(ds, axis=1, kind='stable')[:, :2]
        d.append(np.take_along_axis(ds, order, axis=1))
        ids.append(order)
    d = np.stack(d, axis=1)
    d[~np.isfinite(d)] = np.nan
    ids = np.stack(ids, axis=1)
    valid = np.isfinite(d[:, :, 1]) & (d[:, :, 0] <= 6.0)
    matrices, names = ([], [])

    def add(name, values):
        values = np.asarray(values, float)
        if values.shape == (n,):
            values = np.repeat(values[:, None], 2, axis=1)
        if values.shape != (n, 2):
            raise ValueError('Variable de toques incompatible')
        names.append(name)
        matrices.append(values)
    add('distancia_1', d[:, :, 0])
    add('distancia_2', d[:, :, 1])
    add('distancia_rival', d[:, ::-1, 0])
    margin = d[:, ::-1, 0] - d[:, :, 0]
    add('ventaja_distancia', margin)
    for k_radius in (2.0, 4.0):
        count = np.stack([((teams == k)[None] & (distance < k_radius)).sum(axis=1) for k in (0, 1)], axis=1)
        add(f'apoyos_{k_radius:g}', count)
        add(f'balance_apoyos_{k_radius:g}', count - count[:, ::-1])
    row = np.arange(n)[:, None, None]
    for seconds in (0.4, 0.8, 1.0):
        lag = round(seconds * fps)
        prev, nxt = (_desplazar(smooth, -lag, runs), _desplazar(smooth, lag, runs))
        vi, vo = ((smooth - prev) / seconds, (nxt - smooth) / seconds)
        si, so = (_norm(vi), _norm(vo))
        turn = _norm(vo - vi)
        add(f'rapidez_antes_{seconds}', si)
        add(f'rapidez_despues_{seconds}', so)
        add(f'quiebre_{seconds}', turn)
        add(f'coseno_{seconds}', (vi * vo).sum(axis=1) / np.maximum(si * so, 0.1))
        add(f'quiebre_proximo_{seconds}', turn[:, None] / (1 + d[:, :, 0]))
        before, after = (_desplazar(d[:, :, 0], -lag, runs), _desplazar(d[:, :, 0], lag, runs))
        mi, mo = (_desplazar(margin, -lag, runs), _desplazar(margin, lag, runs))
        add(f'distancia_antes_{seconds}', before)
        add(f'distancia_despues_{seconds}', after)
        add(f'minimo_local_{seconds}', np.minimum(before, after) - d[:, :, 0])
        add(f'ventaja_antes_{seconds}', mi)
        add(f'ventaja_despues_{seconds}', mo)
        add(f'ventaja_transitoria_{seconds}', margin - np.maximum(mi, mo))
        pi, po = (_desplazar(players, -lag, runs), _desplazar(players, lag, runs))
        pvi, pvo = ((players - pi) / seconds, (po - players) / seconds)
        for j in (0, 1):
            slot = ids[:, :, j:j + 1]
            vin, vout = (pvi[row, slot][:, :, 0], pvo[row, slot][:, :, 0])
            add(f'relativa_antes_v{j + 1}_{seconds}', _norm(vin - vi[:, None]))
            add(f'relativa_despues_v{j + 1}_{seconds}', _norm(vout - vo[:, None]))
        collective = []
        for k in (0, 1):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                collective.append(np.nanmedian(_norm(pvo[:, teams == k]), axis=1) - np.nanmedian(_norm(pvi[:, teams == k]), axis=1))
        collective = np.stack(collective, axis=1)
        add(f'reaccion_equipo_{seconds}', collective)
        add(f'reaccion_relativa_{seconds}', collective - collective[:, ::-1])
        if seconds == 0.4:
            valid &= np.isfinite(vi).all(axis=1)[:, None] & np.isfinite(vo).all(axis=1)[:, None]
    x = np.stack(matrices, axis=2)
    add_missing = (~np.isfinite(x)).mean(axis=2)
    x = np.concatenate([np.nan_to_num(x, nan=0.0, posinf=40, neginf=-40), add_missing[:, :, None]], axis=2)
    return (np.clip(x, -80, 80).astype(np.float32), valid, names + ['fraccion_sin_soporte'])

def eventos_micro_referencia(ref, grupos, max_frames=50):
    """Documentación pública en README.md."""
    ref = np.asarray(ref)
    events = set()
    for a, b in bloques_validos(np.isin(ref, ['H', 'A']), grupos):
        cuts = np.r_[a, a + 1 + np.flatnonzero(ref[a + 1:b] != ref[a:b - 1]), b]
        for j in range(1, len(cuts) - 2):
            left, right = cuts[j:j + 2]
            if right - left < max_frames and ref[cuts[j - 1]] == ref[right]:
                events.add((int(left), int(ref[left] == 'A')))
                events.add((int(right), int(ref[right] == 'A')))
    return sorted(events)

def picos_probabilidad(prob, valid, grupos=None, separacion=8, umbral=0.0):
    """Documentación pública en README.md."""
    prob = np.asarray(prob, float)
    out = np.zeros_like(prob)
    for k in (0, 1):
        for a, b in bloques_validos(valid[:, k], grupos):
            z = prob[a:b, k]
            keep = np.ones(len(z), bool)
            for local in np.argsort(-z, kind='stable'):
                if z[local] < umbral:
                    break
                if keep[local]:
                    out[a + local, k] = z[local]
                    keep[max(0, local - separacion):min(len(z), local + separacion + 1)] = False
    return out

class DetectorToquesSupervisado:
    """Documentación pública en README.md."""

    def __init__(self, modelo=None):
        self.modelo = json.loads(MODELO_TOQUES.read_text(encoding='utf-8')) if modelo is None else json.loads(Path(modelo).read_text(encoding='utf-8')) if isinstance(modelo, (str, Path)) else modelo
        if self.modelo['version_variables'] != VERSION_VARIABLES:
            raise ValueError('Versión de variables de toque incompatible')
        protocol = self.modelo.get('protocolo', {})
        if protocol.get('train') != ['J03WMX', 'J03WN1', 'J03WOH'] or protocol.get('dev') != ['J03WOY'] or protocol.get('test_usado') is not False or (protocol.get('etiquetas_como_variables') is not False):
            raise ValueError('Procedencia del detector incompatible con el empalme')

    def probabilidades_toque(self, datos):
        key = 'toques_supervisados_' + hashlib.sha256(json.dumps(self.modelo, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
        extra = getattr(datos, 'extra', None)
        if extra is not None and key in extra:
            return extra[key]
        x, valid, names = extraer_variables(datos.atd, datos.calif, datos.grupos)
        if names != self.modelo['variables']:
            raise ValueError('Variables de toque incompatibles')
        raw = sigmoid(log_odds_stumps(x.reshape(-1, x.shape[2]), self.modelo['clasificador']))
        raw = raw.reshape(len(x), 2)
        raw[~valid] = 0.0
        out = picos_probabilidad(raw, valid, datos.grupos, separacion=self.modelo['separacion_frames'], umbral=self.modelo['umbral_busqueda'])
        if extra is not None:
            out.flags.writeable = False
            extra[key] = out
        return out
