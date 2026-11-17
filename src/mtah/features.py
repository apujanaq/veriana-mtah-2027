"""Documentación pública en README.md."""
import warnings
import numpy as np
import pandas as pd
from .carga import A, H, games
from .parametros import p
SIN_EQUIPO = -1

def _nanmin_seguro(x, axis):
    """Documentación pública en README.md."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmin(x, axis=axis)

def _nanmean_seguro(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(x, axis=axis)

def cambio_direccion(xy, lag):
    """Documentación pública en README.md."""
    n = len(xy)
    v_in = np.full((n, 2), np.nan)
    v_out = np.full((n, 2), np.nan)
    v_in[lag:] = xy[lag:] - xy[:-lag]
    v_out[:-lag] = xy[lag:] - xy[:-lag]
    n_in = np.linalg.norm(v_in, axis=1)
    n_out = np.linalg.norm(v_out, axis=1)
    quieto = (n_in < 0.05) | (n_out < 0.05)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos = np.einsum('ij,ij->i', v_in, v_out) / (n_in * n_out)
        angulo = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        cambio_rel = np.abs(n_out - n_in) / np.maximum(n_in, n_out)
    angulo[quieto] = np.nan
    cambio_rel[quieto] = np.nan
    return (angulo, cambio_rel)

def calcular_features(atd, calif=None, pitch=None):
    """Documentación pública en README.md."""
    balon, jug, eq = (atd['balon'], atd['jugadores'], atd['equipo_slot'])
    bxy = balon[:, :2].astype(np.float64)
    visible = np.isfinite(bxy).all(axis=1)
    calif = mascara_calificada(len(bxy)) if calif is None else np.asarray(calif, dtype=bool)
    if calif.shape != (len(bxy),):
        raise ValueError('Máscara temporal de dimensiones inválidas')
    from .etapa_a import tramos
    bloques = tramos(calif)
    dist = np.linalg.norm(jug[:, :, :2] - balon[:, None, :2], axis=2)
    d_H = _nanmin_seguro(np.where(eq == H, dist, np.nan), axis=1)
    d_A = _nanmin_seguro(np.where(eq == A, dist, np.nan), axis=1)
    d_min = np.fmin(d_H, d_A)
    dh, da = (np.nan_to_num(d_H, nan=np.inf), np.nan_to_num(d_A, nan=np.inf))
    cercano = np.where(~np.isfinite(d_min) | (dh == da), SIN_EQUIPO, np.where(dh < da, H, A))
    velocidades = np.where(np.isfinite(jug[:, :, :2]).all(axis=2), jug[:, :, 2], np.nan)
    v_col = _nanmean_seguro(velocidades, axis=1)
    v_col_suave = np.full(len(bxy), np.nan)
    for a, b in bloques:
        v_col_suave[a:b] = pd.Series(v_col[a:b]).rolling(p('W_SUAVE_COLECTIVA'), center=True, min_periods=1).mean()
    largo, ancho = games.GAMES[games.GAME]['pitch'] if pitch is None else pitch
    m = p('MARGEN_FUERA')
    fuera = visible & ((bxy[:, 0] < -m) | (bxy[:, 0] > largo + m) | (bxy[:, 1] < -m) | (bxy[:, 1] > ancho + m))
    angulo, cambio_rel = (np.full(len(bxy), np.nan), np.full(len(bxy), np.nan))
    for a, b in bloques:
        angulo[a:b], cambio_rel[a:b] = cambio_direccion(bxy[a:b], p('LAG_DIRECCION'))
    fps = games.GAMES[games.GAME]['fps']
    v_balon = np.full(len(bxy), np.nan)
    v_balon_suave = np.full(len(bxy), np.nan)
    for a, b in bloques:
        v_balon[a + 1:b] = np.linalg.norm(np.diff(bxy[a:b], axis=0), axis=1) * fps
        v_balon_suave[a:b] = pd.Series(v_balon[a:b]).rolling(fps, center=True, min_periods=fps // 2).median()
    return pd.DataFrame({'visible': visible, 'bx': bxy[:, 0], 'by': bxy[:, 1], 'bs': balon[:, 2], 'd_H': d_H, 'd_A': d_A, 'd_min': d_min, 'equipo_cercano': cercano.astype(np.int8), 'v_col': v_col, 'v_col_suave': v_col_suave, 'fuera': fuera, 'angulo': angulo, 'cambio_rel': cambio_rel, 'v_balon': v_balon, 'v_balon_suave': v_balon_suave})

def mascara_calificada(n):
    """Documentación pública en README.md."""
    m = np.zeros(n, dtype=bool)
    for _, ini, fin in games.half_video_blocks(games.GAME):
        m[ini:fin + 1] = True
    return m
