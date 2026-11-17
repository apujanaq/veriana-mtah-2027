"""Documentación pública en README.md."""
import numpy as np

def _matrix(h):
    h = np.asarray(h, float)
    if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.norm(h) == 0:
        raise ValueError('Homografía inválida')
    h = h / np.linalg.norm(h)
    if np.linalg.cond(h) >= 1000000000000.0:
        raise ValueError('Homografía degenerada')
    return h

def campo_a_imagen(xy, h, largo, ancho):
    if largo <= 0 or ancho <= 0:
        raise ValueError('Campo inválido')
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('Coordenadas inválidas')
    p = np.c_[xy / [largo, ancho], np.ones(len(xy))] @ _matrix(h).T
    good = np.isfinite(p).all(1) & (abs(p[:, 2]) > 1e-12)
    out = np.full((len(xy), 2), np.nan)
    out[good] = p[good, :2] / p[good, 2, None]
    return out

def pies_a_campo(pixels, h, largo, ancho):
    if largo <= 0 or ancho <= 0:
        raise ValueError('Campo inválido')
    pixels = np.asarray(pixels, float)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError('Pies inválidos')
    p = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(_matrix(h)).T
    good = np.isfinite(p).all(1) & (abs(p[:, 2]) > 1e-12)
    out = np.full((len(pixels), 2), np.nan)
    out[good] = p[good, :2] / p[good, 2, None] * [largo, ancho]
    return out

def asociar_ejecutante(pies_pixels, h, largo, ancho, jugadores_xy, equipos, radio_m=2.0, margen_m=1.0):
    """Documentación pública en README.md."""
    xy = pies_a_campo(pies_pixels, h, largo, ancho)
    players, teams = (np.asarray(jugadores_xy, float), np.asarray(equipos))
    if players.ndim != 2 or players.shape[1] != 2 or teams.shape != (len(players),) or (not np.isin(teams, [0, 1]).all()):
        raise ValueError('Tracking de asociación inválido')
    if radio_m <= 0 or margen_m < 0:
        raise ValueError('Tolerancias inválidas')
    distance = np.linalg.norm(xy[:, None] - players[None], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    nearest = np.column_stack([distance[:, teams == k].min(1) if (teams == k).any() else np.full(len(xy), np.inf) for k in [0, 1]])
    good = np.isfinite(xy).all(1) & np.isfinite(nearest).all(1)
    team = nearest.argmin(1)
    margin = abs(nearest[:, 0] - nearest[:, 1])
    decided = good & (nearest.min(1) <= radio_m) & (margin >= margen_m)
    return np.where(decided, team, -1).astype(np.int8)
