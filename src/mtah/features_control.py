"""Documentación pública en README.md."""
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from .etapa_a import tramos
from .preprocesamiento import features_smart_tagging, validar_arrays

@dataclass(frozen=True)
class ConfigFeatures:
    fps: float = 25.0
    largo: float = 105.3
    ancho: float = 68.0
    max_balon_ms: float = 45.0
    max_jugador_ms: float = 13.0
    radio_apoyo_m: float = 3.0
    continuidad_s: float = 0.2

def bloques(mask, grupos=None):
    """Documentación pública en README.md."""
    mask = np.asarray(mask, dtype=bool)
    if grupos is not None and np.asarray(grupos).shape != mask.shape:
        raise ValueError('Grupos temporales incompatibles')
    result = []
    for first, stop in tramos(mask):
        cuts = [] if grupos is None else (np.flatnonzero(np.asarray(grupos)[first + 1:stop] != np.asarray(grupos)[first:stop - 1]) + first + 1).tolist()
        edges = [first, *cuts, stop]
        result.extend(zip(edges[:-1], edges[1:]))
    return result

def _media(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(x, axis=axis)

def _velocidad(xy, segments, fps, max_speed):
    """Documentación pública en README.md."""
    out = np.full_like(xy, np.nan, dtype=np.float64)
    bad = np.zeros(xy.shape[:-1], bool)
    for first, stop in segments:
        delta = np.diff(xy[first:stop], axis=0) * fps
        finite = np.isfinite(delta).all(axis=-1)
        jump = finite & (np.linalg.norm(delta, axis=-1) > max_speed)
        bad[first + 1:stop] |= jump
        bad[first:stop - 1] |= jump
        delta[~finite | jump] = np.nan
        out[first + 1:stop] = delta
    return (out, bad)

def extraer(atd, eventos=(), calif=None, grupos=None, config=None, homografia=None):
    config = config or ConfigFeatures()
    if config.fps <= 0 or config.max_balon_ms <= 0 or config.max_jugador_ms <= 0:
        raise ValueError('Configuración física inválida')
    validar_arrays(atd)
    ball, players = (np.asarray(atd['balon'], float), np.asarray(atd['jugadores'], float))
    teams = np.asarray(atd['equipo_slot'])
    n = len(ball)
    calif = np.ones(n, bool) if calif is None else np.asarray(calif, bool)
    if calif.shape != (n,):
        raise ValueError('Máscara temporal incompatible')
    segments = bloques(calif, grupos)
    bxy, pxy = (ball[:, :2], players[:, :, :2])
    seen = np.isfinite(bxy).all(axis=1)
    bv, ball_jump = _velocidad(bxy, segments, config.fps, config.max_balon_ms)
    pv, player_jump = _velocidad(pxy, segments, config.fps, config.max_jugador_ms)
    usable = seen & ~ball_jump
    distance = np.linalg.norm(pxy - bxy[:, None], axis=2)
    distance[~usable] = np.nan
    data = dict(visible=seen, balon_util=usable, salto_balon=ball_jump, bx=bxy[:, 0], by=bxy[:, 1], v_balon=np.linalg.norm(bv, axis=1), jugadores_observados=np.isfinite(pxy).all(axis=2).sum(axis=1), saltos_slots=player_jump.sum(axis=1))
    player_speed = np.linalg.norm(pv, axis=2)
    speed_available = np.isfinite(player_speed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        data['v50_jugadores'] = np.nanmedian(player_speed, axis=1)
        data['v75_jugadores'] = np.nanpercentile(player_speed, 75, axis=1)
    count_speed = speed_available.sum(axis=1)
    data['n_velocidades_validas'] = count_speed
    data['fraccion_jugadores_moviles'] = np.divide(((player_speed > 1.4) & speed_available).sum(axis=1), count_speed, out=np.full(n, np.nan), where=count_speed > 0)
    turn = np.full(n, np.nan)
    for first, stop in segments:
        incoming, outgoing = (bv[first:stop - 1], bv[first + 1:stop])
        denominator = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            cosine = np.einsum('ij,ij->i', incoming, outgoing) / denominator
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        angle[denominator < 0.01] = np.nan
        turn[first:stop - 1] = angle
    data['giro_balon_grados'] = turn
    for team, side in ((0, 'H'), (1, 'A')):
        keep = teams == team
        valid_players = np.isfinite(pxy[:, keep]).all(axis=2)
        data['n_' + side] = valid_players.sum(axis=1)
        if not keep.any():
            nearest = np.full(n, -1)
            dmin, relative, persistence = (np.full(n, np.nan), np.full(n, np.nan), np.zeros(n))
            count = np.zeros(n, int)
            centroid = np.full((n, 2), np.nan)
        else:
            d = np.where(np.isfinite(distance[:, keep]), distance[:, keep], np.inf)
            local_slot = np.argmin(d, axis=1)
            dmin = d[np.arange(n), local_slot]
            nearest = np.flatnonzero(keep)[local_slot]
            nearest[~np.isfinite(dmin)] = -1
            dmin[~np.isfinite(dmin)] = np.nan
            relative = np.linalg.norm(bv - pv[np.arange(n), np.maximum(nearest, 0)], axis=1)
            relative[nearest < 0] = np.nan
            count = (d <= config.radio_apoyo_m).sum(axis=1)
            count = np.where(usable, count.astype(float), np.nan)
            centroid = _media(pxy[:, keep], axis=1)
            same = np.zeros(n, float)
            persistence = np.zeros(n, float)
            window = max(1, round(config.continuidad_s * config.fps))
            for first, stop in segments:
                same[first + 1:stop] = (nearest[first + 1:stop] == nearest[first:stop - 1]) & (nearest[first + 1:stop] >= 0)
                persistence[first:stop] = pd.Series(same[first:stop]).rolling(window, min_periods=1).mean().to_numpy()
        data['d_' + side], data['rel_' + side] = (dmin, relative)
        data['continuidad_' + side], data['apoyo_' + side] = (persistence, count)
        data['centro_x_' + side], data['centro_y_' + side] = (centroid[:, 0], centroid[:, 1])
        approach = np.full(n, np.nan)
        for first, stop in segments:
            approach[first + 1:stop] = -np.diff(dmin[first:stop]) * config.fps
        data['aproximacion_' + side] = approach
    data['margen_distancia'] = data['d_A'] - data['d_H']
    data['d_min'] = np.fmin(data['d_H'], data['d_A'])
    data['v_colectiva'] = _media(np.where(np.isfinite(pxy).all(axis=2), players[:, :, 2], np.nan), 1)
    data['fuera'] = usable & ((bxy[:, 0] < -0.5) | (bxy[:, 0] > config.largo + 0.5) | (bxy[:, 1] < -0.5) | (bxy[:, 1] > config.ancho + 0.5))
    previous, following = (np.full(n, np.inf), np.full(n, np.inf))
    for first, stop in segments:
        index = np.arange(stop - first)
        known = usable[first:stop]
        prev = np.maximum.accumulate(np.where(known, index, -np.inf))
        nex = np.minimum.accumulate(np.where(known, index, np.inf)[::-1])[::-1]
        previous[first:stop], following[first:stop] = ((index - prev) / config.fps, (nex - index) / config.fps)
    data['desde_balon_s'], data['hasta_balon_s'] = (previous, following)
    data['contexto_balon_s'] = np.minimum(previous, following)
    frame = pd.DataFrame(data)
    st = features_smart_tagging(eventos, n)
    frame = pd.concat([frame, st], axis=1)
    if homografia is not None:
        if len(homografia['valid']) != n:
            raise ValueError('Homografía de otra línea de tiempo')
        frame['h_valida'] = homografia['valid']
        frame['h_error'] = homografia['error']
    return frame
