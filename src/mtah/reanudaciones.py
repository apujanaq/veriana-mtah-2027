"""Documentación pública en README.md."""
import numpy as np
import pandas as pd
from .preprocesamiento import _normalizar
TYPES = {'throw-in': 'banda', 'throw in': 'banda', 'corner kick': 'corner', 'goal kick': 'meta', 'free kick': 'falta', 'kick off (start)': 'centro', 'kick off (after goal)': 'centro'}

def features_reanudacion(events, n):
    votes = {kind: np.zeros(n + 1, int) for kind in ['banda', 'corner', 'meta', 'falta', 'centro', 'otro']}
    owners = np.zeros((n + 1, 2), int)
    for event in events:
        if _normalizar(event['code']) != 'set pieces':
            continue
        first, stop = (int(event['start_frame']), int(event['end_frame']))
        if not 0 <= first <= stop <= n:
            raise ValueError('Ventana de reanudación inválida')
        tags = {_normalizar(k): _normalizar(v) for k, v in event.get('tags', {}).items()}
        kind = TYPES.get(tags.get('type', ''), 'otro')
        votes[kind][first] += 1
        votes[kind][stop] -= 1
        team = event.get('equipo')
        if team in [0, 1]:
            owners[first, team] += 1
            owners[stop, team] -= 1
    active = {kind: np.cumsum(v)[:-1] > 0 for kind, v in votes.items()}
    team_active = np.cumsum(owners, axis=0)[:-1] > 0
    conflict = team_active.all(1)
    owner = np.where(team_active[:, 0] & ~conflict, 0, np.where(team_active[:, 1] & ~conflict, 1, -1))
    return pd.DataFrame({**{'saque_' + kind: mask for kind, mask in active.items()}, 'saque_equipo': owner, 'saque_conflicto': conflict})

def evidencia_reanudacion(frame, atd, segments, cf):
    """Documentación pública en README.md."""
    n = len(frame)
    owner = np.full(n, -1, np.int8)
    stationary = frame.observada_original.to_numpy() & ~frame.salto_balon.to_numpy() & (frame.v_balon.to_numpy() < 0.7)
    at_edge = (frame.by.to_numpy() < 1.5) | (frame.by.to_numpy() > cf.ancho - 1.5)
    candidates = frame.saque_banda.to_numpy() & stationary & at_edge & ~frame.saque_conflicto.to_numpy()
    dist_h, dist_a = (frame.d_H.to_numpy(), frame.d_A.to_numpy())
    clear_h = (dist_h < 1.8) & (dist_a - dist_h > 1.5)
    clear_a = (dist_a < 1.8) & (dist_h - dist_a > 1.5)
    candidate_team = np.where(clear_h & ~clear_a, 0, np.where(clear_a & ~clear_h, 1, -1))
    tag = frame['saque_contexto'].to_numpy() if 'saque_contexto' in frame else frame.st_saque.to_numpy()
    conflicts = frame.saque_conflicto.to_numpy()
    known_owners = frame.saque_equipo.to_numpy()
    vb, observed, outside = (frame.v_balon.to_numpy(), frame.observada_original.to_numpy(), frame.fuera.to_numpy())
    for first, stop in segments:
        inferred, last_seen = (-1, -np.inf)
        for i in range(first, stop):
            if not tag[i] or conflicts[i]:
                inferred = -1
                continue
            if candidates[i] and candidate_team[i] >= 0:
                inferred, last_seen = (int(candidate_team[i]), i)
            known = int(known_owners[i])
            team = known if known >= 0 else inferred if (i - last_seen) / cf.fps <= 3 else -1
            restarted = observed[i] and (not outside[i]) and np.isfinite(vb[i]) and (vb[i] > 3.0)
            if restarted and team >= 0:
                owner[i] = team
    return owner
