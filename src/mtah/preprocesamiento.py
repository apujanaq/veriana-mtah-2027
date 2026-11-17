"""Documentación pública en README.md."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import BadZipFile
import numpy as np
import pandas as pd
H, A, SIN_EQUIPO = (0, 1, -1)
VERSION_CACHE = 2

def autorizar(permiso):
    if permiso is not True:
        raise PermissionError('Se requiere autorización explícita antes de acceder a datos restringidos')

def _xml(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        raise ValueError('XML inválido; revisar el archivo localmente') from None
    for element in root.iter():
        element.tag = element.tag.rsplit('}', 1)[-1]
    return root

def equipo_por_slot(meta_path):
    """Documentación pública en README.md."""
    root = _xml(meta_path)
    teams = [t.get('id') for t in root.iter('Team')]
    if len(teams) != 2 or not all(teams) or len(set(teams)) != 2:
        raise ValueError('Los metadatos deben contener exactamente dos equipos distintos')
    mapping = {teams[0]: H, teams[1]: A}
    players = list(root.iter('Player'))
    if not players or any((p.get('teamId') not in mapping for p in players)):
        raise ValueError('Slots ausentes o con afiliación desconocida')
    return np.array([mapping[p.get('teamId')] for p in players], dtype=np.int8)

def leer_tracking(raw_path, equipos, n_frames, pitch):
    """Documentación pública en README.md."""
    equipos = np.asarray(equipos)
    if n_frames <= 0 or equipos.ndim != 1 or (not len(equipos)):
        raise ValueError('Dimensiones de tracking inválidas')
    if not np.isin(equipos, [H, A]).all():
        raise ValueError('Equipos de slots inválidos')
    scale = np.asarray([*pitch, 1.0], dtype=np.float32)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Dimensiones del campo inválidas')
    players = np.empty((n_frames, len(equipos), 3), dtype=np.float32)
    ball = np.empty((n_frames, 3), dtype=np.float32)
    count = 0
    with open(raw_path, encoding='utf-8-sig') as handle:
        for count, line in enumerate(handle, start=1):
            parts = line.strip().split(':')
            if count > n_frames or len(parts) != 3:
                raise ValueError('ATD: número o estructura de registros inválidos')
            try:
                frame_id = int(parts[0])
                groups = parts[1].split(';') + [parts[2]]
                if len(groups) != len(equipos) + 1 or any((len(g.split(',')) != 3 for g in groups)):
                    raise ValueError
                values = np.array([g.split(',') for g in groups], dtype=np.float32)
            except (ValueError, OverflowError):
                raise ValueError('ATD: frame_id o grupos x,y,s inválidos') from None
            if frame_id != count:
                raise ValueError('ATD: frame_id debe ser exactamente 1..N, en orden')
            if np.isinf(values).any():
                raise ValueError('ATD: infinitos no permitidos')
            values[~np.isfinite(values[:, :2]).all(axis=1)] = np.nan
            values[values[:, 2] < 0, 2] = np.nan
            with np.errstate(over='ignore'):
                values *= scale
            if np.isinf(values).any():
                raise ValueError('ATD: coordenadas fuera del rango numérico')
            players[count - 1], ball[count - 1] = (values[:-1], values[-1])
    if count != n_frames:
        raise ValueError('ATD: número de registros distinto del vídeo de referencia')
    return {'balon': ball, 'jugadores': players, 'equipo_slot': equipos.astype(np.int8)}

def validar_arrays(atd, n_frames=None):
    ball, players, teams = (np.asarray(atd[k]) for k in ('balon', 'jugadores', 'equipo_slot'))
    n = len(ball) if n_frames is None else n_frames
    if n <= 0 or ball.shape != (n, 3) or teams.ndim != 1 or (not len(teams)):
        raise ValueError('Arrays ATD: dimensiones inválidas')
    if players.shape != (n, len(teams), 3) or not np.isin(teams, [H, A]).all():
        raise ValueError('Arrays ATD: slots o equipos inválidos')
    if not np.issubdtype(ball.dtype, np.number) or not np.issubdtype(players.dtype, np.number):
        raise ValueError('Arrays ATD: coordenadas no numéricas')
    if np.isinf(ball).any() or np.isinf(players).any():
        raise ValueError('Arrays ATD: infinitos no permitidos')

def cargar_tracking(meta_path, raw_path, n_frames, pitch, cache_dir=None):
    """Documentación pública en README.md."""
    if cache_dir is None:
        return leer_tracking(raw_path, equipo_por_slot(meta_path), n_frames, pitch)
    config = json.dumps([VERSION_CACHE, n_frames, list(pitch)], sort_keys=True).encode()
    digest = hashlib.sha256(config)
    for path in (meta_path, raw_path):
        file_digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                file_digest.update(block)
        digest.update(file_digest.digest())
    key = digest.hexdigest()
    cache = Path(cache_dir) / ('atd_' + key + '.npz')
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                if z['firma'].item() != key:
                    raise ValueError
                out = {k: z[k] for k in ('balon', 'jugadores', 'equipo_slot')}
            validar_arrays(out, n_frames)
        except (ValueError, KeyError, OSError, EOFError, BadZipFile):
            raise ValueError('Caché inválida; ejecutar sin caché, sin borrar el archivo') from None
        return out
    out = leer_tracking(raw_path, equipo_por_slot(meta_path), n_frames, pitch)
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cache.open('xb') as handle:
            np.savez(handle, firma=np.array(key), **out)
    except FileExistsError:
        pass
    return out

def _normalizar(text):
    return ' '.join((text or '').strip().casefold().split())

def equipo_tag(text):
    return {'barcelona': H, 'fc barcelona': H, 'atletic club': A, 'athletic club': A}.get(_normalizar(text))

def leer_smart_tagging(path, n_frames, fps):
    """Documentación pública en README.md."""
    if n_frames <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Dimensiones temporales inválidas')
    events = []
    for instance in _xml(path).iter('instance'):
        try:
            start, end = (float(instance.findtext('start')), float(instance.findtext('end')))
        except (TypeError, ValueError):
            raise ValueError('Smart Tagging: timestamps inválidos') from None
        if not np.isfinite([start, end, start * fps, end * fps]).all() or end < start:
            raise ValueError('Smart Tagging: intervalo no finito o invertido')
        tags = {}
        for label in instance.findall('label'):
            group, value = (_normalizar(label.findtext('group')), (label.findtext('text') or '').strip())
            if group in tags and _normalizar(tags[group]) != _normalizar(value):
                raise ValueError('Smart Tagging: grupo con valores contradictorios')
            tags[group] = value
        first = min(n_frames, max(0, int(round(start * fps))))
        stop = min(n_frames, max(0, int(round(end * fps))))
        if stop <= first:
            continue
        events.append({'code': _normalizar(instance.findtext('code')).upper(), 'start_s': start, 'end_s': end, 'start_frame': first, 'end_frame': stop, 'tags': {k.title(): v for k, v in tags.items()}, 'equipo': equipo_tag(tags.get('team'))})
    return sorted(events, key=lambda e: (e['start_frame'], e['end_frame'], e['code']))
ATAQUE = {'build up', 'progression', 'creating chances', 'attacking transition', 'long ball', 'ball in final third', 'ball in the box', 'players in the box'}
DEFENSA = {'defending in defensive third', 'defending in middle third', 'defending in attacking third'}

def features_smart_tagging(events, n_frames):
    """Documentación pública en README.md."""
    votes = np.zeros((n_frames + 1, 2), dtype=np.int32)
    restart = np.zeros(n_frames + 1, dtype=np.int32)
    for event in events:
        first, stop = (int(event['start_frame']), int(event['end_frame']))
        if not 0 <= first <= stop <= n_frames:
            raise ValueError('Smart Tagging no alineado al vídeo de referencia')
        code, team = (_normalizar(event['code']), event.get('equipo'))
        if code == 'set pieces':
            restart[first] += 1
            restart[stop] -= 1
        if team not in (H, A) or code not in ATAQUE | DEFENSA:
            continue
        owner = 1 - team if code in DEFENSA else team
        votes[first, owner] += 1
        votes[stop, owner] -= 1
    active = np.cumsum(votes, axis=0)[:-1] > 0
    conflict = active.all(axis=1)
    owner = np.where(active[:, H] & ~conflict, H, np.where(active[:, A] & ~conflict, A, SIN_EQUIPO))
    return pd.DataFrame({'st_equipo': owner.astype(np.int8), 'st_conflicto': conflict, 'st_saque': np.cumsum(restart)[:-1] > 0})
