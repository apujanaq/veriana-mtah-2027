"""Documentación pública en README.md."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import ROOT
COMMIT_SC = '4340d274572876239c154c90bc507a9b3250a656'
SPLITS = dict(train=[1874553, 1886347, 1899585], dev=[1925299], test=[1927964, 1953632])

def verificar_skillcorner(folder):
    config = json.loads((ROOT / 'config/skillcorner_v3.json').read_text(encoding='utf-8'))
    if config['commit'] != COMMIT_SC or config['source'] != 'SkillCorner/opendata' or config['splits'] != SPLITS:
        raise ValueError('Corpus público diferente del protocolo fijado')
    for spec in config['files']:
        path = (Path(folder) / spec['path']).resolve()
        if not path.is_relative_to(Path(folder).resolve()):
            raise ValueError('Ruta de procedencia inválida')
        size = path.stat().st_size
        if spec['path'].endswith('.jsonl'):
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            content = f'version https://git-lfs.github.com/spec/v1\noid sha256:{digest.hexdigest()}\nsize {size}\n'.encode()
        else:
            content = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\x00' + content).hexdigest()
        if len(content) != spec['size'] or blob != spec['git_sha1']:
            raise ValueError('Archivo público alterado o incompleto')
    return config

def adaptar_registros(records, metadata):
    """Documentación pública en README.md."""
    n = len(records)
    if [r['frame'] for r in records] != list(range(n)):
        raise ValueError('Frames SkillCorner no contiguos desde cero')
    length, width = (float(metadata['pitch_length']), float(metadata['pitch_width']))
    if not 80 <= length <= 130 or not 45 <= width <= 100:
        raise ValueError('Dimensiones externas inválidas')
    home, away = (metadata['home_team']['id'], metadata['away_team']['id'])
    slots = {p['id']: k for k, p in enumerate(metadata['players'])}
    if len(slots) != len(metadata['players']):
        raise ValueError('IDs externos duplicados')
    teams = np.array([0 if p['team_id'] == home else 1 if p['team_id'] == away else -1 for p in metadata['players']], dtype=np.int8)
    if (teams < 0).any() or not (teams == 0).any() or (not (teams == 1).any()):
        raise ValueError('Afiliación externa desconocida')
    players, ball = (np.full((n, len(slots), 3), np.nan), np.full((n, 3), np.nan))
    periods = np.zeros(n, np.int8)
    refs = np.full(n, 'X', dtype='<U1')
    shift = np.array([length / 2, width / 2])
    extrapolated_ball = 0
    for i, row in enumerate(records):
        period = row.get('period')
        if period not in [None, 1, 2]:
            raise ValueError('Periodo externo desconocido')
        periods[i] = period or 0
        b = row['ball_data']
        if b.get('is_detected') is True and b.get('x') is not None and (b.get('y') is not None):
            ball[i, :2] = [b['x'], b['y']] + shift
        elif b.get('x') is not None:
            extrapolated_ball += 1
        used = set()
        for p in row['player_data']:
            pid = p['player_id']
            if pid not in slots or pid in used:
                raise ValueError('Slot externo desconocido o duplicado')
            used.add(pid)
            if p.get('is_detected') is True and p.get('x') is not None and (p.get('y') is not None):
                players[i, slots[pid], :2] = [p['x'], p['y']] + shift
        group = (row.get('possession') or {}).get('group')
        if group not in [None, 'home team', 'away team']:
            raise ValueError('Semántica de posesión externa desconocida')
        if period in [1, 2] and group is not None:
            refs[i] = 'H' if group == 'home team' else 'A'
    atd = dict(balon=ball, jugadores=players, equipo_slot=teams)
    audit = dict(frames=n, fps=10, largo=length, ancho=width, frames_calificados=int((periods > 0).sum()), balon_detectado=int(np.isfinite(ball[:, 0]).sum()), balon_extrapolado_descartado=extrapolated_ball, referencias={s: int((refs == s).sum()) for s in 'HADX'}, referencia='Posesión automática del proveedor; sin D explícito. Null permanece X.')
    return (atd, periods, refs, audit)

def cargar_skillcorner(folder, match_id):
    if match_id not in sum(SPLITS.values(), []):
        raise ValueError('Partido externo fuera del protocolo')
    base = Path(folder) / 'data/matches' / str(match_id)
    metadata = json.loads((base / f'{match_id}_match.json').read_text(encoding='utf-8'))
    with (base / f'{match_id}_tracking_extrapolated.jsonl').open(encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle]
    atd, periods, refs, audit = adaptar_registros(records, metadata)
    phases = pd.read_csv(base / f'{match_id}_phases_of_play.csv')
    phase_refs = np.full(len(refs), 'X', dtype='<U1')
    for row in phases.itertuples():
        a, b = (int(row.frame_start), int(row.frame_end))
        if not 0 <= a < b <= len(refs):
            raise ValueError('Fase externa fuera del tracking')
        if row.team_in_possession_id not in [metadata['home_team']['id'], metadata['away_team']['id']]:
            raise ValueError('Equipo de fase desconocido')
        side = 'H' if row.team_in_possession_id == metadata['home_team']['id'] else 'A'
        phase_refs[a:b] = side
    both = (refs != 'X') & (phase_refs != 'X')
    audit.update(partido=match_id, fases=len(phases), comparables_fase=int(both.sum()), desacuerdos_fase=int(((refs != phase_refs) & both).sum()))
    return (atd, periods, refs, audit)
