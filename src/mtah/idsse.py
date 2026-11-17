"""Documentación pública en README.md."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from . import ROOT
SPLITS_DFL = dict(train=['J03WMX', 'J03WN1', 'J03WOH'], dev=['J03WOY'], test=['J03WPY', 'J03WQQ', 'J03WR9'])

def config_idsse():
    config = json.loads((ROOT / 'config/idsse_v4.json').read_text(encoding='utf-8'))
    if config['article'] != 28196177 or config['version'] != 1 or config['splits'] != SPLITS_DFL:
        raise ValueError('Corpus DFL fuera del protocolo')
    return config

def verificar_idsse(folder):
    config = config_idsse()
    for spec in config['files']:
        path = Path(folder) / spec['name']
        if Path(spec['name']).name != spec['name']:
            raise ValueError('Ruta pública inválida')
        digest = hashlib.md5()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != spec['size'] or digest.hexdigest() != spec['md5']:
            raise ValueError('Archivo DFL diferente al publicado')
    return config

def _timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()

def leer_metadata(path):
    root = ET.parse(path).getroot()
    general, environment = (root.find('.//General'), root.find('.//Environment'))
    if general is None or environment is None or general.get('Season') != '2022/2023':
        raise ValueError('Metadatos DFL no admitidos')
    teams = [general.get('HomeTeamId'), general.get('GuestTeamId')]
    players = {}
    for team in root.findall('.//Teams/Team'):
        if team.get('TeamId') not in teams:
            raise ValueError('Equipo DFL desconocido')
        side = teams.index(team.get('TeamId'))
        for p in team.iter('Player'):
            if p.get('PersonId') in players:
                raise ValueError('Jugador DFL duplicado')
            players[p.get('PersonId')] = side
    if not players or set(players.values()) != {0, 1}:
        raise ValueError('Plantillas DFL incompletas')
    return dict(match_id=general.get('MatchId'), largo=float(environment.get('PitchX')), ancho=float(environment.get('PitchY')), teams=teams, players=players)

def leer_posiciones(path, metadata, fps=5):
    if fps not in [5, 25]:
        raise ValueError('Frecuencia DFL no admitida')
    stride = 25 // fps
    slots = list(metadata['players'])
    slot_index = {p: i for i, p in enumerate(slots)}
    streams, ball_streams = ([], [])
    current, records = (None, [])
    period_map = {'firstHalf': 1, 'secondHalf': 2}
    for action, node in ET.iterparse(path, events=('start', 'end')):
        if action == 'start' and node.tag == 'FrameSet':
            current = dict(node.attrib)
            records = []
            if current.get('MatchId') != metadata['match_id']:
                raise ValueError('Tracking de otro partido')
        elif action == 'end' and node.tag == 'Frame':
            if current is None:
                raise ValueError('Frame fuera de FrameSet')
            team, person, section = (current.get('TeamId'), current.get('PersonId'), current.get('GameSection'))
            is_ball = (team or '').lower() == 'ball' or (person or '').lower() == 'ball'
            if section not in period_map:
                node.clear()
                continue
            if (is_ball or person in slot_index) and int(node.get('N')) % stride == 0:
                a = node.attrib
                speed = float(a.get('S', 'nan')) / 3.6
                row = (int(a['N']), float(a['X']), float(a['Y']), speed)
                if is_ball:
                    row += (_timestamp(a['T']), int(a.get('BallStatus', -1)), int(a.get('BallPossession', -1)))
                records.append(row)
            node.clear()
        elif action == 'end' and node.tag == 'FrameSet':
            if records:
                stream = (period_map[current['GameSection']], current.get('PersonId'), np.asarray(records, float))
                if (current.get('TeamId') or '').lower() == 'ball' or (current.get('PersonId') or '').lower() == 'ball':
                    ball_streams.append(stream)
                else:
                    streams.append(stream)
            current, records = (None, [])
            node.clear()
    if len(ball_streams) != 2 or {s[0] for s in ball_streams} != {1, 2}:
        raise ValueError('Faltan mitades de balón DFL')
    ball_streams.sort(key=lambda s: s[0])
    all_ball = np.concatenate([s[2] for s in ball_streams])
    periods = np.concatenate([np.full(len(s[2]), s[0], np.int8) for s in ball_streams])
    n = len(all_ball)
    players = np.full((n, len(slots), 3), np.nan)
    index = {}
    for i, (period, row) in enumerate(zip(periods, all_ball)):
        key = (int(period), int(row[0]))
        if key in index:
            raise ValueError('Frames DFL duplicados')
        index[key] = i
    for period, person, values in streams:
        if person not in slot_index:
            raise ValueError('Afiliación DFL desconocida')
        seen = set()
        for row in values:
            key = (period, int(row[0]))
            if key in seen:
                raise ValueError('Slot DFL duplicado')
            seen.add(key)
            if key in index:
                players[index[key], slot_index[person]] = row[1:4]
    for period in [1, 2]:
        mask = periods == period
        if not np.allclose(np.diff(all_ball[mask, 0]), stride) or not np.allclose(np.diff(all_ball[mask, 4]), 1 / fps, atol=0.005):
            raise ValueError('Reloj DFL irregular; no rellenar ni recortar')
    shift = np.array([metadata['largo'] / 2, metadata['ancho'] / 2])
    players[:, :, :2] += shift
    ball = all_ball[:, 1:4].copy()
    ball[:, :2] += shift
    refs = np.full(n, 'X', dtype='<U1')
    status, owner = (all_ball[:, 5].astype(int), all_ball[:, 6].astype(int))
    refs[status == 0] = 'D'
    refs[(status == 1) & (owner == 1)] = 'H'
    refs[(status == 1) & (owner == 2)] = 'A'
    atd = dict(balon=ball, jugadores=players, equipo_slot=np.array([metadata['players'][p] for p in slots], np.int8))
    return (atd, periods, refs, all_ball[:, 4])

def cargar_idsse(folder, match_id, fps=5):
    config = config_idsse()
    if match_id not in sum(SPLITS_DFL.values(), []):
        raise ValueError('Partido DFL fuera de partición')
    files = [s['name'] for s in config['files'] if s['name'].endswith(match_id + '.xml')]

    def path(kind):
        found = [f for f in files if kind in f]
        if len(found) != 1:
            raise ValueError('Archivo DFL ambiguo')
        return Path(folder) / found[0]
    meta = leer_metadata(path('matchinformation'))
    atd, periods, refs, times = leer_posiciones(path('positions_raw'), meta, fps)
    events = []
    for event in ET.parse(path('events_raw')).getroot().iter('Event'):
        kinds = [x.tag for x in event.iter() if x.tag in ['ThrowIn', 'GoalKick', 'CornerKick', 'FreeKick', 'KickOff', 'Penalty']]
        if not kinds:
            continue
        stamp = _timestamp(event.get('EventTime'))
        index = int(np.argmin(abs(times - stamp)))
        events.append(dict(type=kinds[0], frame=index, error_tiempo_s=float(abs(times[index] - stamp))))
    audit = dict(partido=match_id, frames=len(refs), fps=fps, referencias={s: int((refs == s).sum()) for s in 'HADX'}, etiquetas_como_features=False, eventos_como_features=False, largo=meta['largo'], ancho=meta['ancho'], reanudaciones=len(events))
    return (atd, periods, refs, events, audit)
