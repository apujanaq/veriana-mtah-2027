"""Documentación pública en README.md."""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from . import ROOT
from .control import ConfigControl, ajustar_control, guardar_modelo, matriz_control, potenciales
from .eda import analizar_tracking, escribir_reporte
from .features_control import ConfigFeatures, bloques, extraer
from .temporal_tasas_fijas import ConfigTemporal, decodificar
COMMIT = 'e706dd506b360d69d9d123d5b8026e7294b13996'
EXPECTED = {'data/Sample_Game_1/Sample_Game_1_RawEventsData.csv': {'size': 139446, 'git_sha': '3c426dd4f442dd433bccf89e3634f91aa310cd8f'}, 'data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv': {'size': 32782073, 'git_sha': 'ff409a5680fd75acc7cebcf786827214c687478b'}, 'data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv': {'size': 32782961, 'git_sha': 'ce2e91c0e633426cec1146f4e728b72417454e3d'}, 'data/Sample_Game_2/Sample_Game_2_RawEventsData.csv': {'size': 153897, 'git_sha': 'decccd47c1d53960d290cb6679337eec83a8a97f'}, 'data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv': {'size': 29627711, 'git_sha': 'e7334d8d762954a215c49e55f211855000243fdc'}, 'data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv': {'size': 31890338, 'git_sha': '8b4d3648888cc8b033e7142919ebaea04f6328e0'}}

def verificar_procedencia(folder):
    folder = Path(folder)
    with (folder / 'PROVENANCE.json').open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('source') != 'https://github.com/metrica-sports/sample-data' or manifest.get('commit') != COMMIT:
        raise ValueError('Se requiere el corpus público externo fijado, no el partido del concurso')
    supplied = {spec['path']: dict(size=spec['size'], git_sha=spec['git_sha']) for spec in manifest.get('files', [])}
    if supplied != EXPECTED:
        raise ValueError('El manifiesto no coincide con los seis archivos públicos fijados')
    for spec in manifest['files']:
        relative = Path(spec['path']).relative_to('data')
        if relative.parts[0] not in ['Sample_Game_1', 'Sample_Game_2'] or '..' in relative.parts:
            raise ValueError('Ruta externa no permitida')
        path = folder / relative
        size = path.stat().st_size
        digest = hashlib.sha1(('blob ' + str(size) + '\x00').encode())
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if size != spec['size'] or digest.hexdigest() != spec['git_sha']:
            raise ValueError('Los samples externos no coinciden con su procedencia')
    return manifest

def cargar_sample(folder, game):
    stem = 'Sample_Game_{}'.format(game)
    base = Path(folder) / stem
    tables = [pd.read_csv(base / (stem + '_RawTrackingData_' + side + '_Team.csv'), skiprows=2) for side in ['Home', 'Away']]
    home, away = tables
    for key in ['Frame', 'Period', 'Time [s]']:
        if not np.array_equal(home[key].to_numpy(), away[key].to_numpy()):
            raise ValueError('Tracking externo: relojes de equipos diferentes')
    frame_ids = home.Frame.to_numpy()
    if not np.array_equal(frame_ids, np.arange(1, len(home) + 1)):
        raise ValueError('Tracking externo: frames no contiguos desde 1')
    periods = home.Period.to_numpy()
    delta = np.diff(home['Time [s]'].to_numpy())
    same = periods[1:] == periods[:-1]
    positive = delta[same & (delta > 0)]
    fps = 1 / np.median(positive)
    if not np.isclose(fps, 25, atol=0.01):
        raise ValueError('Export externo con fps inesperado')
    xy = [table.iloc[:, 3:-2].to_numpy(float).reshape(len(table), -1, 2) * [105.0, 68.0] for table in tables]
    raw_ball = [table.iloc[:, -2:].to_numpy(float) for table in tables]
    both = np.isfinite(raw_ball[0]) & np.isfinite(raw_ball[1])
    if not np.allclose(raw_ball[0][both], raw_ball[1][both], atol=1e-06):
        raise ValueError('Tracking externo: balón inconsistente entre equipos')
    bxy = np.where(np.isfinite(raw_ball[0]), raw_ball[0], raw_ball[1]) * [105.0, 68.0]
    pxy = np.concatenate(xy, axis=1)
    speeds = np.full(pxy.shape[:-1], np.nan)
    for first, stop in bloques(np.ones(len(home), bool), periods):
        speeds[first + 1:stop] = np.linalg.norm(np.diff(pxy[first:stop], axis=0), axis=2) * fps
    speeds[speeds > 13] = np.nan
    atd = dict(balon=np.column_stack([bxy, np.full(len(home), np.nan)]), jugadores=np.concatenate([pxy, speeds[:, :, None]], axis=2), equipo_slot=np.r_[np.zeros(xy[0].shape[1], np.int8), np.ones(xy[1].shape[1], np.int8)])
    events = pd.read_csv(base / (stem + '_RawEventsData.csv'))
    return (atd, periods, float(fps), events)

def referencias(events, n):
    """Documentación pública en README.md."""
    votes = np.zeros((n, 2), np.int16)
    for _, event in events.iterrows():
        if event['Team'] not in ['Home', 'Away'] or event['Type'] not in ['PASS', 'RECOVERY', 'SHOT']:
            continue
        team = 0 if event['Team'] == 'Home' else 1
        points = [event['Start Frame']]
        if event['Type'] == 'PASS' and pd.notna(event['To']):
            points.append(event['End Frame'])
        for value in points:
            if pd.notna(value) and int(value) == value and (1 <= value <= n):
                votes[int(value) - 1, team] += 1
    conflict = (votes > 0).all(axis=1)
    labels = np.full(n, 'X', dtype='<U1')
    labels[(votes[:, 0] > 0) & ~conflict] = 'H'
    labels[(votes[:, 1] > 0) & ~conflict] = 'A'
    return labels

def puntuar_referencias(labels, refs):
    mask = refs != 'X'
    f1 = []
    for side in ['H', 'A']:
        tp = ((labels == side) & (refs == side) & mask).sum()
        fp = ((labels == side) & (refs != side) & mask).sum()
        fn = ((labels != side) & (refs == side) & mask).sum()
        f1.append(float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0)
    return dict(referencias=int(mask.sum()), acuerdo=float((labels[mask] == refs[mask]).mean()) if mask.any() else None, macroF1_referencias_HA=float(np.mean(f1)), advertencia='Referencias parciales de eventos; NO macroF1/spellF1 oficiales ni GT de todo el partido')

def degradar_balon(atd, fps, seed=2027, fraction=0.35):
    rng = np.random.default_rng(seed)
    ball = atd['balon'].copy()
    missing = np.zeros(len(ball), bool)
    while missing.mean() < fraction:
        first = rng.integers(len(ball))
        width = round(rng.choice([0.4, 1.0, 3.0, 10.0]) * fps)
        missing[first:min(first + width, len(ball))] = True
    ball[missing] = np.nan
    return dict(atd, balon=ball)

def desarrollar(folder, destination):
    provenance = verificar_procedencia(folder)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('El destino externo ya existe')
    summaries = dict(fuente=provenance['source'], commit=COMMIT, entrenamiento='Sample_Game_1', validacion='Sample_Game_2', uso_gt_concurso=False, resultados={})
    atd, periods, fps, events = cargar_sample(folder, 1)
    mask = np.isin(periods, [1, 2])
    frame = extraer(atd, calif=mask, grupos=periods, config=ConfigFeatures(fps=fps, largo=105.0, ancho=68.0))
    refs = referencias(events, len(frame))
    train = (refs != 'X') & frame.balon_util.to_numpy() & (frame.d_min.to_numpy() <= 5.0)
    model = ajustar_control(matriz_control(frame)[train], (refs[train] == 'H').astype(float))
    model['procedencia'] = dict(fuente='metrica-sports/sample-data', commit=COMMIT, partido='Sample_Game_1', etiquetas='referencias parciales de PASS/RECOVERY/SHOT', n=int(train.sum()), gt_concurso_usado=False)
    summaries['modelo_control'] = model
    summaries['configuracion'] = dict(control=asdict(ConfigControl()), temporal=asdict(ConfigTemporal()), features=asdict(ConfigFeatures(largo=105.0, ancho=68.0)), oclusiones=dict(seed=2027, fraccion=0.35, duraciones_s=[0.4, 1.0, 3.0, 10.0]))
    summaries['entorno'] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)
    summaries['codigo_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob('*.py')}
    public_eda = dict(version=1, fuentes={'Sample_Game_1': dict(tracking=analizar_tracking(atd, frame, mask, fps, periods))})
    del atd, frame, refs
    atd, periods, fps, events = cargar_sample(folder, 2)
    mask = np.isin(periods, [1, 2])
    refs = referencias(events, len(atd['balon']))
    for name, observed in [('original', atd), ('oclusiones_sinteticas', degradar_balon(atd, fps))]:
        frame = extraer(observed, calif=mask, grupos=periods, config=ConfigFeatures(fps=fps, largo=105.0, ancho=68.0))
        evidence = potenciales(frame, model=model)
        pointwise = np.array(['H', 'A', 'D'])[evidence['logits'].argmax(axis=1)]
        pointwise[~evidence['soporte']] = 'X'
        temporal = decodificar(evidence['logits'], bloques(mask, periods), evidence['soporte'], frame.contexto_balon_s.to_numpy(), ConfigTemporal(fps=fps))
        nearest = np.where(frame.d_H.fillna(np.inf) <= frame.d_A.fillna(np.inf), 'H', 'A')
        nearest[~frame.balon_util.to_numpy()] = 'X'
        summaries['resultados'][name] = {key: puntuar_referencias(value, refs) for key, value in [('vecino_cercano', nearest), ('control_sin_temporal', pointwise), ('control_temporal', temporal['etiquetas'])]}
        if name == 'original':
            public_eda['fuentes']['Sample_Game_2'] = dict(tracking=analizar_tracking(atd, frame, mask, fps, periods))
    destination.mkdir(parents=True, exist_ok=False)
    guardar_modelo(model, destination / 'control.json')
    with (destination / 'evaluacion_externa.json').open('x', encoding='utf-8') as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False, allow_nan=False)
    escribir_reporte(public_eda, destination / 'eda_samples')
    return summaries

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', default=str(ROOT / 'data' / 'external' / 'metrica' / COMMIT))
    parser.add_argument('--salida', default=str(ROOT / 'outputs' / 'desarrollo_externo_v1'))
    args = parser.parse_args(argv)
    report = desarrollar(args.sample_dir, args.salida)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
