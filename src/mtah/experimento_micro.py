"""Documentación pública en README.md."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .boosting import ajustar_stumps, log_odds_stumps, sigmoid
from .idsse import SPLITS_DFL, cargar_idsse, config_idsse
from .toques_supervisados import VERSION_VARIABLES, eventos_micro_referencia, extraer_variables, picos_probabilidad
from .validacion_externa import _ruido_ar1
RUIDOS = (0.0, 0.4, 0.8, 1.0)
UMBRALES = (0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95)

def verificar_particion(folder, split):
    """Documentación pública en README.md."""
    if split not in ('train', 'dev'):
        raise ValueError('El experimento de micro-posesiones solo permite train/dev')
    config = config_idsse()
    for spec in config['files']:
        if not any((spec['name'].endswith(game + '.xml') for game in SPLITS_DFL[split])):
            continue
        path = Path(folder) / spec['name']
        digest = hashlib.md5()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != spec['size'] or digest.hexdigest() != spec['md5']:
            raise ValueError('Archivo DFL diferente al manifiesto público')

def aumentar(atd, periods, sd, seed):
    """Documentación pública en README.md."""
    ball = np.asarray(atd['balon'], float).copy()
    for j, per in enumerate(np.unique(periods)):
        m = periods == per
        ball[m, :2] += _ruido_ar1(int(m.sum()), sd, 0.4, seed + j, fps=25)
    return dict(atd, balon=ball)

def objetivos(n, events, periods, radio=3):
    y = np.zeros((n, 2), bool)
    for frame, side in events:
        a, b = (max(0, frame - radio), min(n, frame + radio + 1))
        y[a:b, side] |= periods[a:b] == periods[frame]
    return y

def contar_eventos(prob, events, threshold, valid, periods, tolerancia=12):
    peaks = picos_probabilidad(prob, valid, periods, separacion=8, umbral=threshold)
    frames, sides = np.nonzero(peaks >= threshold)
    pred = sorted(zip(frames, sides), key=lambda ev: -peaks[ev])
    used = set()
    for f, k in pred:
        candidates = [(abs(f - g), j) for j, (g, kk) in enumerate(events) if kk == k and j not in used and (periods[f] == periods[g]) and (abs(f - g) <= tolerancia)]
        if candidates:
            used.add(min(candidates)[1])
    return dict(tp=len(used), pred=len(pred), ref=len(events))

def entrenar(folder, destino, progreso=lambda msg: None):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=False)
    verificar_particion(folder, 'train')
    verificar_particion(folder, 'dev')
    xs, ys, ws, audit = ([], [], [], [])
    names = None
    for game_index, game in enumerate(SPLITS_DFL['train']):
        atd, periods, ref, _, _ = cargar_idsse(folder, game, fps=25)
        events = eventos_micro_referencia(ref, periods)
        y = objetivos(len(ref), events, periods)
        for j, sd in enumerate(RUIDOS):
            seed = 20271117 + game_index * 100 + j * 10
            x, valid, names = extraer_variables(aumentar(atd, periods, sd, seed), np.ones(len(ref), bool), periods)
            rng = np.random.default_rng(seed)
            keep = valid & (y | (rng.random(y.shape) < 0.08))
            xs.append(x[keep])
            ys.append(y[keep])
            ws.append(np.where(y[keep], 1.0, 1 / 0.08))
            audit.append(dict(partido=game, sd=sd, filas=int(keep.sum()), positivos=int(y[keep].sum())))
            progreso(f'Variables externas listas: {game}, ruido={sd}')
    x, y, w = (np.concatenate(xs), np.concatenate(ys), np.concatenate(ws))
    classifier = ajustar_stumps(x, y, w, n_trees=160, learning_rate=0.12, bins=32, regularizacion=5.0)
    del xs, ys, ws, x, y, w
    progreso('Clasificador externo entrenado; calibrando únicamente en DFL dev')
    totals = {t: dict(tp=0, pred=0, ref=0) for t in UMBRALES}
    validation = []
    for game in SPLITS_DFL['dev']:
        atd, periods, ref, _, _ = cargar_idsse(folder, game, fps=25)
        events = eventos_micro_referencia(ref, periods)
        for j, sd in enumerate(RUIDOS):
            x, valid, _ = extraer_variables(aumentar(atd, periods, sd, 20280117 + j * 10), np.ones(len(ref), bool), periods)
            prob = sigmoid(log_odds_stumps(x.reshape(-1, x.shape[2]), classifier)).reshape(len(ref), 2)
            prob[~valid] = 0.0
            for t in UMBRALES:
                counts = contar_eventos(prob, events, t, valid, periods)
                validation.append(dict(partido=game, sd=sd, umbral=t, **counts))
                for key, value in counts.items():
                    totals[t][key] += value
            progreso(f'Calibración externa lista: {game}, ruido={sd}')
    candidates = []
    for t, c in totals.items():
        precision = c['tp'] / max(1, c['pred'])
        recall = c['tp'] / max(1, c['ref'])
        f1 = 2 * c['tp'] / max(1, c['pred'] + c['ref'])
        c.update(precision=precision, recall=recall, f1=f1)
        if precision >= 0.5 and c['tp'] >= 3:
            candidates.append((f1, t))
    threshold = max(candidates)[1] if candidates else 1.0
    model = dict(version_variables=VERSION_VARIABLES, variables=names, fps=25, clasificador=classifier, umbral_toque=threshold, umbral_busqueda=min(UMBRALES), separacion_frames=8, calibracion_valida=bool(candidates), protocolo=dict(fuente='DFL/IDSSE figshare 28196177 v1', train=SPLITS_DFL['train'], dev=SPLITS_DFL['dev'], test_usado=False, ruido_sd_m=RUIDOS, memoria_ruido_s=0.4, radio_objetivo_frames=3, tolerancia_evento_frames=12, criterio='max F1 eventos dev sujeto a precision >= .5 y TP >= 3', etiquetas_debiles=True, etiquetas_como_variables=False, concurso_accedido=False, metrica_para_calibrar=False))
    report = dict(entrenamiento=audit, calibracion_por_condicion=validation, calibracion_agrupada=totals, umbral_elegido=threshold, calibracion_valida=bool(candidates), aceptacion_metrica_pendiente=True)
    for name, value in (('toques_micro_dfl_v1.json', model), ('calibracion.json', report)):
        with (destino / name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
    progreso(f'Calibración externa: umbral={threshold}; válida={bool(candidates)}')
    return (model, report)
