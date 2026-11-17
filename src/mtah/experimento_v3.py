"""Documentación pública en README.md."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from . import ROOT
from .candidata_v3 import ajustar_logistica, calibracion_binaria, evidencia_v3, extraer_v3, matriz_v3, predecir_v3
from .duracion import ajustar_riesgos, decodificar_duracion
from .externos import COMMIT, cargar_sample, puntuar_referencias, referencias, verificar_procedencia
from .features_control import ConfigFeatures, bloques
from .mejora import _distancia_contexto, cargar_mejora, predecir_mejora
from .referencias_externas import evaluar_apoyo, intervalos_apoyados
from .skillcorner import COMMIT_SC, SPLITS, cargar_skillcorner, verificar_skillcorner
from .temporal_tasas_fijas import ConfigTemporal, decodificar

def degradar_realista(atd, fps, seed=2027):
    """Documentación pública en README.md."""
    rng = np.random.default_rng(seed)
    ball, players = (atd['balon'].copy(), atd['jugadores'].copy())
    n = len(ball)
    distance = np.linalg.norm(players[:, :, :2] - ball[:, None, :2], axis=2)
    crowd = (distance < 5).sum(1)
    velocity = np.r_[0.0, np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1) * fps]
    weight = 1 + 2 * (crowd >= 3) + 2 * (velocity > 10)
    weight = weight / weight.sum()
    missing = np.zeros(n, bool)
    while missing.mean() < 0.25:
        first = int(rng.choice(n, p=weight))
        width = max(1, round(rng.choice([0.4, 1.0, 3.0, 10.0]) * fps))
        missing[first:min(first + width, n)] = True
    ball[missing] = np.nan
    jumps = rng.choice(n, size=max(1, n // 400), replace=False)
    ball[jumps, :2] += rng.normal(0, 12, (len(jumps), 2))
    for team in [0, 1]:
        slots = np.flatnonzero(atd['equipo_slot'] == team)
        if len(slots) < 2:
            continue
        for _ in range(max(1, n // round(fps * 90))):
            a, b = rng.choice(slots, 2, replace=False)
            first = int(rng.integers(n))
            stop = min(n, first + max(1, round(fps * 0.8)))
            players[first:stop, [a, b]] = players[first:stop, [b, a]]
    for _ in range(max(1, n // round(fps * 20))):
        first = int(rng.integers(n))
        players[first:min(n, first + round(fps)), int(rng.integers(players.shape[1]))] = np.nan
    return dict(atd, balon=ball, jugadores=players)

def metricas_parciales(labels, refs, periods):
    from evaluation import possession_metrics as pm
    codes, truth = (pm.label_codes(labels), pm.label_codes(refs))
    runs = bloques(refs != 'X', periods)
    filled = pm.fill_abstentions(codes, runs)
    spells = pm.spell_scores(truth, filled, runs)
    return dict(**{k: v for k, v in puntuar_referencias(labels, refs).items() if k != 'advertencia'}, spellF1_condicional=float(spells['mean_f1']), spells_referencia=int(spells['n_gt']), spells_predichos=int(spells['n_pred']), advertencia='Comparación H/A con posesión automática de otro proveedor. X corta el dominio; spellF1 condicional recorta posesiones y omite regiones desconocidas/D. NO puntuación oficial.')

def _write(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)

def _decode(frame, periods, model, fps, length=105.0, width=68.0):
    cf = ConfigFeatures(fps=fps, largo=length, ancho=width)
    segments = bloques(np.isin(periods, [1, 2]), periods)
    evidence = evidencia_v3(frame, segments, model, cf)
    context = _distancia_contexto(evidence['soporte'], segments, fps)
    if model['opciones']['duracion']:
        result = decodificar_duracion(evidence['logits'], segments, evidence['soporte'], context, fps, model['duracion'])
    else:
        result = decodificar(evidence['logits'], segments, evidence['soporte'], context, ConfigTemporal(fps=fps))
    return result['etiquetas']

def desarrollar(destination, metrica_folder, skillcorner_folder, model_v2):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('El experimento requiere destino nuevo')
    verificar_procedencia(metrica_folder)
    verificar_skillcorner(skillcorner_folder)
    raw_base = json.loads(Path(model_v2).read_text(encoding='utf-8'))
    if raw_base.get('version') == 3:
        from .candidata_v3 import cargar_v3
        base = cargar_v3(model_v2)['base_v2']
    else:
        base = cargar_mejora(model_v2)
    destination.mkdir(parents=True, exist_ok=False)
    protocol = dict(version=3, entrenamiento=SPLITS['train'] + ['Metrica Sample_Game_1'], desarrollo=SPLITS['dev'] + ['Metrica Sample_Game_2 (guardia de regresión, ya utilizado)'], test_reservado=SPLITS['test'], gt_concurso_usado=False, criterio_desarrollo='Maximizar media spellF1 condicional original/estrés con pérdida HA <= .015 frente a v2', criterio_promocion='Ganancia media spellF1 condicional test >= .005, caída HA media <= .015 y caída macroF1 débil Metrica <= .01', limitacion='Son referencias parciales y automáticas; ninguna métrica sustituye evaluación oficial', commits=dict(metrica=COMMIT, skillcorner=COMMIT_SC))
    _write(destination / 'protocolo_previo.json', protocol)
    xs, ys, ws, durations, audits = ([], [], [], [], [])
    for game in SPLITS['train']:
        atd, periods, refs, audit = cargar_skillcorner(skillcorner_folder, game)
        cf = ConfigFeatures(fps=10, largo=audit['largo'], ancho=audit['ancho'])
        frame = extraer_v3(atd, mask=periods > 0, groups=periods, cf=cf)
        valid = (refs != 'X') & frame.observada_original.to_numpy() & ~frame.salto_balon.to_numpy() & (frame.d_min.to_numpy() <= 5)
        valid &= np.arange(len(refs)) % 5 == 0
        if valid.sum() < 100:
            raise ValueError('Fuente externa sin soporte suficiente')
        xs.append(matriz_v3(frame)[valid])
        ys.append((refs[valid] == 'H').astype(float))
        ws.append(np.full(valid.sum(), 1 / valid.sum()))
        durations.append((refs, periods, 10.0))
        audits.append(audit)
        del frame, atd
        gc.collect()
        print('Entrenamiento público preparado: ' + str(game), flush=True)
    atd, periods, fps, events = cargar_sample(metrica_folder, 1)
    frame = extraer_v3(atd, mask=periods > 0, groups=periods, cf=ConfigFeatures(fps=fps, largo=105.0, ancho=68.0))
    refs = referencias(events, len(frame))
    valid = (refs != 'X') & frame.observada_original.to_numpy() & ~frame.salto_balon.to_numpy() & (frame.d_min.to_numpy() <= 5)
    xs.append(matriz_v3(frame)[valid])
    ys.append((refs[valid] == 'H').astype(float))
    ws.append(np.full(valid.sum(), 1 / valid.sum()))
    x, y, weight = (np.concatenate(xs), np.concatenate(ys), np.concatenate(ws))
    weights = []
    for yy, ww in zip(ys, ws):
        positive = float(np.sum(ww * yy) / ww.sum())
        weights.append(ww * np.where(yy == 1, 0.5 / positive, 0.5 / (1 - positive)))
    control = ajustar_logistica(x, y, np.concatenate(weights))
    duration = ajustar_riesgos(durations)
    del frame, atd, xs, ys, ws, durations, x, y
    gc.collect()
    model = dict(version=3, base_v2=base, control_v3=control, duracion=duration, temperatura_control=1.0, opciones=dict(duracion=False, peso_parada_extra=0.0, retro_parada_s=0.0), procedencia=dict(gt_concurso_usado=False, skillcorner_commit=COMMIT_SC, metrica_commit=COMMIT, splits=SPLITS, entrenamiento_metrica='Sample_Game_1', etiquetas='Metrica: anclas parciales; SkillCorner: posesión automática H/A. Sin etiquetas D de SkillCorner.'))
    report = dict(protocolo=protocol, auditorias_entrenamiento=audits, desarrollo={}, test={}, advertencia='Las métricas externas no son el score oficial. Test se abre después de congelar la elección.')
    game = SPLITS['dev'][0]
    atd, periods, refs, audit = cargar_skillcorner(skillcorner_folder, game)
    cf = ConfigFeatures(fps=10, largo=audit['largo'], ancho=audit['ancho'])
    frame = extraer_v3(atd, mask=periods > 0, groups=periods, cf=cf)
    valid = (refs != 'X') & frame.observada_original.to_numpy() & ~frame.salto_balon.to_numpy() & (frame.d_min.to_numpy() <= 5)
    scores = matriz_v3(frame)[valid] @ np.asarray(control['coeficientes'])
    calibration = {str(t): calibracion_binaria(scores, refs[valid] == 'H', t) for t in [0.5, 1.0, 2.0, 4.0]}
    model['temperatura_control'] = float(min(calibration, key=lambda t: calibration[t]['nll']))
    report['calibracion_desarrollo_control_condicional'] = calibration
    variants = {'control': dict(duracion=False, peso_parada_extra=0.0, retro_parada_s=0.0), 'duracion': dict(duracion=True, peso_parada_extra=0.0, retro_parada_s=0.0), 'paradas': dict(duracion=True, peso_parada_extra=1.5, retro_parada_s=0.8), 'paradas_hmm': dict(duracion=False, peso_parada_extra=1.5, retro_parada_s=0.8)}
    for scenario, observed in [('original', atd), ('estres', degradar_realista(atd, 10))]:
        frame = extraer_v3(observed, mask=periods > 0, groups=periods, cf=cf)
        v2 = predecir_mejora(observed, calif=periods > 0, grupos=periods, model=base, config_features=cf, config_temporal=ConfigTemporal(fps=10))['etiquetas']
        results = dict(v2=metricas_parciales(v2, refs, periods))
        for name, options in variants.items():
            candidate = dict(model, opciones=options)
            labels = _decode(frame, periods, candidate, 10, audit['largo'], audit['ancho'])
            results[name] = metricas_parciales(labels, refs, periods)
            print('Desarrollo público: ' + scenario + '/' + name, flush=True)
        report['desarrollo'][scenario] = results
        del frame
        gc.collect()
    eligible = [v for v in variants if all((report['desarrollo'][s][v]['acuerdo'] >= report['desarrollo'][s]['v2']['acuerdo'] - 0.015 for s in report['desarrollo']))]
    if not eligible:
        eligible = ['control']
        report['sin_candidata_elegible_desarrollo'] = True
    chosen = max(eligible, key=lambda v: np.mean([r[v]['spellF1_condicional'] for r in report['desarrollo'].values()]))
    model['opciones'] = variants[chosen]
    report['variante_congelada'] = chosen
    _write(destination / 'modelo_v3.json', model)
    _write(destination / 'seleccion_antes_test.json', dict(variante=chosen, desarrollo=report['desarrollo'], modelo_sha256=hashlib.sha256((destination / 'modelo_v3.json').read_bytes()).hexdigest()))
    del atd, refs, periods
    gc.collect()
    for game in SPLITS['test']:
        atd, periods, refs, audit = cargar_skillcorner(skillcorner_folder, game)
        cf = ConfigFeatures(fps=10, largo=audit['largo'], ancho=audit['ancho'])
        report['test'][str(game)] = dict(auditoria=audit, escenarios={})
        for scenario, observed in [('original', atd), ('estres', degradar_realista(atd, 10, seed=2027 + game))]:
            v2 = predecir_mejora(observed, calif=periods > 0, grupos=periods, model=base, config_features=cf, config_temporal=ConfigTemporal(fps=10))['etiquetas']
            v3 = predecir_v3(observed, calif=periods > 0, grupos=periods, model=model, config_features=cf)
            report['test'][str(game)]['escenarios'][scenario] = dict(v2=metricas_parciales(v2, refs, periods), v3=metricas_parciales(v3['etiquetas'], refs, periods))
            frame = v3['feat']
            valid = (refs != 'X') & frame.observada_original.to_numpy() & ~frame.salto_balon.to_numpy() & (frame.d_min.to_numpy() <= 5)
            logits = matriz_v3(frame)[valid] @ np.asarray(control['coeficientes'])
            report['test'][str(game)]['escenarios'][scenario]['calibracion_control'] = calibracion_binaria(logits, refs[valid] == 'H', model['temperatura_control'])
            del v2, v3, frame
            gc.collect()
            print('Test externo reservado: ' + str(game) + '/' + scenario, flush=True)
        del atd
        gc.collect()
    atd, periods, fps, events = cargar_sample(metrica_folder, 2)
    anchors, (weak, _) = (referencias(events, len(atd['balon'])), intervalos_apoyados(events, periods, fps))
    cf = ConfigFeatures(fps=fps, largo=105.0, ancho=68.0)
    report['regresion_metrica'] = {}
    for scenario, observed in [('original', atd), ('estres', degradar_realista(atd, fps))]:
        results = {}
        for name, method, frozen in [('v2', predecir_mejora, base), ('v3', predecir_v3, model)]:
            labels = method(observed, calif=periods > 0, grupos=periods, model=frozen, config_features=cf)['etiquetas']
            results[name] = dict(anclas=puntuar_referencias(labels, anchors), apoyo=evaluar_apoyo(labels, weak))
        report['regresion_metrica'][scenario] = results
        print('Regresión externa Metrica: ' + scenario, flush=True)
    test_cases = [s for r in report['test'].values() for s in r['escenarios'].values()]
    gains = [r['v3']['spellF1_condicional'] - r['v2']['spellF1_condicional'] for r in test_cases]
    losses = [r['v2']['acuerdo'] - r['v3']['acuerdo'] for r in test_cases]
    macro_loss = max((r['v2']['apoyo']['macroF1_apoyo'] - r['v3']['apoyo']['macroF1_apoyo'] for r in report['regresion_metrica'].values()))
    promotion = bool(not report.get('sin_candidata_elegible_desarrollo') and np.mean(gains) >= 0.005 and (np.mean(losses) <= 0.015) and (macro_loss <= 0.01))
    report['decision'] = dict(promover_v3=promotion, ganancia_media_spellF1_condicional=float(np.mean(gains)), caida_media_acuerdo_HA=float(np.mean(losses)), caida_max_macroF1_apoyo=float(macro_loss), candidata_recomendada='v3' if promotion else 'v2', test_evaluado_una_vez=True, ajuste_despues_test=False)
    report['codigo_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    _write(destination / 'comparacion_publica.json', report)
    print(json.dumps(report['decision'], ensure_ascii=False), flush=True)
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', required=True)
    parser.add_argument('--metrica', default=str(ROOT / 'data/external/metrica' / COMMIT))
    parser.add_argument('--skillcorner', default=str(ROOT / 'data/external/skillcorner' / COMMIT_SC))
    parser.add_argument('--modelo-v2', default=str(ROOT / 'outputs/experimento_v2_01/modelo_v2.json'))
    args = parser.parse_args(argv)
    desarrollar(args.salida, args.metrica, args.skillcorner, args.modelo_v2)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
