"""Documentación pública en README.md."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from . import ROOT
from .boosting import ajustar_stumps, log_odds_stumps
from .candidata_v3 import ajustar_logistica, cargar_v3, predecir_v3
from .candidata_v4 import D_VARIABLES, evidencia_v4, extraer_v4, matriz_control_v4, matriz_pausa_v4, predecir_v4
from .experimento_v3 import degradar_realista
from .externos import COMMIT, cargar_sample, puntuar_referencias, referencias, verificar_procedencia
from .features_control import ConfigFeatures, bloques
from .idsse import SPLITS_DFL, cargar_idsse, verificar_idsse
from .mejora import _distancia_contexto
from .referencias_externas import evaluar_apoyo, intervalos_apoyados
from .skillcorner import COMMIT_SC, SPLITS, cargar_skillcorner, verificar_skillcorner
from .temporal_tasas_fijas import ConfigTemporal, decodificar

def escribir(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)

def metricas_completas(labels, refs, periods):
    from evaluation import possession_metrics as pm
    truth, pred = (pm.label_codes(refs), pm.label_codes(labels))
    runs = bloques(refs != 'X', periods)
    cm = pm.scored_confusion(pm.confusion(truth, pred))
    filled = pm.fill_abstentions(pred, runs)
    spells = pm.spell_scores(truth, filled, runs)
    by_class = {}
    for i, side in enumerate('HAD'):
        p, r, f = pm.per_class_prf(cm, i)
        by_class[side] = dict(precision=p, recall=r, f1=f)
    return dict(macroF1=pm.macro_f1(cm), spellF1=spells['mean_f1'], spells_referencia=spells['n_gt'], spells_predichos=spells['n_pred'], por_clase=by_class, acuerdo=pm.accuracy(cm), advertencia='Referencias de otro proveedor, 5 Hz; no puntuación del concurso')

def _predict_features(frame, atd, periods, model, cf):
    segments = bloques(periods > 0, periods)
    e = evidencia_v4(frame, atd, segments, model, cf)
    context = _distancia_contexto(e['soporte'], segments, cf.fps)
    return decodificar(e['logits'], segments, e['soporte'], context, ConfigTemporal(fps=cf.fps))['etiquetas']

def desarrollar(destination, dfl_folder, model_v3, skillcorner_folder, metrica_folder):
    out = Path(destination)
    if out.exists():
        raise FileExistsError('Experimento necesita destino nuevo')
    fixed = verificar_idsse(dfl_folder)
    verificar_skillcorner(skillcorner_folder)
    verificar_procedencia(metrica_folder)
    raw_base = json.loads(Path(model_v3).read_text(encoding='utf-8'))
    if raw_base.get('version') == 4:
        from .candidata_v4 import cargar_v4
        base = cargar_v4(model_v3)['base_v3']
    else:
        base = cargar_v3(model_v3)
    out.mkdir(parents=True, exist_ok=False)
    variants = {'pausa_media': dict(control_nuevo=False, mezcla_pausa=0.5, peso_reanudacion=0.0), 'pausa_completa': dict(control_nuevo=False, mezcla_pausa=1.0, peso_reanudacion=0.0), 'control_y_pausa': dict(control_nuevo=True, mezcla_pausa=0.5, peso_reanudacion=0.0), 'control_pausa_saques': dict(control_nuevo=True, mezcla_pausa=0.5, peso_reanudacion=0.5)}
    protocol = dict(version=4, gt_concurso_usado=False, idsse_article=28196177, idsse_version=1, splits_dfl=SPLITS_DFL, fps_evaluacion=5, variantes=variants, etiquetas_y_eventos_como_features=False, seleccion='Mayor spellF1 medio en desarrollo original/estrés; caída macroF1 <= .02 frente a v3', promocion='Ganancia spellF1 test DFL >= .01; caída macroF1 test <= .01; caída macroF1 apoyo Metrica <= .03', nota='Metrica Game 2 sigue siendo regresión de desarrollo, no test ciego nuevo')
    escribir(out / 'protocolo_previo.json', protocol)
    x_control, y_control, x_pause, y_pause = ([], [], [], [])
    audits = []
    for game in SPLITS_DFL['train']:
        atd, periods, refs, events, audit = cargar_idsse(dfl_folder, game)
        cf = ConfigFeatures(fps=5, largo=audit['largo'], ancho=audit['ancho'])
        for scene, observed in [('original', atd), ('estres', degradar_realista(atd, 5, 2027 + len(audits)))]:
            frame = extraer_v4(observed, mask=periods > 0, groups=periods, cf=cf)
            valid = np.isin(refs, ['H', 'A']) & frame.observada_original.to_numpy() & ~frame.salto_balon.to_numpy() & (frame.d_min.to_numpy() <= 5)
            x_control.append(matriz_control_v4(frame)[valid])
            y_control.append((refs[valid] == 'H').astype(float))
            xp = matriz_pausa_v4(frame)
            reliable = (refs != 'X') & np.isfinite(xp).all(1) & (frame.n_velocidades_validas.to_numpy() >= 6)
            x_pause.append(xp[reliable])
            y_pause.append((refs[reliable] == 'D').astype(float))
            del frame
        audits.append(audit)
        del atd
        gc.collect()
        print('DFL entrenamiento preparado: ' + game, flush=True)
    for game in SPLITS['train']:
        raw, periods, refs, audit = cargar_skillcorner(skillcorner_folder, game)
        atd = dict(balon=raw['balon'][::2], jugadores=raw['jugadores'][::2], equipo_slot=raw['equipo_slot'])
        periods, refs = (periods[::2], refs[::2])
        frame = extraer_v4(atd, mask=periods > 0, groups=periods, cf=ConfigFeatures(fps=5, largo=audit['largo'], ancho=audit['ancho']))
        valid = np.isin(refs, ['H', 'A']) & frame.observada_original.to_numpy() & ~frame.salto_balon.to_numpy() & (frame.d_min.to_numpy() <= 5)
        x_control.append(matriz_control_v4(frame)[valid])
        y_control.append((refs[valid] == 'H').astype(float))
        del frame, atd, raw
        gc.collect()
        print('SkillCorner control preparado: ' + str(game), flush=True)
    weights = []
    for yy in y_control:
        fraction = yy.mean()
        weights.append(np.where(yy == 1, 0.5 / fraction, 0.5 / (1 - fraction)) / len(yy))
    control = ajustar_logistica(np.concatenate(x_control), np.concatenate(y_control), np.concatenate(weights))
    control['variables'] = ['margen', 'velocidad_relativa', 'continuidad', 'apoyo', 'aproximacion', 'proximidad', 'intercepcion_04', 'intercepcion_08', 'margen_contexto', 'tti_ventaja', 'tti_proximidad']
    weights_pause = np.concatenate([np.full(len(y), 1 / len(y)) for y in y_pause])
    pause = ajustar_stumps(np.concatenate(x_pause), np.concatenate(y_pause), weights_pause, n_trees=120)
    pause['variables'] = D_VARIABLES
    model = dict(version=4, base_v3=base, control_v4=control, pausa_v4=pause, temperatura_pausa=1.0, opciones=variants['pausa_media'], procedencia=dict(gt_concurso_usado=False, idsse_article=28196177, idsse_version=1, splits_dfl=SPLITS_DFL, skillcorner_commit=COMMIT_SC, skillcorner_entrenamiento=SPLITS['train'], referencia_D='BallStatus publicado de otros partidos DFL; no ausencia de balón', entrenamiento='DFL train + SkillCorner train; base v3 conserva Metrica Game 1'))
    del x_control, y_control, x_pause, y_pause
    gc.collect()
    report = dict(protocolo=protocol, auditorias_entrenamiento=audits, desarrollo={}, test={}, regresion_metrica={})
    atd, periods, refs, events, audit = cargar_idsse(dfl_folder, SPLITS_DFL['dev'][0])
    cf = ConfigFeatures(fps=5, largo=audit['largo'], ancho=audit['ancho'])
    frame = extraer_v4(atd, mask=periods > 0, groups=periods, cf=cf)
    xp = matriz_pausa_v4(frame)
    valid = (refs != 'X') & np.isfinite(xp).all(1) & (frame.n_velocidades_validas.to_numpy() >= 6)
    score, target = (log_odds_stumps(xp[valid], pause), (refs[valid] == 'D').astype(float))
    losses = {str(t): float(np.mean(np.logaddexp(0, score / t) - target * score / t)) for t in [0.5, 1.0, 2.0]}
    model['temperatura_pausa'] = float(min(losses, key=losses.get))
    report['calibracion_pausa_desarrollo'] = losses
    for scene, observed in [('original', atd), ('estres', degradar_realista(atd, 5, seed=4004))]:
        frame = extraer_v4(observed, mask=periods > 0, groups=periods, cf=cf)
        labels = predecir_v3(observed, calif=periods > 0, grupos=periods, model=base, config_features=cf)['etiquetas']
        results = dict(v3=metricas_completas(labels, refs, periods))
        for name, options in variants.items():
            labels = _predict_features(frame, observed, periods, dict(model, opciones=options), cf)
            results[name] = metricas_completas(labels, refs, periods)
            print('DFL desarrollo: ' + scene + '/' + name, flush=True)
        report['desarrollo'][scene] = results
        del frame
    eligible = [name for name in variants if all((r[name]['macroF1'] >= r['v3']['macroF1'] - 0.02 for r in report['desarrollo'].values()))]
    if not eligible:
        report['sin_elegible_desarrollo'] = True
        eligible = list(variants)
    chosen = max(eligible, key=lambda name: np.mean([r[name]['spellF1'] for r in report['desarrollo'].values()]))
    model['opciones'] = variants[chosen]
    report['variante_congelada'] = chosen
    escribir(out / 'modelo_v4.json', model)
    escribir(out / 'seleccion_antes_test.json', dict(variante=chosen, desarrollo=report['desarrollo'], modelo_sha256=hashlib.sha256((out / 'modelo_v4.json').read_bytes()).hexdigest()))
    del atd, refs, periods
    gc.collect()
    for game in SPLITS_DFL['test']:
        atd, periods, refs, events, audit = cargar_idsse(dfl_folder, game)
        cf = ConfigFeatures(fps=5, largo=audit['largo'], ancho=audit['ancho'])
        report['test'][game] = dict(auditoria=audit, escenarios={})
        for scene, observed in [('original', atd), ('estres', degradar_realista(atd, 5, seed=7000 + len(report['test'])))]:
            results = {}
            for name, method, frozen in [('v3', predecir_v3, base), ('v4', predecir_v4, model)]:
                labels = method(observed, calif=periods > 0, grupos=periods, model=frozen, config_features=cf)['etiquetas']
                results[name] = metricas_completas(labels, refs, periods)
                restart_mask = np.zeros(len(refs), bool)
                for event in events:
                    if event['type'] == 'ThrowIn' and event['error_tiempo_s'] <= 1:
                        a, b = (max(0, event['frame'] - 25), min(len(refs), event['frame'] + 25))
                        restart_mask[a:b] = True
                if restart_mask.any():
                    local_refs = np.where(restart_mask, refs, 'X')
                    results[name]['alrededor_bandas'] = evaluar_apoyo(labels, local_refs)
            report['test'][game]['escenarios'][scene] = results
            print('DFL test reservado: ' + game + '/' + scene, flush=True)
        del atd
        gc.collect()
    raw, groups, fps, events = cargar_sample(metrica_folder, 2)
    refs, _ = intervalos_apoyados(events, groups, fps)
    atd = dict(balon=raw['balon'][::5], jugadores=raw['jugadores'][::5], equipo_slot=raw['equipo_slot'])
    groups, refs = (groups[::5], refs[::5])
    cf = ConfigFeatures(fps=5, largo=105.0, ancho=68.0)
    for scene, observed in [('original', atd), ('estres', degradar_realista(atd, 5, seed=4004))]:
        results = {}
        for name, method, frozen in [('v3', predecir_v3, base), ('v4', predecir_v4, model)]:
            labels = method(observed, calif=groups > 0, grupos=groups, model=frozen, config_features=cf)['etiquetas']
            results[name] = evaluar_apoyo(labels, refs)
        report['regresion_metrica'][scene] = results
        print('Regresión Metrica a 5 Hz: ' + scene, flush=True)
    cases = [r for match in report['test'].values() for r in match['escenarios'].values()]
    gain = float(np.mean([r['v4']['spellF1'] - r['v3']['spellF1'] for r in cases]))
    loss = float(np.mean([r['v3']['macroF1'] - r['v4']['macroF1'] for r in cases]))
    weak_loss = max((r['v3']['macroF1_apoyo'] - r['v4']['macroF1_apoyo'] for r in report['regresion_metrica'].values()))
    promote = bool(not report.get('sin_elegible_desarrollo') and gain >= 0.01 and (loss <= 0.01) and (weak_loss <= 0.03))
    report['decision'] = dict(promover_v4=promote, candidata_recomendada='v4' if promote else 'v3', ganancia_media_spellF1=gain, caida_media_macroF1=loss, caida_max_macroF1_metrica=weak_loss, ajuste_despues_test=False)
    report['codigo_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    escribir(out / 'comparacion_publica.json', report)
    print(json.dumps(report['decision']), flush=True)
    return report

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--salida', required=True)
    p.add_argument('--dfl', default=str(ROOT / 'data/external/idsse/figshare_28196177'))
    p.add_argument('--modelo-v3', default=str(ROOT / 'modelos/modelo.json'))
    p.add_argument('--skillcorner', default=str(ROOT / 'data/external/skillcorner' / COMMIT_SC))
    p.add_argument('--metrica', default=str(ROOT / 'data/external/metrica' / COMMIT))
    a = p.parse_args(argv)
    desarrollar(a.salida, a.dfl, a.modelo_v3, a.skillcorner, a.metrica)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
