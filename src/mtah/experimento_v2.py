"""Documentación pública en README.md."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from . import ROOT
from .control import ajustar_control, guardar_modelo, matriz_control, potenciales
from .externos import COMMIT, cargar_sample, degradar_balon, puntuar_referencias, referencias, verificar_procedencia
from .features_control import ConfigFeatures, bloques
from .mejora import ConfigMejora, ajustar_pausa, evidencia_mejora, extraer_mejora
from .referencias_externas import evaluar_apoyo, intervalos_apoyados
from .solucion import predecir as predecir_v1
from .temporal_tasas_fijas import ConfigTemporal, decodificar

def desarrollar(sample_dir, destination, modelo_v1):
    verificar_procedencia(sample_dir)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('El experimento necesita un destino nuevo')
    with open(modelo_v1, encoding='utf-8') as handle:
        previous_model = json.load(handle)
    atd, periods, fps, events = cargar_sample(sample_dir, 1)
    mask = np.isin(periods, [1, 2])
    cf = ConfigFeatures(fps=fps, largo=105.0, ancho=68.0)
    c, ct = (ConfigMejora(), ConfigTemporal(fps=fps))
    frame = extraer_mejora(atd, mask=mask, grupos=periods, config_features=cf, config=c)
    anchors = referencias(events, len(frame))
    use = (anchors != 'X') & frame.observada_original.to_numpy() & (frame.d_min.to_numpy() <= 5.0)
    control = ajustar_control(matriz_control(frame)[use], (anchors[use] == 'H').astype(float))
    weak, intervals = intervalos_apoyados(events, periods, fps)
    pause = ajustar_pausa(frame, weak)
    model = dict(version=2, control=control, pausa=pause, procedencia=dict(fuente='metrica-sports/sample-data', commit=COMMIT, partido='Sample_Game_1', gt_concurso_usado=False, n_referencias_control=int(use.sum()), referencias_pausa='interiores de pases y salida/reanudación; referencias débiles externas'), configuracion=dict(features=asdict(cf), mejora=asdict(c), temporal=asdict(ct)))
    report = dict(version=2, entrenamiento='Sample_Game_1', validacion_desarrollo='Sample_Game_2', commit_datos=COMMIT, gt_concurso_usado=False, resultados={}, modelo=model, advertencia='Validación de desarrollo sobre referencias parciales/débiles; no rendimiento oficial')
    del frame, anchors, weak, intervals, atd
    gc.collect()
    atd, periods, fps, events = cargar_sample(sample_dir, 2)
    mask = np.isin(periods, [1, 2])
    anchors = referencias(events, len(atd['balon']))
    weak, intervals = intervalos_apoyados(events, periods, fps)
    segments = bloques(mask, periods)
    report['cobertura_apoyo'] = {label: int((weak == label).sum()) for label in 'HADX'}
    report['n_intervalos_apoyados'] = len(intervals)
    for scenario, observed in [('original', atd), ('oclusiones_sinteticas', degradar_balon(atd, fps))]:
        v1 = predecir_v1(observed, calif=mask, grupos=periods, config_features=cf, config_temporal=ct, model=previous_model)
        report['resultados'][scenario] = dict(v1=dict(anclas=puntuar_referencias(v1['etiquetas'], anchors), apoyo=evaluar_apoyo(v1['etiquetas'], weak)))
        del v1
        gc.collect()
        frame = extraer_mejora(observed, mask=mask, grupos=periods, config_features=cf, config=c)
        evidence = evidencia_mejora(frame, segments, model, cf, c)
        from .mejora import _distancia_contexto
        context = _distancia_contexto(evidence['soporte'], segments, fps)
        v2 = decodificar(evidence['logits'], segments, evidence['soporte'], context, ct)
        report['resultados'][scenario]['v2'] = dict(anclas=puntuar_referencias(v2['etiquetas'], anchors), apoyo=evaluar_apoyo(v2['etiquetas'], weak))
        del frame, evidence, v2
        gc.collect()
    results = report['resultados']
    gain = np.mean([r['v2']['apoyo']['macroF1_apoyo'] - r['v1']['apoyo']['macroF1_apoyo'] for r in results.values()])
    loss_anchor = max((r['v1']['anclas']['acuerdo'] - r['v2']['anclas']['acuerdo'] for r in results.values()))
    sufficient = all((report['cobertura_apoyo'][label] >= 100 for label in 'HAD'))
    promote = bool(sufficient and gain >= 0.005 and (loss_anchor <= 0.02))
    report['seleccion'] = dict(recomendada='v2' if promote else 'v1', mejora_media_macro_apoyo=float(gain), perdida_max_acuerdo_anclas=float(loss_anchor), criterio='ganancia media >= .005; pérdida en anclas <= .02; cobertura H/A/D >=100 frames', elegibilidad_externa=promote, advertencia='Esta selección no demuestra mejora sobre el concurso ni es un test final ciego')
    report['codigo_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    destination.mkdir(parents=True, exist_ok=False)
    guardar_modelo(model, destination / 'modelo_v2.json')
    with (destination / 'comparacion_publica.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', default=str(ROOT / 'data' / 'external' / 'metrica' / COMMIT))
    parser.add_argument('--salida', required=True)
    parser.add_argument('--modelo-v1', default=str(ROOT / 'outputs' / 'desarrollo_externo_v1_verificado' / 'control.json'))
    args = parser.parse_args(argv)
    report = desarrollar(args.sample_dir, args.salida, args.modelo_v1)
    print(json.dumps(dict(resultados=report['resultados'], seleccion=report['seleccion'], referencias_pausa=report['modelo']['pausa']['referencias']), indent=2, ensure_ascii=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
