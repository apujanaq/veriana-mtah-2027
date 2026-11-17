"""Documentación pública en README.md."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from mtah.candidata_v3 import predecir_v3
from mtah.candidata_v4 import cargar_v4, predecir_v4
from mtah.experimento_v4 import metricas_completas
from mtah.externos import COMMIT, cargar_sample, verificar_procedencia
from mtah.features_control import ConfigFeatures
from mtah.idsse import SPLITS_DFL, cargar_idsse, verificar_idsse
from mtah.referencias_externas import evaluar_apoyo, intervalos_apoyados

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modelo', default=str(ROOT / 'outputs/experimento_v4_01/modelo_v4.json'))
    p.add_argument('--salida', required=True)
    a = p.parse_args()
    out = Path(a.salida)
    if out.exists():
        raise FileExistsError('La guardia necesita destino nuevo')
    model = cargar_v4(a.modelo)
    dfl = ROOT / 'data/external/idsse/figshare_28196177'
    metrica = ROOT / 'data/external/metrica' / COMMIT
    verificar_idsse(dfl)
    verificar_procedencia(metrica)
    report = dict(fps=25, gt_concurso_usado=False, reentrenamiento=False, advertencia='Regresión de desarrollo a frecuencia nativa; no test ciego adicional', resultados={})
    for source in ['DFL_J03WOY', 'Metrica_Game_2']:
        if source.startswith('DFL'):
            atd, groups, refs, events, audit = cargar_idsse(dfl, SPLITS_DFL['dev'][0], fps=25)
            cf = ConfigFeatures(fps=25, largo=audit['largo'], ancho=audit['ancho'])
        else:
            atd, groups, fps, events = cargar_sample(metrica, 2)
            refs, _ = intervalos_apoyados(events, groups, fps)
            cf = ConfigFeatures(fps=fps, largo=105.0, ancho=68.0)
        report['resultados'][source] = {}
        for name, method, frozen in [('v3', predecir_v3, model['base_v3']), ('v4', predecir_v4, model)]:
            result = method(atd, calif=groups > 0, grupos=groups, model=frozen, config_features=cf)
            labels = result['etiquetas']
            assert len(labels) == len(refs) and np.isin(labels, list('HADX')).all()
            score = metricas_completas(labels, refs, groups) if source.startswith('DFL') else evaluar_apoyo(labels, refs)
            report['resultados'][source][name] = score
            del result
            print('Guardia pública a 25 Hz: ' + source + '/' + name, flush=True)
        del atd
    report['checks'] = dict(longitudes_y_etiquetas_correctas=True, ajuste_despues_test=False)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'validacion_publica_25hz.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print('Guardia nativa completada con otros partidos públicos', flush=True)
if __name__ == '__main__':
    main()
