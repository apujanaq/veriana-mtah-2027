"""Documentación pública en README.md."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from mtah.control_multiventana import decodificar_ha
from mtah.toques_supervisados import bloques_validos
from mtah.experimento_v4 import metricas_completas

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--entrenamiento', type=Path, required=True)
    ap.add_argument('--salida', type=Path, required=True)
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=False)
    model = json.loads((args.entrenamiento / 'control_multiventana.json').read_text(encoding='utf-8'))
    rows = []
    for j, sd in enumerate((0.0, 0.4, 0.8, 1.0)):
        with np.load(args.entrenamiento / f'dev_control_{j}.npz', allow_pickle=False) as data:
            ref, periods = (data['ref'].copy(), data['periods'])
            ref[ref == 'D'] = 'X'
            runs = bloques_validos(ref != 'X', periods)
            z = np.where(data['valid'], data['z'] / model['temperatura'], 0.0)
            for penalty in (1.0, 2.0, 4.0, 8.0):
                lab = decodificar_ha(z, runs, penalty, 10.0)
                metrics = metricas_completas(lab, ref, periods)
                rows.append(dict(sd=sd, pena=penalty, spellF1=metrics['spellF1'], macroF1=metrics['macroF1']))
        print(f'Persistencia H/A DFL dev: sd={sd}', flush=True)
    _, penalty = max(((float(np.mean([r['spellF1'] for r in rows if r['pena'] == p])), p) for p in (1.0, 2.0, 4.0, 8.0)))
    model['temporal'] = dict(pena=penalty, escala=10.0, criterio='max spellF1 condicional H/A en DFL dev', D_oraculo_solo_en_calibracion=True, D_runtime_bloqueada_a_base=True, metrica_para_elegir=False)
    for name, obj in (('control_multiventana.json', model), ('calibracion_temporal.json', rows)):
        with (args.salida / name).open('x', encoding='utf-8') as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
    print(f'PERSISTENCIA_CONGELADA {penalty}', flush=True)
if __name__ == '__main__':
    main()
