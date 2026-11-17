"""Documentación pública en README.md."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from mtah import validacion_externa as vx
from mtah.aproximaciones import REGISTRO
from mtah.externos import COMMIT, verificar_procedencia
from scripts.validar_microposesiones import calcular_base

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    ap.add_argument('--cache-base', type=Path, required=True)
    ap.add_argument('--json', type=Path, required=True)
    args = ap.parse_args()
    if args.json.exists():
        raise FileExistsError('Informe nuevo obligatorio')
    folder = args.data_dir / 'external/metrica' / COMMIT
    verificar_procedencia(folder)
    rows = []
    for game in (1, 2):
        datos, refs = vx.cargar_partido(folder, game)
        for condition, d in (('limpio', datos), ('ruido', vx.ensuciar_balon(datos, 0.8, semilla=game))):
            base = calcular_base(d, args.cache_base, game, condition, 0.8)
            aprox = REGISTRO['juliana_control_pares']()
            lab = aprox.refinar_etiquetas(d, base)
            for key, value in aprox.params.items():
                if not key.startswith(('MICRO_', 'MV_')):
                    continue
                for factor in (0.8, 1.2):
                    changed = round(value * factor) if isinstance(value, int) else value * factor
                    variant = type(aprox)(params={**aprox.params, key: changed})
                    new = variant.refinar_etiquetas(d, base)
                    score = vx.metricas_completas(new, lab, refs['periods'])['spellF1']
                    rows.append(dict(partido=game, condicion=condition, parametro=key, factor=factor, spellF1_vs_base=score, D_conservada=bool(np.array_equal(new == 'D', base == 'D'))))
            print(f'Robustez externa {game} {condition} lista', flush=True)
    result = dict(alcance='solo parametros nuevos H/A; no C3 completa del pipeline', minimo=min((r['spellF1_vs_base'] for r in rows)), umbral=0.85, pasa=all((r['spellF1_vs_base'] >= 0.85 and r['D_conservada'] for r in rows)), resultados=rows)
    with args.json.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print('ROBUSTEZ_EXTERNA', result['pasa'], round(result['minimo'], 6), flush=True)
if __name__ == '__main__':
    main()
