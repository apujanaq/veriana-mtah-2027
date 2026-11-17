"""Documentación pública en README.md."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from mtah.boosting import ajustar_stumps, log_odds_stumps, sigmoid
from mtah.experimento_micro import RUIDOS, aumentar, verificar_particion
from mtah.idsse import SPLITS_DFL, cargar_idsse
from mtah.toques_supervisados import VERSION_VARIABLES, extraer_variables

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dfl', type=Path, required=True)
    ap.add_argument('--salida', type=Path, required=True)
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'dev'):
        verificar_particion(args.dfl, split)
    xs, ys = ([], [])
    for gi, game in enumerate(SPLITS_DFL['train']):
        atd, periods, ref, _, _ = cargar_idsse(args.dfl, game, fps=25)
        for j, sd in enumerate(RUIDOS):
            seed = 20271119 + gi * 100 + j * 10
            x, valid, names = extraer_variables(aumentar(atd, periods, sd, seed), np.ones(len(ref), bool), periods)
            eligible = valid.any(axis=1) & np.isin(ref, ['H', 'A'])
            keep = eligible & (np.random.default_rng(seed).random(len(ref)) < 0.04)
            xs.append(x[keep].reshape(-1, x.shape[2]))
            ys.append(np.column_stack([ref[keep] == 'H', ref[keep] == 'A']).reshape(-1))
            print(f'Control externo: variables {game}, sd={sd}', flush=True)
    classifier = ajustar_stumps(np.concatenate(xs), np.concatenate(ys), n_trees=200, learning_rate=0.1, bins=32, regularizacion=15.0)
    model = dict(version=1, version_variables=VERSION_VARIABLES, variables=names, clasificador=classifier, train=SPLITS_DFL['train'], dev=SPLITS_DFL['dev'], test_usado=False, etiquetas_como_variables=False, eventos_como_variables=False, concurso_accedido=False, semilla=20271119, muestreo_frames=0.04, fps=25, ruido_sd_m=RUIDOS, combinacion='logit simetrico; temperatura dev; D base bloqueada')
    del xs, ys
    calibration = []
    for game in SPLITS_DFL['dev']:
        atd, periods, ref, _, _ = cargar_idsse(args.dfl, game, fps=25)
        for j, sd in enumerate(RUIDOS):
            x, valid, _ = extraer_variables(aumentar(atd, periods, sd, 20280119 + j * 10), np.ones(len(ref), bool), periods)
            scores = log_odds_stumps(x.reshape(-1, x.shape[2]), classifier).reshape(len(ref), 2)
            z = (scores[:, 0] - scores[:, 1]) / 2
            mask = valid.any(axis=1) & np.isin(ref, ['H', 'A'])
            for temp in (0.5, 1.0, 2.0):
                q = np.clip(sigmoid(z[mask] / temp), 1e-07, 1 - 1e-07)
                y = ref[mask] == 'H'
                loss = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
                calibration.append(dict(sd=sd, temperatura=temp, logloss=loss))
            with (args.salida / f'dev_control_{j}.npz').open('xb') as handle:
                np.savez_compressed(handle, z=z, valid=valid.any(axis=1), ref=ref, periods=periods)
            print(f'Control externo: calibración dev sd={sd}', flush=True)
    model['temperatura'] = min(((float(np.mean([r['logloss'] for r in calibration if r['temperatura'] == t])), t) for t in (0.5, 1.0, 2.0)))[1]
    for filename, value in (('control_multiventana.json', model), ('calibracion.json', calibration)):
        with (args.salida / filename).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
    print('CONTROL_MULTIVENTANA_CONGELADO', model['temperatura'], flush=True)
if __name__ == '__main__':
    main()
