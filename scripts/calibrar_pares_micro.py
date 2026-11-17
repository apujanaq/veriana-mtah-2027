"""Documentación pública en README.md."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from mtah import features, validacion_externa as vx
from mtah.aproximaciones.juliana_v4 import JulianaV4
from mtah.contrato import Datos
from mtah.experimento_micro import RUIDOS, UMBRALES, aumentar, verificar_particion
from mtah.idsse import SPLITS_DFL, cargar_idsse
from mtah.microposesiones import metricas_micro
from mtah.pares_micro import refinar_pares
from mtah.toques_supervisados import DetectorToquesSupervisado

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dfl', type=Path, required=True)
    ap.add_argument('--salida', type=Path, required=True)
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=False)
    verificar_particion(args.dfl, 'dev')
    detector = DetectorToquesSupervisado()
    records = []
    configs = [(t, mode, duration) for t in UMBRALES for mode in (False, True) for duration in ((50,) if not mode else (10, 25, 50))]
    for game in SPLITS_DFL['dev']:
        atd, periods, refs, _, audit = cargar_idsse(args.dfl, game, fps=25)
        for j, sd in enumerate(RUIDOS):
            noisy = aumentar(atd, periods, sd, 20280117 + j * 10)
            mask = np.ones(len(refs), bool)
            datos = Datos(noisy, features.calcular_features(noisy, mask, pitch=(audit['largo'], audit['ancho'])), [], mask, pitch=(audit['largo'], audit['ancho']), grupos=periods)
            aprox = JulianaV4()
            lab = vx.etiquetas_por_periodo(aprox, datos)
            regime = 'v4_sin_tagging'
            pt = detector.probabilidades_toque(datos)
            support = mask & np.isfinite(noisy['balon'][:, :2]).all(axis=1)
            with (args.salida / f'dev_{game}_{j}.npz').open('xb') as handle:
                np.savez_compressed(handle, labels=lab, prob=pt, ref=refs, periods=periods, soporte=support, ruidoso=True)
            base = vx.metricas_completas(lab, refs, periods)
            for threshold, verify, duration in configs:
                new, info = refinar_pares(lab, pt, support, periods, threshold, verify, duration)
                metrics = vx.metricas_completas(new, refs, periods)
                micro = metricas_micro(new, refs, periods)
                records.append(dict(partido=game, sd=sd, umbral=threshold, verificar=verify, max_verificar=duration, base='juliana_v4_sin_tagging', spellF1_base=base['spellF1'], spellF1=metrics['spellF1'], macroF1=metrics['macroF1'], mejora=metrics['spellF1'] - base['spellF1'], precision_micro=micro['precision_micro'], **info))
            print(f'Parejas DFL dev listas: {game}, sd={sd}, régimen={regime}', flush=True)
    ranked = []
    for threshold, verify, duration in configs:
        rows = [r for r in records if (r['umbral'], r['verificar'], r['max_verificar']) == (threshold, verify, duration)]
        delta = [r['mejora'] for r in rows]
        if min(delta) >= -0.005 and sum((r['sd'] > 0 and r['mejora'] > 0 for r in rows)) >= 2:
            ranked.append((float(np.mean(delta)), threshold, verify, duration))
    selected = max(ranked) if ranked else None
    model = dict(version=1, detector='toques_micro_dfl_v1.json', seleccion_valida=selected is not None, umbral=selected[1] if selected else 1.0, verificar=selected[2] if selected else False, max_verificar=selected[3] if selected else 50, tolerancia=12, max_insertar=50, train=SPLITS_DFL['train'], dev=SPLITS_DFL['dev'], test_usado=False, metrica_para_seleccionar=False, concurso_accedido=False, base_calibracion='juliana_v4_sin_tagging', base_destino='alejo_v2_robusta', criterio='max ganancia media spellF1 dev; pérdida por condición <=.005; mejora >=2 ruidos')
    for name, obj in (('modelo_pares_micro.json', model), ('comparacion_dev.json', records)):
        with (args.salida / name).open('x', encoding='utf-8') as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print(f'Selección de parejas DFL: {selected}', flush=True)
if __name__ == '__main__':
    main()
