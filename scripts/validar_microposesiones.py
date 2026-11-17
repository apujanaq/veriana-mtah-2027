"""Documentación pública en README.md."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
for sub in ('src', ''):
    sys.path.insert(0, str(ROOT / sub))
import pandas as pd
from mtah import microposesiones, validacion_externa as vx
from mtah.aproximaciones import REGISTRO
from mtah.aproximaciones.alejo_v1_fusion import FusionV2
from mtah.aproximaciones.alejo_v2_micro import FusionV2Micro
from mtah.externos import COMMIT, verificar_procedencia

def medir(nombre, datos, refs, lab=None):
    """Documentación pública en README.md."""
    if lab is None:
        lab = vx.etiquetas_por_periodo(REGISTRO[nombre](), datos)
    den = vx.metricas_completas(lab, refs['denso'], refs['periods'])
    mic = microposesiones.metricas_micro(lab, refs['denso'], refs['periods'])
    return {'aprox': nombre, 'spellF1': den['spellF1'], 'macroF1': den['macroF1'], 'spells pred/ref': f"{mic['spells_pred']}/{mic['spells_ref']}", 'micro pred/ref': f"{mic['micro_pred']}/{mic['micro_ref']}", 'recall micro': mic['recall_micro'], 'precisión micro': mic['precision_micro']}

def aceptacion(filas):
    """Documentación pública en README.md."""
    checks = []
    for game in (1, 2):
        for condition in ('limpio', 'ruido'):
            rows = [r for r in filas if r['partido'] == game and r['condicion'] == condition]
            base, new = rows
            checks.append(dict(partido=game, condicion=condition, mejora_spell=new['spellF1'] - base['spellF1'], spell_ok=new['spellF1'] >= base['spellF1'] + (0.02 if condition == 'ruido' else -0.01), precision_ok=bool(np.isfinite(new['precisión micro']) and new['precisión micro'] >= 0.5)))
    return dict(aceptada=all((c['spell_ok'] and c['precision_ok'] for c in checks)), criterios=checks, concurso_evaluado=False, ajuste_con_metrica=False)

def calcular_base(datos, cache=None, game=None, cond=None, sd=0.8):
    """Documentación pública en README.md."""
    if cache is None:
        return vx.etiquetas_por_periodo(REGISTRO['alejo_v2_robusta'](), datos)
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'src' / 'mtah').rglob('*.py')):
        if 'micro' in path.name or 'toques_supervisados' in path.name or 'control_multiventana' in path.name or ('control_pares' in path.name) or (path.name == '__init__.py'):
            continue
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    digest.update((ROOT / 'modelos' / 'modelo.json').read_bytes())
    digest.update(f'{COMMIT}:{game}:{cond}:{sd}:seed={game}'.encode())
    signature = digest.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'sample_{game}_{cond}.npz'

    def leer():
        with np.load(path, allow_pickle=False) as saved:
            lab = saved['labels']
            if str(saved['signature']) != signature or lab.shape != datos.calif.shape or (not np.isin(lab, list('HADX')).all()):
                raise ValueError('Caché externa incompatible; elige una carpeta nueva')
            return lab
    if path.exists():
        return leer()
    lab = vx.etiquetas_por_periodo(REGISTRO['alejo_v2_robusta'](), datos)
    try:
        with path.open('xb') as handle:
            np.savez_compressed(handle, labels=lab, signature=signature)
    except FileExistsError:
        if not np.array_equal(leer(), lab):
            raise ValueError('Dos evaluaciones externas produjeron rutas base diferentes') from None
    return lab

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--aprox', default='alejo_v2_micro_ref', help='aproximación registrada con detector')
    ap.add_argument('--sd', type=float, default=0.8, help='ruido del balón (m) para el régimen BAR-ATH')
    ap.add_argument('--data-dir', type=Path, default=Path(os.environ.get('MTAH_DATA_DIR', ROOT / 'data')))
    ap.add_argument('--json', type=Path, help='informe externo reproducible en un archivo nuevo')
    ap.add_argument('--cache-base', type=Path, help='reutiliza decodificaciones externas si código/modelo coinciden')
    ap.add_argument('--solo-base', action='store_true', help='prepara la referencia v2 sin cargar un detector')
    args = ap.parse_args()
    if not np.isfinite(args.sd) or args.sd < 0:
        ap.error('--sd debe ser una desviación finita no negativa')
    if args.json is not None and args.json.exists():
        raise FileExistsError('El informe debe escribirse en un destino nuevo')
    folder = args.data_dir / 'external' / 'metrica' / COMMIT
    verificar_procedencia(folder)
    pd.set_option('display.width', 200)
    filas = []
    for game in (1, 2):
        datos, refs = vx.cargar_partido(folder, game)
        for cond, d in (('limpio', datos), ('ruido', vx.ensuciar_balon(datos, args.sd, semilla=game))):
            print(f'Evaluando Metrica externo {game}, {cond}', flush=True)
            base = calcular_base(d, args.cache_base, game, cond, args.sd)
            if args.solo_base:
                row = dict(partido=game, condicion=cond, **medir('alejo_v2_robusta', d, refs, base))
                filas.append(row)
                print(pd.DataFrame([row]).to_string(index=False), flush=True)
                continue
            aprox = REGISTRO[args.aprox]()
            if isinstance(aprox, FusionV2Micro) and (type(aprox).probabilidades is FusionV2.probabilidades or getattr(aprox, 'mantiene_probabilidades_v2', False)):
                nuevo = aprox.refinar_etiquetas(d, base)
                if not np.array_equal(base == 'D', nuevo == 'D'):
                    raise AssertionError('El empalme de micro-posesiones cambió D')
            else:
                nuevo = vx.etiquetas_por_periodo(aprox, d)
            rows = [medir(n, d, refs, lab) for n, lab in (('alejo_v2_robusta', base), (args.aprox, nuevo))]
            rows[0].update(frames_modificados=0, micro_insertadas=0, micro_retiradas=0, toques_confiados=0)
            rows[1].update(frames_modificados=int((nuevo != base).sum()), micro_insertadas=getattr(aprox, 'info_micro', {}).get('insertadas'), micro_retiradas=getattr(aprox, 'info_micro', {}).get('retiradas', 0), toques_confiados=getattr(aprox, 'info_micro', {}).get('toques_confiados'))
            filas.extend((dict(partido=game, condicion=cond, **r) for r in rows))
            t = pd.DataFrame(rows).set_index('aprox')
            print(t.to_string(float_format=lambda v: f'{v:.4f}'), flush=True)
    checks = dict(modo='solo_base') if args.solo_base else aceptacion(filas)
    result = dict(aprox='alejo_v2_robusta' if args.solo_base else args.aprox, sd=args.sd, resultados=filas, **checks)
    print(f"Aceptación externa: {result.get('aceptada', 'pendiente')}", flush=True)
    if args.json is not None:
        for row in result['resultados']:
            for key, value in row.items():
                if isinstance(value, float) and (not np.isfinite(value)):
                    row[key] = None
        with args.json.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
if __name__ == '__main__':
    main()
