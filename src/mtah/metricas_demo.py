"""Documentación pública en README.md."""
import argparse
from html import escape
import json
from pathlib import Path
import numpy as np
from .validacion import pm

def ejemplos():
    truth = np.array(list('H' * 100 + 'A' * 100 + 'D' * 100 + 'H' * 100))
    cases = {'Perfecto': truth.copy(), 'Un rebote espurio de un frame': truth.copy(), 'Diez frames de retraso': np.r_[np.repeat(truth[0], 10), truth[:-10]], 'Oclusión dentro de una posesión': truth.copy(), 'Oclusión que tapa el cambio': truth.copy(), 'Confundir la parada con posesión': truth.copy(), 'Siempre local': np.full(len(truth), 'H')}
    cases['Un rebote espurio de un frame'][50] = 'A'
    cases['Oclusión dentro de una posesión'][30:60] = 'X'
    cases['Oclusión que tapa el cambio'][90:180] = 'X'
    cases['Confundir la parada con posesión'][200:300] = 'H'
    result = []
    gt = pm.label_codes(truth)
    for name, prediction in cases.items():
        metrics = pm.evaluate(gt, pm.label_codes(prediction))
        result.append(dict(caso=name, prediccion=prediction.tolist(), macroF1=metrics['macro_f1'], spellF1=metrics['spell_f1_mean'], posesiones=metrics['n_pred_spells'], confusion=metrics['confusion'], por_clase=metrics['per_class'], por_iou=metrics['spell_f1']))
    return (truth, result)

def _timeline(labels):
    colors = {'H': '#236ca6', 'A': '#bf5b10', 'D': '#6b7280', 'X': '#dedede'}
    codes = pm.label_codes(labels)
    rectangles = []
    for first, stop, code in pm.segments(codes, [(0, len(labels))]):
        rectangles.append(f'<rect x="{first}" y="0" width="{stop - first}" height="24" fill="{colors[pm.LABELS[code]]}"/>')
    return f'<svg role="img" aria-label="Secuencia de etiquetas sintética" viewBox="0 0 {len(labels)} 24">' + ''.join(rectangles) + '</svg>'

def generar(path):
    truth, rows = ejemplos()
    parts = ['<h1>Cómo castigan los errores las dos métricas</h1>', '<p>Ejemplos inventados: 400 frames. H = azul, A = naranja, D = gris, X = gris claro.</p>', '<h2>Referencia sintética</h2>', _timeline(truth)]
    for row in rows:
        parts += [f"<h2>{escape(row['caso'])}</h2>", _timeline(row['prediccion']), '<p>macroF1 = {:.4f} · spellF1 = {:.4f} · posesiones = {}</p>'.format(row['macroF1'], row['spellF1'], row['posesiones']), '<details><summary>Confusión, F1 por clase y emparejamientos por IoU</summary><pre>' + escape(json.dumps({k: row[k] for k in ['confusion', 'por_clase', 'por_iou']}, indent=2)) + '</pre></details>']
    html = '<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'">\n<title>Ejemplos sintéticos de métricas</title><style>body{max-width:1000px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#163247;background:#fff}\nsvg{width:100%;height:40px}h2{margin-top:40px}pre{white-space:pre-wrap;background:#f3f5f6;padding:16px}</style></head><body>'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(html + ''.join(parts) + '</body></html>')
    return [{k: row[k] for k in ['caso', 'macroF1', 'spellF1', 'posesiones']} for row in rows]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', default='outputs/demo_metricas.html')
    args = parser.parse_args(argv)
    rows = generar(args.salida)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
