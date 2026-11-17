"""Documentación pública en README.md."""
import numpy as np
import pandas as pd

def intervalos_apoyados(events, periods, fps=25, margen_parada_s=0.5):
    periods = np.asarray(periods)
    n = len(periods)
    votes = np.zeros((n + 1, 3), np.int16)
    records = []
    ordered = events.sort_values(['Period', 'Start Frame'], kind='stable')
    rows = [row for _, row in ordered.iterrows()]

    def add(first, stop, label, origin):
        if not 0 <= first < stop <= n or not (periods[first:stop] == periods[first]).all():
            return
        k = 'HAD'.index(label)
        votes[first, k] += 1
        votes[stop, k] -= 1
        records.append(dict(inicio=first, fin=stop, etiqueta=label, origen=origin))
    for i, event in enumerate(rows):
        start, end = (event.get('Start Frame'), event.get('End Frame'))
        if pd.isna(start) or pd.isna(end) or int(start) != start or (int(end) != end):
            continue
        start, end = (int(start) - 1, int(end) - 1)
        subtype = str(event.get('Subtype', '')).upper()
        if event['Type'] == 'PASS' and pd.notna(event.get('To')) and (event['Team'] in ['Home', 'Away']):
            if any((token in subtype for token in ['OFFSIDE', 'RETAKEN', 'REFEREE', 'FORCED', 'VOLUNTARY'])):
                continue
            if 2 <= end - start <= round(6 * fps):
                add(start + 1, end, 'H' if event['Team'] == 'Home' else 'A', 'interior_pase_completado')
        if event['Type'] != 'BALL OUT' or 'END HALF' in subtype:
            continue
        restart = next((r for r in rows[i + 1:] if r['Period'] == event['Period'] and r['Type'] == 'SET PIECE' and (r['Start Frame'] > event['End Frame'])), None)
        if restart is None:
            continue
        last = int(restart['Start Frame']) - 1
        blockers = [r for r in rows[i + 1:] if r['Period'] == event['Period'] and r['Type'] in ['PASS', 'RECOVERY', 'SHOT'] and (event['End Frame'] < r['Start Frame'] < restart['Start Frame'])]
        if blockers or not 0 < last - end <= round(120 * fps):
            continue
        guard = max(1, round(margen_parada_s * fps))
        add(end + guard, last - guard, 'D', 'interior_salida_reanudacion')
    active = np.cumsum(votes, axis=0)[:-1] > 0
    unambiguous = active.sum(axis=1) == 1
    labels = np.full(n, 'X', dtype='<U1')
    labels[unambiguous] = np.array(list('HAD'))[active[unambiguous].argmax(1)]
    return (labels, records)

def evaluar_apoyo(pred, refs):
    pred, refs = (np.asarray(pred), np.asarray(refs))
    if pred.shape != refs.shape:
        raise ValueError('Referencias y predicciones de distinta longitud')
    mask = refs != 'X'
    by_class = {}
    for side in 'HAD':
        tp = int(((pred == side) & (refs == side) & mask).sum())
        fp = int(((pred == side) & (refs != side) & mask).sum())
        fn = int(((pred != side) & (refs == side) & mask).sum())
        by_class[side] = dict(f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0, precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn) if tp + fn else 0.0, frames=int(((refs == side) & mask).sum()))
    return dict(acuerdo=float((pred[mask] == refs[mask]).mean()) if mask.any() else None, macroF1_apoyo=float(np.mean([v['f1'] for v in by_class.values()])), por_clase=by_class, advertencia='Referencias débiles y censuradas de otro partido; no GT completo ni métricas oficiales')
