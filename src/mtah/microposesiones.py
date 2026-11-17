"""Documentación pública en README.md."""
from typing import Protocol
import numpy as np
from .carga import A, H
from .etapa_a import tramos
from .parametros import p
LETRA = {H: 'H', A: 'A'}

class DetectorToques(Protocol):
    """Documentación pública en README.md."""

    def probabilidades_toque(self, datos) -> np.ndarray:
        """Documentación pública en README.md."""

def eventos_toque(p_toques, umbral):
    """Documentación pública en README.md."""
    pt = np.nan_to_num(np.asarray(p_toques, dtype=np.float64), nan=0.0)
    ev = [(a, k) for k in (H, A) for a, _ in tramos(pt[:, k] >= umbral)]
    return sorted(ev)

def insertar_microposesiones(etiquetas, p_toques, umbral=None, max_frames=None, grupos=None, calif=None):
    """Documentación pública en README.md."""
    umbral = p('MICRO_UMBRAL') if umbral is None else umbral
    max_frames = p('MICRO_MAX') if max_frames is None else max_frames
    lab = np.asarray(etiquetas).copy()
    pt = np.asarray(p_toques, float)
    if lab.ndim != 1 or pt.shape != (len(lab), 2):
        raise ValueError('Dimensiones de micro-posesiones incompatibles')
    if not np.isfinite(umbral) or umbral <= 0 or (not np.isfinite(max_frames)) or (max_frames < 1):
        raise ValueError('Parámetros de micro-posesiones inválidos')
    finitos = pt[np.isfinite(pt)]
    if np.isinf(pt).any() or ((finitos < 0) | (finitos > 1)).any():
        raise ValueError('Probabilidades de toque inválidas')
    g = np.zeros(len(lab), int) if grupos is None else np.asarray(grupos)
    m = np.ones(len(lab), bool) if calif is None else np.asarray(calif, bool)
    if g.shape != lab.shape or m.shape != lab.shape:
        raise ValueError('Soporte temporal de micro-posesiones incompatible')
    pt = pt.copy()
    pt[(pt >= umbral).all(axis=1) | ~m] = 0.0
    ev = eventos_toque(pt, umbral)
    n = 0
    for j, (f, k) in enumerate(ev):
        dueno = lab[f]
        if dueno not in ('H', 'A') or dueno == LETRA[k]:
            continue
        for f2, k2 in ev[j + 1:]:
            if f2 - f > max_frames:
                break
            if LETRA[k2] == dueno:
                if f2 > f and (lab[f:f2] == dueno).all() and m[f:f2 + 1].all() and (g[f:f2 + 1] == g[f]).all():
                    lab[f:f2] = LETRA[k]
                    n += 1
                break
    return (lab, n)

def metricas_micro(etiquetas, ref, periods, umbral_s=2.0, iou=0.5, fps=25):
    """Documentación pública en README.md."""
    from evaluation import possession_metrics as pm
    from .experimento_v4 import bloques
    ref = np.asarray(ref)
    runs = bloques(ref != 'X', periods)
    truth = pm.label_codes(ref)
    pred = pm.fill_abstentions(pm.label_codes(etiquetas), runs)
    gts, prs = (pm.possession_spells(truth, runs), pm.possession_spells(pred, runs))
    ok_gt, ok_pr = (np.zeros(len(gts), dtype=bool), np.zeros(len(prs), dtype=bool))
    for v, i, j in pm.spell_overlaps(gts, prs):
        if v < iou:
            break
        if ok_gt[i] or ok_pr[j]:
            continue
        ok_gt[i] = ok_pr[j] = True
    corto_gt = np.array([b - a < umbral_s * fps for a, b, _ in gts], dtype=bool)
    corto_pr = np.array([b - a < umbral_s * fps for a, b, _ in prs], dtype=bool)
    return {'spells_ref': len(gts), 'spells_pred': len(prs), 'micro_ref': int(corto_gt.sum()), 'micro_pred': int(corto_pr.sum()), 'recall_micro': float(ok_gt[corto_gt].mean()) if corto_gt.any() else float('nan'), 'precision_micro': float(ok_pr[corto_pr].mean()) if corto_pr.any() else float('nan')}
