"""Documentación pública en README.md."""
import numpy as np
from .microposesiones import insertar_microposesiones
from .toques_supervisados import bloques_validos

def refinar_pares(etiquetas, p_toques, calif, grupos=None, umbral=0.25, verificar=False, max_verificar=50, tolerancia=12, max_insertar=50):
    lab = np.asarray(etiquetas).copy()
    pt = np.asarray(p_toques, float)
    mask = np.asarray(calif, bool)
    if pt.shape != (len(lab), 2) or mask.shape != lab.shape:
        raise ValueError('Dimensiones de parejas incompatibles')
    removed = 0
    if verificar:
        for a, b in bloques_validos(mask & np.isin(lab, ['H', 'A']), grupos):
            original = lab[a:b].copy()
            cuts = np.r_[a, a + 1 + np.flatnonzero(original[1:] != original[:-1]), b]
            for j in range(1, len(cuts) - 2):
                lo, hi = cuts[j:j + 2]
                owner = original[cuts[j - 1] - a]
                side = original[lo - a]
                if hi - lo >= max_verificar or owner != original[hi - a]:
                    continue
                if lo - tolerancia < a or hi + tolerancia >= b:
                    continue
                k, kk = (int(side == 'A'), int(owner == 'A'))
                first = np.nan_to_num(pt[lo - tolerancia:lo + tolerancia + 1, k], nan=0).max()
                last = np.nan_to_num(pt[hi - tolerancia:hi + tolerancia + 1, kk], nan=0).max()
                if min(first, last) < umbral:
                    lab[lo:hi] = owner
                    removed += 1
    lab, inserted = insertar_microposesiones(lab, pt, umbral, max_insertar, grupos, mask)
    if not np.array_equal(lab == 'D', np.asarray(etiquetas) == 'D'):
        raise AssertionError('El refinamiento H/A alteró D')
    return (lab, dict(retiradas=removed, insertadas=inserted))
