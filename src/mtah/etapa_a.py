"""Documentación pública en README.md."""
import numpy as np
from .parametros import p

def tramos(mask):
    """Documentación pública en README.md."""
    bordes = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(bordes == 1).tolist(), np.flatnonzero(bordes == -1).tolist()))

def otsu(x, bins=256):
    """Documentación pública en README.md."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not len(x):
        return float('nan')
    if np.min(x) == np.max(x):
        return float(x[0])
    hist, bordes = np.histogram(x, bins=bins)
    centros = (bordes[:-1] + bordes[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centros)
    with np.errstate(invalid='ignore', divide='ignore'):
        m0 = s0 / w0
        m1 = (s0[-1] - s0) / w1
        entre = w0 * w1 * (m0 - m1) ** 2
    return float(centros[np.nanargmax(entre)])

def detectar_muerto(feat, calif):
    """Documentación pública en README.md."""
    v = feat['v_col_suave'].to_numpy()
    tau_otsu = otsu(v[calif & np.isfinite(v)])
    lento = np.isfinite(v) & (v < p('V_COL_PARADA'))
    vb = feat['v_balon_suave'].to_numpy()
    balon_lento = np.isfinite(vb) & (vb <= p('V_BALON_EN_JUEGO'))
    muerto = calif & (lento & balon_lento | feat['fuera'].to_numpy())
    for a, b in tramos(calif & ~muerto):
        if b - a < p('MIN_VIVO_ENTRE_PARADAS') and a > 0 and muerto[a - 1] and (b < len(muerto)) and muerto[b]:
            muerto[a:b] = True
    for a, b in tramos(muerto):
        if b - a < p('MIN_PARADA'):
            muerto[a:b] = False
    info = {'umbral_v_col': p('V_COL_PARADA'), 'umbral_otsu_descartado': round(tau_otsu, 3), 'n_paradas': len(tramos(muerto))}
    return (muerto, info)

def detectar_muerto_v01(feat, calif):
    """Documentación pública en README.md."""
    v = feat['v_col_suave'].to_numpy()
    tau_otsu = otsu(v[calif & np.isfinite(v)])
    lento = (v < p('V_COL_PARADA')) | ~np.isfinite(v)
    en_juego = np.nan_to_num(feat['v_balon_suave'].to_numpy()) > p('V_BALON_EN_JUEGO')
    muerto = calif & (lento & ~en_juego | feat['fuera'].to_numpy())
    for a, b in tramos(calif & ~muerto):
        if b - a < p('MIN_VIVO_ENTRE_PARADAS') and a > 0 and muerto[a - 1] and (b < len(muerto)) and muerto[b]:
            muerto[a:b] = True
    for a, b in tramos(muerto):
        if b - a < p('MIN_PARADA'):
            muerto[a:b] = False
    fuera = feat['fuera'].to_numpy()
    for a, b in tramos(muerto):
        ventana = np.flatnonzero(fuera[max(0, a - p('RETRO_FUERA')):a])
        if len(ventana):
            muerto[max(0, a - p('RETRO_FUERA')) + ventana[0]:a] = True
    info = {'umbral_v_col': p('V_COL_PARADA'), 'umbral_otsu_descartado': round(tau_otsu, 3), 'n_paradas': len(tramos(muerto))}
    return (muerto, info)
