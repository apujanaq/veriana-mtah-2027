"""Documentación pública en README.md."""
import numpy as np
import pandas as pd
from .carga import A, H
from .etapa_a import tramos
from .features import SIN_EQUIPO
from .parametros import p
FPS = 25

def frames_quiebre(feat, vivo, umbrales):
    """Documentación pública en README.md."""
    with np.errstate(invalid='ignore'):
        quiebre = (feat['angulo'].to_numpy() > umbrales['angulo']) | (feat['cambio_rel'].to_numpy() > umbrales['cambio_rel'])
        cerca = feat['d_min'].to_numpy() <= p('DIST_NULA')
    return vivo & feat['visible'].to_numpy() & quiebre & cerca

def _f_toque(d, sigma, dmax):
    """Documentación pública en README.md."""
    z = 1 - np.exp(-dmax ** 2 / (2 * sigma ** 2))
    return d / sigma ** 2 * np.exp(-d ** 2 / (2 * sigma ** 2)) / z

def _f_fondo(d, dmax):
    """Documentación pública en README.md."""
    return 2 * d / dmax ** 2

def em_mezcla(d, dmax, pi=0.5, sigma=1.0, tol=1e-09, iter_max=500):
    """Documentación pública en README.md."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 0) & (d <= dmax)]
    ll_prev, ll, it = (-np.inf, np.nan, 0)
    for it in range(1, iter_max + 1):
        a = pi * _f_toque(d, sigma, dmax)
        b = (1 - pi) * _f_fondo(d, dmax)
        w = a / (a + b)
        ll = float(np.log(a + b).sum())
        pi = float(w.mean())
        if w.sum() < 1e-09:
            break
        sigma = float(np.clip(np.sqrt((w * d ** 2).sum() / (2 * w.sum())), 0.05, dmax))
        if np.isfinite(ll_prev) and abs(ll - ll_prev) < tol * abs(ll_prev):
            break
        ll_prev = ll
    return {'pi': pi, 'sigma': sigma, 'n': int(len(d)), 'iteraciones': it, 'loglik': ll}

def posterior_toque(d, mezcla, dmax):
    """Documentación pública en README.md."""
    d = np.asarray(d, dtype=np.float64)
    pi, s = (mezcla['pi'], mezcla['sigma'])
    z = 1 - np.exp(-dmax ** 2 / (2 * s ** 2))
    cociente = 2 * s ** 2 * z / dmax ** 2 * np.exp(np.minimum(np.minimum(d, dmax) ** 2 / (2 * s ** 2), 700))
    post = pi / (pi + (1 - pi) * cociente)
    return np.where(np.isfinite(d) & (d <= dmax), post, 0.0)

def radio_toque(mezcla, dmax, tau):
    """Documentación pública en README.md."""
    pi, s = (mezcla['pi'], mezcla['sigma'])
    if not 0 < pi < 1:
        return 0.0 if pi <= 0 else float(dmax)
    z = 1 - np.exp(-dmax ** 2 / (2 * s ** 2))
    arg = (1 - tau) / tau * pi / (1 - pi) * dmax ** 2 / (2 * s ** 2 * z)
    return 0.0 if arg <= 1 else float(min(dmax, s * np.sqrt(2 * np.log(arg))))

def contactos(feat, vivo, umbrales, mezcla):
    """Documentación pública en README.md."""
    dmax, tau = (p('DIST_NULA'), p('TAU_TOQUE'))
    d = feat['d_min'].to_numpy()
    cercano = feat['equipo_cercano'].to_numpy()
    post = posterior_toque(d, mezcla, dmax)
    contacto = frames_quiebre(feat, vivo, umbrales) & (post >= tau)
    radio = radio_toque(mezcla, dmax, tau)
    with np.errstate(invalid='ignore'):
        cerca = vivo & feat['visible'].to_numpy() & (d <= radio)
    clave = np.where(cerca, cercano, SIN_EQUIPO)
    for k in (H, A):
        for a, b in tramos(clave == k):
            if b - a >= p('MIN_CONDUCCION'):
                contacto[a:b] = True
    equipo = np.where(contacto, cercano, SIN_EQUIPO).astype(np.int8)
    return (equipo, np.where(contacto, post, np.nan), radio)

def toque_sacador(equipo, p_toque, feat, vivo):
    """Documentación pública en README.md."""
    equipo, p_toque = (equipo.copy(), p_toque.copy())
    d = feat['d_min'].to_numpy()
    cercano = feat['equipo_cercano'].to_numpy()
    lag = p('LAG_VIAJE')
    for a, _ in tramos(np.asarray(vivo, dtype=bool)):
        w = np.arange(max(0, a - lag), min(len(d), a + lag + 1))
        w = w[np.isfinite(d[w]) & (cercano[w] >= 0)]
        if len(w) and d[w].min() <= p('R_CONTACTO'):
            equipo[a] = cercano[w[np.argmin(d[w])]]
            p_toque[a] = 1 - p('P_MIN')
    return (equipo, p_toque)

def decaimiento(visible):
    """Documentación pública en README.md."""
    visible = np.asarray(visible, dtype=bool)
    n = len(visible)
    idx = np.arange(n)
    ult_visible = np.maximum.accumulate(np.where(visible, idx, -1))
    sin_balon = np.where(ult_visible >= 0, idx - ult_visible, n)
    return np.exp(-sin_balon / (p('TAU_DECAY') * FPS))

def confianza(dueno, p_toque, visible, vivo):
    """Documentación pública en README.md."""
    n = len(dueno)
    decae = decaimiento(visible)
    out = np.full(n, 0.5)
    for a, b in tramos(np.asarray(vivo, dtype=bool)):
        s = pd.Series(p_toque[a:b]).ffill().bfill().to_numpy()
        s = np.where(np.isfinite(s), s, 0.5)
        out[a:b] = 0.5 + (s - 0.5) * decae[a:b]
    out[np.asarray(dueno) < 0] = 0.5
    return out
