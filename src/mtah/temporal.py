"""Documentación pública en README.md."""
import numpy as np
from .etapa_a import tramos
FPS = 25
EPS = 1e-06
PARAMS_DEFECTO = {'alpha': 0.5, 'dur_inicial_pos_s': 10.0, 'dur_inicial_D_s': 30.0, 'prior_uniforme': False, 'iter_max': 30, 'tol': 0.0001}

def matriz_inicial(dur_pos_s, dur_d_s):
    """Documentación pública en README.md."""
    sp, sd = (1 - 1 / (dur_pos_s * FPS), 1 - 1 / (dur_d_s * FPS))
    return np.array([[sp, 0.7 * (1 - sp), 0.3 * (1 - sp)], [0.7 * (1 - sp), sp, 0.3 * (1 - sp)], [0.5 * (1 - sd), 0.5 * (1 - sd), sd]])

def _emisiones(P, prior, alpha):
    B = (np.clip(P, EPS, 1) / prior) ** alpha
    return B / B.sum(axis=1, keepdims=True)

def _forward_backward(B, T, pi0):
    """Documentación pública en README.md."""
    n = len(B)
    alpha = np.empty_like(B)
    c = np.empty(n)
    a = pi0 * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, n):
        a = alpha[t - 1] @ T * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.empty_like(B)
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = T @ (B[t + 1] * beta[t + 1]) / c[t + 1]
    gamma = alpha * beta
    xi = T * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
    return (gamma, xi, float(np.log(c).sum()))

def _viterbi(B, T, pi0):
    n, k = B.shape
    logT, logB = (np.log(T), np.log(B))
    delta = np.log(pi0) + logB[0]
    atras = np.empty((n, k), dtype=np.int8)
    for t in range(1, n):
        cand = delta[:, None] + logT
        atras[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + logB[t]
    ruta = np.empty(n, dtype=np.int8)
    ruta[-1] = delta.argmax()
    for t in range(n - 1, 0, -1):
        ruta[t - 1] = atras[t, ruta[t]]
    return ruta

def decodificar(P, calif, params=None):
    """Documentación pública en README.md."""
    prm = {**PARAMS_DEFECTO, **(params or {})}
    bloques = tramos(calif)
    prior = np.clip(np.concatenate([P[a:b] for a, b in bloques]).mean(axis=0), EPS, 1)
    if prm['prior_uniforme']:
        prior = np.full(3, 1 / 3)
    Bs = [_emisiones(P[a:b], prior, prm['alpha']) for a, b in bloques]
    T = matriz_inicial(prm['dur_inicial_pos_s'], prm['dur_inicial_D_s'])
    pi0 = np.full(3, 1 / 3)
    ll_prev, ll, it = (-np.inf, float('nan'), 0)
    for it in range(1, prm['iter_max'] + 1):
        xi_tot, ll = (np.zeros((3, 3)), 0.0)
        for B in Bs:
            _, xi, l = _forward_backward(B, T, pi0)
            xi_tot += xi
            ll += l
        T = xi_tot / xi_tot.sum(axis=1, keepdims=True)
        if np.isfinite(ll_prev) and abs(ll - ll_prev) < prm['tol'] * abs(ll_prev):
            break
        ll_prev = ll
    etiquetas = np.full(len(P), 'X', dtype='<U1')
    for (a, b), B in zip(bloques, Bs):
        etiquetas[a:b] = np.array(['H', 'A', 'D'])[_viterbi(B, T, pi0)]
    info = {'T': T.round(6), 'dur_implicita_s': {e: round(1 / max(1 - T[i, i], EPS) / FPS, 1) for i, e in enumerate('HAD')}, 'iteraciones': it, 'loglik': round(ll, 1)}
    return (etiquetas, info)
