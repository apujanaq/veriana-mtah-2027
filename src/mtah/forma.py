"""Documentación pública en README.md."""
import warnings
import numpy as np
import pandas as pd
from .carga import A, H, games
from .etapa_a import tramos
from .parametros import p
FASES_ATAQUE = frozenset({'build up', 'progression', 'creating chances', 'ball in final third', 'ball in the box', 'players in the box', 'attacking transition', 'long ball'})
FASES_DEFENSA = frozenset({'defending in defensive third', 'defending in middle third', 'defending in attacking third', 'defensive transition'})
RIVAL = {H: A, A: H}
SENALES = ('dif_dispersion', 'dif_ancho', 'dif_adelanto', 'dif_velocidad')

def poseedor_st(eventos_st, n):
    """Documentación pública en README.md."""
    votos = {H: np.zeros(n, dtype=bool), A: np.zeros(n, dtype=bool)}
    for e in eventos_st:
        code = (e['code'] or '').strip().lower()
        if e['equipo'] is None or code not in FASES_ATAQUE | FASES_DEFENSA:
            continue
        k = e['equipo'] if code in FASES_ATAQUE else RIVAL[e['equipo']]
        votos[k][max(0, e['start_frame']):min(n, e['end_frame'])] = True
    out = np.full(n, -1, dtype=np.int8)
    out[votos[H] & ~votos[A]] = H
    out[votos[A] & ~votos[H]] = A
    return out

def _percentil_filas(X, q):
    """Documentación pública en README.md."""
    S = np.sort(X, axis=1)
    n = np.isfinite(X).sum(axis=1)
    pos = np.maximum(n - 1, 0) * q / 100
    lo, hi = (np.floor(pos).astype(int), np.ceil(pos).astype(int))
    filas = np.arange(len(X))
    v = S[filas, lo] + (S[filas, hi] - S[filas, lo]) * (pos - lo)
    return np.where(n > 0, v, np.nan)

def senales_forma(atd, calif, pitch=None):
    """Documentación pública en README.md."""
    jug, eq = (atd['jugadores'], np.asarray(atd['equipo_slot']))
    largo = (pitch if pitch is not None else games.GAMES[games.GAME]['pitch'])[0]
    n = len(jug)
    out = {k: np.full(n, np.nan) for k in SENALES}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for a, b in tramos(np.asarray(calif, dtype=bool)):
            x, y, v = (jug[a:b, :, 0], jug[a:b, :, 1], jug[a:b, :, 2])
            cx_medio = {k: np.nanmean(x[:, eq == k]) for k in (H, A)}
            por_equipo = {}
            for k in (H, A):
                cols = np.flatnonzero(eq == k)
                izquierda = cx_medio[k] < cx_medio[RIVAL[k]]
                mx = np.nanmean(x[:, cols], axis=0)
                ok = np.isfinite(mx)
                portero = cols[ok][np.argmin(mx[ok]) if izquierda else np.argmax(mx[ok])]
                campo = cols[cols != portero]
                xc, yc = (x[:, campo], y[:, campo])
                cx = np.nanmean(xc, axis=1)
                por_equipo[k] = {'dispersion': np.sqrt(np.nanvar(xc, axis=1) + np.nanvar(yc, axis=1)), 'ancho': _percentil_filas(yc, 90) - _percentil_filas(yc, 10), 'adelanto': cx if izquierda else largo - cx, 'velocidad': np.nanmean(v[:, campo], axis=1)}
            for nombre in ('dispersion', 'ancho', 'adelanto', 'velocidad'):
                out[f'dif_{nombre}'][a:b] = por_equipo[H][nombre] - por_equipo[A][nombre]
    return pd.DataFrame(out)

def auc(score, y):
    """Documentación pública en README.md."""
    score, y = (np.asarray(score, dtype=np.float64), np.asarray(y, dtype=bool))
    ok = np.isfinite(score)
    score, y = (score[ok], y[ok])
    n1, n0 = (int(y.sum()), int((~y).sum()))
    if n1 == 0 or n0 == 0:
        return float('nan')
    rangos = pd.Series(score).rank().to_numpy()
    return float((rangos[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def logit_irls(X, y, iter_max=50, tol=1e-10):
    """Documentación pública en README.md."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X1.shape[1])
    info = np.eye(X1.shape[1])
    for _ in range(iter_max):
        mu = 1 / (1 + np.exp(-(X1 @ beta)))
        w = mu * (1 - mu)
        info = X1.T @ (X1 * w[:, None])
        paso = np.linalg.solve(info, X1.T @ (y - mu))
        beta = beta + paso
        if np.abs(paso).max() < tol:
            break
    return (beta, np.linalg.inv(info))

def separabilidad(senales, pseudo, mascaras):
    """Documentación pública en README.md."""
    filas = {}
    for s in senales.columns:
        fila = {}
        for nombre, m in mascaras.items():
            sel = m & (pseudo >= 0)
            a = auc(senales[s].to_numpy()[sel], pseudo[sel] == H)
            fila[f'auc_{nombre}'] = a
            fila[f'separa_{nombre}'] = max(a, 1 - a)
        filas[s] = fila
    return pd.DataFrame(filas).T.round(3)

def senales_suavizadas(atd, calif, grupos=None, pitch=None):
    """Documentación pública en README.md."""
    calif = np.asarray(calif, dtype=bool)
    w = p('VENTANA_FORMA')
    S = pd.DataFrame(np.nan, index=range(len(calif)), columns=list(SENALES))
    periodos = np.unique(grupos[calif]) if grupos is not None else [None]
    for g in periodos:
        m = calif if g is None else calif & (grupos == g)
        S.loc[m] = senales_forma(atd, m, pitch=pitch).loc[m].to_numpy()
    S = S.rolling(w, center=True, min_periods=1).mean()
    S['dif_avance'] = S['dif_adelanto'].diff(w).shift(-(w // 2))
    S.loc[~calif] = np.nan
    return S

def prob_forma(senales, dueno, firme, min_frames=1000):
    """Documentación pública en README.md."""
    X = senales.to_numpy(dtype=np.float64)
    ok = np.isfinite(X).all(axis=1)
    tr = np.asarray(firme, dtype=bool) & ok & (np.asarray(dueno) >= 0)
    y = (np.asarray(dueno)[tr] == H).astype(np.float64)
    p_h = np.full(len(X), 0.5)
    if tr.sum() < min_frames or y.min() == y.max():
        return (p_h, {'n': int(tr.sum()), 'beta': None, 'auc_firme': float('nan')})
    mu, sd = (X[tr].mean(axis=0), X[tr].std(axis=0))
    sd[sd == 0] = 1.0
    beta, _ = logit_irls((X[tr] - mu) / sd, y)
    z = beta[0] + (X[ok] - mu) / sd @ beta[1:]
    p_h[ok] = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    p_h = np.clip(p_h, p('P_MIN'), 1 - p('P_MIN'))
    return (p_h, {'n': int(tr.sum()), 'beta': beta.round(3).tolist(), 'auc_firme': auc(p_h[tr], y == 1)})

def puente_huecos(dueno, equipo, visible, vivo, p_forma):
    """Documentación pública en README.md."""
    dueno = np.asarray(dueno).copy()
    equipo = np.asarray(equipo)
    vivo = np.asarray(vivo, dtype=bool)
    q = np.clip(np.asarray(p_forma, dtype=np.float64), 1e-06, 1 - 1e-06)
    tras = p('PUENTE_TRAS_HUECO')
    n_huecos, n_cambio, reasignados = (0, 0, 0)
    for i, j in tramos(vivo & ~np.asarray(visible, dtype=bool)):
        if i == 0 or not vivo[i - 1] or dueno[i - 1] < 0:
            continue
        n_huecos += 1
        antes = dueno[i - 1]
        fin = j + tras
        sigue_vivo = np.flatnonzero(~vivo[j:fin])
        fin = j + sigue_vivo[0] if len(sigue_vivo) else min(fin, len(vivo))
        toques = np.flatnonzero(equipo[j:fin] >= 0)
        if not len(toques) or equipo[j + toques[0]] == antes:
            continue
        despues, t1 = (equipo[j + toques[0]], j + toques[0])
        lp = np.log(np.where(antes == H, q[i:j], 1 - q[i:j]))
        lq = np.log(np.where(despues == H, q[i:j], 1 - q[i:j]))
        ll = np.r_[0.0, np.cumsum(lp)] + np.r_[np.cumsum(lq[::-1])[::-1], 0.0]
        k = i + int(np.argmax(ll))
        dueno[k:t1] = despues
        n_cambio += 1
        reasignados += t1 - k
    return (dueno, {'huecos': n_huecos, 'con_cambio': n_cambio, 'frames_reasignados': int(reasignados)})
