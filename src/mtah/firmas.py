"""Documentación pública en README.md."""
import numpy as np
from .etapa_a import tramos
AREA_LARGO, AREA_ANCHO = (16.5, 40.32)

def lados(jxy, eq, bloques):
    """Documentación pública en README.md."""
    out = np.ones(len(jxy), dtype=bool)
    with np.errstate(invalid='ignore'):
        for a, b in bloques:
            x = jxy[a:b, :, 0]
            out[a:b] = np.nanmean(x[:, eq == 0]) <= np.nanmean(x[:, eq == 1])
    return out

def _goles_propios(izq0, largo, ancho):
    """Documentación pública en README.md."""
    n = len(izq0)
    g = np.empty((n, 2, 2))
    g[:, 0, 0] = np.where(izq0, 0.0, largo)
    g[:, 1, 0] = np.where(izq0, largo, 0.0)
    g[:, :, 1] = ancho / 2
    return g

def firmas(jxy, eq, vel, bloques, pitch, r_grupo=4.0, margen_banda=1.0):
    """Documentación pública en README.md."""
    largo, ancho = pitch
    n = len(jxy)
    izq0 = lados(jxy, eq, bloques)
    x, y = (jxy[..., 0], jxy[..., 1])
    ok = np.isfinite(x) & np.isfinite(y)
    out = {}
    mitad = largo / 2
    with np.errstate(invalid='ignore'):
        en_propio = np.where(eq[None, :] == 0, np.where(izq0[:, None], x <= mitad, x >= mitad), np.where(izq0[:, None], x >= mitad, x <= mitad))
    frac, fuera_propio = ([], [])
    for k in (0, 1):
        m = ok & (eq[None, :] == k)
        frac.append(np.where(m.sum(1) > 0, (en_propio & m).sum(1) / np.maximum(m.sum(1), 1), np.nan))
        fuera_propio.append((m & ~en_propio).sum(1))
    out['formacion'] = np.fmin(frac[0], frac[1])
    out['en_campo_rival_max'] = np.maximum(fuera_propio[0], fuera_propio[1]).astype(np.int16)
    goles = _goles_propios(izq0, largo, ancho)
    out['portero_vel'] = np.full((n, 2), np.nan)
    out['portero_dist'] = np.full((n, 2), np.nan)
    out['rivales_area'] = np.zeros((n, 2), dtype=np.int16)
    for k in (0, 1):
        cols = np.flatnonzero(eq == k)
        d = np.hypot(x[:, cols] - goles[:, k, :1], y[:, cols] - goles[:, k, 1:])
        d = np.where(np.isfinite(d), d, np.inf)
        idx = np.argmin(d, axis=1)
        filas = np.arange(n)
        dmin = d[filas, idx]
        hay = np.isfinite(dmin)
        out['portero_dist'][hay, k] = dmin[hay]
        out['portero_vel'][hay, k] = vel[filas, cols[idx]][hay]
        rivales = np.flatnonzero(eq != k)
        dx = np.abs(x[:, rivales] - goles[:, k, :1])
        dy = np.abs(y[:, rivales] - ancho / 2)
        with np.errstate(invalid='ignore'):
            out['rivales_area'][:, k] = ((dx <= AREA_LARGO) & (dy <= AREA_ANCHO / 2)).sum(1)
    out['grupo_un_equipo'] = np.zeros(n, dtype=np.int16)
    for a in range(0, n, 5000):
        b = min(n, a + 5000)
        p = jxy[a:b]
        dist = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=3)
        cerca = np.nan_to_num(dist, nan=np.inf) <= r_grupo
        mismo = eq[:, None] == eq[None, :]
        comp = (cerca & mismo[None]).sum(2)
        riv = (cerca & ~mismo[None]).sum(2)
        val = np.where((riv == 0) & ok[a:b], comp, 0)
        out['grupo_un_equipo'][a:b] = val.max(1)
    with np.errstate(invalid='ignore'):
        out['fuera_banda'] = ((y < -margen_banda) | (y > ancho + margen_banda)).sum(1).astype(np.int16)
    return out

def episodio_previo(senal, t_fin, lo, hueco_max, pegado_max):
    """Documentación pública en README.md."""
    senal = np.asarray(senal, dtype=bool)
    t_fin, lo = (int(t_fin), int(max(lo, 0)))
    idx = np.flatnonzero(senal[lo:t_fin]) + lo
    if not len(idx) or t_fin - idx[-1] > pegado_max:
        return -1
    ini = idx[-1]
    for t in idx[::-1][1:]:
        if ini - t > hueco_max:
            break
        ini = t
    return int(ini)

def adelantar_por_firma(paradas, senal, tipos, hueco_max, pegado_max):
    """Documentación pública en README.md."""
    d = paradas.copy()
    for i in d.index:
        if str(d.at[i, 'tipo']).strip().lower() not in tipos:
            continue
        k = episodio_previo(senal, d.at[i, 't_fin'], d.at[i, 'ini_busqueda'], hueco_max, pegado_max)
        if 0 <= k < d.at[i, 't_ini']:
            d.at[i, 't_ini'], d.at[i, 'ancla_ini'] = (k, 'firma')
    if 'dur_s' in d:
        d['dur_s'] = (d['t_fin'] - d['t_ini']).clip(lower=0) / 25
    return d
AREA_META_LARGO, AREA_META_ANCHO = (5.5, 18.32)
TIPOS_INFERIBLES = ('throw-in', 'corner kick', 'goal kick', 'free kick')

def tipo_por_lugar(bxy_saque, pitch, fuera_banda=0, r_esquina=3.0, margen_banda=2.0):
    """Documentación pública en README.md."""
    largo, ancho = pitch
    x, y = (float(bxy_saque[0]), float(bxy_saque[1]))
    if not (np.isfinite(x) and np.isfinite(y)):
        return 'throw-in' if fuera_banda >= 1 else 'desconocido'
    esquinas = [(0, 0), (0, ancho), (largo, 0), (largo, ancho)]
    if min((np.hypot(x - a, y - b) for a, b in esquinas)) <= r_esquina:
        return 'corner kick'
    if min(y, ancho - y) <= margen_banda:
        return 'throw-in'
    if min(x, largo - x) <= AREA_META_LARGO and abs(y - ancho / 2) <= AREA_META_ANCHO / 2:
        return 'goal kick'
    return 'free kick'

def lugar_saque(bxy, t, ventana=5):
    """Documentación pública en README.md."""
    a, b = (max(0, int(t) - ventana), min(len(bxy), int(t) + ventana + 1))
    ok = np.flatnonzero(np.isfinite(bxy[a:b, 0]) & np.isfinite(bxy[a:b, 1]))
    return bxy[a + ok[0], :2] if len(ok) else np.array([np.nan, np.nan])

def inferir_tipos(paradas, bxy, fuera_banda, pitch):
    """Documentación pública en README.md."""
    d = paradas.copy()
    d['tipo_lugar'] = [tipo_por_lugar(lugar_saque(bxy, t), pitch, int(fuera_banda[max(0, int(t) - 25):int(t) + 1].max(initial=0))) for t in d['t_fin']]
    return d
