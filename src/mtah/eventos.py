"""Documentación pública en README.md."""
import numpy as np
import pandas as pd
from . import etapa_a
from .carga import games
from .etapa_a import tramos
from .parametros import p
FPS = games.GAMES[games.GAME]['fps']
CODIGO_SET_PIECE = 'set pieces'
TIPO_SAQUE_CENTRO = 'kick off (start)'
TIPO_SAQUE_TRAS_GOL = 'kick off (after goal)'
CODIGO_GOL = 'goals'
TIPO_SIN_SET_PIECE = 'sin set piece'
ANCLAS_NITIDAS = frozenset({'cruce', 'nitido', 'cota_publicada'})

def ventanas_reanudacion(eventos_st):
    """Documentación pública en README.md."""
    filas = [{'tipo': (e['tags'].get('Type') or '').strip().lower(), 'equipo': e['equipo'], 'mitad': e['tags'].get('Half'), 'ini': e['start_frame'], 'fin': e['end_frame']} for e in eventos_st if (e['code'] or '').strip().lower() == CODIGO_SET_PIECE]
    cols = ['tipo', 'equipo', 'mitad', 'ini', 'fin']
    return pd.DataFrame(filas, columns=cols).sort_values('ini', ignore_index=True)

def ventanas_gol(eventos_st):
    """Documentación pública en README.md."""
    filas = [{'equipo': e['equipo'], 'ini': e['start_frame'], 'fin': e['end_frame']} for e in eventos_st if (e['code'] or '').strip().lower() == CODIGO_GOL]
    return pd.DataFrame(filas, columns=['equipo', 'ini', 'fin']).sort_values('ini', ignore_index=True)

def candidatos_arranque(bxy, desde, hasta, v_min, lag, min_viaje):
    """Documentación pública en README.md."""
    desde, hasta = (max(1, desde), min(len(bxy) - lag, hasta))
    if hasta <= desde:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=bool))
    tramo = bxy[desde - 1:hasta + lag]
    desplaz = np.linalg.norm(tramo[lag:] - tramo[:-lag], axis=1)
    visto = np.isfinite(desplaz)
    with np.errstate(invalid='ignore'):
        viaja = desplaz > v_min * lag / FPS
    a = np.array([ini for ini, fin in tramos(viaja) if ini >= 1 and fin - ini >= min_viaje], dtype=np.int64)
    return (desde - 1 + a, visto[a - 1])

def separacion(bxy, jugadores, frames, lag):
    """Documentación pública en README.md."""
    out = np.full(len(frames), np.nan)
    for i, t in enumerate(np.asarray(frames, dtype=np.int64)):
        if t + lag >= len(bxy):
            continue
        d0 = np.linalg.norm(jugadores[t, :, :2] - bxy[t], axis=1)
        if not np.isfinite(d0).any():
            continue
        j = int(np.nanargmin(d0))
        out[i] = np.linalg.norm(jugadores[t + lag, j, :2] - bxy[t + lag]) - d0[j]
    return out

def elegir_arranque(frames, nitido, separa=None):
    """Documentación pública en README.md."""
    creible = separa if separa is not None else np.ones(len(frames), dtype=bool)
    if creible.any():
        j = int(np.flatnonzero(creible)[0])
        return (int(frames[j]), 'nitido' if nitido[j] else 'reaparece')
    if len(frames):
        return (int(frames[0]), 'nitido' if nitido[0] else 'reaparece')
    return (-1, 'sin_balon')

def _resolver_arranque(bxy, jugadores, desde, hasta, centro=None, pitch=None):
    """Documentación pública en README.md."""
    frames, nitido = candidatos_arranque(bxy, desde, hasta, p('V_BALON_EN_JUEGO'), p('LAG_VIAJE'), p('MIN_VIAJE'))
    if pitch is not None and len(frames):
        t2 = np.minimum(frames + 2 * p('LAG_VIAJE'), len(bxy) - 1)
        m, (largo, ancho) = (p('MARGEN_FUERA'), pitch)
        x, y = (bxy[t2, 0], bxy[t2, 1])
        with np.errstate(invalid='ignore'):
            dentro = (x >= -m) & (x <= largo + m) & (y >= -m) & (y <= ancho + m)
        dentro |= ~np.isfinite(x)
        frames, nitido = (frames[dentro], nitido[dentro])
    if centro is None and len(frames) and (p('SAQUE_NETO_M') > 0):
        t1 = np.minimum(frames + FPS, len(bxy) - 1)
        neto = np.linalg.norm(bxy[t1] - bxy[frames], axis=1)
        aleja = ~(neto < p('SAQUE_NETO_M'))
        frames, nitido = (frames[aleja], nitido[aleja])
    hay_centro = None
    if centro is not None and len(frames):
        en_centro = np.linalg.norm(bxy[frames] - np.asarray(centro), axis=1) <= p('R_CENTRO')
        hay_centro = bool(en_centro.any())
        if hay_centro:
            frames, nitido = (frames[en_centro], nitido[en_centro])
    separa = None
    if jugadores is not None and len(frames):
        separa = ~(separacion(bxy, jugadores, frames, p('LAG_VIAJE')) <= p('SEP_MIN'))
    t, ancla = elegir_arranque(frames, nitido, separa)
    nit = frames[nitido]
    return {'t_reanuda': t, 'ancla': ancla, 'n_nitidos': len(nit), 'n_descartados': int((nitido & ~separa).sum()) if separa is not None else 0, 'gap_s': (nit[1] - nit[0]) / FPS if len(nit) > 1 else np.nan, 'en_centro': hay_centro}

def _bloque_de(frame, bloques):
    """Documentación pública en README.md."""
    if not bloques:
        return (0, 10 ** 12)
    return min(bloques, key=lambda ab: 0 if ab[0] <= frame < ab[1] else min(abs(frame - ab[0]), abs(frame - ab[1])))

def fechar_reanudaciones(ventanas, bxy, jugadores=None, bloques=None, pitch=None, cotas_saque=None):
    """Documentación pública en README.md."""
    d = p('DELTA_VENTANA_ST')
    centro = None if pitch is None else (pitch[0] / 2, pitch[1] / 2)
    filas = []
    for ini, fin, tipo, mitad in zip(ventanas['ini'], ventanas['fin'], ventanas['tipo'], ventanas['mitad']):
        a, b = _bloque_de((ini + fin) // 2, bloques)
        if tipo == TIPO_SAQUE_CENTRO and cotas_saque and (mitad in cotas_saque):
            a = max(a, int(cotas_saque[mitad]))
        es_centro = centro if tipo in (TIPO_SAQUE_CENTRO, TIPO_SAQUE_TRAS_GOL) else None
        fila = _resolver_arranque(bxy, jugadores, max(ini - d, a), min(fin + d, b), es_centro, pitch)
        if tipo == TIPO_SAQUE_CENTRO and cotas_saque and (mitad in cotas_saque) and (fila['en_centro'] is not True):
            fila.update(t_reanuda=int(cotas_saque[mitad]), ancla='cota_publicada')
        filas.append(fila)
    out = pd.concat([ventanas, pd.DataFrame(filas, index=ventanas.index)], axis=1)
    hay = out['t_reanuda'] >= 0
    out['desfase_s'] = np.where(hay, (out['t_reanuda'] - out['ini']) / FPS, np.nan)
    out['en_ventana'] = hay & (out['t_reanuda'] >= out['ini']) & (out['t_reanuda'] < out['fin'])
    return out

def resumen_reanudaciones(r):
    """Documentación pública en README.md."""
    agg = dict(n=('ancla', 'size'), pct_nitido=('ancla', lambda s: 100 * (s == 'nitido').mean()), pct_reaparece=('ancla', lambda s: 100 * (s == 'reaparece').mean()), pct_sin_balon=('ancla', lambda s: 100 * (s == 'sin_balon').mean()), pct_ambiguas=('n_nitidos', lambda s: 100 * (s > 1).mean()), pct_descartes=('n_descartados', lambda s: 100 * (s > 0).mean()), pct_en_ventana=('en_ventana', lambda s: 100 * s.mean()), desfase_p10=('desfase_s', lambda s: s.quantile(0.1)), desfase_p50=('desfase_s', 'median'), desfase_p90=('desfase_s', lambda s: s.quantile(0.9)))
    por_tipo = r.groupby('tipo').agg(**agg).sort_values('n', ascending=False)
    total = r.assign(tipo='TOTAL').groupby('tipo').agg(**agg)
    return pd.concat([por_tipo, total]).round(1)

def saques_de_centro(r):
    """Documentación pública en README.md."""
    filas = r[r['tipo'] == TIPO_SAQUE_CENTRO]
    return {m: int(t) for m, t in zip(filas['mitad'], filas['t_reanuda'])}

def quiebre_media(y, min_antes, min_despues):
    """Documentación pública en README.md."""
    y = pd.Series(np.asarray(y, dtype=np.float64)).interpolate(limit_direction='both').to_numpy()
    n = len(y)
    if n < min_antes + min_despues or not np.isfinite(y).all():
        return (-1, np.nan, np.nan)
    s = np.cumsum(y)
    k = np.arange(min_antes, n - min_despues + 1)
    s1 = s[k - 1]
    s2 = s[-1] - s1
    j = int(np.argmax(s1 ** 2 / k + s2 ** 2 / (n - k)))
    return (int(k[j]), float(s1[j] / k[j]), float(s2[j] / (n - k[j])))

def cruces_fuera(fuera, visible, desde, hasta):
    """Documentación pública en README.md."""
    desde, hasta = (max(1, desde), min(len(fuera), hasta))
    if hasta <= desde:
        return np.empty(0, dtype=np.int64)
    f, v = (fuera[desde - 1:hasta], visible[desde - 1:hasta])
    i = np.flatnonzero(f[1:] & ~f[:-1] & v[:-1])
    return (desde + i).astype(np.int64)

def separacion_ventana(bxy, jugadores, desde, hasta, lag):
    """Documentación pública en README.md."""
    desde, hasta = (max(0, desde), min(len(bxy) - lag, hasta))
    if hasta <= desde:
        return np.empty(0)
    t = np.arange(desde, hasta)
    d0 = np.linalg.norm(jugadores[t, :, :2] - bxy[t, None, :], axis=2)
    todo_nan = ~np.isfinite(d0).any(axis=1)
    j = np.where(todo_nan, 0, np.nanargmin(np.where(np.isfinite(d0), d0, np.inf), axis=1))
    d1 = np.linalg.norm(jugadores[t + lag, j, :2] - bxy[t + lag], axis=1)
    out = d1 - d0[np.arange(len(t)), j]
    out[todo_nan] = np.nan
    return out

def ultimo_viaje(bxy, desde, hasta, v_min, lag, jugadores=None, sep_min=None, neto_min=None):
    """Documentación pública en README.md."""
    desde, hasta = (max(0, desde), min(len(bxy) - lag, hasta))
    if hasta <= desde:
        return -1
    tramo = bxy[desde:hasta + lag]
    desplaz = np.linalg.norm(tramo[lag:] - tramo[:-lag], axis=1)
    with np.errstate(invalid='ignore'):
        viaja = desplaz > v_min * lag / FPS
        if jugadores is not None and sep_min is not None:
            sep = separacion_ventana(bxy, jugadores, desde, hasta, lag)
            viaja &= ~(sep <= sep_min)
    neto_min = p('VIAJE_NETO_M') if neto_min is None else neto_min
    for a, b in reversed(tramos(viaja)):
        neto = np.linalg.norm(tramo[b - 1 + lag] - tramo[a])
        if not neto < neto_min:
            return int(desde + b - 1 + lag)
    return -1

def fechar_inicios(r, feat, calif, goles=None, grupos=None, bxy=None, jugadores=None):
    """Documentación pública en README.md."""
    vcol = feat['v_col_suave'].to_numpy()
    fuera, visible = (feat['fuera'].to_numpy(), feat['visible'].to_numpy())
    dentro = np.where(fuera[:, None], np.nan, bxy) if bxy is not None else None
    mitades = mitades_calificadas(calif, grupos)
    t_fin = np.where(r['t_reanuda'] >= 0, r['t_reanuda'], r['fin']).astype(np.int64)
    ini_nucleo = r['ini_nucleo'].to_numpy(dtype=np.float64) if 'ini_nucleo' in r else np.full(len(r), np.nan)
    tipos = r['tipo'].to_numpy()
    ini_goles = goles['ini'].to_numpy(dtype=np.int64) if goles is not None and len(goles) else np.empty(0, np.int64)
    filas, previo = ([None] * len(r), -10 ** 9)
    for i in np.argsort(t_fin, kind='stable'):
        tf = int(t_fin[i])
        a_mitad = next((a for a, b in mitades if tf < b), len(calif))
        fila = {'k_frenazo': -1, 'cruce': -1, 'caida_vcol': np.nan, 'ini_busqueda': a_mitad}
        if tipos[i] == TIPO_SAQUE_CENTRO:
            hay = tf - a_mitad >= p('MIN_PARADA')
            fila.update(t_ini=a_mitad if hay else tf, ancla_ini='inicio_mitad' if hay else 'vacia')
        elif tipos[i] == TIPO_SAQUE_TRAS_GOL:
            lo_gol = int(max(tf - p('RETRO_MAX'), a_mitad, 0))
            tags = ini_goles[(ini_goles >= lo_gol) & (ini_goles < tf)]
            if len(tags):
                lo = int(max(lo_gol, tags[-1] - p('RETRO_FUERA')))
                cruces = cruces_fuera(fuera, visible, lo, tf)
                if len(cruces):
                    t_ini, ancla, cruce = (int(cruces[0]), 'cruce', int(cruces[0]))
                else:
                    t_ini, ancla, cruce = (int(tags[-1]), 'gol_tag', -1)
            else:
                lo = int(max(lo_gol, previo + p('MIN_VIVO_ENTRE_PARADAS')))
                cruces = cruces_fuera(fuera, visible, lo, tf)
                if len(cruces):
                    t_ini, ancla, cruce = (int(cruces[-1]), 'cruce', int(cruces[-1]))
                else:
                    t_ini, ancla, cruce = (max(lo, tf - p('MIN_PARADA')), 'sin_ancla', -1)
            fila.update(t_ini=t_ini, ancla_ini=ancla, cruce=cruce, ini_busqueda=lo)
        else:
            lo = max(tf - p('RETRO_MAX'), a_mitad, previo + p('MIN_VIVO_ENTRE_PARADAS'))
            if np.isfinite(ini_nucleo[i]):
                lo = max(lo, ini_nucleo[i] - p('RETRO_FRENAZO'))
            lo = int(min(lo, tf))
            k_rel, m1, m2 = quiebre_media(vcol[lo:tf], p('MIN_VIVO_ENTRE_PARADAS'), p('MIN_PARADA'))
            k = lo + k_rel if k_rel >= 0 and m2 < m1 else -1
            cruces = cruces_fuera(fuera, visible, lo, tf)
            if k >= 0:
                cruces = cruces[np.abs(cruces - k) < p('RETRO_FUERA')]
                cruce = int(cruces[0]) if len(cruces) else -1
            else:
                cruce = int(cruces[-1]) if len(cruces) else -1
            bp = ultimo_viaje(dentro, max(lo, tf - p('RETRO_BALON')), tf, p('V_BALON_EN_JUEGO'), p('LAG_VIAJE'), jugadores, p('SEP_MIN')) if bxy is not None else -1
            if cruce >= 0 and bxy is not None:
                viaje_dentro = ultimo_viaje(dentro, cruce, tf, p('V_BALON_EN_JUEGO'), p('LAG_VIAJE'), jugadores, p('SEP_MIN'))
                if viaje_dentro > cruce + p('LAG_VIAJE') + p('TOLERANCIA_CRUCE'):
                    cruce = -1
                    if bp < 0 or tf - viaje_dentro >= p('MIN_PARADA'):
                        bp = viaje_dentro
            if cruce >= 0:
                t_ini, ancla = (cruce, 'cruce')
            elif bp >= 0 and tf - bp >= p('MIN_PARADA'):
                t_ini, ancla = (bp, 'balon_para')
            elif k >= 0 and (bxy is None or ultimo_viaje(dentro, k, tf, p('V_BALON_EN_JUEGO'), p('LAG_VIAJE'), jugadores, p('SEP_MIN')) < 0):
                t_ini, ancla = (k, 'frenazo')
            else:
                t_ini, ancla = (max(lo, tf - p('MIN_PARADA')), 'sin_ancla')
            fila.update(t_ini=t_ini, ancla_ini=ancla, k_frenazo=k, cruce=cruce, caida_vcol=m1 - m2 if k >= 0 else np.nan, ini_busqueda=lo)
        filas[i] = fila
        previo = tf
    d = pd.concat([r, pd.DataFrame(filas, index=r.index)], axis=1)
    d['t_fin'] = t_fin
    d['ancla_fin'] = np.where(r['t_reanuda'] >= 0, r['ancla'], 'sin_arranque')
    ambos = (d['cruce'] >= 0) & (d['k_frenazo'] >= 0)
    desfase = int(np.median(d.loc[ambos, 'k_frenazo'] - d.loc[ambos, 'cruce'])) if ambos.any() else 0
    fr = d['ancla_ini'] == 'frenazo'
    d.loc[fr, 't_ini'] = np.maximum(d.loc[fr, 'ini_busqueda'], d.loc[fr, 'k_frenazo'] - desfase)
    d.loc[d['t_ini'] >= d['t_fin'], 'ancla_ini'] = 'vacia'
    d['dur_s'] = np.maximum(d['t_fin'] - d['t_ini'], 0) / FPS
    return (d, desfase)

def mitades_calificadas(calif, grupos=None):
    """Documentación pública en README.md."""
    out = []
    for a, b in tramos(np.asarray(calif, dtype=bool)):
        cortes = [] if grupos is None else (np.flatnonzero(np.diff(np.asarray(grupos)[a:b]) != 0) + a + 1).tolist()
        bordes = [a, *cortes, b]
        out += [(bordes[i], bordes[i + 1]) for i in range(len(bordes) - 1)]
    return out

def mascara_paradas(d, n):
    """Documentación pública en README.md."""
    m = np.zeros(n, dtype=bool)
    for a, b in zip(d['t_ini'], d['t_fin']):
        m[max(0, int(a)):max(0, int(b))] = True
    return m

def resumen_paradas(d, por='ancla_ini'):
    """Documentación pública en README.md."""
    agg = dict(n=('dur_s', 'size'), dur_p10=('dur_s', lambda s: s.quantile(0.1)), dur_p50=('dur_s', 'median'), dur_p90=('dur_s', lambda s: s.quantile(0.9)), min_muertos=('dur_s', lambda s: s.sum() / 60))
    por_valor = d.groupby(por).agg(**agg).sort_values('n', ascending=False)
    total = d.assign(**{por: 'TOTAL'}).groupby(por).agg(**agg)
    return pd.concat([por_valor, total]).round(1)

def nucleos_quietud(feat, calif):
    """Documentación pública en README.md."""
    muerto, _ = etapa_a.detectar_muerto_v01(feat, calif)
    return tramos(muerto)

def reanudaciones_sin_set_piece(nucleos, cubiertas, bxy, jugadores=None, pitch=None, visible=None):
    """Documentación pública en README.md."""
    m = p('DELTA_NUCLEO')
    cub = np.convolve(np.asarray(cubiertas, dtype=np.int32), np.ones(2 * m + 1, dtype=np.int32), mode='same') > 0
    for _, fin_sp in tramos(np.asarray(cubiertas, dtype=bool)):
        cub[fin_sp:fin_sp + p('MIN_VIVO_TRAS_SAQUE')] = True
    filas = []
    for a, b in nucleos:
        if cub[a:b].any():
            continue
        if visible is not None and b > a and (np.asarray(visible)[a:b].mean() > p('VISIBLE_MAX_PARADA')):
            continue
        fila = _resolver_arranque(bxy, jugadores, b - m, b + m, pitch=pitch)
        fila.update(tipo=TIPO_SIN_SET_PIECE, equipo=None, mitad=None, ini=a, fin=b, ini_nucleo=a, desfase_s=np.nan, en_ventana=False)
        filas.append(fila)
    cols = ['tipo', 'equipo', 'mitad', 'ini', 'fin', 't_reanuda', 'ancla', 'n_nitidos', 'n_descartados', 'gap_s', 'desfase_s', 'en_ventana', 'ini_nucleo']
    df = pd.DataFrame(filas, columns=cols).sort_values('ini', ignore_index=True)
    con = df['t_reanuda'] >= 0
    df.loc[con, 'fin'] = df[con].groupby('t_reanuda')['fin'].transform('max')
    df = df[~(con & df.duplicated('t_reanuda'))]
    unidos = []
    for fila in df.to_dict('records'):
        anterior = unidos[-1] if unidos else None
        if anterior is not None and anterior['t_reanuda'] < 0 and (fila['ini'] - anterior['fin'] <= m):
            unidos.pop()
            fila = {**fila, 'ini': anterior['ini'], 'ini_nucleo': anterior['ini_nucleo']}
        unidos.append(fila)
    return pd.DataFrame(unidos, columns=cols)

def _ancho_rampa(ancla):
    if ancla == 'inicio_mitad':
        return 0
    return p('RAMPA_NITIDA') if ancla in ANCLAS_NITIDAS else p('RAMPA_DIFUSA')

def prob_muerto(d, n):
    """Documentación pública en README.md."""
    out = np.zeros(n)
    for a, b, ai, af in zip(d['t_ini'], d['t_fin'], d['ancla_ini'], d['ancla_fin']):
        if b <= a:
            continue
        wi, wf = (max(_ancho_rampa(ai), 1e-09), max(_ancho_rampa(af), 1e-09))
        lo, hi = (max(0, int(a - wi)), min(n, int(b + wf) + 1))
        t = np.arange(lo, hi)
        sube = np.clip(0.5 + (t - a + 0.5) / wi, 0, 1)
        baja = np.clip(0.5 + (b - 0.5 - t) / wf, 0, 1)
        out[lo:hi] = np.maximum(out[lo:hi], np.minimum(sube, baja))
    return np.clip(out, p('P_MIN'), 1 - p('P_MIN'))

def paradas_completas(feat, calif, eventos_st, atd, grupos=None, pitch=None, cotas_saque=None):
    """Documentación pública en README.md."""
    n = len(calif)
    bxy = atd['balon'][:, :2].astype(np.float64)
    jug = atd['jugadores']
    goles = ventanas_gol(eventos_st)
    bloques = mitades_calificadas(calif, grupos)
    if pitch is None:
        pitch = games.GAMES[games.GAME]['pitch']
    r_sp = fechar_reanudaciones(ventanas_reanudacion(eventos_st), bxy, jug, bloques, pitch, cotas_saque)
    d_sp, _ = fechar_inicios(r_sp, feat, calif, goles, grupos, bxy, jug)
    nucleos = nucleos_quietud(feat, calif)
    r_sf = reanudaciones_sin_set_piece(nucleos, mascara_paradas(d_sp, n), bxy, jug, pitch, feat['visible'].to_numpy())
    d, desfase = fechar_inicios(pd.concat([r_sp, r_sf], ignore_index=True), feat, calif, goles, grupos, bxy, jug)
    info = {'nucleos_quietud': len(nucleos), 'paradas_sin_set_piece': len(r_sf), 'paradas_con_set_piece': len(r_sp)}
    return (d, desfase, info)
