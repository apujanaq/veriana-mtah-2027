"""Documentación pública en README.md."""
import numpy as np
import pandas as pd
from . import contrato, features, firmas as firmas_mod, forma, temporal
from .aproximaciones import REGISTRO
from .etapa_a import tramos
from .experimento_v4 import bloques, metricas_completas
from .externos import cargar_sample
from .referencias_externas import intervalos_apoyados
PITCH_METRICA = (105.0, 68.0)
VENTANA_ANTES_S, VENTANA_DESPUES_S = (2.0, 3.0)
TIPOS_SET_PIECE = {'THROW IN': 'Throw-in', 'CORNER KICK': 'Corner Kick', 'FREE KICK': 'Free Kick', 'GOAL KICK': 'Goal Kick'}
MAX_PARADA_S = 120.0

def _mitad(periodo):
    return '1st Half' if int(periodo) == 1 else '2nd Half'

def eventos_emulados(events, periods, fps, n):
    """Documentación pública en README.md."""
    antes, despues = (round(VENTANA_ANTES_S * fps), round(VENTANA_DESPUES_S * fps))
    ordenados = events.sort_values(['Period', 'Start Frame'], kind='stable')
    primeros = ordenados.groupby('Period')['Start Frame'].min().to_dict()
    out = []
    for _, e in ordenados.iterrows():
        tipo, sub = (str(e['Type']).upper(), str(e.get('Subtype', '')).upper())
        s = int(e['Start Frame']) - 1
        equipo = {'Home': 0, 'Away': 1}.get(e.get('Team'))
        if tipo == 'SET PIECE' or (tipo == 'PASS' and 'GOAL KICK' in sub):
            if 'KICK OFF' in sub:
                nombre = 'Kick Off (Start)' if e['Start Frame'] == primeros[e['Period']] else 'Kick Off (After Goal)'
            else:
                nombre = next((v for k, v in TIPOS_SET_PIECE.items() if k in sub), None)
            if nombre is None:
                continue
            if nombre == 'Throw-in':
                equipo = None
            out.append({'code': 'SET PIECES', 'start_frame': max(0, s - antes), 'end_frame': min(n, s + despues), 'tags': {'Type': nombre, 'Half': _mitad(e['Period']), 'Team': 'N/A' if equipo is None else e['Team']}, 'equipo': equipo})
        elif tipo == 'SHOT' and 'GOAL' in sub and ('OWN' not in sub):
            out.append({'code': 'GOALS', 'start_frame': s, 'end_frame': min(n, s + round(5 * fps)), 'tags': {'Half': _mitad(e['Period']), 'Team': e.get('Team')}, 'equipo': equipo})
    return sorted(out, key=lambda e: (e['start_frame'], e['end_frame']))

def referencias_d(events, periods, fps):
    """Documentación pública en README.md."""
    ordenados = events.sort_values(['Period', 'Start Frame'], kind='stable').reset_index(drop=True)
    filas = []
    for i, e in ordenados.iterrows():
        tipo, sub = (str(e['Type']).upper(), str(e.get('Subtype', '')).upper())
        if tipo == 'BALL OUT' and 'END HALF' not in sub:
            causa, ini = ('fuera', int(e['End Frame']) - 1)
        elif tipo == 'SHOT' and 'OUT' in sub and ('GOAL' not in sub):
            causa, ini = ('fuera', int(e['End Frame']) - 1)
        elif tipo == 'PASS' and 'OFFSIDE' in sub:
            causa, ini = ('fuera_de_juego', int(e['End Frame']) - 1)
        elif tipo == 'FAULT RECEIVED':
            causa, ini = ('falta', int(e['Start Frame']) - 1)
        elif tipo == 'BALL LOST' and 'OFFSIDE' in sub:
            causa, ini = ('fuera_de_juego', int(e['Start Frame']) - 1)
        elif tipo == 'SHOT' and 'GOAL' in sub and ('OWN' not in sub):
            causa, ini = ('gol', int(e['Start Frame']) - 1)
        else:
            continue
        despues = ordenados.iloc[i + 1:]
        despues = despues[(despues['Period'] == e['Period']) & (despues['Start Frame'] > e['End Frame'])]
        es_saque = (despues['Type'] == 'SET PIECE') | (despues['Type'] == 'PASS') & despues['Subtype'].fillna('').str.upper().str.contains('GOAL KICK')
        saque = despues[es_saque]
        if saque.empty:
            continue
        fin = int(saque.iloc[0]['Start Frame']) - 1
        vivos = despues[despues['Type'].isin(['PASS', 'RECOVERY', 'SHOT']) & (despues['Start Frame'] < fin + 1) & ~es_saque.loc[despues.index]]
        if len(vivos) or not 0 < fin - ini <= MAX_PARADA_S * fps:
            continue
        filas.append({'ini': ini, 'fin': fin, 'causa': causa, 'periodo': int(e['Period'])})
    return pd.DataFrame(filas, columns=['ini', 'fin', 'causa', 'periodo']).drop_duplicates(['ini', 'fin'])

def referencias_denso(events, periods, refs_d):
    """Documentación pública en README.md."""
    n = len(periods)
    labels = np.full(n, 'X', dtype='<U1')
    sel = events[events['Team'].isin(['Home', 'Away']) & events['Type'].isin(['PASS', 'RECOVERY', 'BALL LOST', 'SHOT', 'CHALLENGE'])]
    sel = sel[~((sel['Type'] == 'CHALLENGE') & sel['Subtype'].fillna('').str.upper().str.contains('LOST'))]
    sel = sel.sort_values(['Period', 'Start Frame'], kind='stable')
    for per in np.unique(periods):
        s = sel[sel['Period'] == per]
        frames = s['Start Frame'].astype(int).to_numpy() - 1
        eq = np.where(s['Team'].to_numpy() == 'Home', 'H', 'A')
        bloque = np.flatnonzero(periods == per)
        if not len(frames) or not len(bloque):
            continue
        t = np.arange(bloque[0], bloque[-1] + 1)
        k = np.searchsorted(frames, t, side='right') - 1
        lab = np.where(k >= 0, eq[np.maximum(k, 0)], 'X')
        labels[bloque[0]:bloque[-1] + 1] = lab
    for r in refs_d.itertuples():
        labels[r.ini:r.fin] = 'D'
    return labels

def referencias_cambios(events):
    """Documentación pública en README.md."""
    rec = events[(events['Type'] == 'RECOVERY') & events['Team'].isin(['Home', 'Away'])]
    perdidas = events[(events['Type'] == 'BALL LOST') & events['Subtype'].fillna('').str.upper().str.contains('INTERCEPTION|THEFT') & events['Team'].isin(['Home', 'Away'])]
    filas = [(int(f) - 1, 0 if t == 'Home' else 1) for f, t in zip(rec['Start Frame'], rec['Team'])]
    filas += [(int(f) - 1, 1 if t == 'Home' else 0) for f, t in zip(perdidas['End Frame'], perdidas['Team'])]
    return pd.DataFrame(sorted(set(filas)), columns=['frame', 'equipo'])

def metricas_cambios(etiquetas, refs_cambios, fps, tol_s=1.0):
    """Documentación pública en README.md."""
    lab = np.asarray(etiquetas)
    vivo = np.isin(lab, ['H', 'A'])
    idx = np.flatnonzero(vivo)
    cambia = idx[1:][lab[idx[1:]] != lab[idx[:-1]]]
    pred = [(int(f), 0 if lab[f] == 'H' else 1) for f in cambia]
    ref = list(zip(refs_cambios['frame'].astype(int), refs_cambios['equipo'].astype(int)))
    tol = round(tol_s * fps)
    acierto = lambda x, lista: any((abs(x[0] - y[0]) <= tol and x[1] == y[1] for y in lista))
    rec = np.mean([acierto(y, pred) for y in ref]) if ref else np.nan
    prec = np.mean([acierto(x, ref) for x in pred]) if pred else np.nan
    return {'cambios_predichos': len(pred), 'cambios_referencia': len(ref), 'recall': round(float(rec), 3), 'precision': round(float(prec), 3)}

def referencias_toques(events):
    """Documentación pública en README.md."""
    sel = events[events['Type'].isin(['PASS', 'RECOVERY', 'SHOT']) & events['Team'].isin(['Home', 'Away'])]
    return pd.DataFrame({'frame': sel['Start Frame'].astype(int) - 1, 'equipo': sel['Team'].map({'Home': 0, 'Away': 1}).astype(int)}).reset_index(drop=True)

def error_bordes(etiquetas, refs_d, fps):
    """Documentación pública en README.md."""
    pred = tramos(np.asarray(etiquetas) == 'D')
    usados = np.zeros(len(pred), dtype=bool)
    filas = []
    for ref in refs_d.itertuples():
        solapes = [(min(b, ref.fin) - max(a, ref.ini), j) for j, (a, b) in enumerate(pred)]
        mejor = max(solapes, default=(0, -1))
        fila = {'causa': ref.causa, 'detectada': mejor[0] > 0, 'err_ini_s': np.nan, 'err_fin_s': np.nan, 'detectada_50': mejor[0] >= 0.5 * (ref.fin - ref.ini)}
        if mejor[0] > 0:
            a, b = pred[mejor[1]]
            usados[mejor[1]] = True
            fila.update(err_ini_s=(a - ref.ini) / fps, err_fin_s=(b - ref.fin) / fps)
        filas.append(fila)
    t = pd.DataFrame(filas, columns=['causa', 'detectada', 'err_ini_s', 'err_fin_s', 'detectada_50'])
    agg = dict(n=('detectada', 'size'), detectadas_pct=('detectada', lambda s: 100 * s.mean()), det_50_pct=('detectada_50', lambda s: 100 * s.mean()), ini_mediana_s=('err_ini_s', 'median'), ini_p90_abs_s=('err_ini_s', lambda s: s.abs().quantile(0.9)), fin_mediana_s=('err_fin_s', 'median'), fin_p90_abs_s=('err_fin_s', lambda s: s.abs().quantile(0.9)))
    por_causa = pd.concat([t.groupby('causa').agg(**agg), t.assign(causa='TOTAL').groupby('causa').agg(**agg)]).round(2)
    largas = [(a, b) for j, (a, b) in enumerate(pred) if not usados[j] and b - a > 2 * fps]
    extra = {'paradas_predichas': len(pred), 'sin_referencia': len(largas), 'minutos_sin_referencia': round(sum((b - a for a, b in largas)) / fps / 60, 1)}
    return (por_causa, extra)

def metricas_toques(equipo_contacto, refs_toques, fps, tol_s=0.5):
    """Documentación pública en README.md."""
    tol = round(tol_s * fps)
    pred = [(a, k) for k in (0, 1) for a, _ in tramos(np.asarray(equipo_contacto) == k)]
    ref = list(zip(refs_toques['frame'].astype(int), refs_toques['equipo'].astype(int)))
    acierto = lambda x, lista: any((abs(x[0] - y[0]) <= tol and x[1] == y[1] for y in lista))
    prec = np.mean([acierto(x, ref) for x in pred]) if pred else np.nan
    rec = np.mean([acierto(y, pred) for y in ref]) if ref else np.nan
    return {'toques_predichos': len(pred), 'toques_referencia': len(ref), 'precision': round(float(prec), 3), 'recall': round(float(rec), 3)}

def cargar_partido(folder, game):
    """Documentación pública en README.md."""
    atd, periods, fps, events = cargar_sample(folder, game)
    n = len(periods)
    calif = np.ones(n, dtype=bool)
    datos = contrato.Datos(atd=atd, feat=features.calcular_features(atd, calif, pitch=PITCH_METRICA), eventos_st=eventos_emulados(events, periods, fps, n), calif=calif, pitch=PITCH_METRICA, grupos=periods)
    refs_d = referencias_d(events, periods, fps)
    refs = {'debiles': intervalos_apoyados(events, periods, fps)[0], 'd': refs_d, 'denso': referencias_denso(events, periods, refs_d), 'cambios': referencias_cambios(events), 'toques': referencias_toques(events), 'fps': fps, 'periods': periods}
    return (datos, refs)

def etiquetas_por_periodo(aprox, datos):
    """Documentación pública en README.md."""
    if hasattr(aprox, 'etiquetas'):
        return np.asarray(aprox.etiquetas(datos))
    P = aprox.probabilidades(datos)
    lab = np.full(len(P), 'X', dtype='<U1')
    for per in np.unique(datos.grupos):
        m = datos.grupos == per
        lab[m] = temporal.decodificar(P, datos.calif & m)[0][m]
    return lab

def evaluar_partido(nombres, datos, refs):
    """Documentación pública en README.md."""
    out = {}
    for nombre in nombres:
        a = REGISTRO[nombre]()
        lab = etiquetas_por_periodo(a, datos)
        res = {'metricas': metricas_completas(lab, refs['debiles'], refs['periods']), 'denso': metricas_completas(lab, refs['denso'], refs['periods']), 'cambios': metricas_cambios(lab, refs['cambios'], refs['fps'])}
        res['bordes'], res['extra'] = error_bordes(lab, refs['d'], refs['fps'])
        if hasattr(a, '_componentes'):
            res['toques'] = metricas_toques(a._componentes(datos)['equipo'], refs['toques'], refs['fps'])
        out[nombre] = res
    return out

def huecos_sinteticos(datos, refs, comp, largos_s=(6, 10, 20), n=150, semilla=0):
    """Documentación pública en README.md."""
    rng = np.random.default_rng(semilla)
    fps = int(refs['fps'])
    den = refs['denso']
    dueno, equipo, conf = (comp['dueno'], comp['equipo'], comp['confianza'])
    vivo = datos.calif & ~comp['parada']
    visible = comp['feat']['visible'].to_numpy()
    senales = forma.senales_suavizadas(datos.atd, datos.calif, datos.grupos, datos.pitch)
    ok = np.isfinite(senales.to_numpy()).all(axis=1)
    firme = vivo & visible & (dueno >= 0) & (conf >= 0.8) & ok
    es_ha = np.isin(den, ['H', 'A'])
    filas = []
    for L_s in largos_s:
        L = int(L_s * fps)
        cand = [s for s in np.flatnonzero(es_ha & vivo)[::5] if 0 < s and s + L < len(den) and es_ha[s:s + L].all() and vivo[s - 1:s + L].all() and (dueno[s - 1] >= 0) and ok[s:s + L].all()]
        inicios = rng.choice(cand, size=min(n, len(cand)), replace=False)
        tapado = np.zeros(len(den), dtype=bool)
        for s in inicios:
            tapado[s:s + L] = True
        p_h, _ = forma.prob_forma(senales, dueno, firme & ~tapado)
        for s in inicios:
            vis_s = np.ones(len(den), dtype=bool)
            vis_s[s:s + L] = False
            ventana = slice(max(0, s - 1), min(len(den), s + L + forma.p('PUENTE_TRAS_HUECO') + 1))
            d_p, _ = forma.puente_huecos(np.full(ventana.stop - ventana.start, dueno[s - 1]), equipo[ventana], vis_s[ventana], vivo[ventana], p_h[ventana])
            ref = den[s:s + L] == 'H'
            cambia = bool((ref[1:] != ref[:-1]).any() or ref[0] != (den[s - 1] == 'H'))
            reglas = {'LOCF': np.full(L, dueno[s - 1] == 0), 'forma': p_h[s:s + L] > 0.5, 'puente': d_p[1:L + 1] == 0}
            t = np.arange(L) / fps
            for nombre, pred in reglas.items():
                filas.append(pd.DataFrame({'largo_s': L_s, 'con_cambio': cambia, 'regla': nombre, 's': t, 'acierto': pred == ref}))
    t = pd.concat(filas, ignore_index=True)
    t['tramo_s'] = pd.cut(t['s'], [-0.01, 2, 5, 10, 20], labels=['0-2', '2-5', '5-10', '10-20'])
    return t.groupby(['largo_s', 'con_cambio', 'tramo_s', 'regla'], observed=True)['acierto'].mean().unstack('regla').round(3)

def ensuciar_balon(datos, sd, tau_s=0.4, semilla=0):
    """Documentación pública en README.md."""
    atd = dict(datos.atd)
    b = np.asarray(atd['balon'], dtype=np.float64).copy()
    b[:, :2] = b[:, :2] + _ruido_ar1(len(b), sd, tau_s, semilla)
    atd['balon'] = b
    feat = features.calcular_features(atd, datos.calif, pitch=datos.pitch)
    return contrato.Datos(atd=atd, feat=feat, eventos_st=datos.eventos_st, calif=datos.calif, pitch=datos.pitch, grupos=datos.grupos)

def _ruido_ar1(n, sd, tau_s, semilla, fps=25):
    rng = np.random.default_rng(semilla)
    rho = np.exp(-1 / (tau_s * fps))
    z = rng.normal(size=(n, 2)) * sd * np.sqrt(1 - rho ** 2)
    e = np.zeros((n, 2))
    e[0] = rng.normal(size=2) * sd
    for t in range(1, n):
        e[t] = rho * e[t - 1] + z[t]
    return e

def tipo_reanudacion(events, fin):
    """Documentación pública en README.md."""
    f0 = events['Start Frame'] - 1
    e = events[(f0 >= fin) & (f0 <= fin + 5)]
    if e.empty:
        return '?'
    sub = str(e.iloc[0]['Subtype']).upper()
    for clave, tipo in (('THROW', 'banda'), ('CORNER', 'córner'), ('KICK OFF', 'centro'), ('FREE', 'falta')):
        if clave in sub:
            return tipo
    return 'meta y otros'

def validar_firmas(folder, games=(1, 2), margen_banda=0.3, semilla=0):
    """Documentación pública en README.md."""
    filas = []
    for game in games:
        datos, refs = cargar_partido(folder, game)
        _, periods, _, events = cargar_sample(folder, game)
        jug = datos.atd['jugadores']
        f = firmas_mod.firmas(jug[..., :2], datos.atd['equipo_slot'], jug[..., 2], bloques(np.ones(len(jug), bool), periods), PITCH_METRICA, margen_banda=margen_banda)
        fps = int(refs['fps'])
        cerca = np.zeros(len(jug), dtype=bool)
        for r in refs['d'].itertuples():
            cerca[max(0, r.ini - 10 * fps):r.fin + 10 * fps] = True
        vivo = np.isin(refs['denso'], ['H', 'A']) & ~cerca

        def fila(idx, tipo, fase):
            pv = np.nanmax(f['portero_vel'][idx], axis=1)
            filas.append({'tipo': tipo, 'fase': fase, 'frames': len(idx), 'celebración %': 100 * np.mean(f['grupo_un_equipo'][idx] >= 4), 'formación de saque de centro %': 100 * np.mean(f['en_campo_rival_max'][idx] <= 1), 'jugador fuera de banda %': 100 * np.mean(f['fuera_banda'][idx] >= 1), 'portero ≥ 2 m/s %': 100 * np.nanmean(pv >= 2)})
        for r in refs['d'].itertuples():
            tipo = 'gol' if r.causa == 'gol' else tipo_reanudacion(events, r.fin)
            for fase, a, b in (('0-5 s', r.ini, r.ini + 5 * fps), ('2-20 s', r.ini + 2 * fps, r.ini + 20 * fps), ('últimos 3 s', r.fin - 3 * fps, r.fin)):
                a, b = (max(a, r.ini), min(b, r.fin))
                if b > a:
                    fila(np.arange(a, b), tipo, fase)
        rng = np.random.default_rng(semilla + game)
        fila(rng.choice(np.flatnonzero(vivo), 20000, replace=False), 'juego vivo', 'muestra')
    t = pd.DataFrame(filas)
    cols = [c for c in t.columns if c.endswith('%')]
    agg = t.groupby(['tipo', 'fase']).apply(lambda g: pd.Series({'frames': int(g['frames'].sum()), **{c: round(float(np.average(g[c].fillna(0), weights=g['frames'])), 1) for c in cols}}), include_groups=False)
    return agg
