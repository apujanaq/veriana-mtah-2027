"""Documentación pública en README.md."""
import numpy as np
import pandas as pd
from .etapa_a import tramos
from .features import SIN_EQUIPO
from .parametros import p

def umbrales_nulos(feat, calif):
    """Documentación pública en README.md."""
    nulo = calif & feat['visible'].to_numpy() & (feat['d_min'].to_numpy() > p('DIST_NULA'))
    ang = feat['angulo'].to_numpy()[nulo]
    rel = feat['cambio_rel'].to_numpy()[nulo]
    ang, rel = (ang[np.isfinite(ang)], rel[np.isfinite(rel)])
    pct = 100 * (1 - p('ALFA_NULO'))
    return {'angulo': float(np.percentile(ang, pct)) if len(ang) else float('inf'), 'cambio_rel': float(np.percentile(rel, pct)) if len(rel) else float('inf'), 'n_nulo': int(len(ang))}

def detectar_contactos(feat, calif, umbrales):
    """Documentación pública en README.md."""
    cercano = feat['equipo_cercano'].to_numpy()
    cerca = calif & feat['visible'].to_numpy() & (feat['d_min'].to_numpy() <= p('R_CONTACTO'))
    with np.errstate(invalid='ignore'):
        quiebre = (feat['angulo'].to_numpy() > umbrales['angulo']) | (feat['cambio_rel'].to_numpy() > umbrales['cambio_rel'])
    contacto = cerca & quiebre
    clave = np.where(cerca, cercano, SIN_EQUIPO)
    for k in (0, 1):
        for a, b in tramos(clave == k):
            if b - a >= p('MIN_CONDUCCION'):
                contacto[a:b] = True
    return np.where(contacto, cercano, SIN_EQUIPO).astype(np.int8)

def asignar_dueno(equipo_contacto, muerto, calif):
    """Documentación pública en README.md."""
    dueno = np.full(len(muerto), SIN_EQUIPO, dtype=np.int8)
    for a, b in tramos(calif & ~muerto):
        s = pd.Series(equipo_contacto[a:b], dtype='float64').replace(SIN_EQUIPO, np.nan)
        dueno[a:b] = s.ffill().bfill().fillna(SIN_EQUIPO).to_numpy().astype(np.int8)
    return dueno
