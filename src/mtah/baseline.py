"""Documentación pública en README.md."""
import numpy as np
from . import ensamble, etapa_a, features
from .parametros import p
from .preprocesamiento import features_smart_tagging, validar_arrays

def predecir(atd, eventos=(), calif=None):
    """Documentación pública en README.md."""
    validar_arrays(atd)
    n = len(atd['balon'])
    calif = features.mascara_calificada(n) if calif is None else np.asarray(calif, dtype=bool)
    if calif.shape != (n,):
        raise ValueError('Máscara temporal de dimensiones inválidas')
    feat = features.calcular_features(atd, calif=calif)
    st = features_smart_tagging(eventos, n)
    muerto, info_a = etapa_a.detectar_muerto(feat, calif)
    near = feat['equipo_cercano'].to_numpy()
    close = feat['visible'].to_numpy() & (feat['d_min'].to_numpy() <= p('R_CONTACTO'))
    candidates = np.where(close, near, -1)
    missing = ~feat['visible'].to_numpy()
    candidates[missing] = st['st_equipo'].to_numpy()[missing]
    contacts = np.full(n, -1, dtype=np.int8)
    owner = np.full(n, -1, dtype=np.int8)
    for first, stop in etapa_a.tramos(calif & ~muerto):
        for team in (0, 1):
            for a, b in etapa_a.tramos(candidates[first:stop] == team):
                if b - a >= p('MIN_CONDUCCION'):
                    contacts[first + a:first + b] = team
        state = -1
        for frame in range(first, stop):
            if contacts[frame] != -1:
                state = contacts[frame]
            owner[frame] = state
    labels = ensamble.etiquetar(muerto, owner, calif)
    return dict(etiquetas=labels, feat=feat, feat_st=st, muerto=muerto, dueno=owner, equipo_contacto=contacts, calif=calif, info_a=info_a, umbrales={}, eventos_st=eventos)
