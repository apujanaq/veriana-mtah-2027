"""Documentación pública en README.md."""
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
import pandas as pd
from . import carga, features
from .carga import games
ESTADOS = ('H', 'A', 'D')
N_FRAMES = games.source_frames()

@dataclass
class Datos:
    """Documentación pública en README.md."""
    atd: dict
    feat: pd.DataFrame
    eventos_st: list
    calif: np.ndarray
    extra: dict = field(default_factory=dict)
    pitch: tuple = None
    grupos: np.ndarray = None

def cargar_datos(autorizar_datos=False):
    """Documentación pública en README.md."""
    atd = carga.cargar_atd(autorizar_datos=autorizar_datos)
    calif = features.mascara_calificada(len(atd['balon']))
    feat = features.calcular_features(atd, calif)
    return Datos(atd=atd, feat=feat, eventos_st=carga.cargar_smart_tagging(autorizar_datos=autorizar_datos), calif=calif)

class Aproximacion(Protocol):
    """Documentación pública en README.md."""
    nombre: str
    params: dict

    def probabilidades(self, datos: Datos) -> np.ndarray:
        ...

def validar_probabilidades(P, n=N_FRAMES, tol=1e-06):
    """Documentación pública en README.md."""
    P = np.asarray(P)
    if P.shape != (n, len(ESTADOS)):
        return f'forma {P.shape}, se esperaba ({n}, {len(ESTADOS)})'
    if not np.isfinite(P).all():
        return f'valores no finitos: {int((~np.isfinite(P)).sum())}'
    if (P < -tol).any():
        return f'probabilidades negativas: {int((P < -tol).sum())}'
    malas = np.abs(P.sum(axis=1) - 1) > tol
    if malas.any():
        return f'filas que no suman 1: {int(malas.sum())}'
    return None

def desde_etiquetas(etiquetas, confianza=1.0):
    """Documentación pública en README.md."""
    etiquetas = np.asarray(etiquetas)
    resto = (1 - confianza) / (len(ESTADOS) - 1)
    P = np.full((len(etiquetas), len(ESTADOS)), 1 / len(ESTADOS))
    for k, e in enumerate(ESTADOS):
        m = etiquetas == e
        P[m] = resto
        P[m, k] = confianza
    return P

def a_etiquetas_argmax(P, calif):
    """Documentación pública en README.md."""
    P = np.asarray(P)
    lab = np.array(ESTADOS)[np.argmax(P, axis=1)]
    sin_opinion = P.max(axis=1) <= 1 / len(ESTADOS) + 1e-09
    return np.where(calif & ~sin_opinion, lab, 'X')
