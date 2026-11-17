"""Documentación pública en README.md."""
import numpy as np
from .. import contrato

class MiAproximacion:
    """Documentación pública en README.md."""
    nombre = 'nombre_idea'

    def __init__(self, params=None):
        self.params = {'ejemplo_umbral_m': 1.5, **(params or {})}

    def probabilidades(self, datos):
        n = len(datos.feat)
        P = np.full((n, 3), 1 / 3)
        assert contrato.validar_probabilidades(P) is None
        return P
