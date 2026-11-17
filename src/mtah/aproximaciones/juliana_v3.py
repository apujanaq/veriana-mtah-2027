"""Documentación pública en README.md."""
import numpy as np
from .. import ROOT
from ..candidata_v3 import cargar_v3, predecir_v3
from ..temporal_tasas_fijas import ConfigTemporal
MODELO = ROOT / 'modelos' / 'modelo_v3.json'

class JulianaV3:
    """Documentación pública en README.md."""
    nombre = 'juliana_v3'

    def __init__(self, params=None):
        self.params = {}
        self.modelo = cargar_v3(MODELO)

    def prediccion(self, datos):
        """Documentación pública en README.md."""
        if 'juliana_v3' not in datos.extra:
            datos.extra['juliana_v3'] = predecir_v3(datos.atd, datos.eventos_st, datos.calif, grupos=datos.grupos, model=self.modelo)
        return datos.extra['juliana_v3']

    def probabilidades(self, datos):
        pred = self.prediccion(datos)
        escala = pred['configuracion']['features']['fps'] * ConfigTemporal().independencia_s
        z = pred['evidencia']['logits'] / escala
        z = z - z.max(axis=1, keepdims=True)
        P = np.exp(z)
        return P / P.sum(axis=1, keepdims=True)

    def etiquetas(self, datos):
        return np.asarray(self.prediccion(datos)['etiquetas'])
