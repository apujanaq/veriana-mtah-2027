"""Documentación pública en README.md."""
import numpy as np
from .. import microposesiones
from ..carga import A, H
from ..parametros import p, sobrescribir
from .alejo_v02_eventos import EventosV02
from .alejo_v1_fusion import FusionV2

class DetectorToqueV02:
    """Documentación pública en README.md."""

    def probabilidades_toque(self, datos):
        c = EventosV02()._componentes(datos)
        pt = np.nan_to_num(np.asarray(c['p_toque'], dtype=np.float64), nan=0.0)
        out = np.zeros((len(pt), 2))
        for k in (H, A):
            out[:, k] = np.where(c['equipo'] == k, pt, 0.0)
        return out

class FusionV2Micro(FusionV2):
    """Documentación pública en README.md."""
    nombre = 'alejo_v2_micro'

    def __init__(self, params=None, detector=None):
        super().__init__(params)
        self.params = {**self.params, 'MICRO_MAX': p('MICRO_MAX'), 'MICRO_UMBRAL': p('MICRO_UMBRAL'), **(params or {})}
        self.detector = detector
        self.info_micro = {}

    def etiquetas(self, datos):
        lab = super().etiquetas(datos)
        return self.refinar_etiquetas(datos, lab)

    def refinar_etiquetas(self, datos, lab):
        """Documentación pública en README.md."""
        if self.detector is None:
            return np.asarray(lab).copy()
        pt = self.detector.probabilidades_toque(datos)
        soporte = datos.calif & np.isfinite(np.asarray(datos.atd['balon'])[:, :2]).all(axis=1)
        with sobrescribir(**self.params):
            lab, n = microposesiones.insertar_microposesiones(lab, pt, grupos=datos.grupos, calif=soporte)
        self.info_micro = {'insertadas': n, 'toques_confiados': int((np.asarray(pt) >= self.params['MICRO_UMBRAL']).sum())}
        return lab

class FusionV2MicroRef(FusionV2Micro):
    """Documentación pública en README.md."""
    nombre = 'alejo_v2_micro_ref'

    def __init__(self, params=None):
        super().__init__(params, detector=DetectorToqueV02())
