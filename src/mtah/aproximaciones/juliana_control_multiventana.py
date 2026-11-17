"""Documentación pública en README.md."""
import json
import numpy as np
from .. import ROOT
from ..control_multiventana import potencial_control, refinar_control
from .alejo_v1_fusion import FusionV2
from .alejo_v2_micro import FusionV2Micro

class JulianaControlMultiventana(FusionV2Micro):
    nombre = 'juliana_control_multiventana'
    mantiene_probabilidades_v2 = True

    def __init__(self, params=None):
        self.model_control = json.loads((ROOT / 'modelos/control_multiventana.json').read_text(encoding='utf-8'))
        temporal = self.model_control['temporal']
        defaults = dict(MV_PENA=temporal['pena'], MV_ESCALA=temporal['escala'])
        super().__init__({**defaults, **(params or {})})
        self.base = FusionV2(params={k: v for k, v in self.params.items() if not k.startswith(('MICRO_', 'MV_'))})

    def probabilidades(self, datos):
        return self.base.probabilidades(datos)

    def refinar_etiquetas(self, datos, lab):
        self.info_micro = dict(insertadas=0, retiradas=0, toques_confiados=0)
        if self.base.regimen(datos) != 'ruidoso':
            return np.asarray(lab).copy()
        z, valid = potencial_control(datos, self.model_control)
        new = refinar_control(lab, z, valid, datos.calif, datos.grupos, self.params['MV_PENA'], self.params['MV_ESCALA'])
        if not np.array_equal(new == 'D', np.asarray(lab) == 'D'):
            raise AssertionError('El control H/A alteró D')
        return new
