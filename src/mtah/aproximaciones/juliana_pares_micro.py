"""Documentación pública en README.md."""
import json
import numpy as np
from .. import ROOT
from ..pares_micro import refinar_pares
from ..toques_supervisados import DetectorToquesSupervisado
from .alejo_v2_micro import FusionV2Micro
from .alejo_v1_fusion import FusionV2

class JulianaParesMicro(FusionV2Micro):
    nombre = 'juliana_pares_micro'
    mantiene_probabilidades_v2 = True

    def __init__(self, params=None):
        self.config_pares = json.loads((ROOT / 'modelos/modelo_pares_micro.json').read_text(encoding='utf-8'))
        if self.config_pares['dev'] != ['J03WOY'] or self.config_pares['test_usado'] is not False or self.config_pares['metrica_para_seleccionar'] is not False:
            raise ValueError('Calibración de parejas fuera de protocolo')
        defaults = dict(MICRO_UMBRAL=self.config_pares['umbral'], MICRO_VERIFICAR_MAX=self.config_pares['max_verificar'], MICRO_TOL=self.config_pares['tolerancia'])
        super().__init__({**defaults, **(params or {})}, detector=DetectorToquesSupervisado())
        self.base = FusionV2(params={k: v for k, v in self.params.items() if not k.startswith('MICRO_')})

    def probabilidades(self, datos):
        return self.base.probabilidades(datos)

    def regimen(self, datos):
        return self.base.regimen(datos)

    def refinar_etiquetas(self, datos, lab):
        if not self.config_pares['seleccion_valida']:
            self.info_micro = dict(insertadas=0, retiradas=0, toques_confiados=0)
            return np.asarray(lab).copy()
        pt = self.detector.probabilidades_toque(datos)
        support = datos.calif & np.isfinite(np.asarray(datos.atd['balon'])[:, :2]).all(axis=1)
        noisy = self.regimen(datos) == 'ruidoso'
        new, info = refinar_pares(lab, pt, support, datos.grupos, self.params['MICRO_UMBRAL'], self.config_pares['verificar'] and noisy, round(self.params['MICRO_VERIFICAR_MAX']), round(self.params['MICRO_TOL']), round(self.params['MICRO_MAX']))
        self.info_micro = {**info, 'toques_confiados': int((pt >= self.params['MICRO_UMBRAL']).sum())}
        return new
