"""Documentación pública en README.md."""
from ..toques_supervisados import DetectorToquesSupervisado
from .alejo_v2_micro import FusionV2Micro

class JulianaV2Micro(FusionV2Micro):
    nombre = 'juliana_v2_micro'

    def __init__(self, params=None):
        detector = DetectorToquesSupervisado()
        defaults = {'MICRO_UMBRAL': detector.modelo['umbral_toque']}
        super().__init__({**defaults, **(params or {})}, detector=detector)
