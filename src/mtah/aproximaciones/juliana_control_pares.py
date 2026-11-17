"""Documentación pública en README.md."""
from .juliana_control_multiventana import JulianaControlMultiventana
from .juliana_pares_micro import JulianaParesMicro

class JulianaControlPares(JulianaControlMultiventana):
    nombre = 'juliana_control_pares'

    def __init__(self, params=None):
        super().__init__(params)
        self.pares = JulianaParesMicro(params={k: v for k, v in (params or {}).items() if not k.startswith('MV_')})
        self.params = {**self.params, **self.pares.params, **(params or {})}

    def refinar_etiquetas(self, datos, lab):
        control = super().refinar_etiquetas(datos, lab)
        out = self.pares.refinar_etiquetas(datos, control)
        self.info_micro = self.pares.info_micro
        return out
