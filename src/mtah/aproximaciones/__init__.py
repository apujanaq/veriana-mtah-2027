"""Documentación pública en README.md."""
from .alejo_v02_eventos import EventosV02, ReglasV02
from .alejo_v1_fusion import FusionV1, FusionV11, FusionV12, FusionV13, FusionV2, FusionV2SinFirma, FusionV2ST
from .alejo_v2_micro import FusionV2Micro, FusionV2MicroRef
from .juliana_v3 import JulianaV3
from .juliana_v4 import JulianaV4
from .juliana_v2_micro import JulianaV2Micro
from .juliana_pares_micro import JulianaParesMicro
from .juliana_control_multiventana import JulianaControlMultiventana
from .juliana_control_pares import JulianaControlPares
from .metrica_baseline import DesdeCSV
from .v01_reglas import ReglasV01
REGISTRO = {'alejo_v01_reglas': ReglasV01, 'alejo_v02_eventos': EventosV02, 'alejo_v02_reglas': ReglasV02, 'alejo_v1_fusion': FusionV1, 'alejo_v11_fusion': FusionV11, 'alejo_v12_fusion': FusionV12, 'alejo_v13_fusion': FusionV13, 'alejo_v2_robusta': FusionV2, 'alejo_v2_sin_firma': FusionV2SinFirma, 'alejo_v2_st': FusionV2ST, 'alejo_v2_micro_ref': FusionV2MicroRef, 'juliana_v3': JulianaV3, 'juliana_v4': JulianaV4, 'juliana_v2_micro': JulianaV2Micro, 'juliana_pares_micro': JulianaParesMicro, 'juliana_control_multiventana': JulianaControlMultiventana, 'juliana_control_pares': JulianaControlPares, 'metrica_baseline_smooth50': lambda: DesdeCSV('metrica_baseline_smooth50', 'evaluation/results/baseline_smooth50.csv')}
