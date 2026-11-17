"""Documentación pública en README.md."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.preparar_entrega_equipo import elegir, preparar
from mtah.aproximaciones.alejo_v1_fusion import FusionV2
from mtah.aproximaciones.juliana_pares_micro import JulianaParesMicro
from mtah.aproximaciones.juliana_control_multiventana import JulianaControlMultiventana
from mtah.aproximaciones.juliana_control_pares import JulianaControlPares

class TestSeleccion(unittest.TestCase):

    def test_sin_evidencia_o_rechazada_conserva_base(self):
        self.assertEqual(elegir(None), 'alejo_v2_robusta')
        self.assertEqual(elegir(dict(aprox='juliana_pares_micro', aceptada=False)), 'alejo_v2_robusta')

    def test_no_admite_aceptacion_incompleta(self):
        r = dict(aprox='juliana_pares_micro', aceptada=True, concurso_evaluado=False, ajuste_con_metrica=False, criterios=[])
        self.assertEqual(elegir(r), 'alejo_v2_robusta')

    def test_admite_solo_candidata_completa_aprobada(self):
        r = dict(aprox='juliana_pares_micro', aceptada=True, concurso_evaluado=False, ajuste_con_metrica=False, criterios=[dict(partido=g, condicion=c, spell_ok=True, precision_ok=True) for g in (1, 2) for c in ('limpio', 'ruido')])
        self.assertEqual(elegir(r), 'juliana_pares_micro')
        r['concurso_evaluado'] = True
        self.assertEqual(elegir(r), 'alejo_v2_robusta')

    def test_autorizacion_antes_de_cualquier_archivo_o_lectura(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'sin_permiso'
            with self.assertRaises(PermissionError):
                preparar('NO_EXISTE', output)
            self.assertFalse(output.exists())

    def test_aproximaciones_delegan_base_sin_parametros_propios(self):
        for cls in (JulianaParesMicro, JulianaControlMultiventana, JulianaControlPares):
            aprox = cls()
            self.assertEqual(aprox.base.params, FusionV2().params)
            p = np.full((4, 3), 1 / 3)
            aprox.base.probabilidades = Mock(return_value=p)
            self.assertIs(aprox.probabilidades(object()), p)
            reconstructed = cls(params={**aprox.params, next((k for k in aprox.params if k.startswith(('MV_', 'MICRO_')))): 30})
            self.assertEqual(reconstructed.base.params, FusionV2().params)
if __name__ == '__main__':
    unittest.main()
