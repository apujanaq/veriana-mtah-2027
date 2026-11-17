"""Documentación pública en README.md."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from mtah.control_multiventana import decodificar_ha, refinar_control

class TestTemporalHA(unittest.TestCase):

    def test_sin_coste_de_cambio_sigue_evidencia(self):
        z = np.r_[np.ones(30), -np.ones(30)]
        out = decodificar_ha(z, [(0, 60)], pena=0)
        self.assertEqual(''.join(out), 'H' * 30 + 'A' * 30)

    def test_persistencia_suprime_pulso_debil_no_cambio_fuerte(self):
        z = np.ones(100) * 3
        z[40:45] = -1
        self.assertTrue((decodificar_ha(z, [(0, 100)], 2) == 'H').all())
        z[40:80] = -5
        self.assertTrue((decodificar_ha(z, [(0, 100)], 2)[45:75] == 'A').all())

    def test_refinamiento_nunca_altera_d_o_x(self):
        base = np.array(list('X' * 10 + 'H' * 40 + 'D' * 10 + 'A' * 40))
        out = refinar_control(base, -np.ones(100) * 8, np.ones(100, bool), np.ones(100, bool))
        np.testing.assert_array_equal(out[base == 'D'], base[base == 'D'])
        np.testing.assert_array_equal(out[base == 'X'], base[base == 'X'])

    def test_empate_respeta_prior_de_base_y_swap(self):
        base = np.full(80, 'A')
        out = refinar_control(base, np.zeros(80), np.ones(80, bool), np.ones(80, bool))
        self.assertTrue((out == 'A').all())
        out = refinar_control(np.full(80, 'H'), np.zeros(80), np.ones(80, bool), np.ones(80, bool))
        self.assertTrue((out == 'H').all())

    def test_periodos_separados_y_fuera_de_mascara(self):
        base = np.full(100, 'H')
        z = np.r_[np.ones(50) * 8, -np.ones(50) * 8]
        mask = np.ones(100, bool)
        mask[20:25] = False
        out = refinar_control(base, z, np.ones(100, bool), mask, np.repeat([1, 2], 50), pena=100)
        self.assertTrue((out[:50] == 'H').all())
        self.assertTrue((out[50:] == 'A').all())
if __name__ == '__main__':
    unittest.main()
