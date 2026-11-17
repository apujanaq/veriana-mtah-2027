"""Documentación pública en README.md."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from mtah.pares_micro import refinar_pares
from scripts.exportar_rama_publicable import sanear_python

class TestPares(unittest.TestCase):

    def setUp(self):
        self.lab = np.array(list('H' * 60 + 'A' * 10 + 'H' * 60))
        self.pt = np.zeros((130, 2))
        self.support = np.ones(130, bool)

    def test_verifica_los_dos_extremos(self):
        self.pt[60, 1] = 0.9
        self.pt[70, 0] = 0.9
        out, info = refinar_pares(self.lab, self.pt, self.support, umbral=0.5, verificar=True)
        np.testing.assert_array_equal(out, self.lab)
        self.assertEqual(info['retiradas'], 0)

    def test_un_solo_toque_no_confirma_posesion_breve(self):
        self.pt[60, 1] = 0.9
        out, info = refinar_pares(self.lab, self.pt, self.support, umbral=0.5, verificar=True)
        self.assertTrue((out == 'H').all())
        self.assertEqual(info['retiradas'], 1)

    def test_no_cruza_ausencia_o_periodo(self):
        self.support[65] = False
        out, _ = refinar_pares(self.lab, self.pt, self.support, umbral=0.5, verificar=True)
        np.testing.assert_array_equal(out, self.lab)
        groups = np.r_[np.ones(65), np.full(65, 2)]
        out, _ = refinar_pares(self.lab, self.pt, np.ones(130, bool), groups, 0.5, True)
        np.testing.assert_array_equal(out, self.lab)

    def test_no_modifica_d_y_no_muta_entradas(self):
        self.lab[80:90] = 'D'
        before, probability = (self.lab.copy(), self.pt.copy())
        out, _ = refinar_pares(self.lab, self.pt, self.support, umbral=0.5, verificar=True)
        np.testing.assert_array_equal(out == 'D', before == 'D')
        np.testing.assert_array_equal(self.lab, before)
        np.testing.assert_array_equal(self.pt, probability)

    def test_intercambiar_equipos_intercambia_salida(self):
        out, _ = refinar_pares(self.lab, self.pt, self.support, umbral=0.5, verificar=True)
        swapped = np.where(self.lab == 'H', 'A', 'H')
        other, _ = refinar_pares(swapped, self.pt[:, ::-1], self.support, umbral=0.5, verificar=True)
        np.testing.assert_array_equal(np.where(out == 'H', 'A', 'H'), other)

class TestExportacion(unittest.TestCase):

    def test_sanea_texto_preserva_operaciones_y_help(self):
        source = '"""PRIVADO ficticio 123"""\n# PRIVADO\nx=dict(valor=.8, origen="PRIVADO ficticio")\ndef f(a):\n    "PRIVADO"\n    return a+x["valor"]\n'
        clean = sanear_python(source, True)
        self.assertNotIn('PRIVADO', clean)
        namespace = {}
        exec(clean, namespace)
        self.assertEqual(namespace['f'](2), 2.8)
        self.assertTrue(namespace['__doc__'].splitlines()[0])
if __name__ == '__main__':
    unittest.main()
