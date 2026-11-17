"""Documentación pública en README.md."""
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from mtah import microposesiones as mp

def toques(n, eventos):
    pt = np.zeros((n, 2))
    for f, k in eventos:
        pt[f, k] = 0.9
    return pt

class TestInsertar(unittest.TestCase):

    def test_rebote_del_rival_se_inserta(self):
        lab = np.array(list('H' * 30))
        nuevo, n = mp.insertar_microposesiones(lab, toques(30, [(10, 1), (20, 0)]), 0.5, 25)
        self.assertEqual(n, 1)
        self.assertEqual(''.join(nuevo[10:20]), 'A' * 10)
        self.assertTrue((nuevo[:10] == 'H').all() and (nuevo[20:] == 'H').all())

    def test_no_cruza_una_parada(self):
        lab = np.array(list('H' * 12 + 'D' * 5 + 'H' * 13))
        nuevo, n = mp.insertar_microposesiones(lab, toques(30, [(10, 1), (20, 0)]), 0.5, 25)
        self.assertEqual(n, 0)
        self.assertTrue((nuevo == lab).all())

    def test_cambio_ya_decodificado_no_se_toca(self):
        lab = np.array(list('H' * 15 + 'A' * 15))
        nuevo, n = mp.insertar_microposesiones(lab, toques(30, [(10, 1), (20, 0)]), 0.5, 25)
        self.assertEqual(n, 0)
        self.assertTrue((nuevo == lab).all())

    def test_sin_recuperacion_a_tiempo_no_es_micro(self):
        lab = np.array(list('H' * 100))
        nuevo, n = mp.insertar_microposesiones(lab, toques(100, [(10, 1), (80, 0)]), 0.5, 50)
        self.assertEqual(n, 0)

class TestMetricas(unittest.TestCase):

    def test_prediccion_perfecta(self):
        ref = np.array(list('H' * 100 + 'A' * 20 + 'H' * 100))
        m = mp.metricas_micro(ref, ref, np.ones(len(ref), int), umbral_s=1.0)
        self.assertEqual((m['micro_ref'], m['recall_micro'], m['precision_micro']), (1, 1.0, 1.0))

    def test_micro_perdida(self):
        ref = np.array(list('H' * 100 + 'A' * 20 + 'H' * 100))
        pred = np.array(list('H' * 220))
        m = mp.metricas_micro(pred, ref, np.ones(len(ref), int), umbral_s=1.0)
        self.assertEqual((m['micro_ref'], m['micro_pred'], m['recall_micro']), (1, 0, 0.0))
if __name__ == '__main__':
    unittest.main()
