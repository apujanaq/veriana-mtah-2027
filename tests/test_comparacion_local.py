import sys
from pathlib import Path
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.comparacion_local import comparar_etiquetas

class ComparacionLocalTests(unittest.TestCase):

    def test_disagreement_is_counted_without_assuming_one_prediction_is_truth(self):
        old, new = (np.array(list('HHADX')), np.array(list('HAADX')))
        result = comparar_etiquetas(old, new, np.ones(5, bool))
        self.assertEqual(result['cambios_calificados'], 1)
        self.assertEqual(result['matriz_cambios']['H']['A'], 1)
        self.assertNotIn('macroF1', result)
        self.assertIn('Ninguna se considera verdad', result['advertencia'])

    def test_ungraded_disagreement_is_ignored(self):
        result = comparar_etiquetas(np.array(list('HHAA')), np.array(list('AHAA')), np.array([False, True, True, True]))
        self.assertEqual(result['cambios_calificados'], 0)
if __name__ == '__main__':
    unittest.main()
