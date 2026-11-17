import sys
from pathlib import Path
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.revision_local import resumir_prediccion, texto_revision

class RevisionLocalTests(unittest.TestCase):

    def test_x_propagation_does_not_cross_a_half_boundary(self):
        labels = np.array(list('HHXXHXXAXX'))
        mask = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1], bool)
        result = resumir_prediccion(labels, mask)
        self.assertEqual(result['posesiones_con_X_como_hueco'], 3)
        self.assertEqual(result['posesiones_con_X_propagado'], 2)
        self.assertEqual(result['conteos_calificados'], dict(H=3, A=1, D=0, X=4))

    def test_review_does_not_present_predictions_as_ground_truth(self):
        result = resumir_prediccion(np.array(list('HHHDDD')), np.ones(6, bool))
        text = texto_revision(dict(fuentes={}), result, dict(csv_valido=True, reproduccion_identica=True))
        self.assertIn('No contiene', text.replace('no contiene', 'No contiene'))
        self.assertIn('no conocemos sus etiquetas', text)
        self.assertIn('RESTRINGIDO', text)
if __name__ == '__main__':
    unittest.main()
