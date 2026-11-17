import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.entrega import extraer_codigo, preparar_codigo, slug_equipo
from mtah.externos import COMMIT

class EntregaTests(unittest.TestCase):

    def model(self, base):
        model = base / 'model.json'
        model.write_text(json.dumps(dict(version=1, coeficientes=[1.0, 0.0, 0.0, 0.0], procedencia=dict(fuente='metrica-sports/sample-data', commit=COMMIT, partido='Sample_Game_1', gt_concurso_usado=False))), encoding='utf-8')
        return model

    def test_public_package_is_allowlisted_and_excludes_private_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            model = self.model(base)
            archive_path = base / 'codigo.zip'
            preparar_codigo(archive_path, model, 'Equipo sintético')
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                self.assertIn('reproducir_entrega.py', names)
                self.assertIn('modelos/modelo.json', names)
                self.assertIn('equipo_sintetico.md', names)
                self.assertFalse(any((name.endswith(('.csv', '.mp4', '.npz', '.ipynb')) for name in names)))
                self.assertFalse(any((name.startswith(('data/', 'outputs/', '.git/')) for name in names)))
                self.assertNotIn('src/mtah/diagnostico.py', names)
                self.assertNotIn('src/mtah/etapa_b.py', names)
                manifest = json.loads(archive.read('MANIFIESTO_PUBLICO.json'))
                self.assertFalse(manifest['datos_restringidos_incluidos'])
            clean = extraer_codigo(archive_path, base / 'limpio')
            self.assertTrue((clean / 'games.py').is_file())
            with self.assertRaises(FileExistsError):
                preparar_codigo(archive_path, model, 'Equipo sintético')

    def test_unverified_model_is_not_packaged(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            model = self.model(base)
            raw = json.loads(model.read_text(encoding='utf-8'))
            raw['procedencia']['gt_concurso_usado'] = True
            model.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaises(ValueError):
                preparar_codigo(base / 'bad.zip', model, 'Equipo')
            self.assertFalse((base / 'bad.zip').exists())

    def test_team_filenames_are_safe_and_windows_reserved_names_rejected(self):
        self.assertEqual(slug_equipo('Más que fútbol'), 'mas_que_futbol')
        for name in ['CON', 'NUL', '\nEquipo', '***']:
            with self.assertRaises(ValueError):
                slug_equipo(name)
if __name__ == '__main__':
    unittest.main()
