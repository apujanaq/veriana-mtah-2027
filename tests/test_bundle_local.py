import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah import carga
from mtah.bundle_local import extraer_zip, main, planificar_zip

class BundleLocalTests(unittest.TestCase):

    def build(self, path, extra=()):
        with zipfile.ZipFile(path, 'w') as archive:
            for source in ('tactical', 'broadcast'):
                for name in carga.games.asset_paths(source=source).values():
                    archive.writestr('bundle/data/' + source + '/' + Path(name).name, b'synthetic asset')
            archive.writestr('bundle/private_possession_gt.csv', b'not an input')
            for name, content in extra:
                archive.writestr(name, content)

    def test_nested_zip_extracts_only_automatic_assets_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            original = base / 'data.zip'
            self.build(original)
            original_bytes = original.read_bytes()
            target = base / 'data' / 'new_bundle'
            result = extraer_zip(original, target, limite=base / 'data')
            self.assertEqual(result, target.resolve())
            self.assertEqual(len(planificar_zip(original)), 9)
            self.assertFalse((target / 'private_possession_gt.csv').exists())
            self.assertEqual(original.read_bytes(), original_bytes)
            metadata = json.loads((target / 'extraccion_local.json').read_text(encoding='utf-8'))
            self.assertEqual(len(metadata['assets']), 9)
            with self.assertRaises(FileExistsError):
                extraer_zip(original, target, limite=base / 'data')

    def test_traversal_rejected_before_creating_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            original, target = (base / 'data.zip', base / 'data' / 'new_bundle')
            self.build(original, [('../escape.txt', b'bad')])
            with self.assertRaises(ValueError):
                extraer_zip(original, target, limite=base / 'data')
            self.assertFalse(target.exists())

    def test_duplicate_asset_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / 'data.zip'
            self.build(original, [('duplicate/BAR-ATH_tactical.mp4', b'ambiguous')])
            with self.assertRaises(ValueError):
                planificar_zip(original)

    def test_destination_cannot_escape_allowed_root(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            original = base / 'data.zip'
            self.build(original)
            with self.assertRaises(ValueError):
                extraer_zip(original, base / 'outside', limite=base / 'data')

    def test_private_error_is_logged_locally_without_leaking_to_console(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            data, output = (base / 'data' / 'bundle', base / 'outputs' / 'run')
            data.mkdir(parents=True)
            model = base / 'model.json'
            model.write_text(json.dumps(dict(version=1, coeficientes=[1.0, 0.0, 0.0, 0.0], procedencia=dict(fuente='metrica-sports/sample-data', gt_concurso_usado=False))), encoding='utf-8')
            console = io.StringIO()
            with patch('mtah.bundle_local.ROOT', base), patch('mtah.bundle_local.correr_eda', side_effect=ValueError('SYNTHETIC_PRIVATE_CONTENT')), contextlib.redirect_stdout(console):
                code = main(['--data-dir', str(data), '--salida', str(output), '--modelo-control', str(model), '--autorizar-datos-restringidos'])
            self.assertEqual(code, 1)
            self.assertNotIn('SYNTHETIC_PRIVATE_CONTENT', console.getvalue())
            self.assertIn('Fallo técnico', console.getvalue())
            self.assertIn('SYNTHETIC_PRIVATE_CONTENT', (output / 'ejecucion_local.log').read_text(encoding='utf-8'))
if __name__ == '__main__':
    unittest.main()
