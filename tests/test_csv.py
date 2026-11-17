import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.ensamble import escribir_csv
from mtah.validacion import main, validar_csv

class CsvTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'synthetic.csv'

    def test_deterministic_utf8_output_and_no_overwrite(self):
        labels = ['H', 'A', 'D', 'X']
        escribir_csv(labels, self.path, 4)
        expected = b'frame,label\n0,H\n1,A\n2,D\n3,X\n'
        self.assertEqual(self.path.read_bytes(), expected)
        with self.assertRaises(FileExistsError):
            escribir_csv(labels, self.path, 4)
        self.assertEqual(self.path.read_bytes(), expected)

    def test_invalid_arrays_do_not_create_output(self):
        for labels in [['H'], ['H', 'B'], ['H', ''], [['H'], ['A']]]:
            with self.assertRaises(ValueError):
                escribir_csv(labels, self.path, 2)
            self.assertFalse(self.path.exists())

    def test_reject_invalid_csv_contract(self):
        cases = ['frame,label\n1,H\n2,A\n', 'frame,label\n0,H\n0,A\n', 'frame,label\n0,H\n', 'frame,label\n0,H\n1,\n', 'frame,label\n0,H\n1,B\n', 'frame,label\n0.0,H\n1.0,A\n', 'frame,label,extra\n0,H,foo\n1,A,foo\n', 'label,frame\nH,0\nA,1\n', 'index,label\n0,H\n1,A\n']
        for index, text in enumerate(cases):
            with self.subTest(index=index):
                self.path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    validar_csv(self.path, 2)

    def test_official_header_and_label_case_tolerance(self):
        self.path.write_text(' Frame , Label \n0, h \n1,a\n', encoding='utf-8')
        self.assertTrue(validar_csv(self.path, 2))

    def test_full_length_synthetic_csv_passes_official_cli(self):
        labels = np.full(199319, 'X')
        escribir_csv(labels, self.path)
        result = subprocess.run([sys.executable, str(ROOT / 'evaluation' / 'validate_submission.py'), str(self.path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('OK', result.stdout)
        other = self.path.with_name('second.csv')
        escribir_csv(labels, other)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).digest(), hashlib.sha256(other.read_bytes()).digest())

    def test_local_cli_invalid_exit_code(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.path)]), 1)

    def test_full_cli_from_synthetic_bundle_is_reproducible(self):
        tactical = Path(self.temp.name) / 'tactical'
        tactical.mkdir()
        meta = '<Root><Team id="h"/><Team id="a"/><Player teamId="h"/><Player teamId="a"/></Root>'
        (tactical / 'BAR-ATH_tactical_atd.xml').write_text(meta, encoding='utf-8')
        with (tactical / 'BAR-ATH_tactical_atd.txt').open('w', encoding='utf-8') as raw:
            raw.writelines((f'{frame}:0.201,0.3,2;0.8,0.8,2:0.2,0.3,0\n' for frame in range(1, 199320)))
        (tactical / 'BAR-ATH_tactical_smart_tagging.xml').write_text('<file/>', encoding='utf-8')
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        outputs = [self.path, self.path.with_name('again.csv')]
        for output in outputs:
            result = subprocess.run([sys.executable, '-m', 'mtah.pipeline', '--autorizar-datos-restringidos', '--sin-cache', '--data-dir', self.temp.name, '--salida', str(output)], env=environment, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(validar_csv(output))
        self.assertEqual(hashlib.sha256(outputs[0].read_bytes()).digest(), hashlib.sha256(outputs[1].read_bytes()).digest())
        result = subprocess.run([sys.executable, '-m', 'mtah.pipeline', '--autorizar-datos-restringidos', '--data-dir', 'no-existe', '--salida', str(outputs[0])], env=environment, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
if __name__ == '__main__':
    unittest.main()
