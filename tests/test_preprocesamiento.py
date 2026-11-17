"""Documentación pública en README.md."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah import carga
from mtah.preprocesamiento import cargar_tracking, equipo_por_slot, features_smart_tagging, leer_smart_tagging, leer_tracking
META = '<Root><Team id="home"/><Team id="away"/><Player teamId="home"/><Player teamId="away"/></Root>'
RAW = '1:0.2,0.3,2;0.8,0.4,3:0.21,0.31,4\n2:NaN,NaN,NaN;0.7,0.4,3:NaN,NaN,NaN\n'

class TrackingTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.meta, self.raw = (self.folder / 'meta.xml', self.folder / 'raw.txt')
        self.meta.write_text(META, encoding='utf-8')
        self.raw.write_text(RAW, encoding='utf-8')

    def load(self):
        return leer_tracking(self.raw, equipo_por_slot(self.meta), 2, (100.0, 50.0))

    def test_alignment_and_metric_scaling(self):
        out = self.load()
        np.testing.assert_allclose(out['balon'][0], [21.0, 15.5, 4.0])
        np.testing.assert_allclose(out['jugadores'][0, 0], [20.0, 15.0, 2.0])
        np.testing.assert_array_equal(out['equipo_slot'], [0, 1])
        self.assertTrue(np.isnan(out['balon'][1]).all())

    def test_missing_xy_is_not_zero(self):
        self.raw.write_text(RAW.replace('0.21,0.31,4', '0.21,NaN,4'), encoding='utf-8')
        self.assertTrue(np.isnan(self.load()['balon'][0, :2]).all())

    def test_invalid_records_are_rejected_without_echoing_data(self):
        invalid = [RAW.replace('1:', '0:', 1), RAW.replace('2:', '1:', 1), RAW.replace('1:', '1.5:', 1), RAW.splitlines()[0] + '\n', RAW + '3:0,0,0;0,0,0:0,0,0\n', '', '\n' + RAW, RAW.replace('0.21,0.31,4', 'inf,0.31,4'), RAW.replace('0.21,0.31,4', 'NO_ES_NUMERO,0.31,4'), RAW.replace(';0.8,0.4,3', ''), RAW.replace('0.21,0.31,4', '0.21,4')]
        for text in invalid:
            with self.subTest(case=invalid.index(text)):
                self.raw.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError) as error:
                    self.load()
                self.assertNotIn('NO_ES_NUMERO', str(error.exception))

    def test_team_metadata_must_be_unambiguous(self):
        for meta in (META.replace('teamId="away"', 'teamId="unknown"'), META.replace('<Team id="away"/>', ''), META.replace('id="away"', 'id="home"'), '<Root/>'):
            self.meta.write_text(meta, encoding='utf-8')
            with self.assertRaises(ValueError):
                equipo_por_slot(self.meta)

    def test_xml_namespace(self):
        self.meta.write_text(META.replace('<Root>', '<Root xmlns="urn:epts">'), encoding='utf-8')
        np.testing.assert_array_equal(equipo_por_slot(self.meta), [0, 1])

    def test_malformed_xml_is_rejected_without_echoing_contents(self):
        self.meta.write_text('<Root><DATO_SINTETICO_SENSIBLE></Root>', encoding='utf-8')
        with self.assertRaises(ValueError) as error:
            equipo_por_slot(self.meta)
        self.assertNotIn('DATO_SINTETICO_SENSIBLE', str(error.exception))

    def test_cache_is_bound_to_content_and_configuration(self):
        cache = self.folder / 'cache'
        first = cargar_tracking(self.meta, self.raw, 2, (100.0, 50.0), cache)
        with patch('mtah.preprocesamiento.leer_tracking', side_effect=AssertionError('reparsed')):
            second = cargar_tracking(self.meta, self.raw, 2, (100.0, 50.0), cache)
        np.testing.assert_array_equal(first['balon'], second['balon'])
        self.raw.write_text(RAW.replace('0.21,0.31,4', '0.41,0.31,4'), encoding='utf-8')
        changed = cargar_tracking(self.meta, self.raw, 2, (100.0, 50.0), cache)
        self.assertAlmostEqual(float(changed['balon'][0, 0]), 41.0)
        scaled = cargar_tracking(self.meta, self.raw, 2, (200.0, 50.0), cache)
        self.assertAlmostEqual(float(scaled['balon'][0, 0]), 82.0)
        self.assertEqual(len(list(cache.glob('*.npz'))), 3)

    def test_corrupt_cache_is_not_overwritten(self):
        cache = self.folder / 'cache'
        cargar_tracking(self.meta, self.raw, 2, (100.0, 50.0), cache)
        path = next(cache.glob('*.npz'))
        path.write_bytes(b'corrupt synthetic cache')
        with self.assertRaises(ValueError):
            cargar_tracking(self.meta, self.raw, 2, (100.0, 50.0), cache)
        self.assertEqual(path.read_bytes(), b'corrupt synthetic cache')

    def test_restricted_access_requires_permission_before_any_open(self):
        with patch('builtins.open', side_effect=AssertionError('opened restricted file')):
            with self.assertRaises(PermissionError):
                carga.cargar_atd(data_dir=self.folder)
            with self.assertRaises(PermissionError):
                carga.cargar_smart_tagging(data_dir=self.folder)

    def test_data_dir_and_disabled_cache(self):
        tactical = self.folder / 'tactical'
        tactical.mkdir()
        (tactical / 'BAR-ATH_tactical_atd.xml').write_text(META, encoding='utf-8')
        (tactical / 'BAR-ATH_tactical_atd.txt').write_text(RAW, encoding='utf-8')
        with patch.object(carga.games, 'source_frames', return_value=2):
            result = carga.cargar_atd(data_dir=self.folder, usar_cache=False, autorizar_datos=True)
        self.assertEqual(result['balon'].shape, (2, 3))
        self.assertFalse((tactical / 'cache').exists())

def instance(start, end, code='BUILD UP', team='Barcelona'):
    return f'<instance><start>{start}</start><end>{end}</end><code>{code}</code><label><group> tEaM </group><text>{team}</text></label></instance>'

class SmartTaggingTests(unittest.TestCase):

    def parse(self, text, n=100, fps=25):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.xml'
            path.write_text('<file>' + text + '</file>', encoding='utf-8')
            return leer_smart_tagging(path, n, fps)

    def test_seconds_rounding_clipping_and_half_open_intervals(self):
        events = self.parse(instance(-1, 0.12) + instance(3.96, 10))
        self.assertEqual([(e['start_frame'], e['end_frame']) for e in events], [(0, 3), (99, 100)])
        features = features_smart_tagging(events, 100)
        self.assertEqual(features.st_equipo.tolist()[:4], [0, 0, 0, -1])
        rounded = self.parse(instance(0.06, 0.14), n=100)
        self.assertEqual((rounded[0]['start_frame'], rounded[0]['end_frame']), (2, 4))

    def test_team_names_and_unknowns(self):
        for text, expected in [(' BARCELONA ', 0), ('Atletic Club', 1), ('Athletic Club', 1), ('N/A', None), ('Equipo desconocido', None), ('', None)]:
            self.assertEqual(self.parse(instance(0, 1, team=text))[0]['equipo'], expected)

    def test_invalid_timestamps(self):
        for start, end in [('nan', 1), (0, 'inf'), (2, 1), ('bad', 1), (0, '1e308')]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.parse(instance(start, end))

    def test_sort_and_drop_empty_or_outside(self):
        events = self.parse(instance(2, 3) + instance(0, 1) + instance(1, 1) + instance(8, 9))
        self.assertEqual([e['start_frame'] for e in events], [0, 50])

    def test_defensive_phase_supports_opponent(self):
        frame = features_smart_tagging(self.parse(instance(0, 1, 'defending in middle third')), 100)
        self.assertEqual(frame.st_equipo.iloc[0], 1)

    def test_overlapping_conflict_abstains_and_duplicates_do_not_win(self):
        events = self.parse(instance(0, 2) * 3 + instance(1, 3, team='Atletic Club'))
        frame = features_smart_tagging(events, 100)
        self.assertEqual(frame.st_equipo.iloc[10], 0)
        self.assertEqual(frame.st_equipo.iloc[30], -1)
        self.assertTrue(frame.st_conflicto.iloc[30])
        self.assertEqual(frame.st_equipo.iloc[60], 1)

    def test_set_piece_is_context_not_a_dead_ball_label(self):
        frame = features_smart_tagging(self.parse(instance(0, 1, 'SET PIECES', 'N/A')), 100)
        self.assertTrue(frame.st_saque.iloc[0])
        self.assertEqual(frame.st_equipo.iloc[0], -1)

    def test_contradictory_tag_groups_rejected(self):
        text = instance(0, 1).replace('</instance>', '<label><group>Team</group><text>Atletic Club</text></label></instance>')
        with self.assertRaises(ValueError):
            self.parse(text)
if __name__ == '__main__':
    unittest.main()
