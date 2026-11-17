import contextlib
import io
import itertools
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.control import ajustar_control, cargar_modelo, matriz_control, potenciales
from mtah.eda import correr as correr_eda, distribucion, escribir_reporte, longitudes_huecos
from mtah.externos import COMMIT, puntuar_referencias, referencias, verificar_procedencia
from mtah.features_control import ConfigFeatures, bloques, extraer
from mtah.fuentes import inspeccionar_mp4, leer_homografia
from mtah.metricas_demo import ejemplos
from mtah.solucion import main, predecir
from mtah.temporal_tasas_fijas import ConfigTemporal, decodificar, inferir, transicion
from mtah.ensamble import escribir_csv
from mtah.validacion import validar_csv

def synthetic(n=140):
    ball = np.column_stack([10.0 + np.arange(n) * 0.02, np.full(n, 30.0), np.full(n, 0.5)])
    players = np.stack([ball.copy(), ball.copy()], axis=1)
    players[:, 0, 0] += 0.3
    players[:, 1, 0] += 12
    players[:, :, 2] = 2.0
    return dict(balon=ball, jugadores=players, equipo_slot=np.array([0, 1]))

class TemporalTests(unittest.TestCase):

    def test_viterbi_and_smoothing_agree_with_exhaustive_enumeration(self):
        emission = np.array([[0.5, -0.2, -0.9], [1, 0.3, -0.1], [0.2, 1.5, 0], [-0.5, 0.8, 0.2]])
        matrix = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
        states, posterior = inferir(emission, matrix)
        sequences = list(itertools.product(range(3), repeat=4))
        weights = []
        for sequence in sequences:
            value = 1 / 3 * np.exp(sum((emission[i, state] for i, state in enumerate(sequence))))
            value *= np.prod([matrix[sequence[i - 1], sequence[i]] for i in range(1, 4)])
            weights.append(value)
        expected = np.zeros((4, 3))
        for weight, sequence in zip(weights, sequences):
            for i, state in enumerate(sequence):
                expected[i, state] += weight / sum(weights)
        np.testing.assert_array_equal(states, sequences[int(np.argmax(weights))])
        np.testing.assert_allclose(posterior, expected, atol=1e-12)

    def decode(self, logits):
        n = len(logits)
        return decodificar(logits, [(0, n)], np.ones(n, bool), np.zeros(n))

    def test_single_rebound_does_not_split_possession(self):
        logits = np.tile([6.0, 0.0, -2.0], (100, 1))
        logits[50] = [0, 6, -2]
        self.assertTrue((self.decode(logits)['etiquetas'] == 'H').all())

    def test_two_fast_but_strong_transitions_survive(self):
        logits = np.tile([8.0, 0.0, -2.0], (100, 1))
        logits[40:50] = [0, 8, -2]
        labels = self.decode(logits)['etiquetas']
        self.assertTrue((labels[40:50] == 'A').all())
        self.assertEqual(labels[39], 'H')
        self.assertEqual(labels[50], 'H')

    def test_both_sides_of_gap_inform_short_occlusion(self):
        logits = np.tile([6.0, 0.0, -2.0], (100, 1))
        logits[30:55] = 0
        self.assertTrue((self.decode(logits)['etiquetas'][30:55] == 'H').all())

    def test_no_evidence_does_not_create_dead_ball(self):
        result = decodificar(np.zeros((80, 3)), [(0, 80)], np.zeros(80, bool), np.full(80, np.inf))
        self.assertTrue((result['etiquetas'] == 'X').all())

    def test_periods_reset_and_leave_ungraded_frames_x(self):
        logits = np.tile([6.0, 0.0, -2.0], (100, 1))
        logits[60:] = [0, 6, -2]
        result = decodificar(logits, [(0, 40), (60, 100)], np.ones(100, bool), np.zeros(100))
        self.assertTrue((result['etiquetas'][:40] == 'H').all())
        self.assertTrue((result['etiquetas'][40:60] == 'X').all())
        self.assertTrue((result['etiquetas'][60:] == 'A').all())

    def test_stochastic_transition_and_posterior(self):
        matrix = transicion()
        np.testing.assert_allclose(matrix.sum(1), 1.0)
        self.assertTrue((matrix >= 0).all())
        _, probability = inferir(np.zeros((5, 3)), matrix)
        np.testing.assert_allclose(probability.sum(1), 1.0)

    def test_invalid_temporal_inputs(self):
        for matrix in [np.zeros((3, 3)), np.ones((2, 2)), np.eye(3) * -1]:
            with self.assertRaises(ValueError):
                inferir(np.zeros((5, 3)), matrix)
        with self.assertRaises(ValueError):
            self.decode(np.full((20, 3), np.nan))

class ControlFeaturesTests(unittest.TestCase):

    def test_relative_velocity_and_distance_units(self):
        frame = extraer(synthetic())
        self.assertAlmostEqual(float(frame.d_H.iloc[20]), 0.3)
        self.assertAlmostEqual(float(frame.rel_H.iloc[20]), 0.0)
        self.assertAlmostEqual(float(frame.v_balon.iloc[20]), 0.5)
        self.assertGreater(frame.margen_distancia.iloc[20], 10)

    def test_team_permutation_is_symmetric(self):
        atd = synthetic()
        original = extraer(atd)
        swapped = extraer(dict(atd, equipo_slot=1 - atd['equipo_slot']))
        np.testing.assert_allclose(matriz_control(original), -matriz_control(swapped))
        logits1, logits2 = (potenciales(original)['logits'], potenciales(swapped)['logits'])
        np.testing.assert_allclose(logits1[:, [1, 0, 2]], logits2)

    def test_slot_identity_jump_does_not_create_relative_speed(self):
        atd = synthetic()
        atd['jugadores'][50:, 0, 0] += 20
        frame = extraer(atd)
        self.assertGreater(frame.saltos_slots.iloc[50], 0)
        self.assertTrue(np.isnan(frame.rel_H.iloc[50]))

    def test_ball_jump_invalidates_proximity_evidence(self):
        atd = synthetic()
        atd['balon'][50, 0] += 30
        frame = extraer(atd)
        self.assertTrue(frame.salto_balon.iloc[50])
        self.assertFalse(frame.balon_util.iloc[50])
        self.assertTrue(np.isnan(frame.d_min.iloc[50]))

    def test_gap_counters_and_period_derivatives(self):
        atd = synthetic()
        atd['balon'][20:45] = np.nan
        periods = np.r_[np.ones(70), np.full(70, 2)]
        frame = extraer(atd, grupos=periods)
        self.assertAlmostEqual(frame.desde_balon_s.iloc[20], 0.04)
        self.assertAlmostEqual(frame.hasta_balon_s.iloc[44], 0.04)
        self.assertTrue(np.isnan(frame.v_balon.iloc[70]))
        self.assertEqual(bloques(np.ones(140, bool), periods), [(0, 70), (70, 140)])

    def test_full_solution_never_reads_gt(self):
        atd = synthetic()
        with patch('possession_metrics.read_label_csv', side_effect=AssertionError('GT read')):
            result = predecir(atd, calif=np.ones(140, bool))
        self.assertTrue((result['etiquetas'] == 'H').all())

    def test_actual_occlusion_can_be_bridged(self):
        atd = synthetic()
        atd['balon'][40:65] = np.nan
        result = predecir(atd, calif=np.ones(140, bool))
        self.assertTrue((result['etiquetas'][40:65] == 'H').all())

    def test_no_ball_does_not_equal_dead_even_if_players_walk(self):
        atd = synthetic()
        atd['balon'][:] = np.nan
        atd['jugadores'][:, :, 2] = 0.2
        result = predecir(atd, calif=np.ones(140, bool))
        self.assertTrue((result['etiquetas'] == 'X').all())

    def test_proximity_outside_pitch_is_not_live_control(self):
        atd = synthetic()
        atd['balon'][:, 0] = -2
        atd['jugadores'][:, 0, 0] = -1.8
        result = predecir(atd, calif=np.ones(140, bool))
        self.assertTrue((result['etiquetas'] == 'D').all())

    def test_logistic_fits_external_style_anchors_and_has_no_intercept(self):
        x = np.array([[1, 0.5, 1, 0.3], [-1, -0.5, -1, -0.3]] * 30)
        y = np.array([1, 0] * 30)
        model = ajustar_control(x, y)
        self.assertEqual(model['intercepto'], 0)
        predicted = x @ np.asarray(model['coeficientes']) > 0
        np.testing.assert_array_equal(predicted, y)

    def test_real_data_cli_needs_explicit_permission(self):
        with patch('mtah.carga.cargar_atd', side_effect=AssertionError('data read')), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([])

    def test_full_length_prediction_validates_without_gt(self):
        atd = synthetic(199319)
        atd['balon'][:, :2] = [10.0, 30.0]
        atd['jugadores'][:, 0, :2] = [10.3, 30.0]
        atd['jugadores'][:, 1, :2] = [22.0, 30.0]
        with patch('mtah.solucion.carga.cargar_atd', side_effect=AssertionError('read actual assets')):
            result = predecir(atd)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'synthetic.csv'
            escribir_csv(result['etiquetas'], output)
            self.assertTrue(validar_csv(output))
        self.assertEqual(len(result['etiquetas']), 199319)
        self.assertEqual(result['etiquetas'][0], 'X')
        self.assertEqual(result['etiquetas'][11185], 'H')
        self.assertEqual(result['etiquetas'][199318], 'X')

class SourcesAndMetricsTests(unittest.TestCase):

    def test_eda_requires_permission_before_reading_paths(self):
        with patch('pathlib.Path.is_file', side_effect=AssertionError('accessed assets')):
            with self.assertRaises(PermissionError):
                correr_eda('no-es-un-bundle')

    def test_external_training_rejects_relabelled_input_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'PROVENANCE.json'
            path.write_text(json.dumps(dict(source='https://github.com/metrica-sports/sample-data', commit=COMMIT, files=[])), encoding='utf-8')
            with self.assertRaises(ValueError):
                verificar_procedencia(folder)

    def test_homography_rejects_wrong_timeline_and_masks_singular(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.csv'
            matrix = np.eye(3).reshape(-1)
            rows = [dict(frame=i, **dict(zip(['h{}{}'.format(a, b) for a in range(3) for b in range(3)], matrix)), reprojection_error=0.2, valid=1) for i in range(2)]
            rows[1]['h22'] = 0
            pd.DataFrame(rows).to_csv(path, index=False)
            h = leer_homografia(path, 2)
            self.assertEqual(h['valid'].tolist(), [True, False])
            with self.assertRaises(ValueError):
                leer_homografia(path, 3)

    def test_mp4_metadata_reader_uses_sample_tables(self):

        def box(kind, payload):
            return struct.pack('>I4s', len(payload) + 8, kind) + payload
        tkhd = box(b'tkhd', bytes(76) + struct.pack('>II', 1920 << 16, 1080 << 16))
        mdhd = box(b'mdhd', bytes(12) + struct.pack('>II', 25, 250) + bytes(4))
        hdlr = box(b'hdlr', bytes(8) + b'vide' + bytes(12))
        stts = box(b'stts', bytes(4) + struct.pack('>III', 1, 250, 1))
        mdia = box(b'mdia', mdhd + hdlr + box(b'minf', box(b'stbl', stts)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.mp4'
            path.write_bytes(box(b'ftyp', b'isom0000') + box(b'mdat', b'not video pixels') + box(b'moov', box(b'trak', tkhd + mdia)))
            result = inspeccionar_mp4(path)
        self.assertEqual(result['muestras_codificadas'], 250)
        self.assertEqual(result['fps_medio'], 25)
        self.assertFalse(result['analisis_pixeles'])

    def test_unavailable_homography_quality_is_not_treated_as_zero_or_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic_quality.csv'
            columns = ['h{}{}'.format(a, b) for a in range(3) for b in range(3)]
            rows = [dict(frame=i, **dict(zip(columns, np.eye(3).reshape(-1))), reprojection_error=value, valid=int(i == 0)) for i, value in enumerate([-1.0, np.inf, np.nan])]
            for column in columns:
                rows[1][column] = np.inf
            pd.DataFrame(rows).to_csv(path, index=False)
            h = leer_homografia(path, 3)
            self.assertTrue(np.isnan(h['error']).all())
            self.assertFalse(h['calidad_disponible'].any())
            self.assertEqual(h['valid'].tolist(), [True, False, False])

    def test_official_synthetic_metric_examples(self):
        _, cases = ejemplos()
        values = {case['caso']: case for case in cases}
        self.assertEqual(values['Perfecto']['spellF1'], 1)
        self.assertGreater(values['Un rebote espurio de un frame']['macroF1'], 0.99)
        self.assertLess(values['Un rebote espurio de un frame']['spellF1'], 0.7)
        self.assertEqual(values['Oclusión dentro de una posesión']['spellF1'], 1)
        self.assertEqual(values['Siempre local']['spellF1'], 0)

    def test_external_references_are_sparse_and_conflicts_abstain(self):
        events = pd.DataFrame([dict(Team='Home', Type='PASS', **{'Start Frame': 1, 'End Frame': 3, 'To': 'P'}), dict(Team='Away', Type='RECOVERY', **{'Start Frame': 3, 'End Frame': 3, 'To': None})])
        refs = referencias(events, 5)
        self.assertEqual(refs.tolist(), ['H', 'X', 'X', 'X', 'X'])
        report = puntuar_referencias(np.full(5, 'H'), refs)
        self.assertEqual(report['referencias'], 1)
        self.assertNotIn('spellF1', report)

    def test_eda_is_offline_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'eda'
            escribir_reporte(dict(fuentes={'synthetic': dict(tracking={})}), output)
            html = (output / 'informe.html').read_text(encoding='utf-8')
            self.assertIn("default-src 'none'", html)
            self.assertNotIn('<script', html)
            with self.assertRaises(FileExistsError):
                escribir_reporte({}, output)
        self.assertEqual(distribucion([np.nan])['n'], 0)
        self.assertEqual(longitudes_huecos([True, False, False], [True, True, True], 25), [0.08])
if __name__ == '__main__':
    unittest.main()
