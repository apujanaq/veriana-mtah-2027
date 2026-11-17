import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from mtah.candidata_v3 import ajustar_logistica, calibracion_binaria, evidencia_v3, extraer_v3, cargar_v3, matriz_v3, predecir_v3
from mtah.duracion import ajustar_riesgos, decodificar_duracion, inferir_duracion
from mtah.experimento_v3 import degradar_realista, metricas_parciales
from mtah.features_control import ConfigFeatures
from mtah.skillcorner import COMMIT_SC, SPLITS, adaptar_registros

def synthetic(n=120):
    ball = np.column_stack([20 + np.arange(n) * 0.08, np.full(n, 30.0), np.full(n, 2.0)])
    players = np.repeat(ball[:, None], 12, axis=1)
    players[:, :, 0] += np.array([0.2, 5, 10, 15, 20, 25, 8, 12, 16, 20, 24, 28])
    return dict(balon=ball, jugadores=players, equipo_slot=np.r_[np.zeros(6, np.int8), np.ones(6, np.int8)])

def model():
    return dict(version=3, control_v3=dict(coeficientes=[2.0, 0.2, 0.1, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5]), temperatura_control=1.0, duracion=ajustar_riesgos([]), opciones=dict(duracion=True, peso_parada_extra=1.5, retro_parada_s=0.8), base_v2=dict(version=2, control=dict(coeficientes=[3.0, 0.2, 0.1, 0.5]), pausa=dict(coeficientes=[2.0, -3.0, -3.0, -2.0])))

class DurationTests(unittest.TestCase):

    def test_exact_marginals_and_viterbi_against_enumeration(self):
        e = np.array([[0.2, -0.1], [-0.4, 0.5], [-0.3, 0.7], [0.9, -0.2]])
        hazards = np.array([[0.1, 0.4], [0.5, 0.2]])
        dest = np.array([[0.0, 1.0], [1.0, 0.0]])
        transition = np.zeros((4, 4))
        for state in range(2):
            for age in range(2):
                index = state * 2 + age
                transition[index, state * 2 + min(age + 1, 1)] = 1 - hazards[state, age]
                transition[index, (1 - state) * 2] = hazards[state, age]
        weights, paths = ([], [])
        for path in itertools.product(range(4), repeat=4):
            if path[0] % 2:
                continue
            weight = 0.5 * np.exp(sum((e[t, p // 2] for t, p in enumerate(path))))
            weight *= np.prod([transition[path[t - 1], path[t]] for t in range(1, 4)])
            weights.append(weight)
            paths.append(path)
        weights = np.array(weights)
        posterior = np.zeros((4, 2))
        for path, w in zip(paths, weights):
            for t, s in enumerate(path):
                posterior[t, s // 2] += w / weights.sum()
        states, actual = inferir_duracion(e, hazards, dest)
        np.testing.assert_allclose(actual, posterior, atol=1e-12)
        np.testing.assert_array_equal(states, np.array(paths[int(weights.argmax())]) // 2)

    def test_short_real_transition_is_possible(self):
        e = np.array([[20.0, -20.0]] * 10 + [[-30.0, 30.0]] * 2 + [[20.0, -20.0]] * 10)
        seq, _ = inferir_duracion(e, np.full((2, 4), 0.05), np.array([[0, 1], [1, 0]]))
        np.testing.assert_array_equal(seq[10:12], [1, 1])

    def test_tail_does_not_force_a_maximum_duration(self):
        seq, _ = inferir_duracion(np.tile([4.0, -4.0], (300, 1)), np.full((2, 3), 0.1), np.array([[0, 1], [1, 0]]))
        self.assertTrue((seq == 0).all())

    def test_unknown_reference_and_half_boundary_censor_transitions(self):
        one = ajustar_riesgos([(np.array(list('HHXXAA')), np.ones(6), 10.0)])
        two = ajustar_riesgos([(np.array(list('HHAA')), np.array([1, 1, 2, 2]), 10.0)])
        self.assertEqual(sum(one['salidas_observadas']), 0)
        self.assertEqual(sum(two['salidas_observadas']), 0)
        real = ajustar_riesgos([(np.array(list('HHAA')), np.ones(4), 10.0)])
        self.assertEqual(sum(real['salidas_observadas']), 1)

    def test_unsupported_blocks_and_ungraded_frames_stay_unknown(self):
        labels = decodificar_duracion(np.tile([8.0, 0.0, 0.0], (50, 1)), [(0, 20), (30, 50)], np.r_[np.ones(20, bool), np.zeros(30, bool)], np.zeros(50), 25.0, ajustar_riesgos([]))['etiquetas']
        self.assertTrue((labels[:20] == 'H').all())
        self.assertTrue((labels[20:] == 'X').all())

    def test_invalid_graph_is_rejected(self):
        for h in [np.zeros((2, 3)), np.full((2, 3), np.nan), np.ones((2, 3))]:
            with self.assertRaises(ValueError):
                inferir_duracion(np.zeros((4, 2)), h, np.array([[0, 1], [1, 0]]))

class SourceTests(unittest.TestCase):

    def fixture(self):
        meta = dict(pitch_length=105, pitch_width=68, home_team=dict(id=1), away_team=dict(id=2), players=[dict(id=3, team_id=1), dict(id=4, team_id=2)])
        record = dict(frame=0, period=1, ball_data=dict(x=0, y=0, is_detected=False), player_data=[dict(player_id=3, x=-2, y=0, is_detected=True), dict(player_id=4, x=2, y=0, is_detected=False)], possession=dict(group=None))
        return ([record], meta)

    def test_extrapolated_coordinates_and_unknown_possession_remain_missing(self):
        records, meta = self.fixture()
        atd, periods, refs, audit = adaptar_registros(records, meta)
        self.assertTrue(np.isnan(atd['balon']).all())
        np.testing.assert_allclose(atd['jugadores'][0, 0, :2], [50.5, 34])
        self.assertTrue(np.isnan(atd['jugadores'][0, 1]).all())
        self.assertEqual(refs[0], 'X')

    def test_reference_cannot_change_inputs(self):
        records, meta = self.fixture()
        first = adaptar_registros(records, meta)[0]
        records[0]['possession']['group'] = 'away team'
        second = adaptar_registros(records, meta)
        for key in first:
            np.testing.assert_array_equal(first[key], second[0][key])
        self.assertEqual(second[2][0], 'A')

    def test_schema_and_alignment_errors_are_rejected(self):
        records, meta = self.fixture()
        records[0]['frame'] = 1
        with self.assertRaises(ValueError):
            adaptar_registros(records, meta)
        records[0]['frame'] = 0
        records[0]['possession']['group'] = 'unknown meaning'
        with self.assertRaises(ValueError):
            adaptar_registros(records, meta)

class CandidateTests(unittest.TestCase):

    def test_v3_loader_and_public_package_preserve_provenance(self):
        import zipfile
        from mtah.entrega import preparar_codigo
        from mtah.externos import COMMIT
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            frozen = model()
            frozen['procedencia'] = dict(gt_concurso_usado=False, skillcorner_commit=COMMIT_SC, splits=SPLITS)
            frozen['base_v2']['procedencia'] = dict(commit=COMMIT, gt_concurso_usado=False)
            path = base / 'model.json'
            path.write_text(json.dumps(frozen), encoding='utf-8')
            self.assertEqual(cargar_v3(path)['version'], 3)
            preparar_codigo(base / 'code.zip', path, 'veriana')
            with zipfile.ZipFile(base / 'code.zip') as archive:
                self.assertIn('config/skillcorner_v3.json', archive.namelist())
                self.assertIn('src/mtah/duracion.py', archive.namelist())
                self.assertIn('THIRD_PARTY_NOTICES.md', archive.namelist())
                self.assertIn('multifuente v3', archive.read('veriana.md').decode('utf-8'))
                self.assertNotIn('{{EQUIPO}}', archive.read('veriana.md').decode('utf-8'))
            frozen['procedencia']['gt_concurso_usado'] = True
            path.write_text(json.dumps(frozen), encoding='utf-8')
            with self.assertRaises(ValueError):
                cargar_v3(path)

    def test_v3_loader_rejects_invalid_duration_and_parameters(self):
        from mtah.externos import COMMIT
        with tempfile.TemporaryDirectory() as folder:
            frozen = model()
            frozen['procedencia'] = dict(gt_concurso_usado=False, skillcorner_commit=COMMIT_SC, splits=SPLITS)
            frozen['base_v2']['procedencia'] = dict(commit=COMMIT, gt_concurso_usado=False)
            path = Path(folder) / 'model.json'
            frozen['duracion']['transferencia_por_s'][0] = -1
            path.write_text(json.dumps(frozen), encoding='utf-8')
            with self.assertRaises(ValueError):
                cargar_v3(path)

    def test_features_are_antisymmetric_and_inputs_not_mutated(self):
        atd = synthetic()
        original = atd['balon'].copy()
        first = matriz_v3(extraer_v3(atd))
        swapped = dict(atd, equipo_slot=1 - atd['equipo_slot'])
        second = matriz_v3(extraer_v3(swapped))
        np.testing.assert_allclose(first, -second, atol=1e-12)
        np.testing.assert_array_equal(original, atd['balon'])

    def test_no_velocity_projection_across_periods(self):
        frame = extraer_v3(synthetic(), groups=np.r_[np.ones(60), np.full(60, 2)])
        self.assertEqual(frame['intercepcion_0.4'].iloc[60], 0.0)

    def test_no_observation_does_not_create_dead_ball(self):
        atd = synthetic()
        atd['balon'][:] = np.nan
        result = predecir_v3(atd, calif=np.ones(120, bool), model=model())
        self.assertTrue((result['etiquetas'] == 'X').all())

    def test_richer_control_and_duration_predict_deterministically(self):
        atd = synthetic()
        a = predecir_v3(atd, calif=np.ones(120, bool), model=model())
        b = predecir_v3(atd, calif=np.ones(120, bool), model=model())
        np.testing.assert_array_equal(a['etiquetas'], b['etiquetas'])
        self.assertTrue((a['etiquetas'][10:] == 'H').all())

    def test_noise_is_seeded_and_does_not_change_source(self):
        atd = synthetic(1000)
        before = {k: v.copy() for k, v in atd.items()}
        a, b = (degradar_realista(atd, 25), degradar_realista(atd, 25))
        for key in atd:
            np.testing.assert_array_equal(atd[key], before[key])
            np.testing.assert_array_equal(a[key], b[key])
        self.assertGreater(np.isnan(a['balon'][:, 0]).mean(), 0.2)

    def test_newton_fit_and_calibration_use_correct_binary_direction(self):
        x = np.array([[-1.0, 0], [-0.9, 0.1], [0.9, -0.1], [1.0, 0.0]])
        fitted = ajustar_logistica(x, np.array([0, 0, 1, 1]))
        self.assertGreater(fitted['coeficientes'][0], 0)
        good = calibracion_binaria(np.array([-3.0, -2.0, 2.0, 3.0]), [0, 0, 1, 1], 1)
        bad = calibracion_binaria(np.array([3.0, 2.0, -2.0, -3.0]), [0, 0, 1, 1], 1)
        self.assertLess(good['nll'], bad['nll'])

    def test_conditional_spell_metric_explicitly_preserves_unknowns(self):
        refs = np.array(list('HHHHXXAAAA'))
        metric = metricas_parciales(refs, refs, np.ones(10))
        self.assertAlmostEqual(metric['spellF1_condicional'], 1.0)
        self.assertEqual(metric['referencias'], 8)
        self.assertEqual(metric['spells_referencia'], 2)
if __name__ == '__main__':
    unittest.main()
