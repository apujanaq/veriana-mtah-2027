from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from mtah.boosting import ajustar_stumps, log_odds_stumps
from mtah.candidata_v4 import D_VARIABLES, extraer_v4, matriz_control_v4, matriz_pausa_v4, cargar_v4, predecir_v4, velocidad_ventana
from mtah.experimento_v4 import metricas_completas
from mtah.features_control import ConfigFeatures
from mtah.geometria_visual import asociar_ejecutante, campo_a_imagen, pies_a_campo
from mtah.idsse import leer_metadata, leer_posiciones, verificar_idsse
from mtah.reanudaciones import evidencia_reanudacion, features_reanudacion

def synthetic(n=120):
    ball = np.c_[20 + np.arange(n) * 0.08, np.full(n, 30.0), np.full(n, 2.0)]
    players = np.repeat(ball[:, None], 12, axis=1)
    players[:, :, 0] += np.array([0.2, 5, 10, 15, 20, 25, 8, 12, 16, 20, 24, 28])
    return dict(balon=ball, jugadores=players, equipo_slot=np.r_[np.zeros(6, np.int8), np.ones(6, np.int8)])

def fixture_model():
    from mtah.duracion import ajustar_riesgos
    return dict(version=4, base_v3=dict(version=3, control_v3=dict(coeficientes=[2.0, 0.2, 0.1, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5]), temperatura_control=1.0, duracion=ajustar_riesgos([]), opciones=dict(duracion=False, peso_parada_extra=0.0, retro_parada_s=0.0), base_v2=dict(version=2, control=dict(coeficientes=[3.0, 0.2, 0.1, 0.5]), pausa=dict(coeficientes=[2.0, -3.0, -3.0, -2.0]))), control_v4=dict(coeficientes=[2.0, 0.2, 0.1, 0.3, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3, 0.3]), pausa_v4=dict(intercepto=-3.0, arboles=[], n_variables=len(D_VARIABLES)), temperatura_pausa=1.0, opciones=dict(control_nuevo=True, mezcla_pausa=0.5, peso_reanudacion=0.5))

class DFLTests(unittest.TestCase):

    def write(self, base, status=1, owner=1):
        meta = base / 'meta.xml'
        meta.write_text('<MatchInformation><General Season="2022/2023" MatchId="synthetic" HomeTeamId="H" GuestTeamId="A"/><Environment PitchX="105" PitchY="68"/><Teams><Team TeamId="H"><Player PersonId="p1"/></Team><Team TeamId="A"><Player PersonId="p2"/></Team></Teams></MatchInformation>')
        root = ET.Element('Positions')
        time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        for period, start in [('firstHalf', 10000), ('secondHalf', 20000)]:
            for team, player in [('H', 'p1'), ('A', 'p2'), ('BALL', 'ball-id')]:
                stream = ET.SubElement(root, 'FrameSet', GameSection=period, MatchId='synthetic', TeamId=team, PersonId=player)
                for i in range(15):
                    attrs = dict(N=str(start + i), T=(time + timedelta(seconds=i / 25)).isoformat(), X='0', Y='0', S='36')
                    if team == 'BALL':
                        attrs.update(BallStatus=str(status), BallPossession=str(owner))
                    ET.SubElement(stream, 'Frame', attrs)
        path = base / 'tracking.xml'
        ET.ElementTree(root).write(path)
        return (leer_metadata(meta), path)

    def test_labels_are_separate_and_units_aligned(self):
        with tempfile.TemporaryDirectory() as folder:
            meta, path = self.write(Path(folder))
            atd, periods, refs, times = leer_posiciones(path, meta)
            self.assertEqual(len(refs), 6)
            np.testing.assert_array_equal(refs, np.full(6, 'H'))
            np.testing.assert_array_equal(periods, [1, 1, 1, 2, 2, 2])
            np.testing.assert_allclose(atd['balon'][:, :2], np.tile([52.5, 34], (6, 1)))
            np.testing.assert_allclose(atd['jugadores'][:, :, 2], 10.0)
            self.assertNotIn('BallStatus', atd)

    def test_response_changes_cannot_change_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            meta, path = self.write(base, status=1, owner=2)
            a = leer_posiciones(path, meta)
            meta, path = self.write(base, status=0, owner=1)
            b = leer_posiciones(path, meta)
            for name in a[0]:
                np.testing.assert_array_equal(a[0][name], b[0][name])
            self.assertTrue((a[2] == 'A').all())
            self.assertTrue((b[2] == 'D').all())

    def test_unknown_live_owner_is_unknown_and_missing_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            meta, path = self.write(Path(folder), status=1, owner=0)
            self.assertTrue((leer_posiciones(path, meta)[2] == 'X').all())
            with self.assertRaises(FileNotFoundError):
                verificar_idsse(folder)

    def test_irregular_clock_is_not_padded(self):
        with tempfile.TemporaryDirectory() as folder:
            meta, path = self.write(Path(folder))
            root = ET.parse(path)
            stream = root.findall('.//FrameSet')[-1]
            stream.remove(list(stream)[5])
            root.write(path)
            with self.assertRaises(ValueError):
                leer_posiciones(path, meta)

class BoostTests(unittest.TestCase):

    def test_boosting_learns_nonlinear_rule_and_does_not_modify_weights(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(600, 3))
        y = (abs(x[:, 0]) > 0.8).astype(float)
        weights = np.ones(600)
        fitted = ajustar_stumps(x, y, weights, n_trees=70)
        pred = log_odds_stumps(x, fitted) > 0
        self.assertGreater((pred == y).mean(), 0.95)
        np.testing.assert_array_equal(weights, np.ones(600))
        self.assertEqual(json.loads(json.dumps(fitted)), fitted)

    def test_determinism_and_invalid_training(self):
        x = np.arange(100)[:, None].astype(float)
        y = (x[:, 0] > 50).astype(float)
        self.assertEqual(ajustar_stumps(x, y, n_trees=4), ajustar_stumps(x, y, n_trees=4))
        with self.assertRaises(ValueError):
            ajustar_stumps(x, np.zeros(100))
        with self.assertRaises(ValueError):
            ajustar_stumps(x * np.nan, y)

class CandidateTests(unittest.TestCase):

    def public_model(self):
        from mtah.externos import COMMIT
        from mtah.skillcorner import COMMIT_SC, SPLITS
        from mtah.idsse import SPLITS_DFL
        model = fixture_model()
        model['procedencia'] = dict(gt_concurso_usado=False, idsse_article=28196177, splits_dfl=SPLITS_DFL)
        model['base_v3']['procedencia'] = dict(gt_concurso_usado=False, skillcorner_commit=COMMIT_SC, splits=SPLITS)
        model['base_v3']['base_v2']['procedencia'] = dict(gt_concurso_usado=False, commit=COMMIT)
        return model

    def test_v4_loader_and_package_only_include_public_code_and_parameters(self):
        import zipfile
        from mtah.entrega import preparar_codigo
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            path = base / 'model.json'
            path.write_text(json.dumps(self.public_model()), encoding='utf-8')
            self.assertEqual(cargar_v4(path)['version'], 4)
            preparar_codigo(base / 'code.zip', path, 'veriana')
            with zipfile.ZipFile(base / 'code.zip') as z:
                self.assertIn('config/idsse_v4.json', z.namelist())
                self.assertIn('src/mtah/reanudaciones.py', z.namelist())
                self.assertIn('src/mtah/geometria_visual.py', z.namelist())
                self.assertIn('Candidata v4', z.read('veriana.md').decode('utf-8'))
                self.assertFalse(any((n.startswith(('data/', 'outputs/')) or n.endswith(('.xml', '.csv', '.mp4', '.npz')) for n in z.namelist())))

    def test_loader_rejects_nested_private_provenance_and_invalid_trees(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            model = self.public_model()
            model['base_v3']['procedencia']['gt_concurso_usado'] = True
            path.write_text(json.dumps(model), encoding='utf-8')
            with self.assertRaises(ValueError):
                cargar_v4(path)
            model = self.public_model()
            model['pausa_v4']['arboles'] = [dict(feature=900, threshold=0, left=0, right=0)]
            path.write_text(json.dumps(model), encoding='utf-8')
            with self.assertRaises(ValueError):
                cargar_v4(path)

    def test_motion_uses_the_same_time_scale_at_5_and_25_hz(self):
        time = np.arange(101) / 25
        xy = np.zeros((101, 2, 2))
        xy[:, :, 0] = time[:, None] * 3
        a = velocidad_ventana(xy, [(0, 101)], 25, 13)
        down = xy[::5]
        b = velocidad_ventana(down, [(0, len(down))], 5, 13)
        np.testing.assert_allclose(a[5::5], b[1:], atol=1e-12)

    def test_5hz_motion_matches_the_original_training_definition(self):
        from mtah.features_control import _velocidad
        atd = synthetic()
        xy = atd['jugadores'][:, :, :2]
        np.testing.assert_array_equal(velocidad_ventana(xy, [(0, 120)], 5, 13), _velocidad(xy, [(0, 120)], 5, 13)[0])
        from mtah.mejora import extraer_mejora
        frame = extraer_v4(atd, cf=ConfigFeatures(fps=5))
        old = extraer_mejora(atd, config_features=ConfigFeatures(fps=5))
        np.testing.assert_array_equal(frame.v50_pausa_suave, old.v50_jugadores_suave)
        np.testing.assert_array_equal(frame.v75_pausa_suave, old.v75_jugadores_suave)
        np.testing.assert_array_equal(frame.fraccion_pausa_suave, old.fraccion_jugadores_moviles_suave)

    def test_control_antisymmetry_and_pause_symmetry(self):
        a = synthetic()
        f = extraer_v4(a)
        g = extraer_v4(dict(a, equipo_slot=1 - a['equipo_slot']))
        np.testing.assert_allclose(matriz_control_v4(f), -matriz_control_v4(g), atol=1e-12)
        np.testing.assert_allclose(matriz_pausa_v4(f), matriz_pausa_v4(g), atol=1e-12, equal_nan=True)

    def test_ball_absence_is_not_a_pause_feature(self):
        a = synthetic()
        b = dict(a, balon=np.full_like(a['balon'], np.nan))
        np.testing.assert_allclose(matriz_pausa_v4(extraer_v4(a)), matriz_pausa_v4(extraer_v4(b)), equal_nan=True)

    def test_no_observations_abstains_and_sequence_is_reproducible(self):
        a = synthetic()
        a['balon'][:] = np.nan
        a['jugadores'][:] = np.nan
        result = predecir_v4(a, calif=np.ones(120, bool), model=fixture_model())
        self.assertTrue((result['etiquetas'] == 'X').all())
        a = synthetic()
        x = predecir_v4(a, calif=np.ones(120, bool), model=fixture_model())
        y = predecir_v4(a, calif=np.ones(120, bool), model=fixture_model())
        np.testing.assert_array_equal(x['etiquetas'], y['etiquetas'])

    def test_period_boundary_has_no_velocity_and_partial_metric_is_exact(self):
        groups = np.r_[np.ones(60), np.full(60, 2)]
        frame = extraer_v4(synthetic(), groups=groups)
        self.assertEqual(frame.tti_ventaja.iloc[60], 0)
        labels = np.array(list('HHHHDDDDAAAA'))
        metrics = metricas_completas(labels, labels, np.ones(12))
        self.assertEqual(metrics['macroF1'], 1.0)
        self.assertEqual(metrics['spellF1'], 1.0)

class RestartTests(unittest.TestCase):

    def event(self, kind='Throw-in', team=None, a=0, b=60):
        return dict(code='SET PIECES', start_frame=a, end_frame=b, equipo=team, tags={'Type': kind})

    def test_types_unknown_teams_conflicts_and_duplicates(self):
        events = [self.event(), self.event(), self.event('Corner Kick', 0, 20, 40), self.event('Free Kick', 1, 20, 40)]
        frame = features_reanudacion(events, 80)
        self.assertTrue(frame.saque_banda.iloc[10])
        self.assertEqual(frame.saque_equipo.iloc[10], -1)
        self.assertTrue(frame.saque_conflicto.iloc[30])
        self.assertEqual(frame.saque_equipo.iloc[30], -1)
        self.assertFalse(frame.saque_banda.iloc[60])

    def test_thrower_is_not_inverted_and_no_live_label_during_wait(self):
        a = synthetic(80)
        a['balon'][:, :2] = [20.0, 0.5]
        a['balon'][30:, 1] += np.arange(50) * 0.4
        a['jugadores'][:, :, 1] = 0.5
        a['jugadores'][:, 0, 0] = 20.2
        frame = extraer_v4(a, [self.event(b=80)], cf=ConfigFeatures(fps=10))
        owners = evidencia_reanudacion(frame, a, [(0, 80)], ConfigFeatures(fps=10))
        self.assertTrue((owners[:30] == -1).all())
        self.assertEqual(owners[31], 0)

    def test_no_ball_or_ambiguous_thrower_does_not_invent_team(self):
        a = synthetic(80)
        a['balon'][:] = np.nan
        frame = extraer_v4(a, [self.event(b=80)])
        owners = evidencia_reanudacion(frame, a, [(0, 80)], ConfigFeatures())
        self.assertTrue((owners == -1).all())

class HomographyTests(unittest.TestCase):

    def test_normalized_plane_mapping_and_scale_invariance(self):
        h = np.array([[1920, 0, 10], [0, 1080, 20], [0, 0, 1.0]])
        xy = np.array([[0, 0], [105, 68], [52.5, 34]])
        pixels = campo_a_imagen(xy, h, 105, 68)
        np.testing.assert_allclose(pixels, [[10, 20], [1930, 1100], [970, 560]])
        np.testing.assert_allclose(pies_a_campo(pixels, h * 1e-05, 105, 68), xy, atol=1e-12)

    def test_singular_matrices_and_ambiguous_association_are_rejected(self):
        with self.assertRaises(ValueError):
            pies_a_campo([[1, 2]], np.zeros((3, 3)), 105, 68)
        h = np.diag([100.0, 100.0, 1.0])
        xy = np.array([[10.0, 20.0], [11.0, 20.0]])
        feet = campo_a_imagen([[10.5, 20]], h, 105, 68)
        self.assertEqual(asociar_ejecutante(feet, h, 105, 68, xy, [0, 1])[0], -1)
        feet = campo_a_imagen([[10.0, 20]], h, 105, 68)
        self.assertEqual(asociar_ejecutante(feet, h, 105, 68, [[10, 20], [15, 20]], [0, 1])[0], 0)
if __name__ == '__main__':
    unittest.main()
