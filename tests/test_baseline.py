"""Documentación pública en README.md."""
import sys
import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.baseline import predecir
from mtah import etapa_a, etapa_b, features, pipeline

def synthetic_atd(n=150):
    ball = np.column_stack([10 + np.arange(n) * 0.02, np.full(n, 20.0), np.full(n, 2.0)])
    players = np.stack([ball.copy(), ball.copy()], axis=1)
    players[:, 0, 0] += 0.3
    players[:, 1, 0] += 10
    return {'balon': ball, 'jugadores': players, 'equipo_slot': np.array([0, 1])}

def run(atd, events=(), mask=None):
    n = len(atd['balon'])
    return predecir(atd, events, np.ones(n, bool) if mask is None else mask)

def event(start, stop, team=1, code='BUILD UP'):
    return dict(start_frame=start, end_frame=stop, equipo=team, code=code)

class BaselineTests(unittest.TestCase):

    def test_sustained_control_and_metric_distances(self):
        result = run(synthetic_atd())
        self.assertTrue((result['etiquetas'] == 'H').all())

    def test_one_frame_touch_does_not_transfer_control(self):
        atd = synthetic_atd()
        atd['jugadores'][40, 0, 0] += 10
        atd['jugadores'][40, 1, 0] = atd['balon'][40, 0] + 0.1
        self.assertTrue((run(atd)['etiquetas'] == 'H').all())

    def test_sustained_opponent_control_changes_owner(self):
        atd = synthetic_atd()
        atd['jugadores'][60:, 0, 0] += 10
        atd['jugadores'][60:, 1, 0] = atd['balon'][60:, 0] + 0.1
        labels = run(atd)['etiquetas']
        self.assertTrue((labels[:60] == 'H').all())
        self.assertTrue((labels[60:] == 'A').all())

    def test_missing_away_side_does_not_assign_away_when_home_is_closer(self):
        atd = synthetic_atd()
        atd['jugadores'][:, 1] = np.nan
        self.assertTrue((run(atd)['etiquetas'] == 'H').all())

    def test_equal_distances_abstain(self):
        atd = synthetic_atd()
        atd['jugadores'][:, 1] = atd['jugadores'][:, 0]
        self.assertTrue((run(atd)['etiquetas'] == 'X').all())

    def test_occlusion_keeps_last_owner_instead_of_declaring_dead(self):
        atd = synthetic_atd()
        atd['balon'][40:100] = np.nan
        labels = run(atd)['etiquetas']
        self.assertTrue((labels[40:100] == 'H').all())

    def test_smart_tagging_supports_inference_during_occlusion(self):
        atd = synthetic_atd()
        atd['balon'][40:100] = np.nan
        labels = run(atd, [event(50, 90)])['etiquetas']
        self.assertTrue((labels[50:90] == 'A').all())
        self.assertEqual(labels[100], 'H')

    def test_conflicting_or_brief_prior_does_not_change_owner(self):
        atd = synthetic_atd()
        atd['balon'][40:100] = np.nan
        events = [event(50, 90, 0), event(50, 90, 1), event(92, 93)]
        self.assertTrue((run(atd, events)['etiquetas'] == 'H').all())

    def test_visible_ball_evidence_wins_over_tag_prior(self):
        self.assertTrue((run(synthetic_atd(), [event(0, 150)])['etiquetas'] == 'H').all())

    def test_no_observations_abstains_without_crashing(self):
        atd = synthetic_atd()
        atd['balon'][:] = np.nan
        atd['jugadores'][:] = np.nan
        result = run(atd)
        self.assertTrue((result['etiquetas'] == 'X').all())
        thresholds = etapa_b.umbrales_nulos(result['feat'], result['calif'])
        self.assertTrue(np.isinf(thresholds['angulo']))

    def test_slow_players_and_stationary_ball_can_be_dead(self):
        atd = synthetic_atd()
        atd['balon'][:] = [10.0, 20.0, 0.0]
        atd['jugadores'][:, :, 2] = 0.2
        self.assertTrue((run(atd)['etiquetas'] == 'D').all())

    def test_missing_ball_alone_does_not_prove_stoppage(self):
        atd = synthetic_atd()
        atd['balon'][:] = np.nan
        atd['jugadores'][:, :, 2] = 0.2
        self.assertTrue((run(atd)['etiquetas'] == 'X').all())

    def test_out_of_pitch_is_dead_for_a_sustained_interval(self):
        atd = synthetic_atd()
        atd['balon'][:, 0] = -2.0
        self.assertTrue((run(atd)['etiquetas'] == 'D').all())

    def test_reset_at_half_boundary_and_no_smoothing_across_gap(self):
        atd = synthetic_atd()
        mask = np.ones(150, bool)
        mask[60:70] = False
        atd['balon'][70:] = np.nan
        labels = run(atd, mask=mask)['etiquetas']
        self.assertTrue((labels[:60] == 'H').all())
        self.assertTrue((labels[60:] == 'X').all())
        feat = features.calcular_features(atd, calif=mask)
        self.assertTrue(np.isnan(feat.v_balon.iloc[70]))
        self.assertTrue(np.isnan(feat.v_col_suave.iloc[65]))

    def test_dead_ball_resets_ownership(self):
        atd = synthetic_atd()
        atd['balon'][40:100, 0] = -2
        atd['balon'][100:] = np.nan
        labels = run(atd)['etiquetas']
        self.assertTrue((labels[40:100] == 'D').all())
        self.assertTrue((labels[100:] == 'X').all())

    def test_public_boundaries_from_official_metadata(self):
        mask = features.mascara_calificada(199319)
        self.assertEqual(int(mask.sum()), 158857)
        for frame in [0, 11184, 90652, 110807, 190198, 199318]:
            self.assertFalse(mask[frame])
        for frame in [11185, 90651, 110808, 190197]:
            self.assertTrue(mask[frame])

    def test_cli_requires_authorization_before_loading(self):
        with patch('mtah.pipeline.correr', side_effect=AssertionError('accessed data')), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                pipeline.main([])
        self.assertEqual(error.exception.code, 2)

    def test_default_pipeline_never_reads_public_labels(self):
        atd = synthetic_atd()
        with patch('mtah.carga.cargar_atd', return_value=atd), patch('mtah.carga.cargar_smart_tagging', return_value=[]), patch('possession_metrics.read_label_csv', side_effect=AssertionError('read GT')):
            result = pipeline.correr(autorizar_datos=True)
        self.assertEqual(len(result['etiquetas']), 150)

    def test_degenerate_and_short_sequences(self):
        for n in [1, 2, 3, 5]:
            self.assertEqual(run(synthetic_atd(n))['etiquetas'].shape, (n,))
        self.assertTrue(np.isnan(etapa_a.otsu([])))
        self.assertEqual(etapa_a.otsu([1.0, 1.0]), 1.0)
if __name__ == '__main__':
    unittest.main()
