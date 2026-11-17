import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtah.features_control import ConfigFeatures, bloques
from mtah.mejora import ConfigMejora, ajustar_pausa, evidencia_mejora, extraer_mejora, matriz_pausa, predecir_mejora, rellenar_huecos_cortos
from mtah.referencias_externas import evaluar_apoyo, intervalos_apoyados

def synthetic(n=160):
    ball = np.column_stack([20 + np.arange(n) * 0.12, np.full(n, 30.0), np.full(n, 3.0)])
    players = np.repeat(ball[:, None], 12, axis=1)
    players[:, :, 0] += np.array([0.2, 5, 10, 15, 20, 25, 8, 12, 16, 20, 24, 28])
    return dict(balon=ball, jugadores=players, equipo_slot=np.r_[np.zeros(6, np.int8), np.ones(6, np.int8)])
MODEL = dict(version=2, control=dict(coeficientes=[3.0, 0.2, 0.1, 0.5]), pausa=dict(coeficientes=[2.0, -3.0, -3.0, -2.0]))

class MejoraTests(unittest.TestCase):

    def test_short_gap_is_bounded_and_original_is_unchanged(self):
        atd = synthetic()
        atd['balon'][40:45] = np.nan
        before = atd['balon'].copy()
        prepared, observed, filled = rellenar_huecos_cortos(atd, [(0, 160)], ConfigFeatures(), ConfigMejora())
        self.assertTrue(filled[40:45].all())
        self.assertFalse(observed[40:45].any())
        self.assertTrue(np.isfinite(prepared['balon'][40:45, :2]).all())
        np.testing.assert_array_equal(before, atd['balon'])

    def test_long_gap_and_period_boundary_are_not_interpolated(self):
        atd = synthetic()
        atd['balon'][30:70] = np.nan
        atd['balon'][79:82] = np.nan
        groups = np.r_[np.ones(80), np.full(80, 2)]
        _, _, filled = rellenar_huecos_cortos(atd, bloques(np.ones(160, bool), groups), ConfigFeatures(), ConfigMejora())
        self.assertFalse(filled.any())

    def test_impossible_jump_or_outside_endpoints_are_not_interpolated(self):
        atd = synthetic()
        atd['balon'][40:45] = np.nan
        atd['balon'][45, 0] += 60
        _, _, filled = rellenar_huecos_cortos(atd, [(0, 160)], ConfigFeatures(), ConfigMejora())
        self.assertFalse(filled.any())
        atd['balon'][45, 0] = -2
        _, _, filled = rellenar_huecos_cortos(atd, [(0, 160)], ConfigFeatures(), ConfigMejora())
        self.assertFalse(filled.any())

    def test_imputed_geometry_has_nonzero_but_reduced_evidence(self):
        atd = synthetic()
        atd['balon'][40:45] = np.nan
        frame = extraer_mejora(atd)
        evidence = evidencia_mejora(frame, [(0, 160)], MODEL)
        self.assertFalse(frame.visible.iloc[42])
        self.assertTrue(frame.interpolada.iloc[42])
        self.assertAlmostEqual(frame.calidad_geometria.iloc[42], 0.35)
        self.assertGreater(evidence['logits'][42, 0], evidence['logits'][42, 1])

    def test_absence_without_context_does_not_become_dead_ball(self):
        atd = synthetic()
        atd['balon'][:] = np.nan
        atd['jugadores'][:, :, :2] = atd['jugadores'][0, :, :2]
        result = predecir_mejora(atd, calif=np.ones(160, bool), model=MODEL)
        self.assertTrue((result['etiquetas'] == 'X').all())

    def test_set_piece_and_independent_motion_support_stoppage_without_ball(self):
        atd = synthetic()
        atd['balon'][:] = np.nan
        atd['jugadores'][:, :, :2] = atd['jugadores'][0, :, :2]
        events = [dict(start_frame=0, end_frame=30, code='SET PIECES', equipo=None)]
        result = predecir_mejora(atd, events, np.ones(160, bool), model=MODEL)
        self.assertTrue((result['etiquetas'][20:140] == 'D').all())

    def test_travelling_ball_clears_stoppage_context(self):
        atd = synthetic()
        atd['balon'][:50] = np.nan
        atd['jugadores'][:50, :, :2] = atd['jugadores'][0, :, :2]
        atd['jugadores'][50:, :, 0] = atd['jugadores'][0, :, 0] + np.arange(110)[:, None] * 0.16
        atd['balon'][50:, 0] = atd['jugadores'][50:, 0, 0] - 0.2
        frame = extraer_mejora(atd, [dict(start_frame=0, end_frame=30, code='SET PIECES', equipo=None)])
        evidence = evidencia_mejora(frame, [(0, 160)], MODEL)
        self.assertFalse(evidence['contexto_parada'][80])

    def test_team_swap_preserves_dead_and_swaps_owner_evidence(self):
        atd = synthetic()
        frame1 = extraer_mejora(atd)
        frame2 = extraer_mejora(dict(atd, equipo_slot=1 - atd['equipo_slot']))
        one, two = [evidencia_mejora(f, [(0, 160)], MODEL)['logits'] for f in [frame1, frame2]]
        np.testing.assert_allclose(one[:, [1, 0, 2]], two)

    def test_pause_model_uses_motion_and_no_ball_visibility_feature(self):
        n = 200
        frame = pd.DataFrame(dict(v50_jugadores_suave=np.r_[np.full(100, 0.1), np.full(100, 3.0)], v75_jugadores_suave=np.r_[np.full(100, 0.2), np.full(100, 4.0)], fraccion_jugadores_moviles_suave=np.r_[np.zeros(100), np.ones(100)], n_velocidades_validas=np.full(n, 12)))
        labels = np.array(list('D' * 100 + 'H' * 100))
        model = ajustar_pausa(frame, labels, stride=1)
        prediction = matriz_pausa(frame) @ np.asarray(model['coeficientes']) > 0
        np.testing.assert_array_equal(prediction, labels == 'D')
        self.assertNotIn('visible', str(model['variables']))

class WeakReferenceTests(unittest.TestCase):

    def row(self, kind, first, last, team='Home', to=None, subtype=''):
        return dict(Team=team, Type=kind, To=to, Subtype=subtype, Period=1, **{'Start Frame': first, 'End Frame': last})

    def test_pass_interior_and_guarded_stoppage(self):
        events = pd.DataFrame([self.row('PASS', 1, 20, to='P2'), self.row('BALL OUT', 30, 40), self.row('SET PIECE', 100, 100)])
        refs, _ = intervalos_apoyados(events, np.ones(120), fps=25)
        self.assertEqual(refs[5], 'H')
        self.assertEqual(refs[19], 'X')
        self.assertEqual(refs[60], 'D')
        self.assertEqual(refs[99], 'X')

    def test_live_event_or_period_boundary_censors_dead_reference(self):
        events = pd.DataFrame([self.row('BALL OUT', 1, 5), self.row('RECOVERY', 30, 30), self.row('SET PIECE', 100, 100)])
        refs, _ = intervalos_apoyados(events, np.ones(120))
        self.assertFalse((refs == 'D').any())
        events = events[events.Type != 'RECOVERY']
        refs, _ = intervalos_apoyados(events, np.r_[np.ones(60), np.full(60, 2)])
        self.assertFalse((refs == 'D').any())

    def test_conflicting_teams_and_offside_pass_are_not_truth(self):
        events = pd.DataFrame([self.row('PASS', 1, 30, to='P2'), self.row('PASS', 1, 30, 'Away', 'P3'), self.row('PASS', 40, 60, to='P4', subtype='OFFSIDE')])
        refs, _ = intervalos_apoyados(events, np.ones(80))
        self.assertTrue((refs == 'X').all())
        result = evaluar_apoyo(np.full(80, 'H'), refs)
        self.assertNotIn('spellF1', result)
if __name__ == '__main__':
    unittest.main()
