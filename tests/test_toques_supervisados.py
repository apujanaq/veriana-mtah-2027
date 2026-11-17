"""Documentación pública en README.md."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from mtah.toques_supervisados import bloques_validos, eventos_micro_referencia, extraer_variables, picos_probabilidad
from mtah.experimento_micro import aumentar, objetivos, verificar_particion
from mtah.microposesiones import insertar_microposesiones
from mtah.aproximaciones.juliana_v2_micro import JulianaV2Micro
from mtah.contrato import Datos
from scripts.validar_microposesiones import aceptacion

def tracking(n=120):
    t = np.arange(n) / 25
    players = np.tile(np.array([[10.0, 20.0], [12.0, 22.0], [13.0, 21.0], [16.0, 23.0]]), (n, 1, 1))
    players[:, :, 0] += t[:, None]
    ball = np.column_stack([11 + t, 20 + np.sin(t), np.full(n, 999.0)])
    return dict(balon=ball, jugadores=np.dstack([players, np.zeros(players.shape[:2])]), equipo_slot=np.array([0, 0, 1, 1]))

class TestVariables(unittest.TestCase):

    def test_intercambiar_equipos_intercambia_todas_las_variables(self):
        atd = tracking()
        x, valid, names = extraer_variables(atd, np.ones(120, bool))
        other = dict(atd, equipo_slot=1 - atd['equipo_slot'])
        xx, vv, nn = extraer_variables(other, np.ones(120, bool))
        np.testing.assert_allclose(x[:, ::-1], xx)
        np.testing.assert_array_equal(valid[:, ::-1], vv)
        self.assertEqual(names, nn)

    def test_ventanas_no_cruzan_periodo(self):
        atd = tracking()
        periods = np.repeat([1, 2], 60)
        x, valid, _ = extraer_variables(atd, np.ones(120, bool), periods)
        modified = {**atd, 'balon': atd['balon'].copy(), 'jugadores': atd['jugadores'].copy()}
        modified['balon'][60:, :2] += 100
        modified['jugadores'][60:, :, :2] -= 100
        xx, vv, _ = extraer_variables(modified, np.ones(120, bool), periods)
        np.testing.assert_allclose(x[:60], xx[:60])
        np.testing.assert_array_equal(valid[:60], vv[:60])
        self.assertFalse(valid[50:70].any())

    def test_ausencia_no_se_rellena_y_s_ignorada(self):
        atd = tracking()
        atd['balon'][50:60, :2] = np.nan
        x, valid, _ = extraer_variables(atd, np.ones(120, bool))
        atd['balon'][:, 2] = -999
        xx, vv, _ = extraer_variables(atd, np.ones(120, bool))
        np.testing.assert_allclose(x, xx)
        np.testing.assert_array_equal(valid, vv)
        self.assertFalse(valid[40:70].any())
        self.assertTrue(np.isfinite(x).all())

    def test_aumento_no_muta_original_y_preserva_nan(self):
        atd = tracking()
        atd['balon'][40:44, :2] = np.nan
        original = atd['balon'].copy()
        noisy = aumentar(atd, np.repeat([1, 2], 60), 0.8, 1)
        np.testing.assert_equal(atd['balon'], original)
        np.testing.assert_array_equal(np.isnan(noisy['balon']), np.isnan(original))
        self.assertFalse(np.allclose(noisy['balon'][:30], original[:30]))

class TestObjetivos(unittest.TestCase):

    def test_solo_cambios_cortos_con_retorno(self):
        ref = np.array(list('H' * 20 + 'A' * 10 + 'H' * 20 + 'D' * 5 + 'A' * 80))
        self.assertEqual(eventos_micro_referencia(ref, np.ones(len(ref), int)), [(20, 1), (30, 0)])

    def test_no_usa_retorno_del_periodo_siguiente(self):
        ref = np.array(list('H' * 20 + 'A' * 10 + 'H' * 20))
        periods = np.r_[np.ones(30), np.full(20, 2)]
        self.assertEqual(eventos_micro_referencia(ref, periods), [])
        y = objetivos(50, [(29, 1)], periods)
        self.assertFalse(y[30:].any())

    def test_prohibe_tests_reservados_antes_de_abrir_archivos(self):
        with self.assertRaises(ValueError):
            verificar_particion('NO_EXISTE', 'test')

    def test_bloques_vacios(self):
        self.assertEqual(bloques_validos(np.zeros(0, bool)), [])

class TestInsercionSegura(unittest.TestCase):

    def test_periodo_y_hueco_no_se_atraviesan(self):
        lab = np.array(list('H' * 40))
        pt = np.zeros((40, 2))
        pt[18, 1] = 0.9
        pt[22, 0] = 0.9
        periods = np.repeat([1, 2], 20)
        out, n = insertar_microposesiones(lab, pt, 0.5, 50, grupos=periods)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(lab, out)
        mask = np.ones(40, bool)
        mask[20] = False
        out, n = insertar_microposesiones(lab, pt, 0.5, 50, calif=mask)
        self.assertEqual(n, 0)

    def test_empate_simultaneo_no_inventa_posesion(self):
        lab = np.array(list('H' * 30))
        pt = np.zeros((30, 2))
        pt[10] = 0.9
        pt[20, 0] = 0.9
        out, n = insertar_microposesiones(lab, pt, 0.5, 25)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(lab, out)

    def test_nms_respeta_huecos(self):
        prob = np.zeros((30, 2))
        prob[9, 0] = 0.8
        prob[13, 0] = 0.9
        valid = np.ones((30, 2), bool)
        valid[11] = False
        peaks = picos_probabilidad(prob, valid, separacion=8, umbral=0.5)
        self.assertEqual(int((peaks >= 0.5).sum()), 2)

    def test_rechaza_probabilidades_incompatibles(self):
        lab = np.array(list('H' * 30))
        for pt in (np.zeros((29, 2)), np.full((30, 2), np.inf), np.full((30, 2), -1)):
            with self.assertRaises(ValueError):
                insertar_microposesiones(lab, pt, 0.5, 25)

    def test_refinamiento_conserva_d_y_constructor_de_c3(self):
        atd = tracking()
        datos = Datos(atd, None, [], np.ones(120, bool), grupos=np.ones(120, int))
        lab = np.array(list('H' * 50 + 'D' * 20 + 'H' * 50))
        pt = np.zeros((120, 2))
        pt[30, 1] = 0.99
        pt[40, 0] = 0.99
        pt[45, 1] = 0.99
        pt[80, 0] = 0.99
        with patch('mtah.aproximaciones.juliana_v2_micro.DetectorToquesSupervisado') as mock:
            mock.return_value.modelo = {'umbral_toque': 0.8}
            mock.return_value.probabilidades_toque.return_value = pt
            original = JulianaV2Micro()
            reconstruida = type(original)(params={**original.params, 'MICRO_UMBRAL': 0.96})
            self.assertIsNotNone(reconstruida.detector)
            self.assertEqual(reconstruida.params['MICRO_UMBRAL'], 0.96)
            out = reconstruida.refinar_etiquetas(datos, lab)
        np.testing.assert_array_equal(out == 'D', lab == 'D')
        self.assertTrue((out[30:40] == 'A').all())
        self.assertEqual(reconstruida.info_micro['insertadas'], 1)

    def test_empalme_no_infiere_micro_a_traves_de_balon_ausente(self):
        atd = tracking()
        atd['balon'][55:60, :2] = np.nan
        datos = Datos(atd, None, [], np.ones(120, bool), grupos=np.ones(120, int))
        lab = np.full(120, 'H')
        pt = np.zeros((120, 2))
        pt[50, 1] = 0.99
        pt[65, 0] = 0.99
        with patch('mtah.aproximaciones.juliana_v2_micro.DetectorToquesSupervisado') as mock:
            mock.return_value.modelo = {'umbral_toque': 0.8}
            mock.return_value.probabilidades_toque.return_value = pt
            aprox = JulianaV2Micro()
            out = aprox.refinar_etiquetas(datos, lab)
        np.testing.assert_array_equal(out, lab)
        self.assertEqual(aprox.info_micro['insertadas'], 0)

class TestAceptacion(unittest.TestCase):

    def filas(self, mejora):
        return [dict(partido=g, condicion=c, spellF1=0.5 + d, **{'precisión micro': 0.6}) for g in (1, 2) for c in ('limpio', 'ruido') for d in (0.0, mejora)]

    def test_criterio_aplica_a_ambos_partidos_sin_redondear(self):
        self.assertFalse(aceptacion(self.filas(0.01999))['aceptada'])
        self.assertTrue(aceptacion(self.filas(0.021))['aceptada'])
        rows = self.filas(0.021)
        rows[-1]['precisión micro'] = 0.49
        self.assertFalse(aceptacion(rows)['aceptada'])
if __name__ == '__main__':
    unittest.main()
