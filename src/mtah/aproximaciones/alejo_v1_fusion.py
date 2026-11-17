"""Documentación pública en README.md."""
import numpy as np
import pandas as pd
from .. import eventos, forma, temporal
from .. import firmas as firmas_mod
from ..carga import games
from ..carga import A, H
from ..parametros import p, sobrescribir
from .alejo_v02_eventos import EventosV02
from .juliana_v3 import JulianaV3
from .juliana_v4 import JulianaV4

def _logit(q):
    return np.log(q / (1 - q))

def pool_log_lineal(c_a, c_b, w, p_min):
    """Documentación pública en README.md."""
    a = np.clip(c_a, max(p_min, 1e-09), 1 - max(p_min, 1e-09))
    b = np.clip(c_b, max(p_min, 1e-09), 1 - max(p_min, 1e-09))
    return 1 / (1 + np.exp(-((1 - w) * _logit(a) + w * _logit(b))))

class FusionV1(EventosV02):
    """Documentación pública en README.md."""
    nombre = 'alejo_v1_fusion'

    def __init__(self, params=None):
        super().__init__()
        self.params = {**self.params, 'PESO_FUSION_V3': p('PESO_FUSION_V3'), **(params or {})}
        self._v3 = JulianaV3()

    def probabilidades(self, datos):
        comp = self._componentes(datos)
        P3 = self._v3.probabilidades(datos)
        with sobrescribir(**self.params):
            c_control = P3[:, 0] / np.maximum(P3[:, 0] + P3[:, 1], 1e-12)
            c_toque = np.where(comp['dueno'] == H, comp['confianza'], np.where(comp['dueno'] == A, 1 - comp['confianza'], 0.5))
            c = pool_log_lineal(c_toque, c_control, p('PESO_FUSION_V3'), p('P_MIN'))
            c = np.clip(c, p('P_MIN'), 1 - p('P_MIN'))
            p_d = comp['p_d']
            return np.column_stack([(1 - p_d) * c, (1 - p_d) * (1 - c), p_d])

def _sigmoide(z):
    with np.errstate(over='ignore'):
        return 1 / (1 + np.exp(-z))

class FusionV11(EventosV02):
    """Documentación pública en README.md."""
    nombre = 'alejo_v11_fusion'

    def __init__(self, params=None):
        super().__init__()
        self.params = {**self.params, 'PESO_CONTROL_V4': p('PESO_CONTROL_V4'), 'PESO_PAUSA_V4': p('PESO_PAUSA_V4'), **(params or {})}
        self._v4 = JulianaV4()

    def probabilidades(self, datos):
        comp = self._componentes(datos)
        pred = self._v4.prediccion(datos)
        P4 = self._v4.probabilidades(datos)
        with sobrescribir(**self.params):
            c_control = P4[:, 0] / np.maximum(P4[:, 0] + P4[:, 1], 1e-12)
            c_toque = np.where(comp['dueno'] == H, comp['confianza'], np.where(comp['dueno'] == A, 1 - comp['confianza'], 0.5))
            c = np.clip(pool_log_lineal(c_toque, c_control, p('PESO_CONTROL_V4'), p('P_MIN')), p('P_MIN'), 1 - p('P_MIN'))
            pausa = np.asarray(pred['evidencia']['pausa_log_odds'], dtype=np.float64)
            fiable = np.isfinite(pausa)
            p_d = comp['p_d'].copy()
            p_d[fiable] = pool_log_lineal(p_d[fiable], _sigmoide(pausa[fiable]), p('PESO_PAUSA_V4'), p('P_MIN'))
            p_d = np.clip(p_d, p('P_MIN'), 1 - p('P_MIN'))
            return np.column_stack([(1 - p_d) * c, (1 - p_d) * (1 - c), p_d])

class FusionV12(FusionV11):
    """Documentación pública en README.md."""
    nombre = 'alejo_v12_fusion'

    def __init__(self, params=None):
        super().__init__(params)
        self.params.pop('PESO_PAUSA_V4', None)

    def probabilidades(self, datos):
        comp = self._componentes(datos)
        P4 = self._v4.probabilidades(datos)
        with sobrescribir(**self.params):
            c_control = P4[:, 0] / np.maximum(P4[:, 0] + P4[:, 1], 1e-12)
            c_toque = np.where(comp['dueno'] == H, comp['confianza'], np.where(comp['dueno'] == A, 1 - comp['confianza'], 0.5))
            c = np.clip(pool_log_lineal(c_toque, c_control, p('PESO_CONTROL_V4'), p('P_MIN')), p('P_MIN'), 1 - p('P_MIN'))
            p_d = comp['p_d']
            return np.column_stack([(1 - p_d) * c, (1 - p_d) * (1 - c), p_d])
ANCLAS_ADELANTABLES = frozenset({'balon_para', 'frenazo', 'sin_ancla'})
TIPOS_ADELANTABLES = frozenset({'throw-in', 'throw in', 'goal kick', 'corner kick', 'corner'})

def adelantar_inicios(paradas, pausa, suave):
    """Documentación pública en README.md."""
    q = pd.Series(np.nan_to_num(np.asarray(pausa, dtype=np.float64), nan=0.0))
    q = q.rolling(int(suave), center=True, min_periods=1).mean().to_numpy()
    d = paradas.copy()
    for i in d.index:
        if d.at[i, 'ancla_ini'] not in ANCLAS_ADELANTABLES or str(d.at[i, 'tipo']).strip().lower() not in TIPOS_ADELANTABLES:
            continue
        s, lo = (int(d.at[i, 't_ini']), int(d.at[i, 'ini_busqueda']))
        k = s
        while k - 1 >= lo and q[k - 1] > 0.5:
            k -= 1
        if k < s:
            d.at[i, 't_ini'], d.at[i, 'ancla_ini'] = (k, 'pausa_v4')
    if 'dur_s' in d:
        d['dur_s'] = (d['t_fin'] - d['t_ini']).clip(lower=0) / 25
    return d

class FusionV13(FusionV12):
    """Documentación pública en README.md."""
    nombre = 'alejo_v13_fusion'

    def __init__(self, params=None):
        super().__init__(params)
        self.params = {**self.params, 'SUAVE_PAUSA_V4': p('SUAVE_PAUSA_V4'), **(params or {})}

    def paradas(self, datos):
        """Documentación pública en README.md."""
        comp = self._componentes(datos)
        pausa = _sigmoide(np.asarray(self._v4.prediccion(datos)['evidencia']['pausa_log_odds'], dtype=np.float64))
        with sobrescribir(**self.params):
            return adelantar_inicios(comp['paradas'], pausa, p('SUAVE_PAUSA_V4'))

    def probabilidades(self, datos):
        P12 = super().probabilidades(datos)
        c = P12[:, 0] / np.maximum(P12[:, 0] + P12[:, 1], 1e-12)
        with sobrescribir(**self.params):
            p_d = eventos.prob_muerto(self.paradas(datos), len(c))
            return np.column_stack([(1 - p_d) * c, (1 - p_d) * (1 - c), p_d])

def regimen_ruidoso(comp):
    """Documentación pública en README.md."""
    return bool(comp['fallback_b'] or comp['radio'] > p('R_CONTACTO'))

class FusionV2(FusionV13):
    """Documentación pública en README.md."""
    nombre = 'alejo_v2_robusta'

    def __init__(self, params=None):
        super().__init__(params)
        self.params = {**self.params, 'MARGEN_BANDA': p('MARGEN_BANDA'), 'HUECO_FIRMA': p('HUECO_FIRMA'), 'PEGADO_FIRMA': p('PEGADO_FIRMA'), **(params or {})}

    def regimen(self, datos):
        return 'ruidoso' if regimen_ruidoso(self._componentes(datos)) else 'limpio'

    def etiquetas(self, datos):
        """Documentación pública en README.md."""
        P = self.probabilidades(datos)
        prm = {'prior_uniforme': True}
        if datos.grupos is None:
            return temporal.decodificar(P, datos.calif, prm)[0]
        lab = np.full(len(P), 'X', dtype='<U1')
        for g in np.unique(datos.grupos):
            m = datos.grupos == g
            lab[m] = temporal.decodificar(P, datos.calif & m, prm)[0][m]
        return lab

    def firmas(self, datos):
        """Documentación pública en README.md."""
        clave = ('firmas', self.params['MARGEN_BANDA'])
        if clave not in datos.extra:
            jug = datos.atd['jugadores']
            pitch = datos.pitch if datos.pitch is not None else games.GAMES[games.GAME]['pitch']
            datos.extra[clave] = firmas_mod.firmas(jug[..., :2].astype(np.float64), np.asarray(datos.atd['equipo_slot']), jug[..., 2].astype(np.float64), eventos.mitades_calificadas(datos.calif, datos.grupos), pitch, margen_banda=self.params['MARGEN_BANDA'])
        return datos.extra[clave]

    def paradas(self, datos):
        """Documentación pública en README.md."""
        d = super().paradas(datos)
        with sobrescribir(**self.params):
            return firmas_mod.adelantar_por_firma(d, self.firmas(datos)['fuera_banda'] >= 1, {'throw-in', 'throw in'}, p('HUECO_FIRMA'), p('PEGADO_FIRMA'))

    def probabilidades(self, datos):
        comp = self._componentes(datos)
        P4 = self._v4.probabilidades(datos)
        with sobrescribir(**self.params):
            c_control = np.clip(P4[:, 0] / np.maximum(P4[:, 0] + P4[:, 1], 1e-12), p('P_MIN'), 1 - p('P_MIN'))
            if regimen_ruidoso(comp):
                c = c_control
            else:
                c_toque = np.where(comp['dueno'] == H, comp['confianza'], np.where(comp['dueno'] == A, 1 - comp['confianza'], 0.5))
                c = np.clip(pool_log_lineal(c_toque, c_control, p('PESO_CONTROL_V4'), p('P_MIN')), p('P_MIN'), 1 - p('P_MIN'))
            p_d = eventos.prob_muerto(self.paradas(datos), len(c))
            return np.column_stack([(1 - p_d) * c, (1 - p_d) * (1 - c), p_d])

class FusionV2SinFirma(FusionV2):
    """Documentación pública en README.md."""
    nombre = 'alejo_v2_sin_firma'

    def paradas(self, datos):
        return FusionV13.paradas(self, datos)

class FusionV2ST(FusionV2):
    """Documentación pública en README.md."""
    nombre = 'alejo_v2_st'

    def __init__(self, params=None):
        super().__init__(params)
        self.params = {**self.params, 'CONF_FASE_ST': p('CONF_FASE_ST'), **(params or {})}

    def probabilidades(self, datos):
        P = super().probabilidades(datos)
        with sobrescribir(**self.params):
            dueno_st = forma.poseedor_st(datos.eventos_st, len(P))
            q = p('CONF_FASE_ST')
            p_st = np.where(dueno_st == H, q, np.where(dueno_st == A, 1 - q, 0.5))
            vivo = np.maximum(P[:, 0] + P[:, 1], 1e-12)
            c = np.clip(P[:, 0] / vivo, p('P_MIN'), 1 - p('P_MIN'))
            c = 1 / (1 + np.exp(-(np.log(c / (1 - c)) + np.log(p_st / (1 - p_st)))))
            c = np.clip(c, p('P_MIN'), 1 - p('P_MIN'))
            return np.column_stack([vivo * c, vivo * (1 - c), P[:, 2]])
