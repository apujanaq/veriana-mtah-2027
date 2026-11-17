"""Documentación pública en README.md."""
import numpy as np
from .. import ensamble, etapa_b, eventos, features, forma, toque
from ..carga import load_smart_tagging
from ..carga import A, H
from ..parametros import p, sobrescribir, valores_defecto
CLAVES = ('W_SUAVE_COLECTIVA', 'LAG_DIRECCION', 'MARGEN_FUERA', 'V_COL_PARADA', 'V_BALON_EN_JUEGO', 'MIN_PARADA', 'MIN_VIVO_ENTRE_PARADAS', 'RETRO_FUERA', 'DIST_NULA', 'ALFA_NULO', 'MIN_CONDUCCION', 'R_CONTACTO', 'DELTA_VENTANA_ST', 'LAG_VIAJE', 'MIN_VIAJE', 'SEP_MIN', 'RETRO_MAX', 'RETRO_BALON', 'VIAJE_NETO_M', 'SAQUE_NETO_M', 'TOLERANCIA_CRUCE', 'R_CENTRO', 'DELTA_NUCLEO', 'MIN_VIVO_TRAS_SAQUE', 'VISIBLE_MAX_PARADA', 'RETRO_FRENAZO', 'RAMPA_NITIDA', 'RAMPA_DIFUSA', 'P_MIN', 'TAU_TOQUE', 'TAU_DECAY', 'VENTANA_FORMA', 'CONF_FIRME', 'PUENTE_TRAS_HUECO')
MIN_FRAMES_MEZCLA = 50

class EventosV02:
    """Documentación pública en README.md."""
    nombre = 'alejo_v02_eventos'

    def __init__(self, params=None):
        self.params = {**valores_defecto(CLAVES), **(params or {})}

    def componentes(self, datos):
        """Documentación pública en README.md."""
        with sobrescribir(**self.params):
            feat = features.calcular_features(datos.atd, datos.calif, pitch=datos.pitch)
            calif = datos.calif
            cotas = None if datos.pitch is not None else load_smart_tagging.KICKOFF_VIDEO_FRAMES
            paradas, desfase, info_a = eventos.paradas_completas(feat, calif, datos.eventos_st, datos.atd, datos.grupos, datos.pitch, cotas)
            p_d = eventos.prob_muerto(paradas, len(calif))
            parada = calif & (p_d > 0.5)
            vivo = calif & ~parada
            umbrales = etapa_b.umbrales_nulos(feat, calif)
            d_quiebre = feat['d_min'].to_numpy()[toque.frames_quiebre(feat, vivo, umbrales)]
            mezcla = toque.em_mezcla(d_quiebre, p('DIST_NULA'))
            radio = toque.radio_toque(mezcla, p('DIST_NULA'), p('TAU_TOQUE'))
            fallback_b = mezcla['n'] < MIN_FRAMES_MEZCLA or radio <= 0
            if fallback_b:
                equipo = etapa_b.detectar_contactos(feat, calif, umbrales)
                p_toque = np.where(equipo >= 0, 0.8, np.nan)
            else:
                equipo, p_toque, radio = toque.contactos(feat, vivo, umbrales, mezcla)
            equipo, p_toque = toque.toque_sacador(equipo, p_toque, feat, vivo)
            dueno = etapa_b.asignar_dueno(equipo, parada, calif)
            visible = feat['visible'].to_numpy()
            conf = toque.confianza(dueno, p_toque, visible, vivo)
            senales = forma.senales_suavizadas(datos.atd, calif, datos.grupos, datos.pitch)
            firme = vivo & visible & (dueno >= 0) & (conf >= p('CONF_FIRME'))
            p_forma, info_c = forma.prob_forma(senales, dueno, firme)
            dueno, info_puente = forma.puente_huecos(dueno, equipo, visible, vivo, p_forma)
            info_c = {**info_c, **info_puente}
            conf = toque.confianza(dueno, p_toque, visible, vivo)
            c_h = np.where(dueno == H, conf, np.where(dueno == A, 1 - conf, 0.5))
            c_h = np.clip(c_h, p('P_MIN'), 1 - p('P_MIN'))
            P = np.column_stack([(1 - p_d) * c_h, (1 - p_d) * (1 - c_h), p_d])
        return dict(feat=feat, paradas=paradas, desfase=desfase, info_a=info_a, p_d=p_d, parada=parada, umbrales=umbrales, mezcla=mezcla, radio=radio, fallback_b=fallback_b, equipo=equipo, p_toque=p_toque, dueno=dueno, confianza=conf, p_forma=p_forma, info_c=info_c, P=P)

    def _componentes(self, datos):
        clave = tuple(sorted(self.params.items()))
        guardado = datos.extra.get('alejo_v02')
        if guardado is None or guardado[0] != clave:
            datos.extra['alejo_v02'] = (clave, self.componentes(datos))
        return datos.extra['alejo_v02'][1]

    def probabilidades(self, datos):
        return self._componentes(datos)['P']

class ReglasV02(EventosV02):
    """Documentación pública en README.md."""
    nombre = 'alejo_v02_reglas'

    def etiquetas(self, datos):
        c = self._componentes(datos)
        return ensamble.etiquetar(c['parada'], c['dueno'], datos.calif)
