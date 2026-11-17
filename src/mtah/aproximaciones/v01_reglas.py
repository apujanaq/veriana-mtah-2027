"""Documentación pública en README.md."""
from .. import contrato, ensamble, etapa_a, etapa_b, features
from ..parametros import CLAVES_V01, sobrescribir, valores_defecto

class ReglasV01:
    """Documentación pública en README.md."""
    nombre = 'alejo_v01_reglas'

    def __init__(self, params=None, confianza=0.8):
        self.params = {**valores_defecto(CLAVES_V01), **(params or {})}
        self.confianza = confianza

    def etiquetas(self, datos):
        with sobrescribir(**self.params):
            feat = features.calcular_features(datos.atd, datos.calif, pitch=datos.pitch)
            calif = datos.calif
            muerto, _ = etapa_a.detectar_muerto_v01(feat, calif)
            ini_2t = features.games.half_video_blocks(features.games.GAME)[1][1]
            muerto[ini_2t:ensamble.KICKOFF_2T] = True
            umbrales = etapa_b.umbrales_nulos(feat, calif)
            contacto = etapa_b.detectar_contactos(feat, calif, umbrales)
            dueno = etapa_b.asignar_dueno(contacto, muerto, calif)
            return ensamble.etiquetar(muerto, dueno, calif)

    def probabilidades(self, datos):
        return contrato.desde_etiquetas(self.etiquetas(datos), self.confianza)
