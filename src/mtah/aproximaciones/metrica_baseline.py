"""Documentación pública en README.md."""
import pandas as pd
from .. import ROOT, contrato

class DesdeCSV:
    """Documentación pública en README.md."""

    def __init__(self, nombre, ruta, confianza=0.8):
        self.nombre, self.ruta, self.confianza = (nombre, ROOT / ruta, confianza)
        self.params = {}

    def etiquetas(self, datos):
        return pd.read_csv(self.ruta, dtype={'label': 'object'})['label'].str.strip().str.upper().to_numpy()

    def probabilidades(self, datos):
        return contrato.desde_etiquetas(self.etiquetas(datos), self.confianza)
