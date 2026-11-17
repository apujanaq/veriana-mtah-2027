"""Documentación pública en README.md."""
import sys
import numpy as np
import pandas as pd
from . import ROOT
from .etapa_a import tramos
sys.path.insert(0, str(ROOT / 'evaluation'))
import possession_metrics as pm
FPS = 25
RANGOS_C1 = {'min_balon_en_juego': (53.0, 60.0, 'alta', "UEFA 53,42 (2022-23); Kicker 'just under 54' (pre-2020); Opta Analyst ≈54:27 (2022-23 a oct. 2024: ~800 partidos, podría incluir este 1 → contaminación despreciable, declarada). No se usa el promedio del Barça 2024-25 (incluye este partido)"), 'n_paradas': (85, 125, 'media', '≈99 reanudaciones por partido (banda 40,5 + córner 9,6 + meta 15 + fuera de juego 3,8 + falta 25 + centro 4,6 + penalti 0,3); Kicker 106. −1 (el 1er tiempo arranca sin D) y + lesiones, VAR y balón a tierra'), 'dur_media_parada_s': (22.0, 38.0, 'media', 'derivada: (105,9 − min_en_juego) × 60 / n_paradas, en los extremos de ambos rangos'), 'dur_media_tramo_vivo_s': (26.0, 40.0, 'media', 'investigación: tramo continuo ≈31-33 s (31-35 según supuestos); ampliado por la diferencia de definición Opta/Metrica'), 'n_posesiones': (200, 500, 'BAJA', 'Opta Pro: 264 (Premier 2012-13) o <200 (Premier 2016-17) cambios de control (otra liga y otras temporadas: sin contaminación); nuestro GT (H1, último toque) suma micro-posesiones → techo amplio. Solo detecta absurdos'), 'pct_posesion_barca': (50.0, 80.0, 'media', 'conocimiento general: el Barça de 2024-25 domina la posesión. Chequeo de DIRECCIÓN (<50% sugeriría H y A invertidos). Por verificar con una fuente')}

def metricas_c1(etiquetas):
    """Documentación pública en README.md."""
    codes = pm.label_codes(np.asarray(etiquetas))
    calif = codes != pm.X
    vivo = calif & ((codes == pm.H) | (codes == pm.A))
    muerto = calif & (codes == pm.D)
    n_paradas = len(tramos(muerto))
    n_vivos = len(tramos(vivo & ~muerto))
    runs = pm.valid_runs(codes)
    n_pos = len(pm.possession_spells(codes, runs))
    seg = lambda m: m.sum() / FPS
    return {'min_balon_en_juego': round(seg(vivo) / 60, 1), 'n_paradas': n_paradas, 'dur_media_parada_s': round(seg(muerto) / n_paradas, 1) if n_paradas else None, 'dur_media_tramo_vivo_s': round(seg(vivo) / n_vivos, 1) if n_vivos else None, 'n_posesiones': n_pos, 'pct_posesion_barca': round(100 * (codes == pm.H).sum() / max(vivo.sum(), 1), 1)}

def evaluar_c1(etiquetas):
    """Documentación pública en README.md."""
    m = metricas_c1(etiquetas)
    filas = []
    for k, (lo, hi, conf, fuente) in RANGOS_C1.items():
        v = m[k]
        filas.append({'regla': k, 'valor': v, 'rango': f'[{lo}, {hi}]', 'pasa': v is not None and lo <= v <= hi, 'confianza': conf, 'fuente': fuente})
    return pd.DataFrame(filas).set_index('regla')
