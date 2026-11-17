"""Documentación pública en README.md."""
import sys
from pathlib import Path
from . import ROOT
from .preprocesamiento import A, H, autorizar, cargar_tracking, equipo_por_slot, leer_smart_tagging
sys.path.insert(0, str(ROOT))
import games
sys.path.insert(0, str(ROOT / 'starter'))
import load_smart_tagging

def cargar_atd(source='tactical', usar_cache=True, data_dir=None, *, autorizar_datos=False):
    autorizar(autorizar_datos)
    paths = games.asset_paths(source=source, data_dir=data_dir)
    cache = Path(paths['atd_raw']).parent / 'cache' if usar_cache else None
    return cargar_tracking(paths['atd_meta'], paths['atd_raw'], games.source_frames(source), games.GAMES[games.GAME]['pitch'], cache_dir=cache)

def cargar_smart_tagging(data_dir=None, *, autorizar_datos=False):
    autorizar(autorizar_datos)
    path = games.asset_paths(data_dir=data_dir)['smart_tagging']
    return leer_smart_tagging(path, games.source_frames(), games.GAMES[games.GAME]['fps'])
