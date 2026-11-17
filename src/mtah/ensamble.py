"""Documentación pública en README.md."""
from pathlib import Path
import numpy as np
from .carga import A, H, games, load_smart_tagging
from .validacion import validar_csv
KICKOFF_2T = load_smart_tagging.KICKOFF_VIDEO_FRAMES['2nd Half']

def etiquetar(muerto, dueno, calif):
    """Documentación pública en README.md."""
    lab = np.full(len(muerto), 'X', dtype='<U1')
    lab[calif & (dueno == H)] = 'H'
    lab[calif & (dueno == A)] = 'A'
    lab[calif & muerto] = 'D'
    return lab

def escribir_csv(etiquetas, path, n_esperados=None, sobrescribir=False):
    """Documentación pública en README.md."""
    n = games.source_frames() if n_esperados is None else n_esperados
    etiquetas = np.asarray(etiquetas)
    if etiquetas.shape != (n,) or not np.isin(etiquetas, ['H', 'A', 'D', 'X']).all():
        raise ValueError('Etiquetas inválidas o número de frames incorrecto')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w' if sobrescribir else 'x', encoding='utf-8', newline='\n') as fh:
        fh.write('frame,label\n')
        fh.writelines((f'{i},{lab}\n' for i, lab in enumerate(etiquetas)))
    validar_csv(path, n)
    return path
