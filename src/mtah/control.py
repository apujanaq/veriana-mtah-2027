"""Documentación pública en README.md."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class ConfigControl:
    radio_m: float = 1.8
    margen_m: float = 2.0
    velocidad_relativa_ms: float = 5.0
    marcha_ms: float = 1.4
    balon_quieto_ms: float = 0.7
    balon_viaja_ms: float = 3.0
    peso_control: float = 6.0
    peso_tag: float = 0.8
    peso_fuera: float = 6.0
    peso_parada: float = 4.0

def matriz_control(frame):
    """Documentación pública en README.md."""
    arrays = [np.tanh(frame['margen_distancia'].to_numpy() / 3.0), np.tanh((frame['rel_A'].to_numpy() - frame['rel_H'].to_numpy()) / 5.0), frame['continuidad_H'].to_numpy() - frame['continuidad_A'].to_numpy(), np.tanh((frame['apoyo_H'].to_numpy() - frame['apoyo_A'].to_numpy()) / 3.0)]
    return np.nan_to_num(np.column_stack(arrays), nan=0.0, posinf=0.0, neginf=0.0)

def ajustar_control(x, y, regularizacion=0.01, iteraciones=1200):
    """Documentación pública en README.md."""
    x, y = (np.asarray(x, float), np.asarray(y, float))
    if x.ndim != 2 or x.shape[1] != 4 or y.shape != (len(x),) or (not len(x)):
        raise ValueError('Referencias externas inválidas')
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('El ajuste requiere referencias finitas de ambos equipos')
    weights = np.zeros(4)
    sample_weight = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    for _ in range(iteraciones):
        predicted = 1 / (1 + np.exp(-np.clip(x @ weights, -30, 30)))
        gradient = x.T @ ((predicted - y) * sample_weight) / len(y) + regularizacion * weights
        weights -= 0.25 * gradient
    return dict(version=1, variables=['margen', 'velocidad_relativa', 'continuidad', 'apoyo'], coeficientes=weights.tolist(), intercepto=0.0, regularizacion=regularizacion, iteraciones=iteraciones)

def guardar_modelo(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(model, handle, indent=2, allow_nan=False)

def cargar_modelo(path):
    with open(path, encoding='utf-8') as handle:
        model = json.load(handle)
    weights = np.asarray(model.get('coeficientes'), float)
    if model.get('version') != 1 or weights.shape != (4,) or (not np.isfinite(weights).all()):
        raise ValueError('Modelo de control incompatible')
    if model.get('procedencia', {}).get('fuente') != 'metrica-sports/sample-data':
        raise ValueError('Modelo sin procedencia externa permitida')
    return model

def potenciales(frame, config=None, model=None):
    config = config or ConfigControl()
    n = len(frame)
    logits = np.zeros((n, 3), float)
    control = np.zeros((n, 2), float)
    for k, side, other in ((0, 'H', 'A'), (1, 'A', 'H')):
        distance = frame['d_' + side].to_numpy()
        rival = frame['d_' + other].to_numpy()
        close = np.exp(-np.square(distance / config.radio_m))
        margin = 0.5 + 0.5 * np.tanh((rival - distance) / config.margen_m)
        margin = np.where(np.isfinite(rival), margin, 0.5)
        relative = frame['rel_' + side].to_numpy()
        compatible = np.where(np.isfinite(relative), 0.25 + 0.75 * np.exp(-relative / config.velocidad_relativa_ms), 0.5)
        continuity = 0.4 + 0.6 * frame['continuidad_' + side].to_numpy()
        control[:, k] = np.nan_to_num(close * margin * compatible * continuity)
    outside_mask = frame.fuera.to_numpy()
    control[outside_mask] = 0
    logits[:, :2] = config.peso_control * control
    logits[:, 0] -= control[:, 1]
    logits[:, 1] -= control[:, 0]
    if model is not None:
        odds = np.clip(matriz_control(frame) @ np.asarray(model['coeficientes']), -4, 4)
        gating = np.exp(-np.square(frame['d_min'].to_numpy() / (2 * config.radio_m)))
        gating = np.nan_to_num(gating)
        gating[outside_mask] = 0
        logits[:, 0] += gating * odds
        logits[:, 1] -= gating * odds
    for k in (0, 1):
        logits[:, k] += config.peso_tag * (frame.st_equipo.to_numpy() == k)
    vb, vc = (frame.v_balon.to_numpy(), frame.v_colectiva.to_numpy())
    still = np.where(np.isfinite(vb), np.exp(-np.square(vb / config.balon_quieto_ms)), 0.0)
    walking = np.where(np.isfinite(vc), np.clip(1 - vc / config.marcha_ms, 0, 1), 0.0)
    stationary = still * walking
    outside = frame.fuera.to_numpy().astype(float)
    travel = np.where(np.isfinite(vb), np.clip(vb / config.balon_viaja_ms, 0, 1), 0.0)
    logits[:, 2] = config.peso_fuera * outside + config.peso_parada * stationary + stationary * frame.st_saque.to_numpy() - 2 * travel - 2 * control.max(axis=1)
    support = frame.balon_util.to_numpy() | (frame.st_equipo.to_numpy() >= 0)
    logits[~support] = 0
    if not np.isfinite(logits).all():
        raise ValueError('Potenciales de control no finitos')
    return dict(logits=logits, control=control, soporte=support, config=asdict(config))
