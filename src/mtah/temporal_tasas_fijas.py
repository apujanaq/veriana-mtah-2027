"""Documentación pública en README.md."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ConfigTemporal:
    fps: float = 25.0
    transferencia_por_s: float = 0.1
    parada_por_s: float = 0.1
    reanudacion_por_s: float = 0.25
    independencia_s: float = 0.1
    concentracion_min: float = 0.55
    contexto_max_s: float = 3.0

def _lse(x, axis=None):
    maximum = np.max(x, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(maximum), maximum, 0.0)
    with np.errstate(divide='ignore'):
        value = safe + np.log(np.sum(np.exp(x - safe), axis=axis, keepdims=True))
    return np.squeeze(value, axis=axis)

def transicion(config=None):
    c = config or ConfigTemporal()
    if c.fps <= 0 or min(c.transferencia_por_s, c.parada_por_s, c.reanudacion_por_s) <= 0:
        raise ValueError('Tasas temporales inválidas')
    rates = np.array([[0, c.transferencia_por_s, c.parada_por_s], [c.transferencia_por_s, 0, c.parada_por_s], [c.reanudacion_por_s / 2, c.reanudacion_por_s / 2, 0]])
    total = rates.sum(axis=1)
    leave = 1 - np.exp(-total / c.fps)
    matrix = rates / total[:, None] * leave[:, None]
    np.fill_diagonal(matrix, 1 - leave)
    return matrix

def inferir(log_potenciales, matriz):
    """Documentación pública en README.md."""
    emission, matrix = (np.asarray(log_potenciales, float), np.asarray(matriz, float))
    if emission.ndim != 2 or emission.shape[1] != 3 or (not len(emission)) or (not np.isfinite(emission).all()):
        raise ValueError('Potenciales temporales inválidos')
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or (matrix < 0).any() or (not np.allclose(matrix.sum(1), 1)):
        raise ValueError('Matriz de transición inválida')
    with np.errstate(divide='ignore'):
        log_a = np.log(matrix)
    n = len(emission)
    initial = np.full(3, -np.log(3.0))
    delta = initial + emission[0]
    back = np.zeros((n, 3), dtype=np.int8)
    alpha = np.empty((n, 3))
    alpha[0] = delta - _lse(delta)
    for frame in range(1, n):
        paths = delta[:, None] + log_a
        back[frame] = np.argmax(paths, axis=0)
        delta = np.max(paths, axis=0) + emission[frame]
        delta -= delta.max()
        forward = _lse(alpha[frame - 1, :, None] + log_a, axis=0) + emission[frame]
        alpha[frame] = forward - _lse(forward)
    states = np.empty(n, np.int8)
    states[-1] = np.argmax(delta)
    for frame in range(n - 2, -1, -1):
        states[frame] = back[frame + 1, states[frame + 1]]
    posterior = np.empty_like(alpha)
    beta = np.zeros(3)
    for frame in range(n - 1, -1, -1):
        combined = alpha[frame] + beta
        posterior[frame] = np.exp(combined - _lse(combined))
        if frame:
            beta = _lse(log_a + emission[frame][None, :] + beta[None, :], axis=1)
            beta -= _lse(beta)
    return (states, posterior)

def decodificar(logits, segmentos, soporte, contexto_s, config=None):
    c = config or ConfigTemporal()
    if c.independencia_s <= 0 or not 1 / 3 <= c.concentracion_min <= 1 or c.contexto_max_s < 0:
        raise ValueError('Configuración temporal inválida')
    logits = np.asarray(logits, float)
    n = len(logits)
    labels = np.full(n, 'X', dtype='<U1')
    posterior = np.full((n, 3), np.nan)
    states = np.full(n, -1, dtype=np.int8)
    matrix = transicion(c)
    previous = 0
    for first, stop in segmentos:
        if not 0 <= first < stop <= n or first < previous:
            raise ValueError('Segmentos temporales inválidos')
        previous = stop
        if not np.asarray(soporte)[first:stop].any():
            continue
        sequence, prob = inferir(logits[first:stop] / (c.fps * c.independencia_s), matrix)
        states[first:stop], posterior[first:stop] = (sequence, prob)
        concentrated = prob[np.arange(stop - first), sequence] >= c.concentracion_min
        context = (np.asarray(contexto_s)[first:stop] <= c.contexto_max_s) | np.asarray(soporte)[first:stop]
        valid = concentrated & context
        labels[first:stop][valid] = np.array(['H', 'A', 'D'])[sequence[valid]]
    return dict(etiquetas=labels, estados=states, posterior=posterior, transicion=matrix)
