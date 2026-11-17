"""Documentación pública en README.md."""
import numpy as np
from .temporal_tasas_fijas import _lse

def inferir_duracion(emission, hazards, destinations, initial=None):
    """Documentación pública en README.md."""
    e, h, dest = (np.asarray(emission, float), np.asarray(hazards, float), np.asarray(destinations, float))
    if e.ndim != 2 or not len(e) or (not np.isfinite(e).all()):
        raise ValueError('Emisiones HSMM inválidas')
    n, k = e.shape
    if h.ndim != 2 or h.shape[0] != k or h.shape[1] < 2 or (not np.isfinite(h).all()) or (not ((h > 0) & (h < 1)).all()):
        raise ValueError('Riesgos HSMM inválidos')
    ages = h.shape[1]
    if dest.shape != (k, k) or not np.isfinite(dest).all() or (dest < 0).any() or (not np.allclose(dest.sum(1), 1)) or np.diag(dest).any():
        raise ValueError('Destinos HSMM inválidos')
    if k * ages >= np.iinfo(np.int16).max:
        raise ValueError('Demasiados estados de edad')
    with np.errstate(divide='ignore'):
        ld = np.log(dest)
    leave, stay = (np.log(h), np.log1p(-h))
    init = np.zeros((k, ages))
    init[:, 0] = 1 / k
    if initial is not None:
        init = np.asarray(initial, float)
    if init.shape != h.shape or (init < 0).any() or (not np.isfinite(init).all()) or (not np.isclose(init.sum(), 1)):
        raise ValueError('Distribución inicial HSMM inválida')
    with np.errstate(divide='ignore'):
        delta = np.log(init) + e[0, :, None]
    alpha = np.empty((n, k, ages))
    alpha[0] = delta - _lse(delta)
    back = np.zeros((n, k, ages), np.int16)
    predecessors = np.arange(k * ages).reshape(k, ages)
    normal_back = predecessors[:, :-1]
    for t in range(1, n):
        end = delta + leave
        end_age = end.argmax(1)
        paths = end[np.arange(k), end_age][:, None] + ld
        source = paths.argmax(0)
        new = np.empty_like(delta)
        new[:, 0] = paths[source, np.arange(k)]
        back[t, :, 0] = source * ages + end_age[source]
        new[:, 1:] = delta[:, :-1] + stay[:, :-1]
        back[t, :, 1:] = normal_back
        tail = delta[:, -1] + stay[:, -1]
        tail_wins = tail > new[:, -1]
        new[tail_wins, -1] = tail[tail_wins]
        back[t, tail_wins, -1] = predecessors[tail_wins, -1]
        delta = new + e[t, :, None]
        delta -= delta.max()
        prev = alpha[t - 1]
        forward = np.empty_like(delta)
        forward[:, 0] = _lse(_lse(prev + leave, axis=1)[:, None] + ld, axis=0)
        forward[:, 1:] = prev[:, :-1] + stay[:, :-1]
        forward[:, -1] = np.logaddexp(forward[:, -1], prev[:, -1] + stay[:, -1])
        forward += e[t, :, None]
        alpha[t] = forward - _lse(forward)
    latent = np.empty(n, np.int16)
    latent[-1] = delta.argmax()
    for t in range(n - 2, -1, -1):
        latent[t] = back[t + 1].ravel()[latent[t + 1]]
    posterior = np.empty((n, k))
    beta = np.zeros((k, ages))
    next_age = np.minimum(np.arange(ages) + 1, ages - 1)
    for t in range(n - 1, -1, -1):
        combined = alpha[t] + beta
        posterior[t] = np.exp(_lse(combined, axis=1) - _lse(combined))
        if t:
            exiting = _lse(ld + (e[t] + beta[:, 0])[None, :], axis=1)
            beta = np.logaddexp(stay + e[t, :, None] + beta[:, next_age], leave + exiting[:, None])
            beta -= _lse(beta)
    return (latent // ages, posterior)

def ajustar_riesgos(samples):
    """Documentación pública en README.md."""
    edges = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf])
    exposure, exits = (np.zeros(6), np.zeros(6))
    for labels, groups, fps in samples:
        age = 0.0
        for i, side in enumerate(labels):
            if side not in 'HA' or i == 0 or groups[i] != groups[i - 1] or (side != labels[i - 1]):
                age = 0.0
            if side not in 'HA':
                continue
            band = np.searchsorted(edges[1:], age, side='right')
            exposure[band] += 1 / fps
            if i + 1 < len(labels) and groups[i] == groups[i + 1] and (labels[i + 1] in 'HA') and (labels[i + 1] != side):
                exits[band] += 1
            age += 1 / fps
    rate = np.clip((exits + 2.0) / (exposure + 20.0), 0.015, 2.0)
    return dict(bordes_s=edges[:-1].tolist(), transferencia_por_s=rate.tolist(), exposicion_s=exposure.tolist(), salidas_observadas=exits.tolist(), parada_por_s=0.1, reanudacion_por_s=0.25, cola_s=8.0, hz=5.0, censura='X y fronteras de periodo no se cuentan como cambios H/A')

def decodificar_duracion(logits, segments, support, context, fps, duration, temperature=1.0, threshold=0.55):
    if fps <= 0 or not np.isfinite(temperature) or temperature <= 0 or (not 1 / 3 <= threshold <= 1):
        raise ValueError('Configuración HSMM inválida')
    n = len(logits)
    labels, states = (np.full(n, 'X', dtype='<U1'), np.full(n, -1, np.int8))
    posterior = np.full((n, 3), np.nan)
    step = max(1, round(fps / duration['hz']))
    dt = step / fps
    ages = max(2, round(duration['cola_s'] / dt) + 1)
    age = np.arange(ages) * dt
    bands = np.searchsorted(duration['bordes_s'], age, side='right') - 1
    transfer = np.asarray(duration['transferencia_por_s'])[bands]
    total = transfer + duration['parada_por_s']
    hazards = np.vstack([1 - np.exp(-total * dt)] * 2 + [np.full(ages, 1 - np.exp(-duration['reanudacion_por_s'] * dt))])
    fraction = float(np.mean(transfer / total))
    destinations = np.array([[0, fraction, 1 - fraction], [fraction, 0, 1 - fraction], [0.5, 0.5, 0]])
    previous = 0
    for first, stop in segments:
        if not 0 <= first < stop <= n or first < previous:
            raise ValueError('Segmentos HSMM inválidos')
        previous = stop
        if not np.asarray(support)[first:stop].any():
            continue
        starts = np.arange(first, stop, step)
        ends = np.minimum(starts + step, stop)
        emissions = np.add.reduceat(np.asarray(logits)[first:stop], starts - first, axis=0) / (fps * 0.1 * temperature)
        seq, prob = inferir_duracion(emissions, hazards, destinations)
        for j, (a, b) in enumerate(zip(starts, ends)):
            states[a:b], posterior[a:b] = (seq[j], prob[j])
            valid = (prob[j, seq[j]] >= threshold) & ((np.asarray(context)[a:b] <= 3.0) | np.asarray(support)[a:b])
            labels[a:b][valid] = np.array(['H', 'A', 'D'])[seq[j]]
    return dict(etiquetas=labels, estados=states, posterior=posterior)
