"""Documentación pública en README.md."""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def ajustar_stumps(x, y, sample_weight=None, n_trees=100, learning_rate=0.12, bins=24, regularizacion=5.0):
    x, y = (np.asarray(x, float), np.asarray(y, float))
    if x.ndim != 2 or y.shape != (len(x),) or (not len(y)) or (not np.isfinite(x).all()) or (not np.isin(y, [0, 1]).all()) or (len(np.unique(y)) != 2):
        raise ValueError('Datos de boosting inválidos')
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float).copy()
    if w.shape != y.shape or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Pesos de boosting inválidos')
    w *= len(y) / w.sum()
    prior = float(np.average(y, weights=w))
    intercept = float(np.log(prior / (1 - prior)))
    score = np.full(len(x), intercept)
    thresholds = [np.unique(np.quantile(x[:, j], np.linspace(0, 1, bins + 1)[1:-1])) for j in range(x.shape[1])]
    membership = [np.searchsorted(t, x[:, j], side='left') for j, t in enumerate(thresholds)]
    trees = []
    for _ in range(n_trees):
        p = sigmoid(score)
        gradient, hessian = (w * (y - p), w * p * (1 - p))
        total_g, total_h = (gradient.sum(), hessian.sum())
        best = None
        for j, (t, groups) in enumerate(zip(thresholds, membership)):
            if not len(t):
                continue
            g = np.cumsum(np.bincount(groups, weights=gradient, minlength=len(t) + 1))[:-1]
            h = np.cumsum(np.bincount(groups, weights=hessian, minlength=len(t) + 1))[:-1]
            count = np.cumsum(np.bincount(groups, minlength=len(t) + 1))[:-1]
            gain = g ** 2 / (h + regularizacion) + (total_g - g) ** 2 / (total_h - h + regularizacion)
            gain[(count < 20) | (len(x) - count < 20)] = -np.inf
            k = int(gain.argmax())
            if np.isfinite(gain[k]) and (best is None or gain[k] > best[0]):
                left = np.clip(g[k] / (h[k] + regularizacion), -2, 2) * learning_rate
                right = np.clip((total_g - g[k]) / (total_h - h[k] + regularizacion), -2, 2) * learning_rate
                best = (gain[k], dict(feature=j, threshold=float(t[k]), left=float(left), right=float(right)))
        if best is None:
            break
        tree = best[1]
        trees.append(tree)
        score += np.where(x[:, tree['feature']] <= tree['threshold'], tree['left'], tree['right'])
    return dict(intercepto=intercept, arboles=trees, n_variables=x.shape[1], referencias=len(y), prevalencia_D=prior, learning_rate=learning_rate, bins=bins)

def log_odds_stumps(x, model):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != model['n_variables'] or (not np.isfinite(x).all()):
        raise ValueError('Variables de boosting incompatibles')
    out = np.full(len(x), model['intercepto'], float)
    for tree in model['arboles']:
        out += np.where(x[:, tree['feature']] <= tree['threshold'], tree['left'], tree['right'])
    return out
