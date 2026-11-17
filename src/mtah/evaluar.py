"""Documentación pública en README.md."""
import argparse
import sys
import time
import numpy as np
import pandas as pd
from . import ROOT, contrato, criterios, temporal
from .aproximaciones import REGISTRO
sys.path.insert(0, str(ROOT / 'evaluation'))
import possession_metrics as pm
UMBRAL_C3 = 0.85
UMBRAL_C2_MIRAR = 0.5
FACTORES_C3 = (0.8, 1.2)

def etiquetas_de(aprox, datos, P=None):
    """Documentación pública en README.md."""
    if hasattr(aprox, 'etiquetas'):
        return np.asarray(aprox.etiquetas(datos))
    P = aprox.probabilidades(datos) if P is None else P
    return temporal.decodificar(P, datos.calif)[0]

def concordancia(lab_a, lab_b):
    """Documentación pública en README.md."""
    return float(pm.evaluate(pm.label_codes(lab_a), pm.label_codes(lab_b))['spell_f1_mean'])

def c5(aprox, P, etiquetas):
    """Documentación pública en README.md."""
    motivo = contrato.validar_probabilidades(P)
    if motivo:
        return f'contrato: {motivo}'
    tabla = pd.DataFrame({'frame': np.arange(len(etiquetas)), 'label': etiquetas})
    motivo = pm.validate_submission(tabla, contrato.N_FRAMES)
    return f'formato: {motivo}' if motivo else None

def c3(aprox, datos, lab_base):
    """Documentación pública en README.md."""
    filas = []
    for k, v in aprox.params.items():
        for f in FACTORES_C3:
            nuevo = round(v * f) if isinstance(v, int) else v * f
            if nuevo == v:
                continue
            variante = type(aprox)(params={**aprox.params, k: nuevo})
            filas.append({'parametro': k, 'factor': f, 'valor': nuevo, 'spellF1_vs_base': round(concordancia(lab_base, etiquetas_de(variante, datos)), 3)})
    t = pd.DataFrame(filas)
    if t.empty:
        return t
    t['fragil'] = t['spellF1_vs_base'] < UMBRAL_C3
    return t

def evaluar(nombres=None, con_c3=True, datos=None, autorizar_datos=False):
    """Documentación pública en README.md."""
    nombres = nombres or list(REGISTRO)
    datos = datos or contrato.cargar_datos(autorizar_datos=autorizar_datos)
    fichas, etiquetas = ({}, {})
    for n in nombres:
        t0 = time.time()
        a = REGISTRO[n]()
        P = a.probabilidades(datos)
        lab = etiquetas_de(a, datos, P)
        etiquetas[n] = lab
        fichas[n] = {'c5': c5(a, P, lab) or 'OK', 'c1': criterios.evaluar_c1(lab), 'segundos': round(time.time() - t0, 1), 'aprox': a}
    c2 = pd.DataFrame(index=nombres, columns=nombres, dtype=float)
    for i in nombres:
        for j in nombres:
            c2.loc[i, j] = 1.0 if i == j else concordancia(etiquetas[i], etiquetas[j])
    c3s = {n: c3(fichas[n]['aprox'], datos, etiquetas[n]) for n in nombres if con_c3 and fichas[n]['aprox'].params}
    return {'fichas': fichas, 'c2': c2.round(3), 'c3': c3s, 'etiquetas': etiquetas}

def imprimir(res):
    for n, f in res['fichas'].items():
        print(f"\n=== {n}  ({f['segundos']} s) ===\nC5 sanidad: {f['c5']}")
        print(f['c1'][['valor', 'rango', 'pasa', 'confianza']].to_string())
    print('\n=== C2 concordancia (spellF1 entre pares; <0,5 = mirar ahí) ===')
    print(res['c2'].to_string())
    for n, t in res['c3'].items():
        frag = t[t['fragil']]
        print(f"\n=== C3 robustez: {n}  (mínimo {t['spellF1_vs_base'].min():.3f}; {('ROBUSTA' if frag.empty else f'{frag.parametro.nunique()} parámetros FRÁGILES')}) ===")
        print(t.sort_values('spellF1_vs_base').to_string(index=False))
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('nombres', nargs='*')
    ap.add_argument('--sin-c3', action='store_true')
    ap.add_argument('--autorizar-datos-restringidos', action='store_true', help='confirma que esta ejecución lee los datos del concurso (T&C, AGENTS.md)')
    args = ap.parse_args()
    imprimir(evaluar(args.nombres or None, con_c3=not args.sin_c3, autorizar_datos=args.autorizar_datos_restringidos))
