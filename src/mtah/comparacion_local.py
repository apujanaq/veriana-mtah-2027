"""Documentación pública en README.md."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import ROOT
from .features import mascara_calificada
from .preprocesamiento import autorizar
from .revision_local import resumir_prediccion
from .validacion import pm, validar_csv

def comparar_etiquetas(old, new, mask=None):
    old, new = (np.asarray(old), np.asarray(new))
    if old.ndim != 1 or old.shape != new.shape:
        raise ValueError('Predicciones incompatibles')
    mask = mascara_calificada(len(old)) if mask is None else np.asarray(mask, bool)
    if mask.shape != old.shape:
        raise ValueError('Máscara incompatible')
    return dict(anterior=resumir_prediccion(old, mask), nueva=resumir_prediccion(new, mask), cambios_calificados=int(((old != new) & mask).sum()), matriz_cambios={a: {b: int(((old == a) & (new == b) & mask).sum()) for b in 'HADX'} for a in 'HADX'}, advertencia='Se comparan dos predicciones. Ninguna se considera verdad; no es una medida de acierto.')

def crear_comparacion(old_path, new_path, destination, *, autorizar_datos=False):
    autorizar(autorizar_datos)
    for path in [old_path, new_path]:
        if not Path(path).resolve().is_relative_to((ROOT / 'outputs').resolve()):
            raise ValueError('Solo se aceptan predicciones locales en outputs/, no archivos GT')
    destination = Path(destination).resolve()
    if not destination.is_relative_to((ROOT / 'outputs').resolve()) or destination.exists():
        raise ValueError('La comparación requiere un destino nuevo dentro de outputs/')
    validar_csv(old_path)
    validar_csv(new_path)
    old = pm.read_label_csv(old_path)['label'].to_numpy()
    new = pm.read_label_csv(new_path)['label'].to_numpy()
    result = comparar_etiquetas(old, new)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / 'comparacion.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ['# Comparación local v1 / v2', '', '**RESTRINGIDO: no subir este informe ni sus cifras a GitHub o servicios externos de IA.**', '', result['advertencia'], '', 'La selección de v2 se realizó exclusivamente con otros partidos públicos. No se escogió por estos conteos ni se consultaron los cinco minutos del concurso.', '', '| Diagnóstico de la predicción | v1 | v2 |', '|---|---:|---:|']
    for key, title in [('pct_X_calificado', 'Abstención X en región evaluada (%)'), ('posesiones_con_X_propagado', 'Posesiones tras propagación de X'), ('posesiones_predichas_menores_1s', 'Posesiones predichas menores de un segundo'), ('transiciones_predichas', 'Transiciones predichas')]:
        values = [result[which][key] for which in ['anterior', 'nueva']]
        lines.append('| {} | {} | {} |'.format(title, *values))
    lines += ['', 'Frames con etiqueta diferente en la región evaluada: {}.'.format(result['cambios_calificados']), '', 'Un cambio de etiqueta no es necesariamente una corrección. Menos abstenciones, menos posesiones o más D tampoco prueban mejora. El cambio se debe a la candidata congelada, no a correcciones manuales.', '', 'Los límites siguen siendo las referencias externas débiles, el cambio de dominio, los huecos largos y las ambigüedades de quietud frente a juego detenido. No se han medido macroF1 ni spellF1 del concurso.']
    with (destination / 'COMPARACION_LOCAL.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anterior', required=True)
    parser.add_argument('--nueva', required=True)
    parser.add_argument('--salida', required=True)
    parser.add_argument('--autorizar-datos-restringidos', action='store_true')
    args = parser.parse_args(argv)
    if not args.autorizar_datos_restringidos:
        parser.error('Se requiere autorización para los resultados locales')
    try:
        crear_comparacion(args.anterior, args.nueva, args.salida, autorizar_datos=True)
    except (ValueError, OSError):
        print('No se creó la comparación; revisar rutas y CSV localmente')
        return 1
    print('Comparación privada guardada; no se imprimen cifras ni etiquetas del partido')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
