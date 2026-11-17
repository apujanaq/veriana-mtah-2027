"""Documentación pública en README.md."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from . import ROOT, carga
from .control import ConfigControl, cargar_modelo, potenciales
from .ensamble import escribir_csv
from .features import mascara_calificada
from .features_control import ConfigFeatures, bloques, extraer
from .temporal_tasas_fijas import ConfigTemporal, decodificar

def predecir(atd, eventos=(), calif=None, grupos=None, config_features=None, config_control=None, config_temporal=None, model=None):
    cf, cc, ct = (config_features or ConfigFeatures(), config_control or ConfigControl(), config_temporal or ConfigTemporal())
    if cf.fps != ct.fps:
        raise ValueError('Las etapas deben compartir fps')
    n = len(atd['balon'])
    calif = mascara_calificada(n) if calif is None else np.asarray(calif, bool)
    frame = extraer(atd, eventos, calif, grupos, cf)
    evidence = potenciales(frame, cc, model)
    decoded = decodificar(evidence['logits'], bloques(calif, grupos), evidence['soporte'], frame.contexto_balon_s.to_numpy(), ct)
    return dict(**decoded, feat=frame, evidencia=evidence, configuracion=dict(features=asdict(cf), control=asdict(cc), temporal=asdict(ct)), modelo_control=model)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir')
    parser.add_argument('--salida', default=str(ROOT / 'outputs' / 'primer_abordaje' / 'equipo.csv'))
    parser.add_argument('--modelo-control', help='JSON ajustado exclusivamente con los samples públicos')
    parser.add_argument('--autorizar-datos-restringidos', action='store_true')
    parser.add_argument('--sin-cache', action='store_true')
    args = parser.parse_args(argv)
    if not args.autorizar_datos_restringidos:
        parser.error('Se requiere autorización explícita para acceder al bundle local')
    output = Path(args.salida)
    manifest = output.with_suffix('.run.json')
    if output.exists() or manifest.exists():
        parser.error('El destino ya existe; elegir uno nuevo')
    try:
        atd = carga.cargar_atd(data_dir=args.data_dir, usar_cache=not args.sin_cache, autorizar_datos=True)
        events = carga.cargar_smart_tagging(data_dir=args.data_dir, autorizar_datos=True)
        model = cargar_modelo(args.modelo_control) if args.modelo_control else None
        result = predecir(atd, events, model=model)
        escribir_csv(result['etiquetas'], output)
        with manifest.open('x', encoding='utf-8') as handle:
            json.dump(dict(configuracion=result['configuracion'], modelo_control=model, gt_leido=False, entrenamiento_sobre_concurso=False), handle, indent=2, allow_nan=False)
    except (ValueError, OSError):
        print('FAIL: revisar el bundle o el destino localmente; ningún contenido se imprime')
        return 1
    print('Primer abordaje generado y CSV validado localmente')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
