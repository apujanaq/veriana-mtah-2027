"""Documentación pública en README.md."""
import argparse
from . import ROOT, carga
from .baseline import predecir
from .ensamble import escribir_csv

def correr(usar_cache=True, data_dir=None, *, autorizar_datos=False):
    atd = carga.cargar_atd(usar_cache=usar_cache, data_dir=data_dir, autorizar_datos=autorizar_datos)
    eventos = carga.cargar_smart_tagging(data_dir=data_dir, autorizar_datos=autorizar_datos)
    return predecir(atd, eventos)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--salida', default=str(ROOT / 'outputs' / 'equipo.csv'))
    parser.add_argument('--data-dir', help='Bundle local; también admite MTAH_DATA_DIR')
    parser.add_argument('--sin-cache', action='store_true')
    parser.add_argument('--autorizar-datos-restringidos', action='store_true', help='Confirmación explícita del usuario para acceder al bundle local')
    args = parser.parse_args(argv)
    if not args.autorizar_datos_restringidos:
        parser.error('Se requiere autorización antes de acceder a los datos restringidos')
    from pathlib import Path
    if Path(args.salida).exists():
        parser.error('El destino ya existe; elegir un archivo nuevo')
    try:
        result = correr(usar_cache=not args.sin_cache, data_dir=args.data_dir, autorizar_datos=True)
        escribir_csv(result['etiquetas'], args.salida)
    except (ValueError, OSError):
        print('FAIL: revisar localmente los archivos de entrada y el destino')
        return 1
    print('CSV generado y validado localmente; ningún dato fue enviado a servicios externos')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
