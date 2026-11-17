"""Documentación pública en README.md."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos', required=True)
    parser.add_argument('--salida', required=True)
    parser.add_argument('--modelo', default=str(ROOT / 'modelos' / 'modelo.json'))
    parser.add_argument('--autorizar-datos-restringidos', action='store_true')
    args = parser.parse_args(argv)
    if not args.autorizar_datos_restringidos:
        parser.error('Se requiere autorización para el procesamiento local')
    output = Path(args.salida)
    log_path, manifest_path = (output.with_suffix('.log'), output.with_suffix('.run.json'))
    if any((path.exists() for path in [output, log_path, manifest_path])):
        parser.error('El destino o sus registros ya existen; elegir uno nuevo')
    output.parent.mkdir(parents=True, exist_ok=True)
    success = False
    with log_path.open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            from mtah import carga
            from mtah.ensamble import escribir_csv
            from mtah.validacion import validar_csv
            with Path(args.modelo).open(encoding='utf-8') as handle:
                raw_model = json.load(handle)
            version = raw_model.get('version')
            if version == 4:
                from mtah.candidata_v4 import cargar_v4, predecir_v4
                model = cargar_v4(args.modelo)
                predict = predecir_v4
            elif version == 3:
                from mtah.candidata_v3 import cargar_v3, predecir_v3
                model = cargar_v3(args.modelo)
                predict = predecir_v3
            elif version == 2:
                from mtah.mejora import cargar_mejora, predecir_mejora
                model = cargar_mejora(args.modelo)
                predict = predecir_mejora
            elif version == 1:
                from mtah.control import cargar_modelo
                from mtah.solucion import predecir
                model = cargar_modelo(args.modelo)
                predict = predecir
            else:
                raise ValueError('Versión de modelo no admitida')
            if model.get('procedencia', {}).get('gt_concurso_usado') is not False:
                raise ValueError('Modelo sin procedencia externa explícita')
            atd = carga.cargar_atd(data_dir=args.datos, usar_cache=False, autorizar_datos=True)
            events = carga.cargar_smart_tagging(data_dir=args.datos, autorizar_datos=True)
            result = predict(atd, events, model=model)
            escribir_csv(result['etiquetas'], output)
            validar_csv(output)
            manifest = dict(version_modelo=version, configuracion=result['configuracion'], gt_leido=False, entrenamiento_sobre_concurso=False, modelo_sha256=hashlib.sha256(Path(args.modelo).read_bytes()).hexdigest(), csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            with manifest_path.open('x', encoding='utf-8') as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
            success = True
        except Exception:
            traceback.print_exc(file=log)
    print('CSV generado y validado localmente' if success else 'FAIL: revisar el registro local; no se imprime contenido restringido')
    return 0 if success else 1
if __name__ == '__main__':
    raise SystemExit(main())
