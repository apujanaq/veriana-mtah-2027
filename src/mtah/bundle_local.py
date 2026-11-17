"""Documentación pública en README.md."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import gc
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time
import traceback
import zipfile
from . import ROOT, carga
from .control import cargar_modelo
from .eda import correr as correr_eda, escribir_reporte
from .ensamble import escribir_csv
from .preprocesamiento import autorizar
from .solucion import predecir
from .validacion import validar_csv

def _dentro(path, limite):
    path, limite = (Path(path).resolve(), Path(limite).resolve())
    if path == limite or not path.is_relative_to(limite):
        raise ValueError('El destino debe quedar dentro de la carpeta autorizada')
    return path

def planificar_zip(path):
    """Documentación pública en README.md."""
    expected = {Path(value).name.casefold(): (source, key, Path(value).name) for source in ('tactical', 'broadcast') for key, value in carga.games.asset_paths(source=source).items()}
    selected = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            normalized = entry.filename.replace('\\', '/')
            relative = PurePosixPath(normalized)
            if relative.is_absolute() or '..' in relative.parts or ':' in normalized or ('\x00' in normalized):
                raise ValueError('ZIP con una ruta no permitida')
            if entry.is_dir():
                continue
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('ZIP con un enlace simbólico no permitido')
            name = relative.name.casefold()
            if name not in expected:
                continue
            if name in selected:
                raise ValueError('ZIP con assets duplicados o ambiguos')
            if entry.flag_bits & 1:
                raise ValueError('ZIP cifrado; requiere preparación local adicional')
            source, key, filename = expected[name]
            selected[name] = dict(miembro=entry.filename, source=source, asset=key, destino=(Path(source) / filename).as_posix(), bytes=entry.file_size, crc=entry.CRC)
    if selected.keys() != expected.keys():
        raise ValueError('El ZIP no contiene exactamente los assets automáticos esperados')
    return sorted(selected.values(), key=lambda e: (e['asset'] == 'video', e['source'], e['asset']))

def extraer_zip(path, destination, limite=None, progreso=None):
    destination = _dentro(destination, limite or ROOT / 'data')
    if destination.exists():
        raise FileExistsError('La extracción requiere una carpeta nueva')
    selected = planificar_zip(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    needed = sum((entry['bytes'] for entry in selected))
    if shutil.disk_usage(destination.parent).free < needed + 1024 ** 3:
        raise OSError('Espacio libre insuficiente para la extracción')
    destination.mkdir(exist_ok=False)
    records = []
    last_notice = time.monotonic()
    with zipfile.ZipFile(path) as archive:
        for entry in selected:
            target = _dentro(destination / entry['destino'], destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with archive.open(entry['miembro']) as original, target.open('xb') as output:
                while True:
                    chunk = original.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    if progreso and time.monotonic() - last_notice >= 20:
                        progreso('La extracción local continúa')
                        last_notice = time.monotonic()
            if count != entry['bytes']:
                raise ValueError('Extracción incompleta')
            records.append(dict(entry, sha256=digest.hexdigest()))
    with (destination / 'extraccion_local.json').open('x', encoding='utf-8') as handle:
        json.dump(dict(original=str(Path(path).resolve()), assets=records), handle, indent=2)
    return destination

def _sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _guardar(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)

def ejecutar(data_dir, output, model_path, *, zip_path=None, autorizar_datos=False, reproducir=True, announce=None):
    autorizar(autorizar_datos)
    data_dir = _dentro(data_dir, ROOT / 'data')
    output = _dentro(output, ROOT / 'outputs')
    if output.exists() or (zip_path is not None and data_dir.exists()):
        raise FileExistsError('Los destinos de esta ejecución deben ser nuevos')
    model = cargar_modelo(model_path)
    if not model.get('procedencia', {}).get('gt_concurso_usado') is False:
        raise ValueError('Modelo sin declaración de procedencia externa')
    output.mkdir(parents=True, exist_ok=False)
    announce = announce or (lambda message: None)
    stage = 'preparacion'
    record = dict(version=1, acceso='solo_local', gt_leido=False, entrenamiento_sobre_concurso=False, modelo_sha256=_sha(model_path), codigo_sha256={p.name: _sha(p) for p in Path(__file__).parent.glob('*.py')})
    _guardar(output / 'modelo_congelado.json', model)
    try:
        with (output / 'ejecucion_local.log').open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            try:
                if zip_path is not None:
                    stage = 'extraccion'
                    extraer_zip(zip_path, data_dir, progreso=announce)
                    announce('Extracción verificada; original conservado')
                elif not data_dir.is_dir():
                    raise FileNotFoundError('No existe el bundle local')
                record['bundle'] = str(data_dir)
                stage = 'eda'
                report = correr_eda(str(data_dir), autorizar_datos=True)
                for source in report['fuentes'].values():
                    if not all(source['estado_archivos'].values()):
                        raise ValueError('Bundle incompleto')
                escribir_reporte(report, output / 'eda')
                del report
                gc.collect()
                announce('EDA local completado')
                stage = 'prediccion'
                atd = carga.cargar_atd(data_dir=str(data_dir), usar_cache=False, autorizar_datos=True)
                events = carga.cargar_smart_tagging(data_dir=str(data_dir), autorizar_datos=True)
                result = predecir(atd, events, model=model)
                csv = output / 'equipo.csv'
                escribir_csv(result['etiquetas'], csv)
                record['configuracion'] = result['configuracion']
                record['csv_sha256'] = _sha(csv)
                labels = result['etiquetas']
                diagnostic = dict(conteos={label: int((labels == label).sum()) for label in 'HADX'}, concentracion='Marginales heurísticos; no probabilidades calibradas de acierto')
                _guardar(output / 'diagnostico_prediccion_local.json', diagnostic)
                del result
                gc.collect()
                announce('Primera predicción generada y CSV validado')
                if reproducir:
                    stage = 'reproduccion'
                    repeated = predecir(atd, events, model=model)
                    second = output / 'equipo_reproduccion.csv'
                    escribir_csv(repeated['etiquetas'], second)
                    record['reproduccion_identica'] = _sha(second) == record['csv_sha256']
                    if not record['reproduccion_identica']:
                        raise ValueError('La reproducción no coincide byte a byte')
                    del repeated
                    gc.collect()
                    announce('Reproducción idéntica verificada')
                validar_csv(csv)
                record['csv_valido'] = True
                record['completado'] = True
            except Exception:
                traceback.print_exc(file=log)
                raise
    except Exception as error:
        record.update(completado=False, etapa_fallida=stage, tipo_error=type(error).__name__)
        _guardar(output / 'estado_ejecucion.json', record)
        own_frames = [f for f in traceback.extract_tb(error.__traceback__) if Path(f.filename).parent == Path(__file__).parent]
        location = '{}:{}'.format(Path(own_frames[-1].filename).name, own_frames[-1].lineno) if own_frames else 'lector_local'
        announce('Fallo técnico en {} ({}, {}); detalle conservado solo localmente'.format(stage, type(error).__name__, location))
        return False
    _guardar(output / 'estado_ejecucion.json', record)
    announce('Proceso local finalizado; no se ha enviado contenido del partido a servicios externos')
    return True

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', dest='zip_path')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--salida', required=True)
    parser.add_argument('--modelo-control', default=str(ROOT / 'outputs' / 'desarrollo_externo_v1_verificado' / 'control.json'))
    parser.add_argument('--autorizar-datos-restringidos', action='store_true')
    args = parser.parse_args(argv)
    if not args.autorizar_datos_restringidos:
        parser.error('Se requiere autorización explícita para el procesamiento local')
    console = sys.stdout
    try:
        success = ejecutar(args.data_dir, args.salida, args.modelo_control, zip_path=args.zip_path, autorizar_datos=True, announce=lambda message: print(message, file=console, flush=True))
    except Exception as error:
        print('No se inició el procesamiento: comprobar destinos y modelo local ({})'.format(type(error).__name__))
        return 1
    return 0 if success else 1
if __name__ == '__main__':
    raise SystemExit(main())
