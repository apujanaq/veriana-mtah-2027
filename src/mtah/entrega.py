"""Documentación pública en README.md."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
import unicodedata
import zipfile
from . import ROOT
from .control import cargar_modelo
from .externos import COMMIT
from .mejora import cargar_mejora
from .candidata_v3 import cargar_v3
from .candidata_v4 import cargar_v4
MODULES = ['__init__', 'baseline', 'bundle_local', 'carga', 'control', 'eda', 'ensamble', 'entrega', 'etapa_a', 'experimento_v2', 'externos', 'features', 'features_control', 'fuentes', 'mejora', 'metricas_demo', 'parametros', 'pipeline', 'preprocesamiento', 'referencias_externas', 'revision_local', 'solucion', 'temporal_tasas_fijas', 'validacion', 'candidata_v3', 'duracion', 'skillcorner', 'experimento_v3', 'boosting', 'idsse', 'reanudaciones', 'candidata_v4', 'experimento_v4', 'geometria_visual']
TESTS = ['test_preprocesamiento', 'test_csv', 'test_primer_abordaje', 'test_mejora', 'test_bundle_local', 'test_revision_local', 'test_entrega', 'test_v3', 'test_v4']

def slug_equipo(name):
    if not name.strip() or len(name) > 100 or any((c in name for c in '\r\n\x00')):
        raise ValueError('Nombre de equipo inválido')
    text = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    text = re.sub('[^a-z0-9_-]+', '_', text).strip('_-')[:80]
    if not text or text in ['con', 'prn', 'aux', 'nul'] or re.fullmatch('(com|lpt)[1-9]', text):
        raise ValueError('Nombre de archivo reservado o vacío')
    return text

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _readme(team, slug):
    return f'# {team} — código de reproducción\n\nEste directorio contiene código y modelo externo; no incluye datos del partido ni resultados.\n\nCon Python 3.12:\n\n```powershell\npython -m venv .venv\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements-pipeline.txt\n.\\.venv\\Scripts\\python.exe reproducir_entrega.py --datos "D:\\bundle-local" --salida "{slug}.csv" --autorizar-datos-restringidos\n.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v\n```\n\nEn macOS/Linux usar `.venv/bin/python`. Conservar los nombres oficiales bajo `tactical/`\ny `broadcast/`. No ajustar parámetros usando los cinco minutos oficiales. No compartir datos,\ncachés, CSV, logs ni reportes del partido con GitHub o servicios de IA. La inferencia es local.\nConsultar `{slug}.md` para método, entrenamiento externo, reglas y límites.\n'

def preparar_codigo(destination, model_path, team, repo=ROOT):
    """Documentación pública en README.md."""
    destination, repo = (Path(destination), Path(repo))
    slug = slug_equipo(team)
    if destination.exists():
        raise FileExistsError('El ZIP debe ser nuevo')
    raw_model = json.loads(Path(model_path).read_text(encoding='utf-8'))
    version = raw_model.get('version')
    model = cargar_v4(model_path) if version == 4 else cargar_v3(model_path) if version == 3 else cargar_mejora(model_path) if version == 2 else cargar_modelo(model_path)
    origin = model.get('procedencia', {})
    if version not in [3, 4] and (origin.get('commit') != COMMIT or origin.get('partido') != 'Sample_Game_1' or origin.get('gt_concurso_usado') is not False):
        raise ValueError('No se empaqueta un modelo sin procedencia pública fijada')
    approved = ['LICENSE', 'THIRD_PARTY_NOTICES.md', 'CITATION.cff', 'games.py', 'requirements-pipeline.txt', 'reproducir_entrega.py', 'evaluation/possession_metrics.py', 'evaluation/validate_submission.py', 'scripts/descargar_metrica.py', 'scripts/descargar_skillcorner.py', 'config/skillcorner_v3.json', 'docs/ENTREGA_TECNICA.md', 'docs/ENTREGA_TECNICA_V3.md', 'docs/02_METRICAS.md', 'docs/07_MULTIFUENTE_Y_REVISION.md', 'scripts/informe_equipo.py', 'config/idsse_v4.json', 'scripts/descargar_idsse.py', 'scripts/validar_nativa_v4.py', 'scripts/informe_equipo_v4.py', 'docs/ENTREGA_TECNICA_V4.md']
    approved += ['src/mtah/' + name + '.py' for name in MODULES]
    approved += ['tests/' + name + '.py' for name in TESTS]
    payload = {name: (repo / name).read_bytes() for name in approved}
    technical_path = repo / ('docs/ENTREGA_TECNICA_V4.md' if version == 4 else 'docs/ENTREGA_TECNICA_V3.md' if version == 3 else 'docs/ENTREGA_TECNICA.md')
    technical = technical_path.read_text(encoding='utf-8').replace('{{EQUIPO}}', team).replace('{{SLUG}}', slug)
    payload[slug + '.md'] = technical.encode('utf-8')
    payload['README.md'] = _readme(team, slug).encode('utf-8')
    payload['modelos/modelo.json'] = Path(model_path).read_bytes()
    payload['.gitignore'] = b'data/\noutputs/\n*.csv\n*.log\n*.run.json\n*.npz\n*.mp4\n.venv/\n__pycache__/\n*.pyc\n'
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(payload.items())}
    payload['MANIFIESTO_PUBLICO.json'] = json.dumps(dict(archivos_sha256=manifest, datos_restringidos_incluidos=False, gt_incluido=False), indent=2).encode('utf-8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(payload.items()):
            if name.endswith(('.csv', '.mp4', '.ipynb', '.npz', '.pkl')) or name.startswith(('data/', 'outputs/', '.git/')):
                raise ValueError('Archivo no permitido en código publicable')
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.external_attr = 33188 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)
    return technical

def extraer_codigo(zip_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(zip_path) as archive:
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination) or entry.is_dir():
                raise ValueError('Estructura de código inválida')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(archive.read(entry))
    return destination

def ejecutar_entrega(data_dir, output, model_path, team='Equipo MTAH', *, autorizar_datos=False, provisional=True):
    from .preprocesamiento import autorizar
    autorizar(autorizar_datos)
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / 'outputs').resolve()) or output.exists():
        raise ValueError('Se requiere un destino nuevo bajo outputs/')
    slug = slug_equipo(team)
    output.mkdir(parents=True, exist_ok=False)
    console = sys.stdout

    def announce(message):
        print(message, file=console, flush=True)
    stage = 'preparacion_codigo'
    record = dict(nombre_equipo=team, nombre_provisional=provisional, gt_leido=False, entrenamiento_sobre_concurso=False, enviado=False, publicado=False)
    with (output / 'preparacion_local.log').open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            public_zip = output / 'codigo_publicable.zip'
            technical = preparar_codigo(public_zip, model_path, team)
            (output / (slug + '.md')).write_text(technical, encoding='utf-8')
            clean = extraer_codigo(public_zip, output / 'verificacion_limpia')
            announce('Código publicable separado de los datos y extraído para verificación')
            import os
            environment = dict(os.environ)
            environment.pop('PYTHONPATH', None)
            environment.pop('MTAH_DATA_DIR', None)
            csv = output / (slug + '.csv')
            stage = 'prediccion_trabajo'
            first = subprocess.run([sys.executable, str(ROOT / 'reproducir_entrega.py'), '--datos', str(Path(data_dir).resolve()), '--salida', str(csv), '--modelo', str(Path(model_path).resolve()), '--autorizar-datos-restringidos'], cwd=ROOT, env=environment, stdout=log, stderr=log)
            if first.returncode:
                raise RuntimeError('Fallo en ejecución de trabajo')
            announce('Nueva candidata local generada y validada')
            stage = 'reproduccion_desde_codigo_publico'
            repeated = output / (slug + '_reproduccion.csv')
            second = subprocess.run([sys.executable, str(clean / 'reproducir_entrega.py'), '--datos', str(Path(data_dir).resolve()), '--salida', str(repeated), '--autorizar-datos-restringidos'], cwd=clean, env=environment, stdout=log, stderr=log)
            if second.returncode or _sha(csv) != _sha(repeated):
                raise RuntimeError('El código público no reprodujo el mismo CSV')
            stage = 'validador_oficial'
            validation = subprocess.run([sys.executable, str(clean / 'evaluation/validate_submission.py'), str(csv)], cwd=clean, env=environment, stdout=log, stderr=log)
            if validation.returncode:
                raise RuntimeError('CSV rechazado por validador oficial')
            record.update(completado=True, reproduccion_limpia_identica=True, validador_oficial_ok=True, csv_sha256=_sha(csv), codigo_zip_sha256=_sha(public_zip), modelo_sha256=_sha(model_path), csv=csv.name, documento=slug + '.md', codigo='codigo_publicable.zip')
            announce('Reproducción limpia idéntica y validador oficial correctos')
        except Exception as error:
            traceback.print_exc(file=log)
            record.update(completado=False, etapa_fallida=stage, tipo_error=type(error).__name__)
            announce('Fallo técnico en {}; detalle conservado solo en el registro local'.format(stage))
    with (output / 'MANIFIESTO_ENTREGA_PRIVADO.json').open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return bool(record.get('completado'))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--salida', required=True)
    parser.add_argument('--modelo', required=True)
    parser.add_argument('--equipo', default='Equipo MTAH')
    parser.add_argument('--nombre-confirmado', action='store_true')
    parser.add_argument('--autorizar-datos-restringidos', action='store_true')
    args = parser.parse_args(argv)
    if not args.autorizar_datos_restringidos:
        parser.error('Se requiere autorización de procesamiento local')
    try:
        success = ejecutar_entrega(args.data_dir, args.salida, args.modelo, args.equipo, autorizar_datos=True, provisional=not args.nombre_confirmado)
    except Exception as error:
        print('No se inició la entrega: revisar destinos y modelo ({})'.format(type(error).__name__))
        return 1
    return 0 if success else 1
if __name__ == '__main__':
    raise SystemExit(main())
