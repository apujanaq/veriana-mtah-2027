"""Documentación pública en README.md."""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from mtah.entrega import MODULES
BASE_OFICIAL = 'f8f2629'
EXTRA_MODULOS = ['contrato', 'criterios', 'evaluar', 'eventos', 'forma', 'toque', 'etapa_b', 'temporal', 'microposesiones', 'toques_supervisados', 'experimento_micro', 'pares_micro', 'validacion_externa', 'comparacion_local', 'control_multiventana', 'firmas']
SCRIPTS = ['descargar_metrica', 'descargar_skillcorner', 'descargar_idsse', 'informe_equipo', 'informe_equipo_v4', 'validar_nativa_v4', 'entrenar_toques_micro', 'validar_microposesiones', 'calibrar_pares_micro', 'preparar_entrega_equipo', 'exportar_rama_publicable', 'entrenar_control_multiventana', 'calibrar_control_temporal', 'robustez_refinamiento', 'generar_entrega']
DOCS = ['THIRD_PARTY_NOTICES.md', 'requirements-pipeline.txt', 'docs/ENTREGA_TECNICA.md', 'docs/ENTREGA_TECNICA_V3.md', 'docs/ENTREGA_TECNICA_V4.md', 'docs/02_METRICAS.md', 'docs/07_MULTIFUENTE_Y_REVISION.md', 'docs/13_MICROPOSESIONES_JULIANA.md', 'docs/14_CIERRE_Y_ENTREGA_EQUIPO.md', 'docs/veriana.md', 'docs/entrega/Veriana_entrega.md']

class QuitarTextoHeredado(ast.NodeTransformer):

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return ast.copy_location(ast.Expr(value=ast.Constant('Documentación pública en README.md.')), node)
        return self.generic_visit(node)

    def generic_visit(self, node):
        result = super().generic_visit(node)
        if isinstance(result, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (not result.body):
            result.body = [ast.Pass()]
        return result

class NormalizarOrigen(ast.NodeTransformer):

    def visit_keyword(self, node):
        if node.arg == 'origen':
            node.value = ast.Constant('Parámetro global del método; documentación pública en README.md')
        return self.generic_visit(node)

def sanear_python(texto, metadatos_parametros=False):
    tree = ast.parse(texto)
    tree = QuitarTextoHeredado().visit(tree)
    if metadatos_parametros:
        tree = NormalizarOrigen().visit(tree)
    tree = ast.fix_missing_locations(tree)
    out = ast.unparse(tree) + '\n'
    if ast.dump(ast.parse(out), include_attributes=False) != ast.dump(tree, include_attributes=False):
        raise ValueError('La exportación cambió el AST del código')
    return out

def canon_sha(content):
    return hashlib.sha256(content.decode('utf-8-sig').replace('\r\n', '\n').encode('utf-8')).hexdigest()

def exportar(source, dest):
    source, dest = (Path(source), Path(dest))
    modules = set(MODULES + EXTRA_MODULOS)
    files = [f'src/mtah/{m}.py' for m in sorted(modules)]
    files += [p.relative_to(source).as_posix() for p in sorted((source / 'src/mtah/aproximaciones').glob('*.py'))]
    files += [p.relative_to(source).as_posix() for p in sorted((source / 'tests').glob('test_*.py'))]
    files += [f'scripts/{m}.py' for m in SCRIPTS]
    files += DOCS + ['reproducir_entrega.py']
    for folder, pattern in (('modelos', '*.json'), ('config', '*.json')):
        files += [p.relative_to(source).as_posix() for p in sorted((source / folder).glob(pattern))]
    for folder in ('docs/revision_micro_juliana_01', 'docs/revision_pares_juliana_01'):
        files += [p.relative_to(source).as_posix() for p in sorted((source / folder).glob('*.json'))]
    files = sorted(set(files))
    manifest = {}
    for relative in files:
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or relative.startswith(('data/', 'outputs/', '.git/')):
            raise ValueError('Ruta no permitida en publicación')
        content = (source / relative).read_text(encoding='utf-8-sig')
        if path.suffix == '.py':
            content = sanear_python(content, relative == 'src/mtah/parametros.py')
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.replace('\r\n', '\n').encode('utf-8'))
        manifest[relative] = canon_sha(target.read_bytes())
    for name, text in (('README.md', (source / 'docs/README_PUBLICABLE_EQUIPO.md').read_text(encoding='utf-8')), ('.gitignore', (source / '.gitignore').read_text(encoding='utf-8'))):
        if name == '.gitignore':
            text += '\n# Publicacion: impedir nuevos artefactos privados incluso fuera de outputs/\n*.csv\n*.ipynb\n*.pdf\n*.log\n*privado*.json\ndocs/estudio/\nnotebooks/\nsalidas/\n'
        (dest / name).write_bytes(text.replace('\r\n', '\n').encode('utf-8'))
        manifest[name] = canon_sha((dest / name).read_bytes())
    report = dict(base_oficial=BASE_OFICIAL, datos_restringidos_incluidos=False, corpus_externos_incluidos=False, notebooks_personales_incluidos=False, historia_del_equipo_incluida=False, normalizacion_hash='UTF-8 LF', comentarios_y_docstrings_heredados_eliminados=True, origen_descriptivo_parametros_normalizado=True, archivos_sha256=manifest)
    with (source.parent / f'MANIFIESTO_{dest.name}.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return report

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destino', type=Path, required=True)
    ap.add_argument('--git', required=True)
    args = ap.parse_args()
    dest = args.destino.resolve()
    if not dest.name.startswith('mtah-2027-veriana-publicable-') or dest.parent != ROOT.parent:
        raise ValueError('Se requiere un worktree de publicación nuevo junto al repositorio')

    def git(*command):
        return subprocess.check_output([args.git, '-C', str(dest), *command], text=True).strip()
    if git('status', '--porcelain') or git('rev-parse', 'HEAD') != git('rev-parse', BASE_OFICIAL):
        raise ValueError('El worktree debe estar limpio y sobre la base oficial')
    result = exportar(ROOT, dest)
    print(f"EXPORTACION_PUBLICA_OK: {len(result['archivos_sha256'])} archivos por lista positiva")
if __name__ == '__main__':
    main()
