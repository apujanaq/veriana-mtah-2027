"""Documentación pública en README.md."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
ROOT = Path(__file__).resolve().parents[1]
ARTICLE = 28196177
API = f'https://api.figshare.com/v2/articles/{ARTICLE}'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventario', action='store_true')
    parser.add_argument('--salida', default=str(ROOT / 'data/external/idsse/figshare_28196177'))
    args = parser.parse_args()
    metadata = json.loads(urllib.request.urlopen(API, timeout=90).read())
    if metadata['id'] != ARTICLE:
        raise ValueError('Artículo externo incorrecto')
    specs = [dict(id=f['id'], name=f['name'], size=f['size'], md5=f['computed_md5'], url=f['download_url']) for f in metadata['files']]
    if args.inventario:
        print(json.dumps(dict(article=ARTICLE, version=metadata['version'], title=metadata['title'], license=metadata.get('license'), files=specs), indent=2))
        return
    fixed = json.loads((ROOT / 'config/idsse_v4.json').read_text(encoding='utf-8'))
    if fixed['article'] != ARTICLE or fixed['version'] != metadata['version'] or fixed['files'] != specs:
        raise ValueError('La fuente cambió respecto al inventario fijado')
    base = Path(args.salida)
    base.mkdir(parents=True, exist_ok=True)

    def one(spec):
        name = spec['name']
        if Path(name).name != name or not name.endswith(('.xml', '.zip')):
            raise ValueError('Nombre público no permitido')
        path = base / name
        digest = hashlib.md5()
        if path.exists():
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
        else:
            with urllib.request.urlopen(spec['url'], timeout=180) as response, path.open('xb') as handle:
                while (chunk := response.read(1024 * 1024)):
                    digest.update(chunk)
                    handle.write(chunk)
        if path.stat().st_size != spec['size'] or digest.hexdigest() != spec['md5']:
            raise ValueError('Archivo incompleto o distinto; no se reemplaza')
        print('Fuente pública DFL verificada: ' + name, flush=True)
        return spec
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(one, specs))
    manifest = dict(article=ARTICLE, version=metadata['version'], license=metadata['license'], source=API, files=records)
    path = base / 'PROVENANCE.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Manifiesto previo diferente')
    else:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2)
    print('Corpus público DFL preparado y verificado', flush=True)
if __name__ == '__main__':
    main()
