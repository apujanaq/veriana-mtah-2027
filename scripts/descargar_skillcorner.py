"""Documentación pública en README.md."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request
ROOT = Path(__file__).resolve().parents[1]

def blob_sha(content):
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\x00' + content).hexdigest()

def descargar(config_path=ROOT / 'config/skillcorner_v3.json', destination=None):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config['source'] != 'SkillCorner/opendata' or not re.fullmatch('[0-9a-f]{40}', config['commit']):
        raise ValueError('Procedencia no admitida')
    destination = Path(destination or ROOT / 'data/external/skillcorner' / config['commit'])
    destination.mkdir(parents=True, exist_ok=True)

    def one(spec):
        relative = spec['path']
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination.resolve()):
            raise ValueError('Ruta no permitida')
        raw_url = f"https://raw.githubusercontent.com/{config['source']}/{config['commit']}/{relative}"
        raw = urllib.request.urlopen(raw_url, timeout=90).read()
        if len(raw) != spec['size'] or blob_sha(raw) != spec['git_sha1']:
            raise ValueError('Blob público no coincide')
        is_lfs = raw.startswith(b'version https://git-lfs.github.com/spec/v1\n')
        if is_lfs:
            match = re.fullmatch(b'version https://git-lfs.github.com/spec/v1\\noid sha256:([a-f0-9]{64})\\nsize ([0-9]+)\\n', raw)
            if not match:
                raise ValueError('Puntero LFS inválido')
            sha, size = (match[1].decode(), int(match[2]))
            url = f"https://media.githubusercontent.com/media/{config['source']}/{config['commit']}/{relative}"
        else:
            sha, size, url = (hashlib.sha256(raw).hexdigest(), len(raw), raw_url)
        if path.exists():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            if path.stat().st_size != size or digest.hexdigest() != sha:
                raise ValueError('Archivo existente incompatible; no se reemplaza')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            digest, total = (hashlib.sha256(), 0)
            with path.open('xb') as handle:
                if is_lfs:
                    with urllib.request.urlopen(url, timeout=180) as response:
                        while (chunk := response.read(1024 * 1024)):
                            handle.write(chunk)
                            digest.update(chunk)
                            total += len(chunk)
                else:
                    handle.write(raw)
                    digest.update(raw)
                    total = len(raw)
            if total != size or digest.hexdigest() != sha:
                raise ValueError('Descarga incompleta; conservar para diagnóstico, no sobreescribir')
        print('Fuente pública verificada: ' + relative, flush=True)
        return dict(path=relative, bytes=size, sha256=sha, git_sha1=spec['git_sha1'], url=url)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(one, config['files']))
    manifest = dict(source=config['source'], commit=config['commit'], splits=config['splits'], files=records)
    target = destination / 'PROVENANCE.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Manifiesto previo diferente')
    else:
        with target.open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2)
    return destination
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida')
    args = parser.parse_args()
    descargar(destination=args.salida)
