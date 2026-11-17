"""Documentación pública en README.md."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
COMMIT = 'e706dd506b360d69d9d123d5b8026e7294b13996'
FILES = [{'path': 'data/Sample_Game_1/Sample_Game_1_RawEventsData.csv', 'size': 139446, 'git_sha': '3c426dd4f442dd433bccf89e3634f91aa310cd8f'}, {'path': 'data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv', 'size': 32782073, 'git_sha': 'ff409a5680fd75acc7cebcf786827214c687478b'}, {'path': 'data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv', 'size': 32782961, 'git_sha': 'ce2e91c0e633426cec1146f4e728b72417454e3d'}, {'path': 'data/Sample_Game_2/Sample_Game_2_RawEventsData.csv', 'size': 153897, 'git_sha': 'decccd47c1d53960d290cb6679337eec83a8a97f'}, {'path': 'data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv', 'size': 29627711, 'git_sha': 'e7334d8d762954a215c49e55f211855000243fdc'}, {'path': 'data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv', 'size': 31890338, 'git_sha': '8b4d3648888cc8b033e7142919ebaea04f6328e0'}]
ROOT = Path(__file__).resolve().parents[1]

def blob_sha(path):
    digest = hashlib.sha1()
    digest.update(('blob ' + str(path.stat().st_size) + '\x00').encode())
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destino', default=str(ROOT / 'data' / 'external' / 'metrica' / COMMIT))
    args = parser.parse_args()
    folder = Path(args.destino)
    for spec in FILES:
        relative = Path(spec['path']).relative_to('data')
        path = folder / relative
        if path.exists():
            if blob_sha(path) != spec['git_sha']:
                raise ValueError('Un archivo existente no coincide; no se sobrescribe')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://raw.githubusercontent.com/metrica-sports/sample-data/' + COMMIT + '/' + spec['path']
        with urllib.request.urlopen(url, timeout=90) as response, path.open('xb') as output:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                output.write(block)
        if path.stat().st_size != spec['size'] or blob_sha(path) != spec['git_sha']:
            raise ValueError('Descarga incompleta o checksum incorrecto; revisar localmente')
        print('Verificado:', relative.as_posix())
    manifest = folder / 'PROVENANCE.json'
    content = json.dumps({'source': 'https://github.com/metrica-sports/sample-data', 'commit': COMMIT, 'credit': 'Metrica Sports', 'files': FILES}, indent=2) + '\n'
    if not manifest.exists():
        with manifest.open('x', encoding='utf-8') as handle:
            handle.write(content)
    print('Otros partidos públicos preparados localmente')
if __name__ == '__main__':
    main()
