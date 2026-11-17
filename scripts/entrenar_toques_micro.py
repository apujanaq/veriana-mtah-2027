"""Documentación pública en README.md."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for sub in ('src', ''):
    sys.path.insert(0, str(ROOT / sub))
from mtah.experimento_micro import entrenar

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dfl', type=Path, required=True)
    ap.add_argument('--salida', type=Path, required=True)
    args = ap.parse_args()
    entrenar(args.dfl, args.salida, progreso=lambda msg: print(msg, flush=True))
if __name__ == '__main__':
    main()
