"""Documentación pública en README.md."""
import argparse
import sys
from . import ROOT
sys.path.insert(0, str(ROOT / 'evaluation'))
import possession_metrics as pm
from games import source_frames

def validar_csv(path, n_esperados=None):
    n = source_frames() if n_esperados is None else n_esperados
    table = pm.read_label_csv(path)
    if list(table.columns) != ['frame', 'label']:
        raise ValueError('El CSV debe contener únicamente frame,label, en ese orden')
    reason = pm.validate_submission(table, n)
    if reason:
        raise ValueError('CSV inválido: revisar filas, índice contiguo y etiquetas H/A/D/X')
    return True

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    args = parser.parse_args(argv)
    try:
        validar_csv(args.csv)
    except (ValueError, OSError):
        print('FAIL: el CSV no cumple el contrato de entrega')
        return 1
    print('OK: CSV válido (validación oficial; sin ground truth)')
    return 0
if __name__ == '__main__':
    sys.exit(main())
