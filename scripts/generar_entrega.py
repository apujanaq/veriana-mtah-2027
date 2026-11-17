"""Documentación pública en README.md."""
import argparse
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for sub in ('src', '', 'evaluation'):
    sys.path.insert(0, str(ROOT / sub))
import pandas as pd
import possession_metrics as pm
from mtah import contrato, ensamble
from mtah.aproximaciones.juliana_control_pares import JulianaControlPares

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--salida', type=Path, required=True, help='CSV de destino (no debe existir)')
    ap.add_argument('--autorizar-datos-restringidos', action='store_true', help='confirma que se leen los datos del concurso (términos y condiciones)')
    args = ap.parse_args()
    if not args.autorizar_datos_restringidos:
        sys.exit('Falta --autorizar-datos-restringidos: el script lee los datos del concurso.')
    datos = contrato.cargar_datos(autorizar_datos=True)
    ruta = ensamble.escribir_csv(JulianaControlPares().etiquetas(datos), args.salida)
    motivo = pm.validate_submission(pd.read_csv(ruta), contrato.N_FRAMES)
    if motivo is not None:
        sys.exit(f'CSV rechazado por el validador oficial: {motivo}')
    print(f'{ruta}: VÁLIDO · sha256 {hashlib.sha256(ruta.read_bytes()).hexdigest()}')
if __name__ == '__main__':
    main()
