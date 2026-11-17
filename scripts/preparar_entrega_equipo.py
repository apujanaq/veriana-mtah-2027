"""Documentación pública en README.md."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import sys
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

def elegir(reporte):
    """Documentación pública en README.md."""
    if not reporte or reporte.get('aprox') not in ('juliana_pares_micro', 'juliana_control_multiventana', 'juliana_control_pares') or reporte.get('aceptada') is not True:
        return 'alejo_v2_robusta'
    if reporte.get('concurso_evaluado') is not False or reporte.get('ajuste_con_metrica') is not False:
        return 'alejo_v2_robusta'
    criteria = reporte.get('criterios', [])
    cases = {(r.get('partido'), r.get('condicion')) for r in criteria}
    if cases != {(g, c) for g in (1, 2) for c in ('limpio', 'ruido')} or len(criteria) != 4:
        return 'alejo_v2_robusta'
    if not all((r.get('spell_ok') is True and r.get('precision_ok') is True for r in criteria)):
        return 'alejo_v2_robusta'
    return reporte['aprox']

def preparar(datos_dir, salida, decision=None, autorizar=False, experimental=None):
    if not autorizar:
        raise PermissionError('Se requiere autorización explícita para procesar el bundle restringido')
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=False)
    with (salida / 'ejecucion_privada.log').open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            import numpy as np
            import pandas as pd
            from mtah import carga, features, temporal
            from mtah.aproximaciones import REGISTRO
            from mtah.contrato import Datos, validar_probabilidades
            from evaluation import possession_metrics as pm
            report = json.loads(Path(decision).read_text(encoding='utf-8')) if decision else None
            name = elegir(report)
            if experimental is not None:
                if experimental not in ('juliana_pares_micro', 'juliana_control_multiventana', 'juliana_control_pares'):
                    raise ValueError('Candidata experimental desconocida')
                name = experimental
            atd = carga.cargar_atd(data_dir=datos_dir, autorizar_datos=True)
            tags = carga.cargar_smart_tagging(data_dir=datos_dir, autorizar_datos=True)
            mask = features.mascara_calificada(len(atd['balon']))
            datos = Datos(atd, features.calcular_features(atd, mask), tags, mask)
            aprox = REGISTRO[name]()
            probabilities = aprox.probabilidades(datos)
            reason = validar_probabilidades(probabilities, len(mask))
            if reason:
                raise ValueError('Contrato de probabilidades inválido: ' + reason)
            lab = aprox.etiquetas(datos) if hasattr(aprox, 'etiquetas') else temporal.decodificar(probabilities, mask)[0]
            frame = pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab})
            reason = pm.validate_submission(frame, len(mask))
            if reason:
                raise ValueError('CSV inválido: ' + reason)
            path = salida / 'veriana.csv'
            with path.open('x', encoding='utf-8', newline='') as handle:
                frame.to_csv(handle, index=False, lineterminator='\n')
            reread = pd.read_csv(path)
            if pm.validate_submission(reread, len(mask)) or not np.array_equal(reread['label'].to_numpy(), lab):
                raise ValueError('La relectura del CSV no coincide')
            receipt = dict(aproximacion=name, experimental=experimental is not None, contrato_valido=True, csv_validado=True, gt_leido=False, datos_enviados_a_servicios_externos=False, csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            with (salida / 'recibo_privado.json').open('x', encoding='utf-8') as handle:
                json.dump(receipt, handle, indent=2)
                handle.write('\n')
            with (salida / 'veriana.md').open('xb') as handle:
                technical = (ROOT / 'docs/veriana.md').read_text(encoding='utf-8').replace('{{APROXIMACION}}', name)
                technical = technical.replace('{{ESTADO}}', 'Experimental; no cumple todos los criterios del empalme.' if experimental else 'Referencia conservadora del empalme.')
                handle.write(technical.encode('utf-8'))
        except Exception:
            traceback.print_exc()
            return False
    return True

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datos', type=Path, required=True, help='carpeta local que contiene tactical/ y broadcast/')
    ap.add_argument('--salida', type=Path, required=True, help='directorio local nuevo, excluido de Git')
    ap.add_argument('--decision-publica', type=Path)
    ap.add_argument('--autorizar-datos-restringidos', action='store_true')
    ap.add_argument('--candidata-experimental', choices=['juliana_pares_micro', 'juliana_control_multiventana', 'juliana_control_pares'])
    args = ap.parse_args()
    ok = preparar(args.datos, args.salida, args.decision_publica, args.autorizar_datos_restringidos, args.candidata_experimental)
    print('ENTREGA_LOCAL_OK' if ok else 'ENTREGA_LOCAL_FALLO: consultar registro privado local')
    return 0 if ok else 1
if __name__ == '__main__':
    raise SystemExit(main())
