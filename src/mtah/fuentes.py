"""Documentación pública en README.md."""
import struct
import numpy as np
import pandas as pd
from .preprocesamiento import autorizar

def leer_homografia(path, n_frames):
    matrix_columns = ['h{}{}'.format(i, j) for i in range(3) for j in range(3)]
    frame = pd.read_csv(path)
    required = ['frame', *matrix_columns, 'reprojection_error', 'valid']
    if any((name not in frame for name in required)):
        raise ValueError('Homografía: columnas incompletas')
    ids = frame['frame'].to_numpy()
    if not np.issubdtype(ids.dtype, np.integer) or not np.array_equal(ids, np.arange(n_frames)):
        raise ValueError('Homografía: no corresponde al vídeo de referencia')
    if not frame['valid'].isin([0, 1]).all():
        raise ValueError('Homografía: valid debe ser 0/1')
    matrices = frame[matrix_columns].to_numpy(dtype=float).reshape(n_frames, 3, 3)
    error = frame.reprojection_error.to_numpy(dtype=float)
    declared = frame.valid.to_numpy(dtype=bool)
    calidad_disponible = np.isfinite(error) & (error >= 0)
    error = np.where(calidad_disponible, error, np.nan)
    finite = np.isfinite(matrices).all(axis=(1, 2))
    determinant = np.full(n_frames, np.nan)
    determinant[finite] = np.linalg.det(matrices[finite])
    norm = np.linalg.norm(matrices, axis=(1, 2))
    usable = declared & finite & (norm > 0)
    normalized = matrices[usable] / norm[usable, None, None]
    condition = np.full(n_frames, np.nan)
    condition[usable] = np.linalg.cond(normalized)
    usable &= np.isfinite(condition) & (condition < 1000000000000.0)
    return dict(matrices=matrices, error=error, valid=usable, declarada_valida=declared, calidad_disponible=calidad_disponible, condicion=condition, determinante=determinant)

def _boxes(handle, first, stop):
    position = first
    while position < stop:
        if stop - position < 8:
            raise ValueError('MP4: cabecera truncada')
        handle.seek(position)
        size, kind = struct.unpack('>I4s', handle.read(8))
        header = 8
        if size == 1:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError('MP4: cabecera extendida truncada')
            size, header = (struct.unpack('>Q', raw)[0], 16)
        elif size == 0:
            size = stop - position
        if size < header or position + size > stop:
            raise ValueError('MP4: estructura de contenedores inválida')
        yield (kind, position + header, position + size)
        position += size

def _child(handle, parent, kind):
    return next((box for box in _boxes(handle, parent[1], parent[2]) if box[0] == kind), None)

def _payload(handle, box, limit=4096):
    if box is None or box[2] - box[1] > limit:
        raise ValueError('MP4: metadatos ausentes o no compatibles')
    handle.seek(box[1])
    return handle.read(box[2] - box[1])

def inspeccionar_mp4(path):
    """Documentación pública en README.md."""
    with open(path, 'rb') as handle:
        handle.seek(0, 2)
        size = handle.tell()
        moov = next((box for box in _boxes(handle, 0, size) if box[0] == b'moov'), None)
        if moov is None:
            raise ValueError('MP4 sin metadatos moov')
        for track in _boxes(handle, moov[1], moov[2]):
            if track[0] != b'trak':
                continue
            mdia = _child(handle, track, b'mdia')
            if mdia is None:
                continue
            handler = _payload(handle, _child(handle, mdia, b'hdlr'))
            if handler[8:12] != b'vide':
                continue
            mdhd = _payload(handle, _child(handle, mdia, b'mdhd'))
            if mdhd[0] == 0:
                timescale, duration = struct.unpack('>II', mdhd[12:20])
            elif mdhd[0] == 1:
                timescale, duration = struct.unpack('>IQ', mdhd[20:32])
            else:
                raise ValueError('MP4: versión mdhd no compatible')
            if timescale <= 0:
                raise ValueError('MP4: reloj inválido')
            tkhd = _payload(handle, _child(handle, track, b'tkhd'))
            width, height = [value / 65536 for value in struct.unpack('>II', tkhd[-8:])]
            minf = _child(handle, mdia, b'minf')
            stbl = _child(handle, minf, b'stbl') if minf else None
            stts = _child(handle, stbl, b'stts') if stbl else None
            if stts is None:
                raise ValueError('MP4 fragmentado o sin stts: requiere un lector externo local')
            handle.seek(stts[1] + 4)
            count = struct.unpack('>I', handle.read(4))[0]
            if count > (stts[2] - stts[1] - 8) // 8:
                raise ValueError('MP4: tabla stts truncada')
            frames, ticks, deltas = (0, 0, set())
            for _ in range(count):
                samples, delta = struct.unpack('>II', handle.read(8))
                frames += samples
                ticks += samples * delta
                deltas.add(delta)
            return dict(bytes=size, ancho_px=width, alto_px=height, muestras_codificadas=frames, duracion_s=duration / timescale, duracion_muestras_s=ticks / timescale, fps_medio=frames * timescale / ticks if ticks else None, intervalos_muestreo_distintos=len(deltas), analisis_pixeles=False, analisis_cortes=False)
    raise ValueError('MP4 sin pista de vídeo')

def cargar_homografia(path, n_frames, *, autorizar_datos=False):
    autorizar(autorizar_datos)
    return leer_homografia(path, n_frames)
