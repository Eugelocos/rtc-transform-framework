"""
Test de compresión diferencial RTC para datos similares.
Genera 100 casos basados en un estado base con pequeñas modificaciones aleatorias.
Evalúa compresión, reconstrucción y eficiencia.
"""

import numpy as np
import gzip
import time
import random
from collections import Counter, defaultdict

class Transformacion:
    def __init__(self, tipo, params=None):
        self.tipo = tipo
        self.params = params or []
    def __repr__(self):
        return f"Transformacion(tipo={self.tipo!r}, params={self.params!r})"

def encontrar_mejor_T_optimizado(dataset):
    if len(dataset) <= 1:
        return dataset[0] if dataset else b''
    frecuencias = Counter(dataset)
    candidatos = [item for item, _ in frecuencias.most_common(3)]
    otros = [d for d in set(dataset) if d not in candidatos]
    if otros:
        candidatos.extend(np.random.choice(otros, min(2, len(otros)), replace=False))
    mejor_T = candidatos[0]
    menor_costo = float('inf')
    for T in candidatos:
        costo_total = estimar_costo_dataset(T, dataset)
        if costo_total < menor_costo:
            menor_costo = costo_total
            mejor_T = T
    return mejor_T

def estimar_costo_dataset(T, dataset):
    costo = len(T)
    for data in dataset:
        if T == data:
            costo += 1
        elif len(T) == len(data):
            T_arr = np.frombuffer(T, dtype=np.uint8)
            D_arr = np.frombuffer(data, dtype=np.uint8)
            diff_count = np.sum(T_arr != D_arr)
            if diff_count == 0:
                costo += 1
            elif diff_count <= 8:
                costo += 1 + 2 + diff_count * 2
            else:
                costo += 1 + len(T)
        else:
            costo += max(len(T), len(data))
    return costo

def optimizar_transformacion_rapido(T, D):
    if T == D:
        return Transformacion('IDENTIDAD')
    T_arr = np.frombuffer(T, dtype=np.uint8)
    D_arr = np.frombuffer(D, dtype=np.uint8)
    if len(T) != len(D):
        if len(D) > len(T) and T == D[:len(T)]:
            return Transformacion('INSERCION', [len(T), D[len(T):]])
        elif len(D) < len(T) and D == T[:len(D)]:
            return Transformacion('ELIMINACION', [len(D), len(T)])
        return Transformacion('SUBSTITUCION', [T, D, 0])
    diff_mask = T_arr != D_arr
    diff_count = np.sum(diff_mask)
    if diff_count == 0:
        return Transformacion('IDENTIDAD')
    if diff_count <= min(8, len(T)//4):
        diff_positions = np.where(diff_mask)[0]
        diff_values = D_arr[diff_mask]
        return Transformacion('PATCH', [diff_positions, diff_values])
    xor_delta = T_arr ^ D_arr
    return Transformacion('DELTA_XOR', [xor_delta])

def posiciones_a_deltas(posiciones):
    posiciones = np.sort(posiciones)
    deltas = []
    prev = 0
    for p in posiciones:
        deltas.append(p - prev)
        prev = p
    max_delta = max(deltas) if deltas else 0
    dtype = np.uint8 if max_delta <= 255 else np.uint16
    return np.array(deltas, dtype=dtype)

def deltas_a_posiciones(deltas):
    pos = []
    acum = 0
    for d in deltas:
        acum += int(d)
        pos.append(acum)
    return np.array(pos, dtype=np.uint16)

def serializar_transformaciones_optimizado(T, transformaciones):
    datos = [T]
    tipo_a_codigo = {
        'IDENTIDAD': 0,
        'PATCH': 1,
        'DELTA_XOR': 2,
        'ROTACION': 3,
        'INVERSION': 4,
        'TRANSPOSICION': 5,
        'SUBSTITUCION': 6,
        'INSERCION': 7,
        'ELIMINACION': 8
    }
    patch_groups = defaultdict(list)
    otros_transform = []
    for t in transformaciones:
        if t.tipo == 'PATCH':
            pos_deltas = posiciones_a_deltas(t.params[0])
            key = pos_deltas.tobytes()
            patch_groups[key].append(t.params[1])
        else:
            otros_transform.append(t)
    for key, valores_list in patch_groups.items():
        datos.append(bytes([tipo_a_codigo['PATCH']]))
        n = len(deltas_a_posiciones(np.frombuffer(key, dtype=np.uint8)))
        datos.append(n.to_bytes(1 if n <= 255 else 2, byteorder='big'))
        datos.append(key)
        for valores in valores_list:
            datos.append(valores.tobytes())
    for t in otros_transform:
        codigo = tipo_a_codigo.get(t.tipo, 255)
        datos.append(bytes([codigo]))
        if t.tipo == 'DELTA_XOR':
            datos.append(t.params[0].tobytes())
        elif t.tipo == 'ROTACION':
            datos.append(bytes([t.params[0] & 0xFF]))
        elif t.tipo == 'INVERSION':
            inicio, fin = t.params
            datos.append(inicio.to_bytes(2, byteorder='big'))
            datos.append(fin.to_bytes(2, byteorder='big'))
        elif t.tipo == 'INSERCION':
            pos, bytes_ins = t.params
            datos.append(pos.to_bytes(2, byteorder='big'))
            datos.append(len(bytes_ins).to_bytes(2, byteorder='big'))
            datos.append(bytes_ins)
        elif t.tipo == 'ELIMINACION':
            start, end = t.params
            datos.append(start.to_bytes(2, byteorder='big'))
            datos.append(end.to_bytes(2, byteorder='big'))
        elif t.tipo == 'SUBSTITUCION':
            original, nuevo, pos = t.params
            datos.append(len(original).to_bytes(2, byteorder='big'))
            datos.append(len(nuevo).to_bytes(2, byteorder='big'))
            datos.append(original)
            datos.append(nuevo)
            datos.append(pos.to_bytes(2, byteorder='big'))
    return b''.join(datos)

def aplicar_transformacion(T, transf):
    T_arr = np.frombuffer(T, dtype=np.uint8)
    if transf.tipo == 'IDENTIDAD':
        return T
    elif transf.tipo == 'PATCH':
        positions, values = transf.params
        arr = T_arr.copy()
        arr[positions] = values
        return arr.tobytes()
    elif transf.tipo == 'DELTA_XOR':
        xor = transf.params[0]
        if len(xor) != len(T_arr):
            raise ValueError("Error tamaño XOR distinto del tamaño base")
        arr = np.bitwise_xor(T_arr, xor)
        return arr.tobytes()
    elif transf.tipo == 'INSERCION':
        pos, bytes_ins = transf.params
        return T[:pos] + bytes_ins + T[pos:]
    elif transf.tipo == 'ELIMINACION':
        start, end = transf.params
        return T[:start] + T[end:]
    elif transf.tipo == 'SUBSTITUCION':
        original, nuevo, pos = transf.params
        if T[pos:pos+len(original)] == original:
            return T[:pos] + nuevo + T[pos+len(original):]
        else:
            raise ValueError("SUBSTITUCION no válida")
    elif transf.tipo == 'ROTACION':
        offset = transf.params[0]
        arr = np.frombuffer(T, dtype=np.uint8)
        arr = np.roll(arr, offset)
        return arr.tobytes()
    elif transf.tipo == 'INVERSION':
        inicio, fin = transf.params
        arr = np.frombuffer(T, dtype=np.uint8)
        arr[inicio:fin] = arr[inicio:fin][::-1]
        return arr.tobytes()
    else:
        raise NotImplementedError(f"Transformacion {transf.tipo} no implementada")

def rtc_optimizado(dataset):
    if not dataset:
        return {'size_con_compresion': 0, 'FE': 0}
    T = encontrar_mejor_T_optimizado(dataset)
    transformaciones = [optimizar_transformacion_rapido(T, d) for d in dataset]
    costo_total = len(T) + sum(len(t.tipo) + sum(len(p) if isinstance(p, (bytes, np.ndarray)) else 1 for p in t.params) for t in transformaciones)
    datos_serializados = serializar_transformaciones_optimizado(T, transformaciones)
    compressed = gzip.compress(datos_serializados)
    tipo_counts = Counter(t.tipo for t in transformaciones)
    return {
        'estado_base': T,
        'transformaciones': transformaciones,
        'transformaciones_counts': tipo_counts,
        'size_sin_compresion': costo_total,
        'datos_serializados': datos_serializados,
        'size_con_compresion': len(compressed),
        'FE': sum(len(d) for d in dataset) / len(compressed) if len(compressed) > 0 else 0,
    }

def ejecutar_experimento():
    print("=== EXPERIMENTO RTC OPTIMIZADO EXTENDIDO ===")
    np.random.seed()
    base = np.random.randint(0, 256, size=128, dtype=np.uint8).tobytes()
    datos_similares = [base]
    for _ in range(99):
        mod = np.frombuffer(base, dtype=np.uint8).copy()
        num_mods = np.random.randint(1, 4)
        for __ in range(num_mods):
            pos = np.random.randint(0, len(mod))
            mod[pos] = np.random.randint(0, 256)
        datos_similares.append(mod.tobytes())

    start = time.time()
    resultado = rtc_optimizado(datos_similares)
    tiempo = time.time() - start

    print(f"\n\U0001F537 DATOS SIMILARES (100 elementos)")
    print(f"Estado base: {len(resultado['estado_base'])} bytes")
    print(f"Tipos de transformaciones: {resultado['transformaciones_counts']}")
    print(f"Tama\u00f1o sin compresi\u00f3n estimado: {resultado['size_sin_compresion']} bytes")
    print(f"Datos serializados (sin compresi\u00f3n): {len(resultado['datos_serializados'])} bytes")
    print(f"RTC+gzip final: {resultado['size_con_compresion']} bytes")
    print(f"Factor de eficiencia: {resultado['FE']:.2f}")
    print(f"Tiempo RTC: {tiempo:.4f}s")

    tamaño_total_original = sum(len(d) for d in datos_similares)
    print(f"\nTama\u00f1o total original datos: {tamaño_total_original} bytes")

    datos_concatenados = b''.join(datos_similares)
    gzip_original = gzip.compress(datos_concatenados)
    gzip_rtc = gzip.compress(resultado['datos_serializados'])
    print(f"\nCompresi\u00f3n gzip tama\u00f1o datos originales: {len(gzip_original)} bytes")
    print(f"Compresi\u00f3n gzip tama\u00f1o RTC serializados: {len(gzip_rtc)} bytes")

    # Mostrar un dato al azar con detalle
    indice = random.randint(0, len(datos_similares) - 1)
    dato_original = datos_similares[indice]
    transformacion = resultado['transformaciones'][indice]
    dato_reconstruido = aplicar_transformacion(resultado['estado_base'], transformacion)

    print("\n\U0001F50D An\u00e1lisis detallado de un dato al azar:")
    print(f"\u00cdndice: {indice}")
    print(f"\nEstado base (hex):\n{resultado['estado_base'].hex(' ')}")

    print("\nTransformaci\u00f3n aplicada:")
    if transformacion.tipo == 'PATCH':
        posiciones = ', '.join(str(p) for p in transformacion.params[0])
        valores = ' '.join(f"{v:02x}" for v in transformacion.params[1])
        print(f"  Tipo: PATCH")
        print(f"  Posiciones modificadas: {posiciones}")
        print(f"  Valores insertados: {valores}")
    elif transformacion.tipo == 'DELTA_XOR':
        delta = ' '.join(f"{b:02x}" for b in transformacion.params[0])
        print(f"  Tipo: DELTA_XOR")
        print(f"  Delta XOR aplicado:\n{delta}")
    elif transformacion.tipo == 'IDENTIDAD':
        print("  Tipo: IDENTIDAD (sin cambios)")
    else:
        print(f"  Tipo: {transformacion.tipo}")
        print(f"  Par\u00e1metros: {transformacion.params}")

    print("\nDato original (hex):")
    print(dato_original.hex(' '))
    print("\nDato reconstruido (hex):")
    print(dato_reconstruido.hex(' '))

    print("\n\u00bfCoinciden original y reconstruido?", "\u2705 S\u00cd" if dato_original == dato_reconstruido else "\u274C NO")

if __name__ == "__main__":
    ejecutar_experimento()
