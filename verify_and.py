"""
verify_and.py
=============
Construye y verifica una AND real vía cascade:
    AND(A, B) = INHIBIT(A_copia_E, ANDNOT(A_copia_N, B_S))

donde ANDNOT(X, Y) = X AND NOT Y se implementa con N+S colision offset=0
leyendo la salida al NORTE (N sobrevive sii X=1 y Y=0).

Inputs:
  A -> se replica como E glider (entra a INHIBIT) y N glider (entra a ANDNOT)
  B -> S glider (entra a ANDNOT)

Verifica la tabla de verdad completa AND(A,B):
    A=0,B=0 -> 0
    A=0,B=1 -> 0
    A=1,B=0 -> 0
    A=1,B=1 -> 1  <-- el unico caso en que E sobrevive
"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.tca_sim.simulator import _cpu_step

RT = np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
               0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,1,
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int8)

NORTH = np.array([[-1,-1],[1,1]], dtype=np.int8)
SOUTH = np.array([[1,1],[-1,-1]], dtype=np.int8)
EAST  = np.array([[-1,0,0],[0,0,-1],[0,0,-1],[-1,0,0]], dtype=np.int8)

def place(g, pat, r, c):
    H, W = g.shape
    ph, pw = pat.shape
    for dr in range(ph):
        for dc in range(pw):
            g[(r+dr)%H, (c+dc)%W] = pat[dr, dc]

def has_east_glider(g):
    H, W = g.shape
    for r in range(H-3):
        for c in range(W-2):
            if np.array_equal(g[r:r+4, c:c+3], EAST):
                return True
    return False

def run(setup, G, steps):
    g = np.zeros((G, G), dtype=np.int8)
    setup(g, G)
    peak = int((g != 0).sum())
    for _ in range(steps):
        g = _cpu_step(g, RT, G, G)
        p = int((g != 0).sum())
        if p > peak: peak = p
    return g, peak

# Layout: G=60, con E fijo en (rE-1, cE) y columna cCol para N+S.
G = 60
rE, cE_start = 25, 2       # fila del INHIBIT y columna inicial de E
cCol = 24                  # columna del N+S (y ruta por la que sube el superviviente)

def build_setup(A, B, rAN, rBS, steps):
    """Construye una funcion de setup para los inputs (A,B)."""
    def s(g, G):
        if A:  # A -> E glider + N glider
            place(g, EAST, rE-1, cE_start)
            place(g, NORTH, rAN, cCol)
        if B:  # B -> S glider
            place(g, SOUTH, rBS, cCol)
    return s

def truth_row(rAN, rBS, steps=120):
    """Evalua AND(A,B) para los 4 inputs con una config concreta."""
    out = {}
    for A in (0, 1):
        for B in (0, 1):
            setup = build_setup(A, B, rAN, rBS, steps)
            g_final, peak = run(setup, G, steps)
            e_out = has_east_glider(g_final)
            out[(A,B)] = (e_out, peak)
    return out

# Barrido: buscar (rAN, rBS) que de la tabla AND y sin explosion
print("=" * 70)
print("BARRIDO: buscando (rAN, rBS) con tabla AND perfecta")
print("=" * 70)

expected = {(0,0): False, (0,1): False, (1,0): False, (1,1): True}

clean = []
all_results = []
# rAN: fila inicial de N. Debe ser > rE (N esta abajo y sube).
# rBS: fila inicial de S. Debe ser < rAN pero > rE preferentemente.
for rAN in range(35, 58):
    for rBS in range(28, rAN - 2):
        res = truth_row(rAN, rBS, steps=120)
        matches = all(res[k][0] == expected[k] for k in expected)
        peak = max(r[1] for r in res.values())
        if matches and peak < 60:
            clean.append((rAN, rBS, peak))
        all_results.append((rAN, rBS, matches, peak))

print(f"Total configs probadas: {len(all_results)}")
print(f"Configs con tabla AND perfecta + sin explosion: {len(clean)}")
print()

if clean:
    print("Primeras 10 configs AND limpias:")
    print(f"  {'rAN':>4} {'rBS':>4} {'peak':>5}")
    for r in clean[:10]:
        print(f"  {r[0]:>4} {r[1]:>4} {r[2]:>5}")
    print()
    rAN_best, rBS_best, _ = clean[0]
    print("=" * 70)
    print(f"VERIFICACION: tabla de verdad AND en (rAN={rAN_best}, rBS={rBS_best})")
    print("=" * 70)
    res = truth_row(rAN_best, rBS_best, steps=120)
    for (A,B), (e_out, peak) in sorted(res.items()):
        expected_v = expected[(A,B)]
        ok = "OK" if e_out == expected_v else "FAIL"
        print(f"  A={A},B={B} -> E_out={int(e_out)}  esperado={int(expected_v)}  peak={peak}  [{ok}]")
    truth_ok = all(res[k][0] == expected[k] for k in expected)
    print()
    print(f"  AND VERIFICADO: {truth_ok}")
else:
    print("NO SE ENCONTRO configuracion limpia.")
    print("Diagnostico: revisar si la columna cCol, la fila rE o el timing necesitan ajuste.")
    # Mostrar casos parciales
    partial = sorted(all_results, key=lambda x: (not x[2], x[3]))[:5]
    print("Mejores 5 intentos parciales:")
    for r in partial:
        print(f"  rAN={r[0]} rBS={r[1]} tabla_ok={r[2]} peak={r[3]}")
