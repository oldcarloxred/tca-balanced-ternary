"""
verify_trit_adder.py
====================
Experimento: puede la CA hacer balanced-ternary trit addition de forma nativa?

Encoding de un trit:
  +1 -> NORTH glider (going up)
  -1 -> SOUTH glider (going down)
   0 -> no glider

Arquitectura: A en fila 40, B en fila 10, misma columna. Collision zone ~row 25.
Readout: contar N y S gliders sobrevivientes en el grid final.

Balanced ternary: a+b produce digit d y carry c, donde a+b = 3c + d, d in {-1,0,+1}.
  (0,0)   -> d=0, c=0
  (+1,0)  -> d=+1, c=0
  (0,+1)  -> d=+1, c=0
  (-1,0)  -> d=-1, c=0
  (0,-1)  -> d=-1, c=0
  (+1,-1) -> d=0, c=0
  (-1,+1) -> d=0, c=0
  (+1,+1) -> d=-1, c=+1  (porque +2 = 3*1 + (-1))
  (-1,-1) -> d=+1, c=-1  (porque -2 = 3*(-1) + 1)

El experimento lee el output real (N count, S count) y ve cuantos casos
casan con el balanced-ternary esperado.
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

def place(g, pat, r, c):
    for dr in range(pat.shape[0]):
        for dc in range(pat.shape[1]):
            g[(r+dr)%g.shape[0], (c+dc)%g.shape[1]] = pat[dr,dc]

def count_north(g):
    n = 0
    for r in range(g.shape[0]-1):
        for c in range(g.shape[1]-1):
            if np.array_equal(g[r:r+2, c:c+2], NORTH): n += 1
    return n

def count_south(g):
    n = 0
    for r in range(g.shape[0]-1):
        for c in range(g.shape[1]-1):
            if np.array_equal(g[r:r+2, c:c+2], SOUTH): n += 1
    return n

G = 60
COL = 30

def trit_to_pattern(t):
    """+1 -> NORTH, -1 -> SOUTH, 0 -> None."""
    if t == 1: return NORTH
    if t == -1: return SOUTH
    return None

def run_trit_sum(a, b, rowA=40, rowB=10, steps=40):
    g = np.zeros((G, G), dtype=np.int8)
    pa = trit_to_pattern(a)
    pb = trit_to_pattern(b)
    if pa is not None: place(g, pa, rowA, COL)
    if pb is not None: place(g, pb, rowB, COL)
    peak = int((g != 0).sum())
    for _ in range(steps):
        g = _cpu_step(g, RT, G, G)
        p = int((g != 0).sum())
        if p > peak: peak = p
    return g, peak

def balanced_ternary_add(a, b):
    """Devuelve (digit, carry) segun balanced ternary."""
    s = a + b
    if s == 2:  return (-1, +1)
    if s == -2: return (+1, -1)
    return (s, 0)

# -------- Ejecutar los 9 casos --------
print("=" * 78)
print("9 CASOS DE BALANCED TERNARY ADDITION (trit + trit)")
print("=" * 78)
print(f"  {'A':>3} + {'B':>3} | {'N#':>3} {'S#':>3} | {'output_read':>12} | "
      f"{'expected':>18} | match?")
print(f"  --- --- | --- --- | -----------  | ------------------ | ------")

results = []
matches = 0
for a in (-1, 0, 1):
    for b in (-1, 0, 1):
        g_final, peak = run_trit_sum(a, b)
        nN = count_north(g_final)
        nS = count_south(g_final)

        # Interpretacion del output:
        #   (N count, S count):
        #     (0,0) -> d=0
        #     (1,0) -> d=+1
        #     (0,1) -> d=-1
        #     (1,1) -> two gliders ignoring each other -> sum=0 (no annihilation)
        #     (2,0) -> d=-1 carry=+1   (SI interpretamos "2 N" como bt overflow)
        #     (0,2) -> d=+1 carry=-1
        if (nN, nS) == (0, 0):       read = (0, 0)
        elif (nN, nS) == (1, 0):     read = (+1, 0)
        elif (nN, nS) == (0, 1):     read = (-1, 0)
        elif (nN, nS) == (1, 1):     read = ("bifur", "bifur")
        elif (nN, nS) == (2, 0):     read = (-1, +1)   # overflow balanced
        elif (nN, nS) == (0, 2):     read = (+1, -1)
        else:                         read = (f"N={nN}", f"S={nS}")

        expected = balanced_ternary_add(a, b)
        ok = (read == expected)
        if ok: matches += 1

        read_s = f"d={read[0]},c={read[1]}" if isinstance(read[0], int) else f"{read[0]}"
        exp_s  = f"d={expected[0]},c={expected[1]}"
        mark = "OK" if ok else "--"
        print(f"  {a:>3} + {b:>3} | {nN:>3} {nS:>3} | {read_s:>12}  | {exp_s:>18} | {mark}")
        results.append({"a": a, "b": b, "nN": nN, "nS": nS, "read": read,
                        "expected": expected, "ok": ok, "peak": peak})

print()
print(f"COINCIDENCIAS CON BALANCED TERNARY: {matches}/9")
print()

# Analisis
print("=" * 78)
print("ANALISIS")
print("=" * 78)
for r in results:
    if not r["ok"]:
        print(f"  FALLA ({r['a']:+d}) + ({r['b']:+d}):")
        print(f"     CA output: N={r['nN']}, S={r['nS']}  peak_pop={r['peak']}")
        print(f"     Esperado:  d={r['expected'][0]:+d}, c={r['expected'][1]:+d}")

print()
if matches == 9:
    print("*** TRIT-ADDER BALANCED TERNARY NATIVO EN LA CA — RESULTADO NOTABLE ***")
elif matches >= 7:
    print(f"*** TRIT-ADDER PARCIAL NATIVO ({matches}/9 casos) ***")
    print("    Los casos que fallan requieren composicion (fuera del scope de un")
    print("    colision unica). Caracterizacion honesta del limite.")
else:
    print(f"   trit-adder incompleto ({matches}/9). El encoding limita el resultado.")
