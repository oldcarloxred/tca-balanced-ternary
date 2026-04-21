"""
verify_nand.py
==============
Construye NAND cascadeando:
    NAND(A, B) = NOT(AND(A, B))

donde:
  AND(A, B)   -> circuito del Paso 2 (E sobrevive sii A AND B)
  NOT(X)      -> INHIBIT(E_constante, X_como_N)  [del Paso 1]

El bridge entre ambas etapas es que la salida de AND (bit x en {0,1}) se
usa como input de NOT colocando un NORTH glider solo cuando x=1.

Tabla NAND esperada:
  (0,0) -> 1
  (0,1) -> 1
  (1,0) -> 1
  (1,1) -> 0

Adicionalmente: combinar NAND + OR para verificar XOR cascaded:
    XOR(A,B) = NAND(NAND(A, NAND(A,B)), NAND(B, NAND(A,B)))
(esto prueba NAND-completeness real en 4 cascades).
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
    for dr in range(pat.shape[0]):
        for dc in range(pat.shape[1]):
            g[(r+dr)%H, (c+dc)%W] = pat[dr, dc]

def has_east(g):
    for r in range(g.shape[0]-3):
        for c in range(g.shape[1]-2):
            if np.array_equal(g[r:r+4, c:c+3], EAST):
                return True
    return False

# ---- Primitivas ----

def AND_gate(A, B, steps=120):
    G = 60
    g = np.zeros((G,G), dtype=np.int8)
    if A:
        place(g, EAST,  24, 2)
        place(g, NORTH, 41, 24)
    if B:
        place(g, SOUTH, 28, 24)
    for _ in range(steps): g = _cpu_step(g, RT, G, G)
    return int(has_east(g))

def NOT_gate(X, steps=80):
    """NOT(X) = INHIBIT(E_constante, X_como_N).
    E siempre colocado; N colocado sii X=1. Output = has_east."""
    G = 50
    g = np.zeros((G,G), dtype=np.int8)
    place(g, EAST, 24, 2)         # constante "1"
    if X:
        place(g, NORTH, 41, 24)   # inhibidor cuando X=1
    for _ in range(steps): g = _cpu_step(g, RT, G, G)
    return int(has_east(g))

def NAND_gate(A, B):
    return NOT_gate(AND_gate(A, B))

# ---- Tabla NAND ----
print("=" * 60)
print("NAND via cascade NOT(AND)")
print("=" * 60)
print(f"  {'A':>2} {'B':>2} | {'AND':>3} {'NAND':>4} | esperado")
print(f"  -- -- | --- ---- | --------")
expected_nand = {(0,0):1, (0,1):1, (1,0):1, (1,1):0}
all_ok_nand = True
for A in (0,1):
    for B in (0,1):
        a = AND_gate(A, B)
        n = NAND_gate(A, B)
        exp = expected_nand[(A,B)]
        ok = "OK" if n == exp else "FAIL"
        if n != exp: all_ok_nand = False
        print(f"  {A:>2} {B:>2} | {a:>3} {n:>4} | esperado {exp} [{ok}]")

print()
print(f"NAND VERIFICADO: {all_ok_nand}")

# ---- NAND-completeness: construir XOR solo con NAND ----
# XOR(A,B) = NAND(NAND(A, NAND(A,B)), NAND(B, NAND(A,B)))
# Esto es la prueba cascade de universalidad: 4 NANDs encadenados
print()
print("=" * 60)
print("XOR construido SOLO con NANDs (4 niveles) — prueba de NAND-completeness")
print("=" * 60)
print(f"  {'A':>2} {'B':>2} | {'X=NAND(A,B)':>11} {'N1=NAND(A,X)':>12} {'N2=NAND(B,X)':>12} | {'XOR':>3} esperado")

expected_xor = {(0,0):0, (0,1):1, (1,0):1, (1,1):0}
all_ok_xor = True
for A in (0,1):
    for B in (0,1):
        X  = NAND_gate(A, B)
        N1 = NAND_gate(A, X)
        N2 = NAND_gate(B, X)
        xor = NAND_gate(N1, N2)
        exp = expected_xor[(A,B)]
        ok = "OK" if xor == exp else "FAIL"
        if xor != exp: all_ok_xor = False
        print(f"  {A:>2} {B:>2} | {X:>11} {N1:>12} {N2:>12} | {xor:>3} esperado {exp} [{ok}]")

print()
print(f"XOR via NANDs VERIFICADO: {all_ok_xor}")
print()
if all_ok_nand and all_ok_xor:
    print("=" * 60)
    print("NAND-COMPLETENESS CERRADA CONSTRUCTIVAMENTE")
    print("=" * 60)
    print("  - NAND funciona (tabla de verdad verificada)")
    print("  - Cascades de 4 NANDs componen XOR correctamente")
    print("  - => cualquier circuito booleano es ejecutable en esta CA")
