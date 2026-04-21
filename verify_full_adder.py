"""
verify_full_adder.py
====================
Full-adder de 1 bit: dado (A, B, Cin), computa
  SUM  = A XOR B XOR Cin
  Cout = (A AND B) OR (Cin AND (A XOR B))

Arquitectura (composicion clasica de dos half-adders + OR):
  HA1: S1 = A XOR B,   C1 = A AND B
  HA2: SUM = S1 XOR Cin,  C2 = S1 AND Cin
  Cout = OR(C1, C2)

Cada operacion basica se ejecuta en una simulacion CA aislada con las
primitivas ya verificadas:
  - AND via Paso 2 (E + N + S)
  - XOR via N+S offset=0 (lectura "glider sobrevive en cualquier sitio")
  - OR via ANDNOT de NOT: OR(X,Y) = NOT(NOT X AND NOT Y)
         Alternativa equivalente y mas simple aqui: OR(X,Y) se simula
         con 2 E gliders en dos lanes paralelas; si alguno sobrevive -> 1.
         Como no hay inhibidores, ambos sobreviven si estan presentes.
         Implementacion: E_X + E_Y en grid aislada, has_east sobre todo
         el grid -> True sii X=1 o Y=1.

Tabla de verdad esperada (A+B+Cin en binario = 2*Cout + SUM):
  (0,0,0) -> SUM=0, Cout=0, total=0
  (0,0,1) -> SUM=1, Cout=0, total=1
  (0,1,0) -> SUM=1, Cout=0, total=1
  (0,1,1) -> SUM=0, Cout=1, total=2
  (1,0,0) -> SUM=1, Cout=0, total=1
  (1,0,1) -> SUM=0, Cout=1, total=2
  (1,1,0) -> SUM=0, Cout=1, total=2
  (1,1,1) -> SUM=1, Cout=1, total=3
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

# ---- Primitivas ejecutadas como simulaciones CA ----

def AND_gate(A, B, steps=120):
    """AND via Paso 2: E + A_N + B_S -> E sobrevive sii A AND B."""
    G = 60
    g = np.zeros((G,G), dtype=np.int8)
    if A:
        place(g, EAST,  24, 2)
        place(g, NORTH, 41, 24)
    if B:
        place(g, SOUTH, 28, 24)
    for _ in range(steps):
        g = _cpu_step(g, RT, G, G)
    return int(has_east(g))

def XOR_gate(A, B, steps=60):
    """XOR via N+S offset=0: pop>0 sii exactamente uno presente."""
    G = 40
    g = np.zeros((G,G), dtype=np.int8)
    c = G//2
    if A: place(g, NORTH, G-6, c)
    if B: place(g, SOUTH, 4, c)
    for _ in range(steps):
        g = _cpu_step(g, RT, G, G)
    return 1 if int((g != 0).sum()) > 0 else 0

def OR_gate(X, Y, steps=60):
    """OR: colocamos un E glider en una lane si X=1 y otro en otra lane si Y=1.
    E sobrevive en CUALQUIER lane sii X=1 OR Y=1."""
    G = 50
    g = np.zeros((G,G), dtype=np.int8)
    if X: place(g, EAST, 10, 2)    # lane 1
    if Y: place(g, EAST, 30, 2)    # lane 2
    for _ in range(steps):
        g = _cpu_step(g, RT, G, G)
    return int(has_east(g))

# ---- Full-adder compuesto ----

def full_adder(A, B, Cin):
    # HA1
    S1 = XOR_gate(A, B)
    C1 = AND_gate(A, B)
    # HA2
    SUM = XOR_gate(S1, Cin)
    C2  = AND_gate(S1, Cin)
    # Cout
    Cout = OR_gate(C1, C2)
    return SUM, Cout

# ---- Tabla de verdad ----
print("=" * 70)
print("FULL-ADDER de 1 bit")
print("=" * 70)
print(f"  {'A':>2} {'B':>2} {'Cin':>3} | {'S1':>2} {'C1':>2} {'C2':>2} | {'SUM':>3} {'Cout':>4} | {'total':>5} esperado")
print(f"  -- -- --- | -- -- -- | --- ---- | -----  --------")

all_ok = True
results = []
for A in (0, 1):
    for B in (0, 1):
        for Cin in (0, 1):
            S1 = XOR_gate(A, B)
            C1 = AND_gate(A, B)
            SUM = XOR_gate(S1, Cin)
            C2  = AND_gate(S1, Cin)
            Cout = OR_gate(C1, C2)
            total = 2*Cout + SUM
            expected = A + B + Cin
            ok = "OK" if total == expected else "FAIL"
            if total != expected: all_ok = False
            results.append((A, B, Cin, SUM, Cout, total, expected))
            print(f"  {A:>2} {B:>2} {Cin:>3} | {S1:>2} {C1:>2} {C2:>2} | {SUM:>3} {Cout:>4} | {total:>5} esperado {expected} [{ok}]")

print()
print(f"FULL-ADDER CORRECTO: {all_ok}")
if all_ok:
    print()
    print("Implicacion: dado que tenemos full-adder 1-bit verificado, podemos")
    print("encadenar N full-adders para sumar N-bits. Cualquier suma entera")
    print("binaria es ejecutable en esta CA.")
