"""
verify_half_adder.py
====================
Half-adder de 1 bit: dado (A, B), computa SUM = A XOR B, CARRY = A AND B.

Arquitectura:
  CARRY  = AND(A, B)    <- usamos el circuito del Paso 2
  SUM    = XOR(A, B)    <- colision N+S offset=0, la salida es
                          "cualquier glider sobrevive" (simetrico):
                            A=1,B=0 -> N pasa  -> pop > 0 -> 1
                            A=0,B=1 -> S pasa  -> pop > 0 -> 1
                            A=1,B=1 -> annihil -> pop = 0 -> 0
                            A=0,B=0 -> nada    -> pop = 0 -> 0

Aritmetica verificable:
  0+0 = 0   (CARRY=0, SUM=0)
  0+1 = 1   (CARRY=0, SUM=1)
  1+0 = 1   (CARRY=0, SUM=1)
  1+1 = 2   (CARRY=1, SUM=0)  <- binario 10

Cada subcircuito en su propio grid (misma regla). La composicion
al mismo grid se intentara despues si este paso va limpio.
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

def has_east(g):
    H, W = g.shape
    for r in range(H-3):
        for c in range(W-2):
            if np.array_equal(g[r:r+4, c:c+3], EAST):
                return True
    return False

# -------- CARRY = AND (config Paso 2 canonica) --------
def carry(A, B, steps=120):
    G = 60
    g = np.zeros((G, G), dtype=np.int8)
    if A:
        place(g, EAST, 24, 2)      # A_E
        place(g, NORTH, 41, 24)    # A_N
    if B:
        place(g, SOUTH, 28, 24)    # B_S
    for _ in range(steps):
        g = _cpu_step(g, RT, G, G)
    return int(has_east(g))

# -------- SUM = XOR via N+S annihilation --------
def sum_bit(A, B, steps=60):
    # Grid aislado, suficiente para que la colision se resuelva y
    # el superviviente se aleje. Usamos un toroide pequeno pero con
    # suficiente holgura vertical para que la S-surviving y N-surviving
    # no se encuentren consigo mismas por wrap en pocos pasos.
    G = 40
    g = np.zeros((G, G), dtype=np.int8)
    cCol = G // 2
    # N viene de abajo y sube; S viene de arriba y baja. Meeting ~row G/2.
    if A:
        place(g, NORTH, G - 6, cCol)   # N glider cerca del fondo
    if B:
        place(g, SOUTH, 4, cCol)       # S glider cerca del techo
    for _ in range(steps):
        g = _cpu_step(g, RT, G, G)
    # Cualquier celda no cero post-transitorio indica "glider sobrevivio"
    pop = int((g != 0).sum())
    return 1 if pop > 0 else 0

# -------- Tabla de verdad half-adder --------
print("=" * 60)
print("HALF-ADDER  (A + B = 2*CARRY + SUM)")
print("=" * 60)
print(f"  {'A':>2} {'B':>2} | {'SUM':>3} {'CARRY':>5} | {'total':>5}  esperado")
print(f"  {'-'*2} {'-'*2} | {'-'*3} {'-'*5} | {'-'*5}  --------")

all_ok = True
for A in (0, 1):
    for B in (0, 1):
        s = sum_bit(A, B)
        c = carry(A, B)
        total = 2*c + s
        expected = A + B
        ok = "OK" if total == expected else "FAIL"
        if total != expected:
            all_ok = False
        print(f"  {A:>2} {B:>2} | {s:>3} {c:>5} | {total:>5}  esperado {expected}  [{ok}]")

print()
print(f"HALF-ADDER CORRECTO: {all_ok}")
