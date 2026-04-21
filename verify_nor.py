"""
verify_nor.py
=============
Construye una puerta NOR directa usando 1 E (constante) + 2 N inhibidores.
NOR es funcionalmente universal (igual que NAND), con lo cual verificar NOR
constructivamente cierra el teorema de universalidad combinatoria.

Semantica:
  E_out = E ∧ ¬(A_N ∨ B_N) = NOR(A, B)  cuando E=1 constante

Tabla de verdad esperada:
  A=0, B=0 -> 1   (E sobrevive, nada le pega)
  A=0, B=1 -> 0   (B_N inhibe)
  A=1, B=0 -> 0   (A_N inhibe)
  A=1, B=1 -> 0   (ambos inhiben, E muere una vez)

Estrategia: A_N en la config canonica de Paso 1 (rAN=41, cAN=24). Barrer
B_N por columnas distintas (cBN != cAN) para que ataque a E en un punto
posterior de su trayectoria.
"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.tca_sim.simulator import _cpu_step

RT = np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
               0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,1,
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int8)

NORTH = np.array([[-1,-1],[1,1]], dtype=np.int8)
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

def run(setup, G, steps):
    g = np.zeros((G, G), dtype=np.int8)
    setup(g, G)
    peak = int((g != 0).sum())
    for _ in range(steps):
        g = _cpu_step(g, RT, G, G)
        p = int((g != 0).sum())
        if p > peak: peak = p
    return g, peak

G = 60
rE, cE_start = 25, 2
# A_N canonico
cAN, rAN_canonico = 24, 41

def build(A, B, cBN, rBN):
    def s(g, G):
        # E siempre presente (constante = 1)
        place(g, EAST, rE-1, cE_start)
        if A:
            place(g, NORTH, rAN_canonico, cAN)
        if B:
            place(g, NORTH, rBN, cBN)
    return s

def truth(cBN, rBN, steps=120):
    out = {}
    for A in (0, 1):
        for B in (0, 1):
            setup = build(A, B, cBN, rBN)
            g_final, peak = run(setup, G, steps)
            out[(A,B)] = (has_east(g_final), peak)
    return out

expected = {(0,0): True, (0,1): False, (1,0): False, (1,1): False}

# Primero verifico que B_N sola (cuando A=0, B=1) puede inhibir E.
# Para eso barrer (cBN, rBN) donde cBN != cAN.
print("=" * 70)
print("BARRIDO: buscar segundo inhibidor B_N que cumpla tabla NOR")
print("=" * 70)

clean = []
for cBN in range(28, 48):
    for rBN in range(32, 50):
        res = truth(cBN, rBN, steps=120)
        matches = all(res[k][0] == expected[k] for k in expected)
        peak = max(r[1] for r in res.values())
        if matches and peak < 50:
            clean.append((cBN, rBN, peak, res))

print(f"Configs con tabla NOR perfecta + sin explosion: {len(clean)}")

if clean:
    print()
    print("Primeras 10:")
    print(f"  {'cBN':>4} {'rBN':>4} {'peak':>5}")
    for c in clean[:10]:
        print(f"  {c[0]:>4} {c[1]:>4} {c[2]:>5}")

    cBN_b, rBN_b, peak_b, res_b = clean[0]
    print()
    print("=" * 70)
    print(f"TABLA DE VERDAD NOR en A_N=(41,24)  B_N=(rBN={rBN_b}, cBN={cBN_b})")
    print("=" * 70)
    for (A,B), (e_out, peak) in sorted(res_b.items()):
        exp = expected[(A,B)]
        ok = "OK" if e_out == exp else "FAIL"
        print(f"  A={A} B={B} -> E_out={int(e_out)}  esperado={int(exp)}  peak={peak}  [{ok}]")
    all_ok = all(res_b[k][0] == expected[k] for k in expected)
    print()
    print(f"  NOR VERIFICADO: {all_ok}")
    if all_ok:
        print()
        print("=" * 70)
        print("IMPLICACION: NOR-completo -> universalidad combinatoria constructiva")
        print("  NOT(A)   = NOR(A, A)")
        print("  OR(A,B)  = NOT(NOR(A, B)) = NOR(NOR(A,B), NOR(A,B))")
        print("  AND(A,B) = NOR(NOT A, NOT B)")
        print("=" * 70)
else:
    print()
    print("NO se encontro NOR limpio con solo 2 N gliders mas E.")
    print("Posible causa: en Paso 1 solo probamos cN=24. Otras columnas")
    print("pueden requerir ajuste de fila para estar en la trayectoria de E.")
