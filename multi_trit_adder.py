"""
multi_trit_adder.py
===================
Sumador multi-trit en balanced ternary, encadenando trit-adders CA.

Cada trit-adder es la primitiva nativa de verify_trit_adder.py:
  add(a, b) -> (digit, carry)   en una sola colision CA.

Full-trit-adder (con carry de entrada): 2 etapas encadenadas
  stage1: (d1, c1) = CA_add(a, b)
  stage2: (d2, c2) = CA_add(d1, cin)
  digit_out = d2
  carry_out = c1 + c2          (siempre en {-1,0,+1}, ver prueba en comentarios)

Sumamos numeros balanced-ternary de N trits y verificamos que coincide
con la suma entera.

Bonus: balanced ternary hace la NEGACION trivial (flip signs de cada trit).
Por tanto a - b = a + (-b) es GRATIS. Demostramos resta tambien.
"""
import numpy as np, sys, json
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

def count_nN(g):
    n=0
    for r in range(g.shape[0]-1):
        for c in range(g.shape[1]-1):
            if np.array_equal(g[r:r+2,c:c+2], NORTH): n+=1
    return n

def count_nS(g):
    n=0
    for r in range(g.shape[0]-1):
        for c in range(g.shape[1]-1):
            if np.array_equal(g[r:r+2,c:c+2], SOUTH): n+=1
    return n

G=60; COL=30

def trit_add_ca(a, b, steps=40):
    """Primitiva nativa: single-collision trit addition."""
    g = np.zeros((G,G), dtype=np.int8)
    if a==1:  place(g, NORTH, 40, COL)
    if a==-1: place(g, SOUTH, 40, COL)
    if b==1:  place(g, NORTH, 10, COL)
    if b==-1: place(g, SOUTH, 10, COL)
    for _ in range(steps): g = _cpu_step(g, RT, G, G)
    nN = count_nN(g); nS = count_nS(g)
    if (nN,nS)==(0,0): return (0, 0)
    if (nN,nS)==(1,0): return (+1, 0)
    if (nN,nS)==(0,1): return (-1, 0)
    if (nN,nS)==(2,0): return (-1, +1)
    if (nN,nS)==(0,2): return (+1, -1)
    raise RuntimeError(f"unexpected output ({nN},{nS}) for a={a} b={b}")

def trit_full_adder(a, b, cin):
    """Full trit adder: suma 3 trits, devuelve (digit, carry)."""
    d1, c1 = trit_add_ca(a, b)
    d2, c2 = trit_add_ca(d1, cin)
    return d2, c1 + c2

# ---- Representacion balanced-ternary ----
def int_to_bt(n, n_trits):
    """Convierte entero a lista de trits LSB-first, balanced ternary."""
    trits = []
    for _ in range(n_trits):
        r = n % 3
        if r == 2: r = -1; n += 3
        trits.append(r)
        n //= 3
    return trits

def bt_to_int(trits):
    return sum(t * (3**i) for i, t in enumerate(trits))

def add_bt_numbers(A, B, n_trits):
    """Suma balanced-ternary A+B usando la cadena de trit-full-adders CA."""
    a_tr = int_to_bt(A, n_trits)
    b_tr = int_to_bt(B, n_trits)
    out = []
    carry = 0
    for i in range(n_trits):
        d, cout = trit_full_adder(a_tr[i], b_tr[i], carry)
        out.append(d)
        carry = cout
    out.append(carry)  # overflow trit
    return bt_to_int(out), out, a_tr, b_tr

# ---- Tests ----
print("="*72)
print("SUMADOR MULTI-TRIT BALANCED-TERNARY — cadena de trit-full-adders CA")
print("="*72)

# Rango balanced ternary 4 trits: +-40
tests = [
    (5, 3),       # simple
    (13, -7),     # con negativos
    (-20, -15),   # dos negativos (carry negativo)
    (40, 1),      # overflow positivo (limite 4 trits: max=40)
    (-40, -1),    # overflow negativo
    (27, 14),     # dos positivos grandes
    (0, 0),       # trivial
    (21, -21),    # cancelacion
]

N_TRITS = 4
all_ok = True
results = []
for A, B in tests:
    total, out_trits, a_tr, b_tr = add_bt_numbers(A, B, N_TRITS)
    expected = A + B
    ok = total == expected
    if not ok: all_ok = False
    a_str = ''.join({-1:'T',0:'0',1:'1'}[t] for t in a_tr[::-1])
    b_str = ''.join({-1:'T',0:'0',1:'1'}[t] for t in b_tr[::-1])
    out_str = ''.join({-1:'T',0:'0',1:'1'}[t] for t in out_trits[::-1])
    mark = "OK" if ok else "FAIL"
    print(f"  {A:>+4} + {B:>+4} = {total:>+4} | bt: {a_str} + {b_str} = {out_str} | exp {expected:>+4} [{mark}]")
    results.append({"A":A,"B":B,"got":total,"expected":expected,"ok":ok,
                    "bt_A":a_str,"bt_B":b_str,"bt_result":out_str})

print()
print(f"TODAS CORRECTAS: {all_ok}")

# ---- BONUS: resta gratis via negacion trivial ----
print()
print("="*72)
print("BONUS: RESTA via NEGACION TRIVIAL (ventaja unica de balanced ternary)")
print("="*72)
print("  negar un trit = flip de signo. negar un numero = flip de cada trit.")
print("  => a - b = a + (-b), sin hardware adicional.")
print()

sub_tests = [(15, 7), (8, 20), (-5, -12), (30, 30)]
sub_ok = True
for A, B in sub_tests:
    # negar B: flip cada trit
    a_tr = int_to_bt(A, N_TRITS)
    b_tr = int_to_bt(B, N_TRITS)
    neg_b_tr = [-t for t in b_tr]
    # sumar a + neg_b usando CA
    out = []; carry = 0
    for i in range(N_TRITS):
        d, cout = trit_full_adder(a_tr[i], neg_b_tr[i], carry)
        out.append(d); carry = cout
    out.append(carry)
    total = bt_to_int(out)
    expected = A - B
    ok = total == expected
    if not ok: sub_ok = False
    mark = "OK" if ok else "FAIL"
    print(f"  {A:>+4} - {B:>+4} = {total:>+4} | esperado {expected:>+4} [{mark}]")

print()
print(f"RESTA CORRECTA: {sub_ok}")
print()
if all_ok and sub_ok:
    print("="*72)
    print("MULTI-TRIT BALANCED-TERNARY ALU OPERATIVA EN CA lambda=0.059")
    print("="*72)
    print("  + suma multi-trit con propagacion de carry")
    print("  + resta gratis via negacion trivial")
    print("  + todo ejecutado sobre primitiva nativa single-collision")

out = Path(__file__).parent / "multi_trit_adder_results.json"
out.write_text(json.dumps({"tests": results, "all_ok": all_ok, "sub_ok": sub_ok}, indent=2))
print(f"\nJSON: {out}")
