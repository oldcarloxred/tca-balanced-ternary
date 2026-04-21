"""
multi_bit_adder.py
==================
Sumador de N bits: encadena N full-adders de 1 bit, propagando el carry.
Cada bit se computa en una simulacion CA real.

Demuestra que:
  - 0101 + 0011 = 01000  (5 + 3 = 8,    carry unico)
  - 0111 + 0001 = 01000  (7 + 1 = 8,    carry chain de 3)
  - 1111 + 0001 = 10000  (15 + 1 = 16,  overflow a 5-bit)
  - 1100 + 0101 = 10001  (12 + 5 = 17,  mixed)
  - 1010 + 0101 = 01111  (10 + 5 = 15,  sin carries)

Export: resumen JSON + tabla ascii.
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

def XOR_gate(A, B, steps=60):
    G = 40
    g = np.zeros((G,G), dtype=np.int8)
    c = G//2
    if A: place(g, NORTH, G-6, c)
    if B: place(g, SOUTH, 4, c)
    for _ in range(steps): g = _cpu_step(g, RT, G, G)
    return 1 if int((g != 0).sum()) > 0 else 0

def OR_gate(X, Y, steps=60):
    G = 50
    g = np.zeros((G,G), dtype=np.int8)
    if X: place(g, EAST, 10, 2)
    if Y: place(g, EAST, 30, 2)
    for _ in range(steps): g = _cpu_step(g, RT, G, G)
    return int(has_east(g))

def full_adder(A, B, Cin):
    S1 = XOR_gate(A, B)
    C1 = AND_gate(A, B)
    SUM = XOR_gate(S1, Cin)
    C2  = AND_gate(S1, Cin)
    Cout = OR_gate(C1, C2)
    return SUM, Cout

def add_nbits(A_val, B_val, n_bits=4):
    """Suma A+B encadenando n_bits full-adders. LSB primero."""
    A_bits = [(A_val >> i) & 1 for i in range(n_bits)]
    B_bits = [(B_val >> i) & 1 for i in range(n_bits)]
    sum_bits = []
    carry = 0
    carry_chain = []
    for i in range(n_bits):
        s, c_out = full_adder(A_bits[i], B_bits[i], carry)
        sum_bits.append(s)
        carry_chain.append((carry, c_out))  # (carry in, carry out) por bit
        carry = c_out
    # resultado: sum_bits (LSB a MSB) + carry final como bit n+1
    total = sum(b << i for i, b in enumerate(sum_bits)) + (carry << n_bits)
    return total, sum_bits, carry, carry_chain

# ---- Tests ----
print("=" * 70)
print("SUMADOR DE 4 BITS — cadena de 4 full-adders CA")
print("=" * 70)

tests = [
    (5, 3),    # 0101 + 0011 = 01000
    (7, 1),    # 0111 + 0001 = 01000
    (15, 1),   # 1111 + 0001 = 10000 (overflow)
    (12, 5),   # 1100 + 0101 = 10001
    (10, 5),   # 1010 + 0101 = 01111
    (6, 6),    # 0110 + 0110 = 01100
    (8, 8),    # 1000 + 1000 = 10000
    (13, 14),  # 1101 + 1110 = 11011
]

all_ok = True
results = []
for A, B in tests:
    total, bits, final_carry, chain = add_nbits(A, B, n_bits=4)
    expected = A + B
    ok = "OK" if total == expected else "FAIL"
    if total != expected: all_ok = False
    bits_str = ''.join(str(b) for b in bits[::-1])  # MSB primero
    full_result = f"{final_carry}{bits_str}"
    print(f"  {A:>3} + {B:>3} = {total:>3}  (bits = {full_result})  esperado {expected}  [{ok}]")
    results.append({
        "A": A, "B": B, "sum": total, "expected": expected,
        "bits_msb_first": full_result,
        "carry_chain": [{"bit": i, "cin": cin, "cout": cout} for i, (cin, cout) in enumerate(chain)]
    })

print()
print(f"TODOS CORRECTOS: {all_ok}")

out = Path(__file__).parent / "multi_bit_adder_results.json"
out.write_text(json.dumps({"tests": results, "all_ok": all_ok}, indent=2))
print(f"Resultado JSON: {out}")
