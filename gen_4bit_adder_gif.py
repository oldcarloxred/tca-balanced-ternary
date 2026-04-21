"""
gen_4bit_adder_gif.py
=====================
WOW visual: sumador de 4 bits de 11+5=16 computando bit-a-bit en la CA.

  A = 1011 (decimal 11)
  B = 0101 (decimal  5)
  ----------------------
  R =10000 (decimal 16)  <- 5-bit, overflow

Para cada bit i (de LSB a MSB), muestra el subcircuito AND(A_i, B_i)
(primitivo de Paso 2) animado. Cada bit tiene inputs distintos y por
tanto dinamica distinta. La propagacion del carry se muestra como
anotacion textual debajo de cada panel.

Output: adder_4bit_11+5.gif
"""
import numpy as np, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim

sys.path.insert(0, str(Path(__file__).parent))
from src.tca_sim.simulator import _cpu_step
from multi_bit_adder import full_adder

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

# ---- Entradas ----
A = 11  # 1011
B = 5   # 0101
N_BITS = 4

A_bits = [(A >> i) & 1 for i in range(N_BITS)]   # LSB-first: [1,1,0,1]
B_bits = [(B >> i) & 1 for i in range(N_BITS)]   # LSB-first: [1,0,1,0]

# Computar full-adder por bit (para obtener los SUM/Cout/Cin reales)
carry = 0
per_bit = []
for i in range(N_BITS):
    s, cout = full_adder(A_bits[i], B_bits[i], carry)
    per_bit.append({'i': i, 'A': A_bits[i], 'B': B_bits[i],
                    'Cin': carry, 'SUM': s, 'Cout': cout})
    carry = cout
final_carry = carry

expected = A + B
computed = sum(p['SUM'] << i for i, p in enumerate(per_bit)) + (final_carry << N_BITS)
assert computed == expected, f"Fallo: {computed} vs {expected}"

# ---- Simular el subcircuito AND(A_i, B_i) por bit ----
G = 60
STEPS = 110

def simulate_and(A_in, B_in):
    g = np.zeros((G, G), dtype=np.int8)
    if A_in:
        place(g, EAST, 24, 2)
        place(g, NORTH, 41, 24)
    if B_in:
        place(g, SOUTH, 28, 24)
    frames = [g.copy()]
    for _ in range(STEPS):
        g = _cpu_step(g, RT, G, G)
        frames.append(g.copy())
    return frames

bit_frames = {}
for p in per_bit:
    bit_frames[p['i']] = simulate_and(p['A'], p['B'])

# ---- Plot ----
# Layout: 1 fila x 4 cols, MSB izquierda (bit 3) -> LSB derecha (bit 0)
fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
fig.patch.set_facecolor('#0b0b14')
title_txt = f"11 + 5 = 16   |   0b1011 + 0b0101 = 0b10000   |   cadena de carry: " \
            + " -> ".join([str(p['Cin']) for p in per_bit] + [str(final_carry)])
fig.suptitle(title_txt, fontsize=11.5, color='white', y=0.97)

ims = []
labels = []
for panel_idx, bit_i in enumerate([3, 2, 1, 0]):   # MSB->LSB visual
    ax = axes[panel_idx]
    ax.set_facecolor('#1a1a24')
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(bit_frames[bit_i][0], cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')
    p = per_bit[bit_i]
    # Etiquetas
    title = f"bit {bit_i}  (2^{bit_i})"
    sub = f"A={p['A']}  B={p['B']}  Cin={p['Cin']}\nSUM={p['SUM']}  Cout={p['Cout']}"
    ax.set_title(title, fontsize=11, color='#ffd24a', pad=6)
    txt = ax.text(0.5, -0.08, sub, transform=ax.transAxes,
                  ha='center', va='top', fontsize=9.5, color='white',
                  family='monospace')
    ims.append((im, bit_i))
    labels.append(txt)

# Anotaciones grandes a la derecha del conjunto
fig.text(0.5, 0.01,
         f"result = {final_carry}{per_bit[3]['SUM']}{per_bit[2]['SUM']}{per_bit[1]['SUM']}{per_bit[0]['SUM']}  "
         f"(= {computed} decimal)    <<  overflow bit = {final_carry}",
         ha='center', fontsize=11, color='#5efc82', family='monospace')

def update(t):
    arts = []
    for im, bit_i in ims:
        im.set_data(bit_frames[bit_i][t])
        arts.append(im)
    fig.suptitle(f"{title_txt}   •  step {t:>3}/{STEPS}",
                 fontsize=11.5, color='white', y=0.97)
    return arts

plt.subplots_adjust(wspace=0.08, top=0.88, bottom=0.18, left=0.02, right=0.98)
ani = anim.FuncAnimation(fig, update, frames=STEPS+1, interval=80, blit=False)
OUT = Path(__file__).parent / "adder_4bit_11plus5.gif"
ani.save(str(OUT), writer='pillow', fps=10, dpi=80)
plt.close(fig)

print(f"GIF: {OUT}  ({OUT.stat().st_size//1024} KB)")
print(f"Resultado computado: {computed}  esperado: {expected}  OK={computed==expected}")
print()
print("Bit breakdown:")
for p in per_bit:
    print(f"  bit {p['i']}: A={p['A']} B={p['B']} Cin={p['Cin']} -> SUM={p['SUM']} Cout={p['Cout']}")
print(f"  final_carry = {final_carry}")
print(f"  result bits (MSB-first, with overflow): {final_carry}"
      + ''.join(str(p['SUM']) for p in per_bit[::-1]))
