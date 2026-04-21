"""
gen_and_circuit_gif.py
======================
Genera un GIF reproducible de la puerta AND real construida vía cascade
ANDNOT -> INHIBIT. Reemplaza el showcase_09_and_circuit.gif original
(huerfano, sin script).

Layout: 4 subplots lado a lado mostrando los 4 inputs de la tabla de verdad.
Etiquetado con el valor esperado y el resultado observado.

Output: and_circuit_verified.gif
"""
import numpy as np, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim

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

# Config canonica verificada en verify_and.py
G = 60
rE, cE_start = 25, 2
cCol = 24
rAN, rBS = 41, 28
STEPS = 90

def build_initial(A, B):
    g = np.zeros((G, G), dtype=np.int8)
    if A:
        place(g, EAST, rE-1, cE_start)
        place(g, NORTH, rAN, cCol)
    if B:
        place(g, SOUTH, rBS, cCol)
    return g

# Simular los 4 casos
cases = [(0,0), (0,1), (1,0), (1,1)]
case_frames = {}
case_final = {}
for (A, B) in cases:
    g = build_initial(A, B)
    frames = [g.copy()]
    for _ in range(STEPS):
        g = _cpu_step(g, RT, G, G)
        frames.append(g.copy())
    case_frames[(A,B)] = frames
    case_final[(A,B)] = int(has_east(g))

# Panel 2x2
fig, axes = plt.subplots(2, 2, figsize=(9, 9))
fig.suptitle('AND = INHIBIT(A_E, ANDNOT(A_N, B_S))  — verified circuit',
             fontsize=12)

axes_flat = axes.flatten()
ims = []
for idx, (A, B) in enumerate(cases):
    ax = axes_flat[idx]
    ax.axis('off')
    expected = int(A and B)
    obs = case_final[(A,B)]
    title = f'A={A}, B={B}  ->  AND={obs}  (expected {expected})'
    im = ax.imshow(case_frames[(A,B)][0], cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ims.append((im, ax, A, B))

def update(t):
    artists = []
    for im, ax, A, B in ims:
        im.set_data(case_frames[(A,B)][t])
        artists.append(im)
    fig.suptitle(f'AND = INHIBIT(A_E, ANDNOT(A_N, B_S))  — step={t}',
                 fontsize=12)
    return artists

ani = anim.FuncAnimation(fig, update, frames=len(case_frames[(0,0)]),
                        interval=120, blit=False)

OUT = Path(__file__).parent / 'and_circuit_verified.gif'
ani.save(str(OUT), writer='pillow', fps=8, dpi=80)
plt.close(fig)

print(f"Generado: {OUT}")
print(f"Tamaño: {OUT.stat().st_size} bytes")
print()
print("Resultados por caso:")
for (A,B) in cases:
    print(f"  A={A}, B={B} -> AND={case_final[(A,B)]}  (esperado {int(A and B)})")
print()
ok = all(case_final[(A,B)] == (A and B) for (A,B) in cases)
print(f"TODOS LOS CASOS CORRECTOS: {ok}")
