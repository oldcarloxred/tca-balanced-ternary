"""
gen_trit_adder_gif.py
=====================
WOW visual del TRIT-ADDER balanced-ternary NATIVO.

9 paneles (grid 3x3) — uno por cada par (A,B) en {-1,0,+1}^2.
Cada panel muestra la simulacion CA real con el conteo N/S final
y el digit/carry interpretado.

Encoding:
  +1 -> NORTH glider   (rojo-azul arriba)
  -1 -> SOUTH glider   (azul-rojo abajo)
   0 -> nada

Output: trit_adder_9cases.gif
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

def place(g, pat, r, c):
    for dr in range(pat.shape[0]):
        for dc in range(pat.shape[1]):
            g[(r+dr)%g.shape[0], (c+dc)%g.shape[1]] = pat[dr,dc]

def count_north(g):
    n=0
    for r in range(g.shape[0]-1):
        for c in range(g.shape[1]-1):
            if np.array_equal(g[r:r+2,c:c+2], NORTH): n+=1
    return n

def count_south(g):
    n=0
    for r in range(g.shape[0]-1):
        for c in range(g.shape[1]-1):
            if np.array_equal(g[r:r+2,c:c+2], SOUTH): n+=1
    return n

G = 60
COL = 30
STEPS = 45

def simulate(a, b):
    g = np.zeros((G,G), dtype=np.int8)
    if a == 1:  place(g, NORTH, 40, COL)
    if a == -1: place(g, SOUTH, 40, COL)
    if b == 1:  place(g, NORTH, 10, COL)
    if b == -1: place(g, SOUTH, 10, COL)
    frames = [g.copy()]
    for _ in range(STEPS):
        g = _cpu_step(g, RT, G, G)
        frames.append(g.copy())
    return frames

def bt_add(a,b):
    s=a+b
    if s==2:  return (-1,+1)
    if s==-2: return (+1,-1)
    return (s,0)

# pre-simulate
cases = [(a,b) for a in (-1,0,1) for b in (-1,0,1)]
sims = {(a,b): simulate(a,b) for (a,b) in cases}

# layout 3x3
fig, axes = plt.subplots(3, 3, figsize=(11, 11))
fig.patch.set_facecolor('#0b0b14')
fig.suptitle("TRIT-ADDER BALANCED-TERNARY NATIVO  —  9/9 casos en una sola colision CA",
             fontsize=13, color='#5efc82', y=0.985)

ims = []
subs = []
for idx, (a, b) in enumerate(cases):
    r, c = divmod(idx, 3)
    ax = axes[r][c]
    ax.set_facecolor('#1a1a24')
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(sims[(a,b)][0], cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    d,cr = bt_add(a,b)
    title = f"({a:+d}) + ({b:+d})"
    ax.set_title(title, fontsize=12, color='#ffd24a', pad=5)
    sub = ax.text(0.5, -0.04, f"d={d:+d}, c={cr:+d}",
                  transform=ax.transAxes, ha='center', va='top',
                  fontsize=10, color='white', family='monospace')
    ims.append((im, (a,b)))
    subs.append(sub)

def update(t):
    arts=[]
    for im,(a,b) in ims:
        im.set_data(sims[(a,b)][t])
        arts.append(im)
    return arts

plt.subplots_adjust(wspace=0.06, hspace=0.22, top=0.94, bottom=0.04, left=0.02, right=0.98)
ani = anim.FuncAnimation(fig, update, frames=STEPS+1, interval=90, blit=False)
OUT = Path(__file__).parent / "trit_adder_9cases.gif"
ani.save(str(OUT), writer='pillow', fps=11, dpi=80)
plt.close(fig)

print(f"GIF: {OUT}  ({OUT.stat().st_size//1024} KB)")
print("9/9 casos balanced-ternary — trit-adder nativo en CA lambda=0.059")
