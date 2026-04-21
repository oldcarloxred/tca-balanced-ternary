"""
gen_half_adder_gif.py
=====================
GIF de 4x2 paneles:
  4 filas = (A,B) in {00,01,10,11}
  2 columnas = [CARRY subcircuit (AND)] | [SUM subcircuit (XOR via N+S)]

Muestra ambos sub-circuitos ejecutandose en paralelo (grids separados).
Resultado del bit A+B (2*CARRY + SUM) mostrado en la suptitle.
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
    for r in range(g.shape[0]-3):
        for c in range(g.shape[1]-2):
            if np.array_equal(g[r:r+4, c:c+3], EAST):
                return True
    return False

# CARRY (AND Paso 2)
GC = 60
STEPS_C = 120
def run_carry(A, B):
    g = np.zeros((GC, GC), dtype=np.int8)
    if A:
        place(g, EAST, 24, 2)
        place(g, NORTH, 41, 24)
    if B:
        place(g, SOUTH, 28, 24)
    frames = [g.copy()]
    for _ in range(STEPS_C):
        g = _cpu_step(g, RT, GC, GC)
        frames.append(g.copy())
    return frames, int(has_east(frames[-1]))

# SUM (N+S annihilation)
GS = 40
STEPS_S = 60
def run_sum(A, B):
    g = np.zeros((GS, GS), dtype=np.int8)
    cCol = GS // 2
    if A: place(g, NORTH, GS-6, cCol)
    if B: place(g, SOUTH, 4, cCol)
    frames = [g.copy()]
    for _ in range(STEPS_S):
        g = _cpu_step(g, RT, GS, GS)
        frames.append(g.copy())
    return frames, 1 if int((frames[-1] != 0).sum()) > 0 else 0

cases = [(0,0), (0,1), (1,0), (1,1)]

# Ejecutar y almacenar todos los frames
data = {}
for (A, B) in cases:
    fc, c = run_carry(A, B)
    fs, s = run_sum(A, B)
    data[(A,B)] = (fc, fs, c, s)
    total = 2*c + s
    print(f"  A={A} B={B} -> SUM={s} CARRY={c}  total={total}  (A+B={A+B})  {'OK' if total==A+B else 'FAIL'}")

# Animacion: 4 filas x 2 cols, cada panel avanza en sus propios frames.
# Alineamos: normalizamos a max(STEPS_C, STEPS_S) frames, saturando al final
# del sub-stream mas corto.
TOTAL_FRAMES = max(STEPS_C, STEPS_S) + 1

def frame_at(frames, t):
    return frames[min(t, len(frames)-1)]

fig, axes = plt.subplots(4, 2, figsize=(8, 14))
fig.suptitle('Half-Adder en la regla lambda=0.059', fontsize=13, y=0.995)

im_refs = []
for row_idx, (A, B) in enumerate(cases):
    fc, fs, c, s = data[(A,B)]
    total = 2*c + s

    axC = axes[row_idx, 0]
    axS = axes[row_idx, 1]
    axC.axis('off'); axS.axis('off')
    imC = axC.imshow(fc[0], cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    imS = axS.imshow(fs[0], cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    axC.set_title(f'A={A} B={B} : CARRY circuit = AND  ->  C={c}', fontsize=10)
    axS.set_title(f'A={A} B={B} : SUM circuit = XOR  ->  S={s}    (A+B={total})', fontsize=10)
    im_refs.append((imC, imS, fc, fs))

def update(t):
    artists = []
    for imC, imS, fc, fs in im_refs:
        imC.set_data(frame_at(fc, t))
        imS.set_data(frame_at(fs, t))
        artists.extend([imC, imS])
    fig.suptitle(f'Half-Adder (step={t})  |  CARRY = AND(A,B)  |  SUM = XOR(A,B)', fontsize=12, y=0.995)
    return artists

ani = anim.FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=80, blit=False)
OUT = Path(__file__).parent / 'half_adder_verified.gif'
ani.save(str(OUT), writer='pillow', fps=10, dpi=75)
plt.close(fig)
print(f"\nGIF: {OUT} ({OUT.stat().st_size//1024} KB)")
