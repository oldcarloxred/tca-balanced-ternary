"""
gen_multi_trit_alu_gif.py
=========================
WOW visual: ALU balanced-ternary sumando 27 + 14 = 41 en la CA.

27 = 1000  (bt, trits MSB->LSB: 1 0 0 0)
14 = 1TTT  (bt:                 1 T T T)   (T = -1)
41 = 1TTTT (bt, overflow):      1 T T T T

Cada trit-full-adder usa 2 colisiones CA encadenadas (stage1: a+b, stage2: d1+cin).
Mostramos los 4 trits (LSB->MSB) en paneles horizontales, cada uno con stage1 y
stage2 superpuestos por animacion. Debajo, anotaciones textuales del resultado.

Output: multi_trit_alu.gif
"""
import numpy as np, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim

sys.path.insert(0, str(Path(__file__).parent))
from src.tca_sim.simulator import _cpu_step
from multi_trit_adder import int_to_bt, trit_add_ca

RT = np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
               0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,1,
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int8)

NORTH = np.array([[-1,-1],[1,1]], dtype=np.int8)
SOUTH = np.array([[1,1],[-1,-1]], dtype=np.int8)

def place(g, pat, r, c):
    for dr in range(pat.shape[0]):
        for dc in range(pat.shape[1]):
            g[(r+dr)%g.shape[0], (c+dc)%g.shape[1]] = pat[dr,dc]

G=60; COL=30; STEPS=40

def sim_trit_add(a, b):
    g = np.zeros((G,G), dtype=np.int8)
    if a==1:  place(g, NORTH, 40, COL)
    if a==-1: place(g, SOUTH, 40, COL)
    if b==1:  place(g, NORTH, 10, COL)
    if b==-1: place(g, SOUTH, 10, COL)
    frames=[g.copy()]
    for _ in range(STEPS):
        g = _cpu_step(g, RT, G, G)
        frames.append(g.copy())
    return frames

# ---- Inputs ----
A = 27; B = 14
N_TRITS = 4
a_tr = int_to_bt(A, N_TRITS)  # LSB first: [0,0,0,1]
b_tr = int_to_bt(B, N_TRITS)  # LSB first: [-1,-1,-1,1]

# Ejecutar cadena trit-full-adder y guardar frames de cada stage
carry = 0
stages = []  # por cada trit: {'a', 'b', 'cin', 'stage1_frames', 'stage2_frames', 'd', 'cout'}
for i in range(N_TRITS):
    a = a_tr[i]; b = b_tr[i]; cin = carry
    s1_frames = sim_trit_add(a, b)
    d1, c1 = trit_add_ca(a, b)
    s2_frames = sim_trit_add(d1, cin)
    d2, c2 = trit_add_ca(d1, cin)
    cout = c1 + c2
    stages.append({'i':i,'a':a,'b':b,'cin':cin,'d1':d1,'c1':c1,
                   'd':d2,'cout':cout,'s1':s1_frames,'s2':s2_frames})
    carry = cout
final_carry = carry
result_trits = [s['d'] for s in stages] + [final_carry]
result_int = sum(t * 3**i for i, t in enumerate(result_trits))
assert result_int == A + B

# ---- Plot: 2 filas (stage1, stage2) x 4 cols (trits MSB->LSB visual) ----
fig, axes = plt.subplots(2, 4, figsize=(14, 7.6))
fig.patch.set_facecolor('#0b0b14')
TSYM = {-1:'T', 0:'0', 1:'1'}
a_str = ''.join(TSYM[t] for t in a_tr[::-1])
b_str = ''.join(TSYM[t] for t in b_tr[::-1])
res_str = ''.join(TSYM[t] for t in result_trits[::-1])
title = (f"ALU balanced-ternary:  {A} + {B} = {result_int}   |   "
         f"{a_str} + {b_str} = {res_str}   (bt, MSB->LSB)")
fig.suptitle(title, fontsize=12.5, color='#5efc82', y=0.985)

# Total frames = 2*(STEPS+1): primer segmento stage1, segundo stage2
TOTAL = 2*(STEPS+1)

ims = []
for row_idx, stage_key in enumerate(['s1','s2']):
    stage_name = "stage1: a + b" if row_idx==0 else "stage2: d1 + cin"
    for col_idx, bit_i in enumerate([3,2,1,0]):  # MSB->LSB visual
        ax = axes[row_idx][col_idx]
        ax.set_facecolor('#1a1a24')
        ax.set_xticks([]); ax.set_yticks([])
        s = stages[bit_i]
        im = ax.imshow(s[stage_key][0], cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        if row_idx==0:
            ax.set_title(f"trit {bit_i} (3^{bit_i}):  a={s['a']:+d}  b={s['b']:+d}",
                         fontsize=10.5, color='#ffd24a', pad=5)
        else:
            ax.set_title(f"d1={s['d1']:+d}  cin={s['cin']:+d}  ->  d={s['d']:+d}  cout={s['cout']:+d}",
                         fontsize=10, color='#ffa0c0', pad=5)
        ims.append((im, bit_i, stage_key))

# Etiqueta de fila a la izquierda
for row_idx, lbl in enumerate(["STAGE 1:  a + b", "STAGE 2:  d1 + cin"]):
    fig.text(0.005, 0.72 - row_idx*0.45, lbl, rotation=90,
             va='center', ha='left', color='white', fontsize=10, family='monospace')

fig.text(0.5, 0.02,
         f"result trits (LSB->MSB): {[s['d'] for s in stages]}  +  final_carry = {final_carry}    "
         f"=>  {result_int} decimal",
         ha='center', fontsize=10.5, color='#5efc82', family='monospace')

def update(t):
    arts=[]
    if t <= STEPS:
        stage_active = 's1'; local_t = t
    else:
        stage_active = 's2'; local_t = t - (STEPS+1)
    for im, bit_i, skey in ims:
        s = stages[bit_i]
        # cada panel muestra SIEMPRE su stage propio; ambas filas avanzan juntas en t local
        if skey == 's1':
            im.set_data(s['s1'][min(t, STEPS)])
        else:
            # stage2 empieza a moverse solo cuando t > STEPS
            idx2 = max(0, t - (STEPS+1))
            im.set_data(s['s2'][min(idx2, STEPS)])
        arts.append(im)
    phase = "STAGE 1 (a+b)" if t <= STEPS else "STAGE 2 (d1+cin)"
    fig.suptitle(f"{title}   •  {phase}  step {t:>3}/{TOTAL-1}",
                 fontsize=12.5, color='#5efc82', y=0.985)
    return arts

plt.subplots_adjust(wspace=0.08, hspace=0.28, top=0.90, bottom=0.07, left=0.03, right=0.99)
ani = anim.FuncAnimation(fig, update, frames=TOTAL, interval=80, blit=False)
OUT = Path(__file__).parent / "multi_trit_alu.gif"
ani.save(str(OUT), writer='pillow', fps=11, dpi=72)
plt.close(fig)
print(f"GIF: {OUT}  ({OUT.stat().st_size//1024} KB)")
print(f"{A} + {B} = {result_int}   bt: {a_str} + {b_str} = {res_str}")
