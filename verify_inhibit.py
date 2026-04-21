"""
verify_inhibit.py
=================
Verifica empíricamente la puerta INHIBIT: E+N colisión perpendicular.

Tabla de verdad esperada (output_E = E AND NOT(N)):
  E=1, N=0  ->  E sobrevive (pop contiene E a la derecha)
  E=1, N=1  ->  E muere (absorbed); N sobrevive (arriba)
  E=0, N=0  ->  vacío
  E=0, N=1  ->  N sobrevive sola

Busca el timing relativo (dt, dc_offset) que da absorción limpia:
  - E absorbido totalmente
  - N sin daño

No asume ningún timing del handoff; barre una grid y reporta las combinaciones
que producen INHIBIT limpio.
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

def has_east_glider(g):
    """Detecta si existe un patrón EAST intacto en el grid (forma 4x3)."""
    H, W = g.shape
    for r in range(H-3):
        for c in range(W-2):
            if np.array_equal(g[r:r+4, c:c+3], EAST):
                return True, (r, c)
    return False, None

def has_north_glider(g):
    H, W = g.shape
    for r in range(H-1):
        for c in range(W-1):
            if np.array_equal(g[r:r+2, c:c+2], NORTH):
                return True, (r, c)
    return False, None

def run(setup, G=50, steps=60):
    g = np.zeros((G, G), dtype=np.int8)
    setup(g, G)
    pops = [int((g != 0).sum())]
    for _ in range(steps):
        g = _cpu_step(g, RT, G, G)
        pops.append(int((g != 0).sum()))
    return g, pops

# Barrido: E va hacia la derecha desde (rE, cE_start),
# N va hacia arriba desde (rN_start, cN). Buscamos que sus caminos se crucen.
# rE y cN fijos; variamos (rN_start) y (cE_start) para cambiar timing relativo.

G = 50
# E crusa la fila rE=25. Su punta (cabeza) está en cE_start+2 y avanza +1/step.
# N cruza la columna cN=25. Su punta (parte de arriba) está en rN_start y avanza -1/step.
# Para que coincidan espacialmente: E llega a columna cN en t = cN - (cE_start+2).
# N llega a fila rE   en t = (rN_start+1) - rE  (N tiene altura 2, parte superior en rN_start).
# Delta t := t_E - t_N
#   = (cN - cE_start - 2) - (rN_start + 1 - rE)

rE, cN = 25, 25

def test_only_E():
    """E=1, N=0: E solo. Debe sobrevivir."""
    def s(g, G):
        place(g, EAST, rE-1, 5)  # rE-1 por la forma 4 de alto
    g_final, pops = run(s, G=G, steps=60)
    ok, _ = has_east_glider(g_final)
    return ok, g_final, pops

def test_only_N():
    """E=0, N=1: N sola. Debe sobrevivir."""
    def s(g, G):
        place(g, NORTH, 45, cN)
    g_final, pops = run(s, G=G, steps=60)
    ok, _ = has_north_glider(g_final)
    return ok, g_final, pops

def test_E_and_N(cE_start, rN_start, steps=80):
    """E=1, N=1 con timing concreto. ¿E muere, N sobrevive?"""
    def s(g, G):
        place(g, EAST, rE-1, cE_start)
        place(g, NORTH, rN_start, cN-1)  # cN-1 por el ancho 2 de N
    g_final, pops = run(s, G=G, steps=steps)
    east_ok, _ = has_east_glider(g_final)
    north_ok, _ = has_north_glider(g_final)
    return east_ok, north_ok, int((g_final != 0).sum()), max(pops)

print("=" * 70)
print("PASO 1: Sanity checks (solo E, solo N)")
print("=" * 70)
ok_e, _, _ = test_only_E()
ok_n, _, _ = test_only_N()
print(f"  E solo sobrevive 60 pasos: {ok_e}")
print(f"  N sola sobrevive 60 pasos: {ok_n}")

print()
print("=" * 70)
print("PASO 2: Barrido (cE_start, rN_start) buscando INHIBIT limpio")
print("       (E ausente, N presente, sin explosión)")
print("=" * 70)

results = []
for cE_start in range(2, 24):
    for rN_start in range(28, 48):
        e, n, final_pop, peak = test_E_and_N(cE_start, rN_start, steps=80)
        # delta t calculado
        dt = (cN - cE_start - 2) - (rN_start + 1 - rE)
        is_inhibit = (not e) and n and peak < 50  # sin explosión
        results.append((cE_start, rN_start, dt, e, n, final_pop, peak, is_inhibit))

clean = [r for r in results if r[-1]]
explode = [r for r in results if r[6] >= 50]
print(f"  Total configs: {len(results)}")
print(f"  Explotan (peak>=50): {len(explode)}")
print(f"  INHIBIT limpio (E=0, N=1, sin explosión): {len(clean)}")

if clean:
    print()
    print("  Primeras 15 configs INHIBIT limpias:")
    print(f"  {'cE':>4} {'rN':>4} {'dt':>4} {'E_out':>6} {'N_out':>6} {'peak':>5}")
    for r in clean[:15]:
        print(f"  {r[0]:>4} {r[1]:>4} {r[2]:>4} {str(r[3]):>6} {str(r[4]):>6} {r[6]:>5}")

print()
print("=" * 70)
print("PASO 3: Tabla de verdad INHIBIT en la PRIMERA config limpia")
print("=" * 70)

if clean:
    cE_best, rN_best = clean[0][0], clean[0][1]
    print(f"  Config: cE_start={cE_best}, rN_start={rN_best}, dt={clean[0][2]}")
    # E=1, N=0
    def s10(g, G):
        place(g, EAST, rE-1, cE_best)
    g10f, _ = run(s10, steps=80)
    e10 = has_east_glider(g10f)[0]; n10 = has_north_glider(g10f)[0]

    def s01(g, G):
        place(g, NORTH, rN_best, cN-1)
    g01f, _ = run(s01, steps=80)
    e01 = has_east_glider(g01f)[0]; n01 = has_north_glider(g01f)[0]

    def s11(g, G):
        place(g, EAST, rE-1, cE_best)
        place(g, NORTH, rN_best, cN-1)
    g11f, _ = run(s11, steps=80)
    e11 = has_east_glider(g11f)[0]; n11 = has_north_glider(g11f)[0]

    def s00(g, G):
        pass
    g00f, _ = run(s00, steps=80)
    e00 = has_east_glider(g00f)[0]; n00 = has_north_glider(g00f)[0]

    print(f"  E=0,N=0 -> E_out={e00} N_out={n00}  (esperado: 0,0)")
    print(f"  E=1,N=0 -> E_out={e10} N_out={n10}  (esperado: 1,0)  [E sobrevive]")
    print(f"  E=0,N=1 -> E_out={e01} N_out={n01}  (esperado: 0,1)  [N sobrevive]")
    print(f"  E=1,N=1 -> E_out={e11} N_out={n11}  (esperado: 0,1)  [INHIBIT]")

    truth_ok = (e00==False and n00==False and
                e10==True  and n10==False and
                e01==False and n01==True  and
                e11==False and n11==True)
    print()
    print(f"  INHIBIT VERIFICADO: {truth_ok}")
else:
    print("  NO SE ENCONTRÓ configuración INHIBIT limpia en el barrido.")
    print("  => El teorema NAND-complete del handoff NO es reproducible con estas primitivas.")
