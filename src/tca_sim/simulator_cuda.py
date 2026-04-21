"""
simulator_cuda.py
=================
GPU-accelerated CA simulator using CuPy vectorised operations.

Does NOT require the CUDA Toolkit / NVRTC — only the CuPy binary and
the CUDA driver.  Uses cp.roll + advanced indexing (all pre-compiled
CUDA kernels shipped with CuPy).

Neighbourhood sums
------------------
2D  Moore  8-cell:   sum in [-8,  +8]  → 17 values
3D  Moore 26-cell:   sum in [-26, +26] → 53 values

Rule table layout (identical to CPU version)
--------------------------------------------
2D:  rule[center_idx * 17 + (sum +  8)]  len = 51
3D:  rule[center_idx * 53 + (sum + 26)]  len = 159
     center_idx = state + 1   (0 = state -1, 1 = state 0, 2 = state +1)

Usage
-----
    from src.tca_sim.simulator_cuda import CUDASimulator
    sim  = CUDASimulator(dims=2)
    grid = sim.random_grid(128, 128)
    rt   = sim.random_rule()
    for _ in range(200):
        grid = sim.step(grid, rt)
    host = sim.to_host(grid)
"""
from __future__ import annotations
import os, numpy as np

# Auto-add CUDA Toolkit bin to DLL search path (Windows only)
# so nvrtc64_120_0.dll is found even when not in PATH.
def _fix_cuda_dll_path() -> None:
    import glob
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if not os.path.isdir(base):
        return
    for ver in sorted(glob.glob(f"{base}\\v12.*"), reverse=True):
        bin_dir = os.path.join(ver, "bin")
        if os.path.isdir(bin_dir):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass
            return

_fix_cuda_dll_path()

try:
    import cupy as cp
    CUPY_OK = True
except ImportError:
    CUPY_OK = False

# Neighbour offsets for 2D (8 offsets) and 3D (26 offsets)
_OFFSETS_2D = [
    (dy, dx)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dy == 0 and dx == 0)
]

_OFFSETS_3D = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


class CUDASimulator:
    """
    GPU-accelerated balanced-ternary CA simulator (2D or 3D).

    Parameters
    ----------
    dims : int
        2 or 3.
    """

    RULE_LEN = {2: 51, 3: 159}

    def __init__(self, dims: int = 2):
        if not CUPY_OK:
            raise RuntimeError("CuPy not installed.  pip install cupy-cuda12x")
        assert dims in (2, 3), "dims must be 2 or 3"
        self.dims    = dims
        self.offsets = _OFFSETS_2D if dims == 2 else _OFFSETS_3D
        # Pre-allocate nothing; CuPy handles GPU memory automatically

    # ── grid / rule helpers ───────────────────────────────────────

    def random_grid(self, *shape: int, rng=None) -> "cp.ndarray":
        if rng is None:
            rng = np.random.default_rng()
        host = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=shape)
        return cp.array(host)

    def grid_from_numpy(self, arr: np.ndarray) -> "cp.ndarray":
        return cp.array(np.asarray(arr, dtype=np.int8))

    def empty_grid(self, *shape: int) -> "cp.ndarray":
        return cp.zeros(shape, dtype=cp.int8)

    def sparse_seed_grid(self, *shape: int, radius: int = 5,
                         density: float = 0.4, rng=None) -> "cp.ndarray":
        """Empty grid with a small random cluster in the centre."""
        if rng is None:
            rng = np.random.default_rng()
        host = np.zeros(shape, dtype=np.int8)
        # Centre of last two dims
        cy = shape[-2] // 2
        cx = shape[-1] // 2
        patch = rng.choice(
            np.array([-1, 0, 1], dtype=np.int8),
            p=[density / 2, 1 - density, density / 2],
            size=(2 * radius + 1, 2 * radius + 1),
        )
        host[..., cy - radius: cy + radius + 1,
                   cx - radius: cx + radius + 1] = patch
        return cp.array(host)

    def random_rule(self, rng=None) -> "cp.ndarray":
        if rng is None:
            rng = np.random.default_rng()
        host = rng.choice(
            np.array([-1, 0, 1], dtype=np.int8),
            size=self.RULE_LEN[self.dims],
        )
        return cp.array(host)

    def rule_from_numpy(self, rt: np.ndarray) -> "cp.ndarray":
        return cp.array(np.asarray(rt, dtype=np.int8))

    def to_host(self, grid: "cp.ndarray") -> np.ndarray:
        return cp.asnumpy(grid)

    # ── single step ───────────────────────────────────────────────

    def step(self, grid: "cp.ndarray", rule: "cp.ndarray") -> "cp.ndarray":
        """One CA step on GPU using vectorised roll + gather."""
        # Neighbourhood sum (int16 to avoid overflow at ±26)
        nbr = cp.zeros(grid.shape, dtype=cp.int16)
        if self.dims == 2:
            for dy, dx in self.offsets:
                shifted = cp.roll(grid, dy, axis=0)
                shifted = cp.roll(shifted, dx, axis=1)
                nbr    += shifted.astype(cp.int16)
        else:
            for dz, dy, dx in self.offsets:
                shifted = cp.roll(grid, dz, axis=0)
                shifted = cp.roll(shifted, dy, axis=1)
                shifted = cp.roll(shifted, dx, axis=2)
                nbr    += shifted.astype(cp.int16)

        # Rule index: center_idx * stride + (sum + offset)
        stride = 17 if self.dims == 2 else 53
        offset =  8 if self.dims == 2 else 26
        ci  = grid.astype(cp.int32) + 1          # 0,1,2
        idx = ci * stride + (nbr.astype(cp.int32) + offset)
        return rule[idx].astype(cp.int8)

    # ── run loop ──────────────────────────────────────────────────

    def run(
        self,
        grid: "cp.ndarray",
        rule: "cp.ndarray",
        steps: int,
        record_every: int = 0,
    ) -> tuple["cp.ndarray", list[np.ndarray]]:
        """
        Run `steps` steps.  Returns (final_grid_gpu, host_snapshots).
        record_every=0 → no snapshots.
        """
        frames: list[np.ndarray] = []
        for i in range(steps):
            grid = self.step(grid, rule)
            if record_every > 0 and (i + 1) % record_every == 0:
                frames.append(self.to_host(grid))
        return grid, frames

    # ── static helpers ────────────────────────────────────────────

    @staticmethod
    def langton_lambda(rule: "cp.ndarray") -> float:
        rt = cp.asnumpy(rule)
        return float(np.count_nonzero(rt)) / len(rt)

    @staticmethod
    def rule_len(dims: int) -> int:
        return CUDASimulator.RULE_LEN[dims]


# ── benchmark ─────────────────────────────────────────────────────

def benchmark(dims: int = 2, grid_side: int = 256, steps: int = 500) -> None:
    import time
    sim  = CUDASimulator(dims=dims)
    rng  = np.random.default_rng(42)
    rule = sim.random_rule(rng)

    if dims == 2:
        grid  = sim.random_grid(grid_side, grid_side, rng=rng)
        cells = grid_side ** 2
    else:
        grid  = sim.random_grid(grid_side, grid_side, grid_side, rng=rng)
        cells = grid_side ** 3

    # warm-up (triggers any lazy cupy compilation)
    for _ in range(5):
        grid = sim.step(grid, rule)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        grid = sim.step(grid, rule)
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0

    ms_per_step  = dt / steps * 1000
    mcells_per_s = cells * steps / dt / 1e6
    print(f"  CUDA {dims}D  {grid_side}^{dims}  "
          f"{ms_per_step:.3f} ms/step  "
          f"{mcells_per_s:.1f} Mcells/s")


if __name__ == "__main__":
    print("=== CuPy Vectorised CA Benchmark ===")
    if not CUPY_OK:
        print("CuPy not available.")
    else:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"GPU: {gpu_name}\n")
        print("2-D benchmarks:")
        benchmark(dims=2, grid_side=128,  steps=1000)
        benchmark(dims=2, grid_side=256,  steps=500)
        benchmark(dims=2, grid_side=512,  steps=200)
        benchmark(dims=2, grid_side=1024, steps=100)
        print("\n3-D benchmarks:")
        benchmark(dims=3, grid_side=32,  steps=500)
        benchmark(dims=3, grid_side=64,  steps=200)
        benchmark(dims=3, grid_side=128, steps=50)
