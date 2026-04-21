"""TCA simulator with toroidal boundary conditions."""
import numpy as np

try:
    from numba import jit
except Exception:
    def jit(*_a, **_kw):
        def dec(fn): return fn
        return dec

from .gpu_kernels import load_cuda_backend


@jit(nopython=True, cache=True)
def _cpu_step(grid, rule_table, width, height):
    """Single CA step on CPU — toroidal Moore neighbourhood."""
    new_grid = np.zeros_like(grid)
    for y in range(height):
        for x in range(width):
            s = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    s += grid[(y + dy) % height, (x + dx) % width]
            state_idx = grid[y, x] + 1          # {-1,0,1} → {0,1,2}
            new_grid[y, x] = rule_table[state_idx * 17 + (s + 8)]
    return new_grid


class TCASimulator:
    def __init__(self, width: int = 256, height: int = 256, use_gpu: bool = True):
        self.width  = width
        self.height = height
        self.use_gpu = False
        self.cuda = None
        self.tca_update_kernel = None
        self.grid  = np.zeros((height, width), dtype=np.int8)
        self.pitch = width

        if use_gpu:
            cuda, kernel, error = load_cuda_backend()
            if cuda is None:
                print(f"GPU backend unavailable, using CPU: {error}")
            else:
                try:
                    self.cuda = cuda
                    self.tca_update_kernel = kernel
                    self.d_grid = cuda.mem_alloc(self.grid.nbytes)
                    self.d_next = cuda.mem_alloc(self.grid.nbytes)
                    self.d_rule = cuda.mem_alloc(51)
                    cuda.memcpy_htod(self.d_grid, self.grid)
                    self.use_gpu = True
                    print("Simulation backend: GPU (toroidal)")
                except Exception as exc:
                    self.cuda = None
                    self.tca_update_kernel = None
                    print(f"GPU allocation failed, using CPU: {exc}")

        if not self.use_gpu:
            print("Simulation backend: CPU (toroidal)")

        self.reset_random(seed=0)

    # ------------------------------------------------------------------
    def reset_random(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.grid = rng.choice([-1, 0, 1], size=(self.height, self.width)).astype(np.int8)
        if self.use_gpu:
            self.cuda.memcpy_htod(self.d_grid, self.grid)
            self.cuda.memcpy_htod(self.d_next, self.grid)

    def set_grid(self, grid: np.ndarray) -> None:
        """Inject an arbitrary grid (must match H×W)."""
        self.grid = grid.astype(np.int8)
        if self.use_gpu:
            self.cuda.memcpy_htod(self.d_grid, self.grid)
            self.cuda.memcpy_htod(self.d_next, self.grid)

    def step(self, rule_table: np.ndarray) -> None:
        if self.use_gpu:
            rt = rule_table.astype(np.int8)
            self.cuda.memcpy_htod(self.d_rule, rt)
            block = (16, 16, 1)
            grid  = ((self.width + 15) // 16, (self.height + 15) // 16)
            self.tca_update_kernel(
                self.d_grid, self.d_next, self.d_rule,
                np.int32(self.width), np.int32(self.height), np.int32(self.pitch),
                block=block, grid=grid,
            )
            self.d_grid, self.d_next = self.d_next, self.d_grid
        else:
            self.grid = _cpu_step(self.grid, rule_table, self.width, self.height)

    def get_grid(self) -> np.ndarray:
        if self.use_gpu:
            out = np.empty_like(self.grid)
            self.cuda.memcpy_dtoh(out, self.d_grid)
            return out
        return self.grid.copy()
