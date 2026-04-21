"""CUDA kernel for 2D ternary outer-totalistic CA — toroidal boundary."""

kernel_code = """
__global__ void tca_update(
    const signed char* __restrict__ cur,
    signed char*       __restrict__ next,
    const signed char* __restrict__ rule_table,
    int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int sum = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            /* toroidal wrap */
            int nx = (x + dx + width)  % width;
            int ny = (y + dy + height) % height;
            sum += (int)cur[ny * pitch + nx];
        }
    }
    int state_idx = (int)cur[y * pitch + x] + 1;
    int sum_idx   = sum + 8;
    next[y * pitch + x] = rule_table[state_idx * 17 + sum_idx];
}
"""


def load_cuda_backend():
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit          # noqa: F401
        from pycuda.compiler import SourceModule

        mod    = SourceModule(kernel_code)
        kernel = mod.get_function("tca_update")
        return cuda, kernel, None
    except Exception as exc:
        return None, None, exc
