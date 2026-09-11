#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CUDA_OK(expr) do { auto e = (expr); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); std::exit(1); } } while (0)

// b16 data use integer tags so every source element is distinguishable.
template<int N, bool Swizzle, bool Trans = false>
__global__ void load_tile(uint32_t* out) {
  __shared__ __align__(128) uint16_t smem[16 * 64];
  const int lane = threadIdx.x;
  for (int i = lane; i < 16 * 64; i += 32) {
    const int r = i / 64, c = i % 64;
    const int p = Swizzle ? r * 64 + (c ^ ((r & 7) << 3)) : i;
    smem[p] = static_cast<uint16_t>(i);
  }
  __syncthreads();
  const int r = lane & 15;
  const int c = (lane >> 4) * 8;
  const int p = r * 64 + (Swizzle ? c ^ ((r & 7) << 3) : c);
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + p));
  uint32_t d[4];
  if constexpr (N == 1 && !Trans) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(d[0]) : "r"(addr) : "memory");
  } else if constexpr (N == 1 && Trans) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(d[0]) : "r"(addr) : "memory");
  } else if constexpr (N == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(d[0]), "=r"(d[1]) : "r"(addr) : "memory");
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr) : "memory");
  }
  for (int m = 0; m < N; ++m) out[lane * N + m] = d[m];
}

template<int N, bool Swizzle, bool Trans = false>
void run(uint32_t* device) {
  load_tile<N, Swizzle, Trans><<<1, 32>>>(device);
  CUDA_OK(cudaGetLastError());
  uint32_t host[128];
  CUDA_OK(cudaMemcpy(host, device, 32*N*sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (int lane = 0; lane < 32; ++lane) for (int m = 0; m < N; ++m) {
    const int g = lane/4, t = lane%4;
    const int r = (m%2)*8 + (Trans ? 2*t : g);
    const int c = (m/2)*8 + (Trans ? g : 2*t);
    const uint32_t lo = r*64+c, hi = Trans ? (r+1)*64+c : r*64+c+1;
    const uint32_t expected = lo | (hi << 16);
    if (host[lane*N+m] != expected) {
      std::fprintf(stderr, "FAIL N=%d swizzle=%d trans=%d lane=%d m=%d expected=%u actual=%u\n", N, Swizzle, Trans, lane, m, expected, host[lane*N+m]);
      std::exit(2);
    }
  }
  std::printf("PASS x%d swizzle=%d trans=%d: all %d register values match\n", N, Swizzle, Trans, 32*N);
}

int main() {
  uint32_t* device;
  CUDA_OK(cudaMalloc(&device, 128*sizeof(uint32_t)));
  run<1, false>(device); run<1, true>(device);
  run<2, false>(device); run<2, true>(device);
  run<4, false>(device); run<4, true>(device);
  run<1, false, true>(device); run<1, true, true>(device);
  CUDA_OK(cudaFree(device));
}
