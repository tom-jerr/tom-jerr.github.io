#include "ffi_common.h"

#include <cute/tensor.hpp>

#include <cmath>
#include <cstdint>

using tvm::ffi::TensorView;

namespace cutlass3_fa2 {

using namespace cute;

using Element = half_t;
using Accumulator = float;
// blockm 64, blockn 64, head_dim 64
using BlockM = Int<64>;
using BlockN = Int<64>;
using HeadDim = Int<64>;
constexpr int kDefaultStages = 2;

using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
// 4 个 warp 全部分布在 M 方向, FA2 algorithm 创新
using TiledMma = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_4, _1, _1>>{},
                                         Tile<_64, _64, _16>{}));

// XOR-swizzled shared layouts. K is consumed as a normal LDSM operand; V is
// exposed as logical (D,N) and consumed with the transposed LDSM atom.
using SmemRowAtom = decltype(composition(
    Swizzle<3, 3, 3>{}, Layout<Shape<_8, HeadDim>, Stride<HeadDim, _1>>{}));
using SmemColAtom = decltype(composition(
    Swizzle<3, 3, 3>{}, Layout<Shape<HeadDim, _8>, Stride<_1, HeadDim>>{}));
using SmemLayoutQ =
    decltype(tile_to_shape(SmemRowAtom{}, make_shape(BlockM{}, HeadDim{})));
using SmemLayoutO = SmemLayoutQ;

template <int kStages> struct SmemConfig {
  static_assert(kStages >= 2, "the cp.async pipeline needs at least 2 stages");

  using LayoutK = decltype(tile_to_shape(
      SmemRowAtom{}, make_shape(BlockN{}, HeadDim{}, Int<kStages>{})));
  using LayoutV = decltype(tile_to_shape(
      SmemColAtom{}, make_shape(HeadDim{}, BlockN{}, Int<kStages>{})));

  static constexpr int kElements =
      cosize(SmemLayoutQ{}) + cosize(LayoutK{}) + cosize(LayoutV{});
  static constexpr int kBytes = kElements * sizeof(Element);
};

// One transaction moves eight FP16 values (16 bytes).
using G2SAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>;
using G2SRow = decltype(make_tiled_copy(
    G2SAtom{}, Layout<Shape<_32, _4>, Stride<_4, _1>>{},
    Layout<Shape<_1, _8>>{})); // copy tile 是 32×32, 64×64 tile
                               // 在两个方向各重复两次
using G2SCol = decltype(make_tiled_copy(
    G2SAtom{}, Layout<Shape<_4, _32>, Stride<_1, _4>>{},
    Layout<Shape<_8, _1>>{}));
using S2RAtomN = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using S2RAtomT = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
using S2GAtom = Copy_Atom<UniversalCopy<uint128_t>, Element>;
using S2GRow = decltype(make_tiled_copy(
    S2GAtom{}, Layout<Shape<_32, _4>, Stride<_4, _1>>{},
    Layout<Shape<_1, _8>>{}));

constexpr int kThreads = 128;
constexpr int kTile = 64;
constexpr int kHeadDim = 64;

static_assert(size(TiledMma{}) == kThreads);
static_assert(SmemConfig<2>::kBytes == 40 * 1024);

__device__ __forceinline__ float subgroup4_max(float value) {
#pragma unroll
  for (int delta = 1; delta < 4; delta <<= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, delta, 4));
  }
  return value;
}

__device__ __forceinline__ float subgroup4_sum(float value) {
#pragma unroll
  for (int delta = 1; delta < 4; delta <<= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, delta, 4);
  }
  return value;
}

template <class Copy, class Src, class Dst>
__device__ __forceinline__ void copy_kv_tile(Copy const &copy_op,
                                             Src const &src, Dst const &dst,
                                             int tile, int stage) {
  copy(copy_op, src(_, _, _, tile), dst(_, _, _, stage));
}

template <bool Causal, int kStages>
__global__ __launch_bounds__(kThreads, 2) void flash_attn_cutlass3_kernel(
    const Element *__restrict__ q, const Element *__restrict__ k,
    const Element *__restrict__ v, Element *__restrict__ out, int heads,
    int seqlen) {
  using SmemLayoutK = typename SmemConfig<kStages>::LayoutK;
  using SmemLayoutV = typename SmemConfig<kStages>::LayoutV;

  extern __shared__ __align__(16) Element smem[];
  Element *q_smem = smem;
  Element *k_smem = q_smem + cosize(SmemLayoutQ{});
  Element *v_smem = k_smem + cosize(SmemLayoutK{});

  const int q_block = int(blockIdx.x);
  const size_t bh_base =
      (size_t(blockIdx.z) * heads + blockIdx.y) * size_t(seqlen) * kHeadDim;

  Tensor mQ =
      make_tensor(make_gmem_ptr(q + bh_base), make_shape(seqlen, HeadDim{}),
                  make_stride(HeadDim{}, _1{}));
  Tensor mK =
      make_tensor(make_gmem_ptr(k + bh_base), make_shape(seqlen, HeadDim{}),
                  make_stride(HeadDim{}, _1{}));
  // Logical V^T[D,N], physical V[N,D].
  Tensor mVt =
      make_tensor(make_gmem_ptr(v + bh_base), make_shape(HeadDim{}, seqlen),
                  make_stride(_1{}, HeadDim{}));
  Tensor mO =
      make_tensor(make_gmem_ptr(out + bh_base), make_shape(seqlen, HeadDim{}),
                  make_stride(HeadDim{}, _1{}));

  Tensor gQ =
      local_tile(mQ, make_tile(BlockM{}, HeadDim{}), make_coord(q_block, 0));
  Tensor gK = local_tile(mK, make_tile(BlockN{}, HeadDim{}),
                         make_coord(_, 0)); // gK shape ≈ (64,64,num_kv_tiles)
  Tensor gVt =
      local_tile(mVt, make_tile(HeadDim{}, BlockN{}), make_coord(0, _));
  Tensor gO =
      local_tile(mO, make_tile(BlockM{}, HeadDim{}), make_coord(q_block, 0));

  Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(k_smem), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(v_smem), SmemLayoutV{});

  G2SRow g2s_row;
  auto g2s_row_thr = g2s_row.get_slice(threadIdx.x);
  /**
   * tQgQ : (8,2,2) (per-thread val, CPY_M, CPY_N)
   * tKgK : (8,2,2,kv_tile) (per-thread val, CPY_M, CPY_N, K_TILE)
   * tQsQ : (8,2,2)
   * tKsK : (8,2,2,stage)
   */
  Tensor tQgQ = g2s_row_thr.partition_S(gQ);
  Tensor tQsQ = g2s_row_thr.partition_D(sQ);
  Tensor tKgK = g2s_row_thr.partition_S(gK);
  Tensor tKsK = g2s_row_thr.partition_D(sK);

  G2SCol g2s_col;
  auto g2s_col_thr = g2s_col.get_slice(threadIdx.x);
  Tensor tVgV = g2s_col_thr.partition_S(gVt);
  Tensor tVsV = g2s_col_thr.partition_D(sV);

  const int kv_tiles = Causal ? q_block + 1 : seqlen / kTile;

  // Pipeline prologue: Q is invariant. Fill kStages - 1 K/V stages, leaving
  // one stage free so the mainloop can prefetch while consuming stage zero.
  copy(g2s_row, tQgQ, tQsQ);
  const int prologue_tiles = kv_tiles < kStages - 1 ? kv_tiles : kStages - 1;
#pragma unroll
  for (int tile = 0; tile < prologue_tiles; ++tile) {
    copy_kv_tile(g2s_row, tKgK, tKsK, tile, tile);
    copy_kv_tile(g2s_col, tVgV, tVsV, tile, tile);
    cp_async_fence();
  }
  cp_async_wait<0>();
  __syncthreads();

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  /**
   * @brief trQ shape ≈ (8, 1, 4)
             │  │  └─ 64/16 = 4 个 K MMA
             │  └──── 每个 warp 在 M 上不再重复
             └─────── MMA A atom 每 lane 的 8 个 half
   */
  Tensor trQ = thr_mma.partition_fragment_A(gQ);

  // s2r 依赖 mma:
  // S2RAtomN: 决定用 ldmatrix，及该指令的 source/destination value 排列
  // tiled_mma.get_layoutA_TV(): 决定最终哪个 thread、哪个 value 属于 MMA A
  // operand auto q_s2r =
  auto q_s2r = make_tiled_copy_A(S2RAtomN{}, tiled_mma);
  auto q_s2r_thr = q_s2r.get_slice(threadIdx.x);
  Tensor tQsQ_mma = q_s2r_thr.partition_S(sQ);
  Tensor tQrQ = q_s2r_thr.retile_D(trQ);
  copy(q_s2r, tQsQ_mma, tQrQ);

  auto score_layout =
      make_layout(make_shape(BlockM{}, BlockN{}), make_stride(BlockN{}, _1{}));
  auto score_tensor = make_tensor(
      make_gmem_ptr(static_cast<Accumulator *>(nullptr)), score_layout);
  /**
   * @brief trS: ((2,2), 1, 8)
                  │ │   │  │
                  │ │   │  └─ 8 个 N 方向 MMA
                  │ │   └──── M repeat，当前配置只有 1
                  │ └──────── 当前 lane 的两个 M row
                  └────────── 每行相邻的两个 N column
   */
  Tensor trS = thr_mma.partition_fragment_C(score_tensor);
  Tensor trO = thr_mma.partition_fragment_C(gO);
  clear(trO);

  // Compile-time C-fragment -> A-fragment re-layout for P @ V.
  auto compact_score = make_tensor(static_cast<Element *>(nullptr),
                                   make_layout(make_shape(BlockM{}, BlockN{})));
  auto layout_as_c = thr_mma.partition_C(compact_score).layout();
  auto layout_as_a = thr_mma.partition_A(compact_score).layout();
  auto a_to_c = left_inverse(layout_as_c).compose(layout_as_a);

  auto k_s2r = make_tiled_copy_B(S2RAtomN{}, tiled_mma);
  auto k_s2r_thr = k_s2r.get_slice(threadIdx.x);
  Tensor tKsK_mma = k_s2r_thr.partition_S(sK);
  auto v_s2r = make_tiled_copy_B(S2RAtomT{}, tiled_mma);
  auto v_s2r_thr = v_s2r.get_slice(threadIdx.x);
  Tensor tVsV_mma = v_s2r_thr.partition_S(sV);

  float running_max[2] = {-INFINITY, -INFINITY};
  float running_sum[2] = {0.0f, 0.0f};
  constexpr float kScale = 1.0f / 8.0f;
  const int lane = int(threadIdx.x) & 31;
  const int warp = int(threadIdx.x) >> 5;
  const int row0 = warp * 16 + lane / 4;
  const int row1 = row0 + 8;
  const int lane_col = lane % 4;

#pragma unroll 1
  for (int tile = 0; tile < kv_tiles; ++tile) {
    const int next_tile = tile + 1;
    const int read_stage = tile % kStages;
    const int prefetch_tile = tile + kStages - 1;

    // Keep kStages - 1 tiles between producer and consumer. The destination
    // is a circular shared-memory buffer; this stage was last consumed by
    // tile - 1, so it is safe to reuse here.
    if (prefetch_tile < kv_tiles) {
      const int write_stage = prefetch_tile % kStages;
      copy_kv_tile(g2s_row, tKgK, tKsK, prefetch_tile, write_stage);
      copy_kv_tile(g2s_col, tVgV, tVsV, prefetch_tile, write_stage);
      cp_async_fence();
    }

    Tensor trK = thr_mma.partition_fragment_B(gK(_, _, 0));
    Tensor tKrK = k_s2r_thr.retile_D(trK);
    copy(k_s2r, tKsK_mma(_, _, _, read_stage), tKrK);
    clear(trS);
    gemm(tiled_mma, trQ, trK, trS); // QK mainloop

    // 一个 thread 同时负责两个 query row
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
      for (int row_item = 0; row_item < 2; ++row_item) {
#pragma unroll
        for (int col_item = 0; col_item < 2; ++col_item) {
          if constexpr (Causal) {
            if (tile == q_block) {
              const int query_row = row_item == 0 ? row0 : row1;
              const int key_col = nj * 8 + lane_col * 2 + col_item;
              if (key_col > query_row) {
                trS(make_coord(col_item, row_item), 0, nj) = -INFINITY;
              }
            }
          }
        }
        tile_max[row_item] = fmaxf(tile_max[row_item],
                                   fmaxf(trS(make_coord(0, row_item), 0, nj),
                                         trS(make_coord(1, row_item), 0, nj)));
      }
    }
    tile_max[0] = subgroup4_max(tile_max[0]) * kScale;
    tile_max[1] = subgroup4_max(tile_max[1]) * kScale;

    const float new_max0 = fmaxf(running_max[0], tile_max[0]);
    const float new_max1 = fmaxf(running_max[1], tile_max[1]);
    const float alpha0 =
        running_sum[0] == 0.0f ? 0.0f : __expf(running_max[0] - new_max0);
    const float alpha1 =
        running_sum[1] == 0.0f ? 0.0f : __expf(running_max[1] - new_max1);

    float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
      for (int row_item = 0; row_item < 2; ++row_item) {
#pragma unroll
        for (int col_item = 0; col_item < 2; ++col_item) {
          const float shifted =
              kScale * trS(make_coord(col_item, row_item), 0, nj) -
              (row_item == 0 ? new_max0 : new_max1);
          const float probability = __expf(shifted);
          trS(make_coord(col_item, row_item), 0, nj) = probability;
          tile_sum[row_item] += probability;
        }
      }
    }
    tile_sum[0] = subgroup4_sum(tile_sum[0]);
    tile_sum[1] = subgroup4_sum(tile_sum[1]);
    running_sum[0] = alpha0 * running_sum[0] + tile_sum[0];
    running_sum[1] = alpha1 * running_sum[1] + tile_sum[1];
    running_max[0] = new_max0;
    running_max[1] = new_max1;

    // Online-softmax correction rescales the old PV numerator.
#pragma unroll
    for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
      for (int col_item = 0; col_item < 2; ++col_item) {
        trO(make_coord(col_item, 0), 0, nj) *= alpha0;
        trO(make_coord(col_item, 1), 0, nj) *= alpha1;
      }
    }

    Tensor trP_as_c = make_tensor_like<Element>(trS);
#pragma unroll
    for (int i = 0; i < size(trS); ++i) {
      trP_as_c(i) = Element(trS(i));
    }
    // layout_as_c ∘ (layout_as_c⁻¹ ∘ layout_as_a) = layout_as_a
    auto trP_as_a = trP_as_c.compose(a_to_c);

    Tensor trV = thr_mma.partition_fragment_B(gVt(_, _, 0));
    Tensor tVrV = v_s2r_thr.retile_D(trV);
    copy(v_s2r, tVsV_mma(_, _, _, read_stage), tVrV);
    gemm(tiled_mma, trP_as_a, trV, trO); // PV mainloop

    if (next_tile < kv_tiles) {
      if (prefetch_tile < kv_tiles) {
        cp_async_wait<kStages - 2>();
      } else {
        // Drain the tail: without a new producer group, wait_group<kStages-2>
        // may legally return while the final prefetched tile is still pending.
        cp_async_wait<0>();
      }
      __syncthreads();
    }
  }

  // Attention epilogue: normalize FP32 O, convert to FP16, scatter the MMA C
  // fragment into swizzled shared memory, then issue coalesced 128-bit stores.
  const float inv_sum0 = 1.0f / running_sum[0];
  const float inv_sum1 = 1.0f / running_sum[1];
  Tensor trO_half = make_tensor_like<Element>(trO);
#pragma unroll
  for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
    for (int col_item = 0; col_item < 2; ++col_item) {
      trO_half(make_coord(col_item, 0), 0, nj) =
          Element(trO(make_coord(col_item, 0), 0, nj) * inv_sum0);
      trO_half(make_coord(col_item, 1), 0, nj) =
          Element(trO(make_coord(col_item, 1), 0, nj) * inv_sum1);
    }
  }

  Tensor sO = make_tensor(make_smem_ptr(q_smem), SmemLayoutO{});
  Tensor tOsO = thr_mma.partition_C(sO);
  copy(trO_half, tOsO);
  __syncthreads();

  S2GRow s2g;
  auto s2g_thr = s2g.get_slice(threadIdx.x);
  Tensor tOsO_vec = s2g_thr.partition_S(sO);
  Tensor tOgO_vec = s2g_thr.partition_D(gO);
  copy(s2g, tOsO_vec, tOgO_vec);
}

void check_fp16_tensor(const TensorView &tensor, const char *name) {
  if (tensor.data_ptr() == nullptr || tensor.ndim() != 4 ||
      tensor.dtype().code != kDLFloat || tensor.dtype().bits != 16 ||
      tensor.device().device_type != kDLCUDA) {
    TVM_FFI_THROW(RuntimeError)
        << name << ": expected a non-null float16 CUDA [B,H,N,64] tensor";
  }
}

void check_same_shape(const TensorView &tensor, const TensorView &q,
                      const char *name) {
  for (int axis = 0; axis < 4; ++axis) {
    if (dim(tensor, axis) != dim(q, axis)) {
      TVM_FFI_THROW(RuntimeError) << name << ": expected the same shape as q";
    }
  }
  if (tensor.device().device_id != q.device().device_id) {
    TVM_FFI_THROW(RuntimeError) << name << ": expected the same device as q";
  }
}

template <bool Causal, int kStages>
void launch_flash_attn_cutlass3(dim3 grid, cudaStream_t stream,
                                const Element *q, const Element *k,
                                const Element *v, Element *out, int heads,
                                int seqlen) {
  constexpr int kSmemBytes = SmemConfig<kStages>::kBytes;
  if constexpr (kSmemBytes > 48 * 1024) {
    static const cudaError_t smem_status = cudaFuncSetAttribute(
        flash_attn_cutlass3_kernel<Causal, kStages>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    CUDA_LEARN_CHECK(smem_status);
  }
  flash_attn_cutlass3_kernel<Causal, kStages>
      <<<grid, kThreads, kSmemBytes, stream>>>(q, k, v, out, heads, seqlen);
}

} // namespace cutlass3_fa2

void flash_attn_cutlass3(TensorView q, TensorView k, TensorView v,
                         TensorView out, int64_t causal64) {
  using namespace cutlass3_fa2;
  check_fp16_tensor(q, "q");
  check_fp16_tensor(k, "k");
  check_fp16_tensor(v, "v");
  check_fp16_tensor(out, "out");
  check_same_shape(k, q, "k");
  check_same_shape(v, q, "v");
  check_same_shape(out, q, "out");

  const int64_t batch64 = dim(q, 0);
  const int64_t heads64 = dim(q, 1);
  const int64_t seqlen64 = dim(q, 2);
  if (dim(q, 3) != kHeadDim || batch64 <= 0 || heads64 <= 0 || seqlen64 <= 0 ||
      seqlen64 % kTile != 0) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn_cutlass3: expected [B,H,N,64] with positive B/H "
           "and N divisible by 64";
  }
  if (batch64 > 65535 || heads64 > 65535 || seqlen64 > 2147483647LL) {
    TVM_FFI_THROW(RuntimeError)
        << "flash_attn_cutlass3: shape exceeds CUDA grid limits";
  }

  const int batch = static_cast<int>(batch64);
  const int heads = static_cast<int>(heads64);
  const int seqlen = static_cast<int>(seqlen64);
  const dim3 grid(seqlen / kTile, heads, batch);
  const cudaStream_t stream = get_stream(q);
  if (causal64 != 0) {
    launch_flash_attn_cutlass3<true, kDefaultStages>(
        grid, stream, static_cast<const Element *>(q.data_ptr()),
        static_cast<const Element *>(k.data_ptr()),
        static_cast<const Element *>(v.data_ptr()),
        static_cast<Element *>(out.data_ptr()), heads, seqlen);
  } else {
    launch_flash_attn_cutlass3<false, kDefaultStages>(
        grid, stream, static_cast<const Element *>(q.data_ptr()),
        static_cast<const Element *>(k.data_ptr()),
        static_cast<const Element *>(v.data_ptr()),
        static_cast<Element *>(out.data_ptr()), heads, seqlen);
  }
  CUDA_LEARN_CHECK(cudaGetLastError());
}

CUDA_LEARN_REGISTER("cuda_learn.flash_attn_cutlass3", flash_attn_cutlass3);
