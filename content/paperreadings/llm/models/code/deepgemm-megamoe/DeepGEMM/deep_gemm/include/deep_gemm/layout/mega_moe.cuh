// [中文注释] 来源：deepseek-ai/DeepGEMM@057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7
// [中文注释] 原文件：deep_gemm/include/deep_gemm/layout/mega_moe.cuh；保留全部原代码，仅添加本前缀的中文注释。
// [中文注释] L 号均指原文件；本地行号映射见 manifest.json。阅读包需原仓 CUDA/CUTLASS 等依赖，未进行 GPU 编译运行。
#pragma once

#include <cstddef>
#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm::layout {

// [中文注释] 原始 L13：host 必须对全部候选 BLOCK_M 估计容量，因为 runtime tile 由 token 数选择。1920 是这些候选的最小公倍数；按它对齐可使 ring 对所有候选都整块划分。
static constexpr int kNumCandidateBlockMs = 8;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192, 240};
static constexpr int kMaxCandidateBlockM = 240;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 1920;

// [中文注释] 原始 L19：完整 logical pool 保存可能接收的最大路由数和各 expert 的 tile padding。这里的满容量供 source metadata 使用，不能等同 global activation ring 的容量。
// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// [中文注释] 原始 L30：每个 data M block 的 scale 行跨度向 128 对齐。比如 BLOCK_M=16，scale 行数比例是 128/16=8；scale 实际 payload 每行又比 activation 小得多。
// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sf_ring_tokens(T num_ring_tokens, T block_m) {
    return (num_ring_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Shared L2 input SF capacity: worst-case aligned SF pages over all candidate BLOCK_M.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_shared_sf_tokens(const T& num_max_tokens_per_rank) {
    return math::constexpr_ceil_div<T>(num_max_tokens_per_rank, kMinCandidateBlockM) * 128;
}

// [中文注释] 原始 L42：写回目的三元组：source rank、source token、source top-k slot。source metadata 属于完整 logical pool，保持到 Linear2 写回读取完毕。
// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;

    TokenSrcMetadata() = default;

    CUTLASS_DEVICE TokenSrcMetadata(const uint32_t& rank_idx, const uint32_t& token_idx, const uint32_t& topk_idx):
        rank_idx(rank_idx), token_idx(token_idx), topk_idx(topk_idx) {}
};

// [中文注释] 原始 L54：这些固定位置 signals 保证每个 rank 的对应字段在相同偏移。环形 signals 是计数/mask，不是模型 KV Cache 或 attention 的 C4 压缩状态。
struct alignas(128) MegaMoESignals {
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;
    static constexpr uint32_t kNumMaxExperts = 2048;
    static constexpr uint32_t kNumMaxExpertsPerRank = 512;
    static constexpr uint32_t kNumMaxRingBlocks = 1u << 20;  // 20 MiB for the ring signals
    static constexpr uint32_t kNumMaxSharedL2Blocks = 1u << 15;  // 128 KiB

    // Grid and NVLink synchronization
    uint32_t grid_sync_count[kNumMaxGridSyncCounters];
    uint32_t nvl_barrier_counter;
    int nvl_barrier_signals[2];

// [中文注释] 原始 L66：l1/l2 task count 是 scheduler 领取任务进度；不得将它们当作算完进度。combine-ready 用每次 launch 的 gridid 标记，无需按轮清零。
    // Task scheduling
    uint32_t l1_task_count[kNumDeviceLocalityDomains];
    uint32_t l2_task_count[kNumDeviceLocalityDomains];
    uint32_t shared_l1_task_count[kNumDeviceLocalityDomains];
    uint32_t shared_l2_task_count[kNumDeviceLocalityDomains];

    // Combine readiness: `combine_ready_grid_idx[peer] == own grid index` means the peer's L2 writes into this
    // rank are done; grid indices are unique per launch, so no reset is needed. Peers push theirs during dispatch.
    alignas(128) uint64_t combine_ready_grid_idx[kNumMaxRanks];
    alignas(128) uint64_t peer_grid_idx[kNumMaxRanks];

// [中文注释] 原始 L77：send/recv packed 64-bit count 高位追踪贡献 CTA 数，低位追踪 token 数。recv_count 保存各 source rank 的数量，recv_sum 汇总所有 source rank。
    // Expert token counts
    alignas(128) uint64_t expert_send_count[kNumMaxExperts];
    uint64_t expert_recv_count[kNumMaxExperts];
    uint64_t expert_recv_count_sum[kNumMaxExpertsPerRank];

// [中文注释] 原始 L82：l1_full=activation 到齐；l1_empty=L1 消费完输入；l2_full_mask=L1 产出对应 intermediate channels；l2_empty=L2 消费完中间输入。四个信号保护两级 data rings。
    // Routed-expert ring signals, `l2_full_mask` has one bit per L1 N block
    uint32_t l1_full_count[kNumMaxRingBlocks];
    uint32_t l1_empty_count[kNumMaxRingBlocks];
    uint64_t l2_full_mask[kNumMaxRingBlocks];
    uint32_t l2_empty_count[kNumMaxRingBlocks];

    // Shared-expert signals
    uint32_t shared_l2_full_count[kNumMaxSharedL2Blocks];
};
DG_STATIC_ASSERT(offsetof(MegaMoESignals, combine_ready_grid_idx) == 128, "Invalid control signal layout");

struct Workspace {
    MegaMoESignals* signals;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_shared_l2_pool_blocks;

// [中文注释] 原始 L100：num_max_pool_tokens 是不取模的 metadata 容量，num_ring_tokens 是可复用的激活容量。两者不同正是为了不等远端输出结束就能复用输入 ring。
    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;

    Workspace() = default;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens):
        signals(static_cast<MegaMoESignals*>(base)),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_shared_l2_pool_blocks = math::ceil_div<uint32_t>(num_max_tokens_per_rank, kMinCandidateBlockM);

        DG_UNIFIED_ASSERT(num_ranks > 0);
        DG_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        DG_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        DG_UNIFIED_ASSERT(num_experts <= MegaMoESignals::kNumMaxExperts);
        DG_UNIFIED_ASSERT(num_experts_per_rank <= MegaMoESignals::kNumMaxExpertsPerRank);
        DG_UNIFIED_ASSERT(num_ring_tokens <= MegaMoESignals::kNumMaxRingBlocks * kMinCandidateBlockM);
        DG_UNIFIED_ASSERT(num_shared_l2_pool_blocks <= MegaMoESignals::kNumMaxSharedL2Blocks);
    }

// [中文注释] 原始 L128：workspace 除 signals，还包含 [local expert][source rank][token slot] uint32 索引与 full-pool TokenSrcMetadata。显存估算只乘 activation 宽度会漏这些项。
    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Fixed-position global and expert signals
        num_bytes += sizeof(MegaMoESignals);

        // Source token-topk: [local expert][source rank][token], with distinct experts per token
        num_bytes += num_experts * num_max_tokens_per_rank * sizeof(uint32_t);

        // Combine push source indices (full)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(signals, get_num_bytes());
    }

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < MegaMoESignals::kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return signals->grid_sync_count + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return &signals->nvl_barrier_counter;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return signals->nvl_barrier_signals + phase;
    }

// [中文注释] 原始 L169：本 rank 的 combine_ready_grid_idx[peer] 表示 peer 向本 rank 的全部本轮 L2 stores 已完成。peer_grid_idx[peer] 存 peer 自己本轮的 id，发布时用于目的端匹配。
    CUTLASS_DEVICE
    uint64_t* get_combine_ready_grid_idx_ptr(const uint32_t& peer_rank_idx = 0) const {
        return signals->combine_ready_grid_idx + peer_rank_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_peer_grid_idx_ptr(const uint32_t& peer_rank_idx = 0) const {
        return signals->peer_grid_idx + peer_rank_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_task_count_ptr() const {
        return signals->l1_task_count;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_task_count_ptr() const {
        return signals->l2_task_count;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l1_task_count_ptr() const {
        return signals->shared_l1_task_count;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_task_count_ptr() const {
        return signals->shared_l2_task_count;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return signals->expert_send_count + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return signals->expert_recv_count + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return signals->expert_recv_count_sum + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l1_full_count + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l1_empty_count + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_full_mask_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l2_full_mask + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        return signals->l2_empty_count + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_full_count_ptr(const uint32_t& block_idx = 0) const {
        return signals->shared_l2_full_count + block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
// [中文注释] 原始 L242：routing 列表每个 local expert、source rank 预留 max_tokens_per_rank 个 slot；同一 token 的 top-k experts 应互异，符合容量估算前提。
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto offset = (expert_idx * num_ranks + rank_idx) * num_max_tokens_per_rank + token_idx;
        return math::advance_ptr<uint32_t>(signals, sizeof(MegaMoESignals)) + offset;
    }

// [中文注释] 原始 L248：get_token_src_metadata_ptr 返回完整 pool 地址，无 ring 取模；最终 combine 的 slots 也没有随 expert ring generation 复用。
    // For combine usages (full)
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    Data() = default;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

// [中文注释] 原始 L287：Buffer 是通用二维 buffer-view 工具。get_rank_buffer 的 rank_idx 在 combine buffer 上实际代表 top-k slot，并不代表 physical EP rank。
struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    Buffer() = default;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& num_max_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

// [中文注释] 原始 L345：同一 symmetric allocation 串联 input、routing、shared inputs、routed L1/L2 rings、combine slots。专家 weights 不属于这份 symmetric activation workspace。
struct MegaMoEBuffer {
    Workspace workspace;

    // Input buffers (per-rank)
    Buffer input_token_buffer,
           input_sf_buffer,
           input_topk_idx_buffer,
           input_topk_weights_buffer;

    // Routed expert ring buffers
    // NOTE: shared L1 tokens reuse `input_token_buffer`.
    Buffer shared_l1_token_buffer, shared_l1_sf_buffer,
           shared_l2_token_buffer, shared_l2_sf_buffer;

    // Routed expert ring buffers
    Buffer l1_token_buffer,
           l1_sf_buffer,
           l1_topk_weights_buffer,
           l2_token_buffer,
           l2_sf_buffer,
           combine_token_buffer;

    CUTLASS_HOST_DEVICE
    MegaMoEBuffer(void* base,
                  const uint32_t& hidden,
                  const uint32_t& intermediate_hidden,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_ring_tokens,
                  const uint32_t& num_sf_ring_tokens,
                  const bool& with_sf,
                  const uint32_t& num_shared_experts = 0) {
        // Workspace
        workspace = Workspace(base, num_ranks, num_experts,
                              num_max_tokens_per_rank, num_topk, num_ring_tokens);

        // Shared
        const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const auto num_max_shared_sf_tokens = with_sf ? get_num_max_shared_sf_tokens(num_max_tokens_per_rank) : 0u;

        // Layouts
        const uint32_t num_mma_elem_bytes = with_sf ? 1 : 2;
        const auto input_token_layout = layout::Data(hidden * num_mma_elem_bytes);
        const auto bf16_token_layout = layout::Data(hidden * 2);
        const auto intermediate_token_layout = layout::Data(intermediate_hidden * num_mma_elem_bytes);
        const auto shared_intermediate_token_layout = layout::Data(shared_intermediate_hidden * num_mma_elem_bytes);
// [中文注释] 原始 L393：FP8 input 每 32 元素一个 UE8M0 byte；H/32 scale bytes 又打包为 H/128 个 uint32。TMA descriptor 可能另有 alignment/stride 要求。
        const auto input_sf_layout = layout::Data(with_sf ? hidden / 32 : 0, false);
        const auto intermediate_sf_layout = layout::Data(with_sf ? intermediate_hidden / 32 : 0, false);
        const auto shared_intermediate_sf_layout = layout::Data(with_sf ? shared_intermediate_hidden / 32 : 0, false);
        const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

        // Input buffers
        input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            workspace.get_end_ptr());
        input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            input_token_buffer.get_end_ptr());
        input_topk_idx_buffer = Buffer(
            input_topk_idx_layout, 1, num_max_tokens_per_rank,
            with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr());
        input_topk_weights_buffer = Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank,
            input_topk_idx_buffer.get_end_ptr());

// [中文注释] 原始 L414：shared L1 可直接复用源 rank input，无 dispatch pull；shared L2 输入按本 rank token 容量独立分配，不使用 routed generation mask。
        // Shared expert buffers
        shared_l1_token_buffer = input_token_buffer;
        shared_l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            input_topk_weights_buffer.get_end_ptr());
        shared_l2_token_buffer = Buffer(
            shared_intermediate_token_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            with_sf ? shared_l1_sf_buffer.get_end_ptr() : input_topk_weights_buffer.get_end_ptr());
        shared_l2_sf_buffer = Buffer(
            shared_intermediate_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            shared_l2_token_buffer.get_end_ptr());

// [中文注释] 原始 L426：L1/L2 activation 和 scale 为可复用 ring。L1 top-k weight 也随 L1 ring 使用；metadata 则留在 workspace 的完整 pool。
        // Routed expert ring buffers
        l1_token_buffer = Buffer(
            input_token_layout, 1, num_ring_tokens,
            num_shared_experts > 0 ?
                (with_sf ? shared_l2_sf_buffer.get_end_ptr() : shared_l2_token_buffer.get_end_ptr()) :
                input_topk_weights_buffer.get_end_ptr()
        );
        l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_sf_ring_tokens,
            l1_token_buffer.get_end_ptr());
        l1_topk_weights_buffer = Buffer(
            l1_topk_weights_layout, 1, num_ring_tokens,
            with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr());

        l2_token_buffer = Buffer(
            intermediate_token_layout, 1, num_ring_tokens,
            l1_topk_weights_buffer.get_end_ptr());
        l2_sf_buffer = Buffer(
            intermediate_sf_layout, 1, num_sf_ring_tokens,
            l2_token_buffer.get_end_ptr());

        combine_token_buffer = Buffer(
            bf16_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), num_max_tokens_per_rank,
            with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr());
    }

    CUTLASS_HOST_DEVICE
    int64_t get_num_bytes() const {
        return static_cast<uint8_t*>(combine_token_buffer.get_end_ptr())
               - reinterpret_cast<uint8_t*>(workspace.signals);
    }
};

} // namespace deep_gemm::layout
