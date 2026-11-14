// [中文注释] 来源：deepseek-ai/DeepGEMM@057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7
// [中文注释] 原文件：deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh；保留全部原代码，仅添加本前缀的中文注释。
// [中文注释] L 号均指原文件；本地行号映射见 manifest.json。阅读包需原仓 CUDA/CUTLASS 等依赖，未进行 GPU 编译运行。
#pragma once

#include <cutlass/arch/barrier.h>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// TODO: add MegaMoE namespace
namespace deep_gemm::sched {

// [中文注释] 原始 L15：persistent kernel 中，L2 等 L1 输入。如果所有驻留 CTA pair 都被等待中的 L2 占住，未调度 L1 无法运行，会形成死锁；warmup 要先给足 L1 任务。wave 以全 GPU 的 CTA pairs 计。
// Get minimal L1 warmup waves to ensure no L1 -> L2 deadlock
// constexpr func, no runtime overhead in kernel
CUTLASS_HOST_DEVICE constexpr
int get_num_l1_warmup_waves(
    const int& num_total_m_blocks,
    const int& num_clusters,
    const int& num_l1_n_clusters,
    const int& num_l2_n_clusters) {
// [中文注释] 原始 L23：C=num_clusters，B1=L1 N clusters，B2=L2 N clusters。第一波 C 个 L2 tasks 最多触及 ceil(C/B2) 个 M blocks，因此这些块的全部 L1 tasks 至少需 ceil(ceil(C/B2)*B1/C) 波。
    // The first L2 wave may touch multiple M blocks; all their L1 N tasks must be issued first.
    const int num_first_l2_wave_m_blocks = math::constexpr_ceil_div(num_clusters, num_l2_n_clusters);
    const int num_l1_warmup_clusters_for_first_l2_wave = math::constexpr_ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters);

// [中文注释] 原始 L28：交替发 L2/L1 时，若 B1>B2，每个 M block 会欠 B1-B2 个 L1 tasks。先预发 ceil((B1+(M-1)*max(B1-B2,0))/C)+1 波；额外一波覆盖 rounding。最终取两个约束的 max。
    // When each M block has more L1 tasks than L2 tasks, the interleaved schedule
    // leaves `(num_l1_pair_n_blocks - num_l2_pair_n_blocks)` extra L1 tasks behind
    // per M block. To avoid deadlock, no L2 task for an M block should be scheduled
    // before that block's L1 tasks have been issued. The last M block is the
    // bottleneck: it needs its own `num_l1_pair_n_blocks` L1 tasks, and the preceding
    // `num_total_m_blocks - 1` M blocks accumulate
    // `(num_total_m_blocks - 1) * (num_l1_pair_n_blocks - num_l2_pair_n_blocks)`
    // pending L1 tasks, so we issue them during the warmup phase. Add one extra
    // CTA-pair wave to cover partial-wave rounding.
    const int num_interleave_cluster_diff_per_m_block =
        num_l1_n_clusters > num_l2_n_clusters ? num_l1_n_clusters - num_l2_n_clusters : 0;
    const int num_warmup_waves_for_interleave_schedule = math::constexpr_ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block,
        num_clusters) + 1;

    // TODO: may delay combine NVLink
    return cute::max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule);
}

// [中文注释] 原始 L47：host ring 容量也依赖调度策略：根据 warmup 和交替尾部的最坏 live M blocks 估容量。修改 scheduler 次序后不能只保留旧 ring 字节数；mask parity 正确性与这条容量约束相连。
// Host-side ring capacity helper for the MegaMoE schedule.
CUTLASS_HOST_DEVICE constexpr int get_num_max_live_pool_blocks(
    const int& num_total_m_blocks,
    const int& num_sms,
    const int& hidden,
    const int& intermediate_hidden) {
    constexpr int kMegaMoEBlockN = 128;
    constexpr int kNumCTAsPerCluster = 2;

    DG_UNIFIED_ASSERT((intermediate_hidden * 2) % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0);
    DG_UNIFIED_ASSERT(hidden % (kNumCTAsPerCluster * kMegaMoEBlockN) == 0);
    const int num_clusters = num_sms / kNumCTAsPerCluster;
    const int num_l1_n_clusters = intermediate_hidden * 2 / (kNumCTAsPerCluster * kMegaMoEBlockN);
    const int num_l2_n_clusters = hidden / (kNumCTAsPerCluster * kMegaMoEBlockN);
    const int num_l1_clusters = num_total_m_blocks * num_l1_n_clusters;
    const int num_l1_waves = math::constexpr_ceil_div(num_l1_clusters, num_clusters);
    const int num_min_l1_warmup_waves = get_num_l1_warmup_waves(
        num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters);
    const int num_l1_warmup_waves = cute::min(num_min_l1_warmup_waves, num_l1_waves);
    const int num_l1_warmup_clusters = cute::min(num_l1_warmup_waves * num_clusters, num_l1_clusters);
    const int num_live_blocks_after_warmup = math::constexpr_ceil_div(num_l1_warmup_clusters, num_l1_n_clusters);

// [中文注释] 原始 L69：若 B1<B2，L1 在 M-block 空间推进得更快，未完成 L2 的 live blocks 会增加；frontier_growth 加上 wave_margin 给保守容量。上限为全部 M blocks。
    // Conservative closed-form bound for the L2/L1 alternating tail. If L1 advances faster in
    // M-block space (B1 < B2), live blocks can grow by about total_m_blocks * (B2 - B1) / B2.
    // Add one global-wave margin to cover partial waves / pipeline lag.
    // This is slightly larger than the tight bound, but cheaper and simpler.
    // TODO: refactor here
    const int frontier_growth = num_l2_n_clusters > num_l1_n_clusters ?
        math::constexpr_ceil_div(
            num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters), num_l2_n_clusters) : 0;
    const int wave_margin = math::constexpr_ceil_div(
        num_clusters, cute::min(num_l1_n_clusters, num_l2_n_clusters));
    return cute::min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin);
}

// [中文注释] 原始 L82：BlockPhase 指 GEMM 的阶段：Linear1/Linear2、可选 shared Linear1/Linear2；None 是退出 sentinel。这里的 phase 不等同 mbarrier 的一位轮次。
// Computation phase for the current block
enum class BlockPhase : uint32_t {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    SharedLinear1 = 3,
    SharedLinear2 = 4
};

// [中文注释] 原始 L91：TaskInfo 描述一个 expert 的 M block 和相邻两个 N tiles。pool_block_idx 是全局逻辑块，m_block_idx 是 expert 内块；valid_m 区分真实 token 与 tile padding。
template <bool kHasSharedExperts>
struct alignas(16) TaskInfo {
    BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_cluster_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;

    CUTLASS_HOST_DEVICE
    TaskInfo(): TaskInfo(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}

    CUTLASS_HOST_DEVICE
    TaskInfo(const BlockPhase& block_phase,
             const uint32_t& local_expert_idx,
             const uint32_t& m_block_idx,
             const uint32_t& pair_n_block_idx,
             const uint32_t& pool_block_idx,
             const uint32_t& valid_m,
             const uint32_t& shape_n,
             const uint32_t& shape_k):
        block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx),
        n_cluster_idx(pair_n_block_idx),
        pool_block_idx(pool_block_idx),
        valid_m(valid_m), shape_n(shape_n), shape_k(shape_k) {}

    CUTLASS_DEVICE
    uint32_t is_valid() const {
        return (block_phase != BlockPhase::None);
    }

    CUTLASS_DEVICE
// [中文注释] 原始 L127：UMMA effective token rows 向 16 对齐，只调整指令的有效维度；source metadata 和远端 BF16 store 仍按真实 valid_m 过滤 padding。
    uint32_t get_umma_aligned_valid_m() const {
        return math::align(valid_m, 16u);
    }

    CUTLASS_DEVICE uint32_t is_shared() const {
        return kHasSharedExperts ? (block_phase > BlockPhase::Linear2) : false;
    }
};

DG_STATIC_ASSERT(sizeof(sched::TaskInfo<true>) == sizeof(sched::TaskInfo<false>), "Invalid layout");

#if defined(__CUDACC__) or defined(__CLION_IDE__)

// [中文注释] 原始 L140：每完成一个 L1 N block，XOR 翻一个 bit。generation 0 从 0→全 1，generation 1 从全 1→0，随后交替。bit 完成态不能永远解释为 1。
// Each finished L1 N block toggles its bit in `l2_full_mask`, so L2 K blocks can start as soon as they are fed
// NOTES: the mask parity per ring generation relies on the ring capacity, see `get_num_max_live_pool_blocks`
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency {
// [中文注释] 原始 L144：L1 的每个 N block 经 SwiGLU 输出 BLOCK_N/2 通道，因此一个 L2 K block 依赖 q=BLOCK_K/(BLOCK_N/2) 个 L1 N blocks。W4A8 常用 BLOCK_N=BLOCK_K=128，q=2。
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N / 2);
    static constexpr uint64_t kFullMask = kNumL1BlockNs == 64 ? ~0ull : (1ull << kNumL1BlockNs) - 1;
    DG_STATIC_ASSERT(kNumL1BlockNs <= 64, "Too many L1 N blocks for the L2 readiness mask");
    DG_STATIC_ASSERT(BLOCK_K % (BLOCK_N / 2) == 0, "Invalid L1/L2 shape relationship");

    const uint64_t* mask_ptr;
    uint64_t expected_mask, pending_mask = kFullMask;

// [中文注释] 原始 L153：expected_mask 表示本 generation 完成后应看到的 bit 值：偶数代全 1、奇数代全 0。pending_mask 初值全 1，迫使第一次 wait 至少执行一次 acquire load。
    CUTLASS_DEVICE L2KBlockDependency(const uint64_t* mask_ptr, const uint32_t& generation_idx):
        mask_ptr(mask_ptr), expected_mask(generation_idx & 1 ? 0ull : kFullMask) {}

// [中文注释] 原始 L156：release XOR 在数据/scale stores 完成后发布 bit；多个 L1 N tiles 对不同 bit 做 atomic XOR，不丢更新。每个 N tile 每代必须且只能 arrive 一次。
    CUTLASS_DEVICE static void arrive(const uint64_t* mask_ptr, const uint32_t& n_block_idx) {
        ptx::red_xor_rel(mask_ptr, 1ull << n_block_idx);
    }

// [中文注释] 原始 L160：k_block_mask 只选择当前 K block 的 q 个 bits。pending=(observed XOR expected)，选中 bits 都为 0 才放行。其他 K blocks 尚未完成不妨碍当前块继续。
    // Re-read the mask only when the K block is not fed yet
    CUTLASS_DEVICE void wait(const uint32_t& k_block_idx) {
        const auto k_block_mask = ((1ull << kNumL1BlockNsPerL2KBlock) - 1) << (k_block_idx * kNumL1BlockNsPerL2KBlock);
        while (pending_mask & k_block_mask)
            pending_mask = ptx::ld_acq_gpu(mask_ptr) ^ expected_mask;
    }
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumRingBlocks, uint32_t kNumLocalityDomains,
          uint32_t kNumSharedExperts = 0,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1Clusters = kNumL1BlockNs / 2,
          uint32_t kNumL2Clusters = kNumL2BlockNs / 2>
struct MegaMoEScheduler {
    static constexpr bool kHasShared = kNumSharedExperts > 0;
    static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;
    static constexpr uint32_t SHARED_L1_SHAPE_K = L1_SHAPE_K;
    static constexpr uint32_t SHARED_L2_SHAPE_N = L2_SHAPE_N;
    static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;
    using task_info_t = TaskInfo<kHasShared>;

    DG_STATIC_ASSERT(L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");

// [中文注释] 原始 L197：N blocks 偶数是 2-CTA cluster 的结构约束；一任务含相邻两 N blocks，两 CTA 的 n_block_idx 只差 1、m_block_idx 相同。
    // NOTES: N block counts must be even so that 2 adjacent CTAs in a cluster
    // always land on the same m_block_idx with n_block_idx differing by 1
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumRingBlocks > 0, "Invalid ring buffer config");

    // Workspace
    const layout::Workspace& workspace;

// [中文注释] 原始 L205：调度 payload 双缓冲。producer 必须先等 empty，再领取全局任务、再 publish，防止任务计数跑得超过真正可执行的 task，从而破坏 L1/L2 依赖和 ring 容量假设。
    // Scheduler configs
    static constexpr uint32_t kNumScheduleStages = 2;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    Barrier* task_info_full_barriers = nullptr;
    Barrier* task_info_empty_barriers = nullptr;
    task_info_t* task_infos = nullptr;

    // Pre-cached per-expert token counts.
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count.
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;

// [中文注释] 原始 L219：warmup 状态属于每个 CTA-pair scheduler；所有 schedulers 各发一个本地 L1 task 合起来形成一波全局 L1 tasks。不是单 scheduler 独自发完整全 GPU wave。
    // Per-scheduler warmup waves; all CTA-pair schedulers together form one global wave.
    static constexpr uint32_t kNumSchedL1WavesDone = 0xffffffffu;
    uint32_t num_sched_l1_waves = 0;

// [中文注释] 原始 L223：先领取本 locality domain 队列；任务足够多时才允许向另一 domain work stealing。少量 tasks 时跨 domain 的额外访问可能抵消负载均衡收益。
    // Locality domain affinity. When a queue is drained, we try work stealing to ensure load balancing.
    // But when there are few tasks, we disable work stealing to avoid cross-die latency.
    static constexpr uint32_t kNumMinStealWaves = 2;
    DG_STATIC_ASSERT(kNumLocalityDomains == 1 or kNumLocalityDomains == kNumDeviceLocalityDomains, "Invalid locality domain count");
    DG_STATIC_ASSERT(kNumL1Clusters % kNumLocalityDomains == 0 and kNumL2Clusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
    uint32_t sm_locality_domain_idx = 0;

    CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace):
        workspace(workspace) {}

    CUTLASS_DEVICE MegaMoEScheduler(const layout::Workspace& workspace,
                                    Barrier* task_info_full_barriers,
                                    Barrier* task_info_empty_barriers,
                                    task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    CUTLASS_DEVICE void advance_sched_pipeline() {
        DG_STATIC_ASSERT(kNumScheduleStages == 2, "Invalid stages");
        sched_stage_idx ^= 1;
        sched_phase ^= sched_stage_idx == 0;
    }

// [中文注释] 原始 L248：consumer 等 payload full_barrier 后复制 TaskInfo 到寄存器，再推进自己的 stage/phase。producer 将同一个 payload async-store 到两个 CTA 的 shared memory。
    CUTLASS_DEVICE bool get_next_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        task_info = task_infos[sched_stage_idx];
        advance_sched_pipeline();
        return task_info.is_valid();
    }

// [中文注释] 原始 L255：epilogue 等 MMA 完成后 release，fence_acq_rel_cta 完成 payload 的 metadata reads。所有 epilogue consumers 到齐后，producer 才能覆写这个双缓冲 slot。
    CUTLASS_DEVICE void release_task_info() const {
        // Complete metadata reads before the scheduler can overwrite this slot.
        ptx::fence_acq_rel_cta();
        task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

// [中文注释] 原始 L271：expert counts 按 warp lanes 分布保存。对前面所有 experts 的 ceil(token_count/BLOCK_M) 做前缀和，得到 expert 在完整 logical pool 的 block offset。
    // Get pool block offset for a given expert index from a per-lane token count array.
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE uint32_t get_num_total_pool_blocks() const {
        return get_pool_block_offset(kNumExpertsPerRank);
    }

// [中文注释] 原始 L286：recv_sum 高 32 位达到 NumSMs×NumRanks 说明所有源 rank 的 CTA count 已发布；低位是 token 总数。这里只保证路由数量 metadata 已齐，实际 activation tiles 仍由 L1 full_count 保证。
    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();

// [中文注释] 原始 L301：M=sum_expert ceil(received_tokens/BLOCK_M)。L1 task 数=M*N1/2；cluster 数=NumSMs/2。warmup=min(所需最小波数,全部 L1 波数)。
        num_total_m_blocks = get_num_total_pool_blocks();
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1Clusters;
        const uint32_t num_total_l1_waves = math::ceil_div(num_total_l1_tasks, kNumSMs / 2);
        const uint32_t min_l1_warmup_waves = get_num_l1_warmup_waves(
            num_total_m_blocks, kNumSMs / 2, kNumL1Clusters, kNumL2Clusters);
        num_sched_l1_waves = cute::min(min_l1_warmup_waves, num_total_l1_waves);
    }

// [中文注释] 原始 L309：把扁平 M-block task 映射回 expert 内 m block 和 valid_m；每 lane 暂存若干 expert count，以 ballot/shuffle 选出对应 expert，减少反复 global 读计数。
    CUTLASS_DEVICE task_info_t create_task(const BlockPhase& block_phase,
                                        const uint32_t& m_block_idx,
                                        const uint32_t& n_cluster_idx,
                                        const uint32_t& shape_n,
                                        const uint32_t& shape_k) const {
        const uint32_t lane_idx = ptx::get_lane_idx();

        task_info_t result(block_phase, 0, 0, n_cluster_idx, m_block_idx, 0, shape_n, shape_k);
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            // Reduce whether task fall in the expert
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank and
                m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            // Exchange the expert info
            if (owner_mask) {
                const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        return result;
    }

// [中文注释] 原始 L344：elected lane atomic_add 从全局 task counter 领取编号，warp shuffle 广播。counter 统计已领取任务，不统计已经完成的 GEMM；二者必须区别。
    static CUTLASS_DEVICE uint32_t get_next_local_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add(task_count_ptr + locality_domain_idx, 1u);
        return ptx::exchange(result, 0);
    }

// [中文注释] 原始 L351：task_idx 按 N-cluster 快维/M-block 慢维映射，使同一个 M block 的全部 N tasks 连续领取。locality domain 再分割 N clusters。
    template <uint32_t kNumClusters>
    static CUTLASS_DEVICE bool get_next_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx, const uint32_t& num_m_blocks,
                                                 uint32_t& m_block_idx, uint32_t& n_cluster_idx) {
        constexpr uint32_t kNumLocalClusters = kNumClusters / kNumLocalityDomains;
        const uint32_t task_idx = get_next_local_task_idx(task_count_ptr, locality_domain_idx);
        if (task_idx >= num_m_blocks * kNumLocalClusters)
            return false;
        m_block_idx = task_idx / kNumLocalClusters;
        n_cluster_idx = locality_domain_idx * kNumLocalClusters + task_idx % kNumLocalClusters;
        return true;
    }

    template <uint32_t kNumClusters>
    CUTLASS_DEVICE bool get_next_task_idx(const uint32_t* task_count_ptr, const uint32_t& num_m_blocks,
                                          uint32_t& m_block_idx, uint32_t& n_cluster_idx) const {
        if (get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx, num_m_blocks, m_block_idx, n_cluster_idx))
            return true;
        return kNumLocalityDomains > 1 and num_m_blocks * kNumClusters >= kNumMinStealWaves * (kNumSMs / 2) and
               get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx ^ 1, num_m_blocks, m_block_idx, n_cluster_idx);
    }

// [中文注释] 原始 L372：warmup 阶段优先 L1；后续通常 L2、L1 交替。scheduler 本身不扫描 readiness bits 来选任务，真正的 K block 就绪等待位于 activation TMA role。
    CUTLASS_DEVICE task_info_t get_next_task() {
        uint32_t m_block_idx, n_cluster_idx;
        while (true) {
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                // One local L1 task per scheduler; globally this is one CTA-pair wave.
                -- num_sched_l1_waves;

                // No more L1 tasks
                if (not get_next_task_idx<kNumL1Clusters>(workspace.get_l1_task_count_ptr(), num_total_m_blocks, m_block_idx, n_cluster_idx)) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }

                // Create task
                return create_task(BlockPhase::Linear1, m_block_idx, n_cluster_idx, L1_SHAPE_N, L1_SHAPE_K);
            } else {
                if (not get_next_task_idx<kNumL2Clusters>(workspace.get_l2_task_count_ptr(), num_total_m_blocks, m_block_idx, n_cluster_idx))
                    break;

                // The next task should be L1
                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;

                // Create task
                auto task_info = create_task(BlockPhase::Linear2, m_block_idx, n_cluster_idx, L2_SHAPE_N, L2_SHAPE_K);

// [中文注释] 原始 L398：发某 M block 的 L2 前，还等所有 locality queues 的 L1 counter 跨过该块所有 L1 tasks 的领取边界。这只保证生产任务已发出，数据完成仍由 mask 判断。
                // Wait until all required L1 tasks are fetched from all scheduling queues
                const auto num_required_l1_tasks = (task_info.pool_block_idx + 1) * (kNumL1Clusters / kNumLocalityDomains);
                #pragma unroll
                for (uint32_t i = 0; i < kNumLocalityDomains; ++ i)
                    while (ptx::ld_volatile(workspace.get_l1_task_count_ptr() + i) < num_required_l1_tasks) {}
                return task_info;
            }
        }
        return task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0);
    }

// [中文注释] 原始 L409：lane 0/1 分别把同一个 TaskInfo async-store 到 CTA 0/1，并对各自 full barrier 声明 transaction bytes。两个 CTA 得到同一个 expert/M/N-cluster 任务。
    CUTLASS_DEVICE void publish_task(const task_info_t& task_info, const uint32_t& lane_idx) {
        if (lane_idx < 2) {
            task_info_full_barriers[sched_stage_idx].arrive_and_expect_tx(sizeof(task_info_t), lane_idx);
            ptx::st_async_cluster(
                task_infos + sched_stage_idx, task_info,
                lane_idx, task_info_full_barriers[sched_stage_idx]
            );
        }
        __syncwarp();
        advance_sched_pipeline();
    }

    template <BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
    CUTLASS_DEVICE void shared_mainloop(const uint32_t& num_tokens, const uint32_t& lane_idx, const uint32_t* task_count_ptr) {
        constexpr uint32_t kNumNClusters = kShapeN / BLOCK_N / 2;
        DG_STATIC_ASSERT(kNumNClusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
        const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
        uint32_t m_block_idx, n_cluster_idx;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);

            // Use dynamic scheduling to reduce tailing across shared L1/L2 tile shapes.
            if (not get_next_task_idx<kNumNClusters>(task_count_ptr, num_m_blocks, m_block_idx, n_cluster_idx))
                break;
            const uint32_t valid_m = cute::min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
            publish_task(task_info_t(kBlockPhase, 0, m_block_idx, n_cluster_idx, m_block_idx, valid_m, kShapeN, kShapeK), lane_idx);
        }
    }

// [中文注释] 原始 L438：shared L1 不依赖跨 rank dispatch，可填通信间隙。shared L2 是否提前插入取决于能否在一波 CTA pairs 内放下；routed 部分仍保留自己的 warmup/依赖顺序。
    CUTLASS_DEVICE void mainloop(const uint32_t& num_tokens, const uint8_t* sm_locality_domains) {
        const auto lane_idx = ptx::get_lane_idx();
        sm_locality_domain_idx = kNumLocalityDomains == 1 ? 0u : sm_locality_domains[ptx::get_sm_idx()];

        // Shared L2 tasks go before the routed tasks if one wave fits the dispatch gap, else at the tail
        constexpr uint32_t kNumSharedL2Clusters = SHARED_L2_SHAPE_N / BLOCK_N / 2;
        const bool is_shared_l2_early = math::ceil_div(num_tokens, BLOCK_M) * kNumSharedL2Clusters <= kNumSMs / 2;
        if constexpr (kHasShared) {
            // Shared expert L1 tasks do not depend on dispatch.
            shared_mainloop<BlockPhase::SharedLinear1, SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l1_task_count_ptr());
            if (is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Wait dispatch's results
        fetch_expert_recv_count();

// [中文注释] 原始 L457：必须保持 wait(empty)→claim(global counter)→publish 顺序。先 claim 再等 empty 会使计数代表的生产进度提前，L2 可能占住执行资源等待尚不能发布的 L1。
        // Generate routed tasks. Keep the original wait -> claim -> publish ordering:
        // `get_next_task()` advances global task counters and must not run before the
        // schedule slot is released by consumers.
        task_info_t task_info;
        do {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            task_info = get_next_task();
            if (task_info.is_valid()) publish_task(task_info, lane_idx);
        } while (task_info.is_valid());

        if constexpr (kHasShared) {
            if (not is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

// [中文注释] 原始 L473：向两个 CTA 发布 None sentinel，让各角色结束 persistent loop；epilogue 随后进入 combine。sentinel 也要占一个正常调度 slot，不能越过 empty 保护。
        // Sentinel.
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
    }
};

#endif

} // namespace deep_gemm::sched
