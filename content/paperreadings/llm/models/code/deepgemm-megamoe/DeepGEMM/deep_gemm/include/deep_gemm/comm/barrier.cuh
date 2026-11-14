// [中文注释] 来源：deepseek-ai/DeepGEMM@057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7
// [中文注释] 原文件：deep_gemm/include/deep_gemm/comm/barrier.cuh；保留全部原代码，仅添加本前缀的中文注释。
// [中文注释] L 号均指原文件；本地行号映射见 manifest.json。阅读包需原仓 CUDA/CUTLASS 等依赖，未进行 GPU 编译运行。
#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::comm {

// [中文注释] 原始 L11：timeout 以 clock64 cycle 估计，常量假定 2 GHz，不能把它当作精确墙钟 60 秒。超时用于报告死锁/peer 未到达。
// 60s timeout, at 2 GHz
constexpr int64_t kNumTimeoutCycles = 60ll * 2000000000ll;

// Spin until `pred()` holds; on timeout, `print_timeout()` runs before the assertion
template <typename pred_t, typename print_timeout_t>
CUTLASS_DEVICE void wait_until(const pred_t& pred, const print_timeout_t& print_timeout) {
    const auto start_clock = clock64();
    while (not pred()) {
        if (clock64() - start_clock >= kNumTimeoutCycles) {
            print_timeout();
            DG_DEVICE_ASSERT(false and "Timeout");
        }
    }
}

// [中文注释] 原始 L26：relaxed cluster arrive 只适用于调用点的已知 ordering 前提；函数注释明确 weaker memory ordering，不应随意替代其他完成 barrier。
CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

// [中文注释] 原始 L33：grid_sync 只汇合调用者指定的线程组（dispatch 或 epilogue）。每 CTA 先 sync_scope，再一个线程 release 原子到达全局计数，再 acquire 轮询，最后再次组内同步。
template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename sync_scope_t>
CUTLASS_DEVICE void grid_sync(const layout::Workspace& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
// [中文注释] 原始 L38：每轮全部 CTA 的 add 总和为 0x80000000，使最高位翻转；sm_idx=0 加 tag-(NumSMs-1)，其余各加 1。根据 old_value 最高位变化识别本轮完成，无需每轮手工清计数。
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        wait_until([&]() { return ((new_value = ptx::ld_acq(count_ptr)) ^ old_value) & kFinishSumTag; }, [&]() {
            printf("DeepGEMM grid sync timeout: sm=%u, thread=%u, grid_sync_idx=%u, old=%u, current=%u, expected_tag=%u\n",
                   sm_idx, thread_idx, kGridSyncIndex, old_value, new_value, old_value ^ kFinishSumTag);
        });
    }
    sync_scope();
}

// [中文注释] 原始 L53：NVLink barrier 将本 GPU 多 CTA 进度汇合到 SM0，再由 SM0 对每个 peer 发送 system-release 信号；完成后可选 grid sync 把结果传播回本 GPU 的其他 CTA。
template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads, uint32_t kGridSyncIndex, uint32_t kTag, typename sync_scope_t>
CUTLASS_DEVICE void nvlink_barrier(const layout::Workspace& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    // Grid sync before NVLink signaling
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

// [中文注释] 原始 L66：两 signal slots 按 phase 交替；每个 slot 又交替 +1/-1，使目标在 NumRanks 与 0 之间切换。counter 的低两位编码 phase/sign，避免每次清 signal 再写造成竞态。
    // NVLink cross-rank barrier, only SM 0 participates
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

// [中文注释] 原始 L73：red_add_rel_sys 与 ld_acq_sys 使用 system scope，使普通 peer memory writes 的发布/观察覆盖 GPU 间通信；本 GPU ring signals 用 gpu scope。
        // Send signals to remote ranks
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // Update status and wait arrival
        if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            wait_until([&]() { return ptx::ld_acq_sys(signal_ptr) == target; }, [&]() {
                printf("DeepGEMM NVLink barrier timeout: rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                       sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
            });
        }
    }

// [中文注释] 原始 L89：sync_prologue/sync_epilogue 由调用点关闭时，附近代码必须已提供等价 grid 汇合或后续不再需要传播。不能孤立复制此 barrier 而忽略调用者的同步顺序。
    // Grid sync after NVLink completion
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
}

} // namespace deep_gemm::comm
