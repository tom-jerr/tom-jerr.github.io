// [中文注释] 来源：deepseek-ai/DeepGEMM@057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7
// [中文注释] 原文件：deep_gemm/include/deep_gemm/layout/sym_buffer.cuh；保留全部原代码，仅添加本前缀的中文注释。
// [中文注释] L 号均指原文件；本地行号映射见 manifest.json。阅读包需原仓 CUDA/CUTLASS 等依赖，未进行 GPU 编译运行。
#pragma once

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

constexpr static uint32_t kNumMaxRanks = 72;

template <uint32_t kNumRanks = kNumMaxRanks>
// [中文注释] 原始 L10：SymBuffer 是预先建立的 peer 地址映射表，不负责创建 NVLink 连接或注册 RDMA。host 的 symmetric-memory rendezvous 先取得可访问的各 rank base pointers。
struct SymBuffer {
    int64_t base;
    int64_t offsets[kNumMaxRanks];
    uint32_t rank_idx;

    DG_STATIC_ASSERT(kNumRanks <= kNumMaxRanks, "Too many ranks");

    SymBuffer() = default;

    template <typename Container>
// [中文注释] 原始 L20：offset[peer]=peer_base-local_base。各 rank 的数据布局一致，故同一字段/row 的地址差只需加这个 offset。
    explicit SymBuffer(const Container& c, const uint32_t& rank_idx): rank_idx(rank_idx) {
        const auto size = static_cast<uint32_t>(c.size());
        base = c[rank_idx];
        for (uint32_t i = 0; i < kNumMaxRanks; ++ i)
            offsets[i] = i < size ? (c[i] - base) : 0;
    }

#if defined(__CUDA_ARCH__) or defined(__CLION_IDE__)
    template <typename ptr_t = void*>
    CUTLASS_DEVICE ptr_t get_base_ptr() const {
        return reinterpret_cast<ptr_t>(base);
    }

    template <typename ptr_t>
// [中文注释] 原始 L34：map(local_ptr,peer) 返回可做 device load/store 的 peer pointer。真正通信由随后指令的内存访问触发；单 rank 时直接返回本地地址。
    CUTLASS_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
        if constexpr (kNumRanks == 1)
            return ptr;

        int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<ptr_t*>(&mapped_ptr);
    }
#endif
};

} // namespace deep_gemm::layout
