---
status: open
priority: high
scheduled: 2025-11-25T11:00
dateCreated: 2025-11-24T14:50:27.371+08:00
dateModified: 2025-11-24T15:05:15.694+08:00
reminders:
  - id: rem_1763967009353
    type: relative
    relatedTo: scheduled
    offset: -PT15M
    description: 15 minutes before
tags:
  - task
---

## 反问问题

1. **我们这个部门是做什么的? 是传统LLM 还是现在的视频生成大模型？**

2.  **我进去之后做什么？是在 Rollout(采样生成)阶段优化吗？瓶颈在哪？**
    - 显存占用
    - or 生成吞吐（Throughput）跟不上训练更新的速度？"
        
3. **推理引擎方案是什么样子的，基于 SGLang 吗，我们需要适配的是因为MoE架构原因吗？**

4. **现在这个领域比较核心的技术有哪些呢？我应该往哪个方向来提升呢(CUDA or Triton or RL or Diffusion)？**

5. **如果我有幸加入，您对我这段实习期间最核心的预期产出是什么？比如，是希望看到某个核心算子（如 MoE 的 Gate 计算或 Attention）有 X% 的加速，还是希望我能把某套新的推理框架集成到现有的 RL Pipeline 里？**
        
6. **咱们这个岗位的工作内容，是偏向于 **Engineering**（把现有的 SOTA 论文快速落地、修 bug、稳系统），还是偏向于 **Research**（探索新的 Attention 机制、新的量化方案）？这两者的比例大概是多少？**

7. **入职的话会有培训或者有mentor来带我吗？组里现在的技术氛围大概是怎样的（比如是否会有定期的 Paper Reading 或技术分享）？**
        
8. **虽然现在谈这个有点早，但我想了解一下 Minimax 对于核心 Infra 组实习生的转正机制大概是怎样的？**
    