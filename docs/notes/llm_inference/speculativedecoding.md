---

title: Speculative Decoding
created: 2026-04-5
update:
comments: true
katex: true
tags:

- LLMInference

---
# Speculative Decoding


## Draft
- 负责便宜地提一个多步候选树


## Verify
Target Model 在每一轮验证时的逻辑输入由两部分构成：

* **已验证的前缀 (Verified Prefix):** 即截至上一轮已确认正确的 Token 序列，记为 $x_{<n}$。这部分通常已经存在于 KV Cache 中。
* **本轮候选 Token / Tree (Draft Proposals):** 由 Draft Model 生成的 $k$ 个候选 Token $\{x_n, x_{n+1}, \dots, x_{n+k}\}$，或者是树状结构的候选节点。

对于 Draft Model 提出的线性序列，Target Model 计算的是每个位置基于前序所有内容的条件概率：

$$
P(x_1 \mid \text{prefix}), \quad P(x_2 \mid \text{prefix}, x_1), \quad P(x_3 \mid \text{prefix}, x_1, x_2), \dots
$$

> [!IMPORTANT]
> 但工程实现里，重新喂给模型的 input_ids 只有本轮 draft_token；verified prefix 不会再作为一整段 token 重新输入，而是通过已有 KV cache 参与 attention。


## Sampling Method
### 拒绝采样（Rejection Sampling）
- 如果 target 觉得这个 token 比 draft 还更可能 $$p(x)>q(x)$$，那 draft 反而“保守”，直接接收不会引入偏差（甚至能提高接收率），所以必接收。
- 如果 target 觉得这个 token 没 draft 说的那么可能（$$p(x)<q(x)$$），那 draft 过度相信了这个 token。为了不让它把输出分布推偏，就要按比例接收——接收概率设成 p/q，恰好把 draft 过量的概率质量抵消掉。

主要的算法循环
- Produce k draft tokens using the draft model q. Evaluate their likelihoods under p, which produces k+1 logits. For each token:
  - If $p(x)>q(x)$, accept the token and continue. 
  - Otherwise, accept the token with probability $\frac{p(x)}{q(x)}$. If accepted, move onto the next draft token.
- On the first rejection, sample a token from the modified distribution $p’(x)=norm(max(0,p(x)−q(x)))$ and exit.
- If no tokens are rejected, sample an additional token from $p(x)$
  > [!IMPORTANT]
  > remember we get k+1 logits!


### Typical Sampling
- 类似 Medusa 的采样方法，sglang 工程上对 eagle 的采样也是类似的方式

**Motivation**
- 拒绝采样在温度升高时效率会明显变差，而真实应用里我们往往只想更有创造性，不一定非得做到分布严格等价

**Process**
以 Medusa 的树候选 + 一次并行验证为背景，Typical Acceptance 的验证/选择大致是：
1. 生成候选：Medusa heads 为不同未来位置给出 top-k，组成若干条候选路径（tree branches）。
2. 并行算目标模型 logits：通过 tree attention，一次 forward 得到这些候选路径上各位置的目标模型分布 $$p_{orig}(x_{n+k}∣x_{1:n+k−1})>min(ϵ,δ⋅e−H(p_{orig}(⋅∣x_{1:n+k−1})))$$
3. 对每条候选路径从前往后检查：
  - 第 1 个 token：论文为了保证每步至少推进 1 token，对第一个 token 用 greedy 并无条件接受（或等价地强行通过）。
  - 从第 2 个 token 开始：逐 token 计算阈值 $\min(\epsilon, \delta e^{-H})$，若该 token 的 $p_{\text{orig}}$ 大于阈值则接受，否则在此截断。
4. 选择最长被接受前缀：在所有候选路径里，选通过长度最长的那条前缀作为本轮输出（然后进入下一轮 decoding）。

> [!NOTE]
>对比严格 speculative sampling：后者一旦某位置拒绝，后续 token 全丢弃，并且要按 residual 分布重采样以保证分布不变；Typical Acceptance 则是超过阈值就接受，不做严格的分布校正。


### SGLang 工程化 Tree Sampling
不像拒绝分布采样需要 draft probs，SGLang 的实现里，tree_speculative_sampling_target_only() 只需要 target_probs 就够了。
同时在失败后需要拒绝分布采样，保证最后分布回到 target。它的核心逻辑是：
 - 当前 draft token 在 target 看来概率足够高，即 target_probs[i] > threshold-single，直接接受
 - 按树的层次从根往下走。在当前层 siblings 之间累计 prob_acc。满足 coin <= prob_acc / threshold_acc时接受该 draft token。
 - 如果某一层没有继续接受，就从 relu(target_probs - draft_probs) 这部分剩余 target 质量里再采一个最终 token。
     - 再用 uniform_samples_for_final_sampling 采样
     - U 是采样阈值已经是 coin * total_mass，范围在 [0, total_mass)
       ```shell
       sampled_id = argmin i such that CDF(i) > u
       where u ~ Uniform(0, total_mass)
       ```
 - 而是“尽量沿 draft 树接受，但保证最终分布回到 target”

## 完整流程
### Prefill Phase
- 先跑 target 的 extend，再把 target 的 hidden states 和 next token 喂给 draft 做一次 draft_extend，把 draft 状态“对齐”到当前前缀。
- draft_extend 主要在构造 forward batch，然后调用 draft model 进行前向推理，然后得到 logits 的 hidden_states，并用 logits 得到 topk_p/topk_index 这些后续 draft 轮次要用的状态。

### Decode Phase
- 如果已经进入 decode，就走完整的 speculative round：draft() -> verify() -> forward_draft_extend_after_decode()
- KV Cache 申请
  - Draft 对每个 req 申请 num_step * topk 个 KV 槽，保证每条分支都能有足够的 KV 空间；但实际使用时，只有被 verify 接受的分支会真正占用这些槽，未接受的分支占用的槽会在 verify 后被释放掉。
  - Verify 盛情的是 draft_token_num 个 KV 槽，专门给这轮 draft tree 的 token 用；而 verified prefix 相关的 KV 已经在之前的 decode 轮次里申请好了，并且会一直保留到 decode 结束。
- V2 的 KV Cache 释放
  - 当前 v2 主路径里，未接受的 speculative slot 通常不是像 v1 那样每轮立即 free 掉，而是留在这个 req 的“已分配但未提交”窗口里，后面继续覆盖复用。也就是说，v2 更偏向：
    - kv_committed_len：真正已经被 target verify 接受的长度
    - kv_allocated_len：这个请求已经预留好的物理窗口长度
  - 下一轮再通过 req_to_token_pool 重新映射、覆盖这些 slot。batch.out_cache_loc 是临时视图，不是所有权本身

**Draft Phase**
1. 先由上一轮留下的 EagleDraftInput 作为 draft 的起点。
     - 这里面最关键的是三样：verified_id、hidden_states、topk_p/topk_index。
     - 它们来自上一轮 target verify 后进行 draft extend 后的结果，是这轮 draft 展树的种子。

2. draft() 先做预处理，给整棵 speculative tree 预分配 cache 位置。
     - 这一步会根据 speculative_num_steps、topk、page_size 给所有分支预留 KV 槽；topk>1 时还会处理“复制最后 partial page”这类工程细节。
     - 分配 KV Cache 会过量分配，每个 req 分配 speculative_num_steps * self.topk 个槽，保证每条分支都能有足够的 KV 空间；但实际使用时，只有被 verify 接受的分支会真正占用这些槽，未接受的分支占用的槽会在 verify 后被释放掉。

3. draft_forward() 按步展开候选树。
     - 每一步都要执行 draft model 的前向推理，得到当前层所有候选节点的 hidden_states 和 logits，然后用 logits 得到下一步的 topk_p/topk_index，继续展开树。
     - 第 0 步：直接取当前 topk_p/topk_index 作为第一层候选。
     - 后续步：把上一层累计分数和当前步 top-k 概率相乘，再从所有展开分支里选新的 top-k，继续向下滚动。
     - draft 并不会把整棵满树都交给 target。organize_draft_results() 会把每一步收集到的分数和 token 合起来，最终只保留 num_draft_tokens - 1 个得分最高的 draft 节点；再把当前已验证 token 作为根拼进去，然后去构建 draft tree 和对应的 tree mask。

4. build_tree_kernel_efficient() 把这批候选变成 target verify 需要的结构化输入。
它会产出：

     - draft_token: 后续真正送入 verify 的 token 序列，不包含 verified token
     - positions
     - custom_mask/tree mask
     - retrive_index / retrive_next_token / retrive_next_sibling

**Verify Phase**

- verify() 先把 batch 切到 TARGET_VERIFY 模式，然后调用 prepare_for_verify()
- 这里有个很关键的实现细节：batch.input_ids = self.draft_token。重新喂给 target 的“新 token”只有这轮 draft tree；已验证前缀不会重新作为 token 序列整段输入，而是通过已有 KV cache 参与 attention。
- 同时，prepare_for_verify() 会按 batch.seq_lens -> batch.seq_lens + draft_token_num 给这些 draft token 分配 KV 位置，并把它们写进 req_to_token_pool。
    - 于是 target verify 的 attention 逻辑上看到的是：verified prefix KV + 本轮 draft tree KV。


**Tree Sampling**
- 贪心场景：
    - verify_tree_greedy 会从树根开始，逐层拿 target 在“当前已接受节点位置”的 argmax，去当前层兄弟节点里找是否有匹配的 draft token。
    - 找到就接受，继续往下一层。找不到就停止，并把当前节点的 target argmax 作为最后输出 token。
    - 所以它本质上是在树里找“一条和 target 真正预测一致的最长路径”。
- 非贪心采样场景：
    - spec_info.verify() 会先对 target logits 做 temperature / top-k / top-p，得到每个 draft 节点位置上的 target_probs，然后交给 tree_speculative_sampling_target_only()。

    - 当前 draft token 在 target 看来概率足够高，直接接受
    - 按树的层次从根往下走。在当前层 siblings 之间累计 prob_acc。满足 coin <= prob_acc / threshold_acc时接受该 draft token。
    - 如果某一层没有继续接受，就从 relu(target_probs - draft_probs) 这部分剩余 target 质量里再采一个最终 token。
        - 再用 uniform_samples_for_final_sampling 采样
        - U 是采样阈值已经是 coin * total_mass，范围在 [0, total_mass)
          ```shell
          sampled_id = argmin i such that CDF(i) > u
          where u ~ Uniform(0, total_mass)
          ```
    - 而是尽量沿 draft 树接受，但保证最终分布回到 target


**Draft Extend**
1. verify 完成后，SGLang 会真正把接受的 token 写入 req.output_ids，更新 grammar、kv_committed_len、seq_lens
     - 释放未接受分支占掉的 KV 槽；topk>1 时还会做 cache compaction，把保留下来的 accepted path 挪到连续位置。
2. verify 得到的 prefict 构造 forward batch，draft 模型沿着这次真正接受的 token 序列做一次 extend，重新得到下一轮要用的 topk_p/topk_index/hidden_states。

3. 构造下一次的 EagleDraftInput，主要是把这轮 verify 接受的 token(最后一个被验证成功的 token) 以及得到的 topk_p/topk_index/hidden_states。这一步非常关键，它让 draft 始终跟 target 已确认的前缀同步。

> [!IMPORTANT]
> 这里是因为 EAGLE 本身的性质决定的，它需要 target model 的 feature 和 sample 后的 token 作为 draft model 的输入来预测 draft token，所以每次 draft->verify->sampling 之后，我们都需要再次推进 draft，即用 target model 刚刚得到的所有 token 的 hidden_state 以及采样得到的 bonus token 来恢复 KV Cache 以供下一轮 draft 使用。
## 工程细节 SGLang
### Mamba Cache Problem
- 普通 Transformer 在 speculative verify 后，只需要把接受的 token 写回 target KV cache 就够了。
- 但 Mamba / hybrid GDN 还维护 conv_states 和 ssm_states。这些状态不按 token 存，而是按请求的当前状态槽位存，所以 **verify 后必须把被接受那一步的状态也同步回去**。
  
**Process**
- 为此，`prepare_for_v2_verify()` 现在会额外准备 `batch.mamba_track_indices`。这不是工作态 slot，而是 prefix-cache 用的持久跟踪 slot。
- 接着 verify forward 跑完后，`_mamba_verify_update()` 根据 accept_length/accept_index 算出两组 step。
  - 第一组是 accepted_steps，表示每个请求真正接受到的最后一步，要把这一步的 Mamba 状态写回当前请求工作态的 `mamba_cache_indices`。
  - 第二组是 mamba_steps_to_track，表示如果这次 verify 跨过了一个 mamba track interval，那么还要把跨点那一步的状态额外写回 `mamba_track_indices`，供后面 radix prefix cache 保存。
- 最终真正做回写的是 hybridlinearattn 后端的 `update_mamba_state_after_mtp_verify()`。
  - 实际上是两次 scatter 操作
    - 一次把 accepted_steps scatter 到 mamba_cache_indices，让请求工作态追上真正接受的那一步。
    - 一次把 mamba_steps_to_track scatter 到 mamba_track_indices，让 prefix-cache 的 snapshot 落在正确的 tracking 点上。

**Important** 
- Mamba cache 不是单纯 token->KV 的映射，而是序列状态快照。
- speculative verify 会先算出一串候选，再只接受其中一部分；真正该缓存的 Mamba state，不一定是 verify 最后一步，而是 accepted 路径里跨过 track interval 的那一步。
- verify 路径和 extend 路径用的 tracking 语义不一样，所以这里只保留 mamba_track_indices，并显式把 mamba_track_mask、mamba_track_seqlens 置空；否则会把 extend 阶段遗留的 metadata 误带进 verify



