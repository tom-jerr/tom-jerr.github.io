---
title: 一条 Request 在 SGLang 的前世今生
tags:
  - LLMInference
date: 2025/11/10
---
# 一条 Request 在 SGLang 的前世今生

> 这里我们设置一些参数更接近真实的推理场景，启用 `mixed chunked` 参数，prefill 开启 `chunked_prefill`，cache 使用 `page_size > 1`，scheduler 使用 `overlap scheduler`  

通过这种设置，我们可以观察 SGLang 中的 **Page Attention** 和 **Continuous Batching** 是如何实现的

> SGLang 通过 **Chunked Prefill** 实现真正的 **Continuous Batching**。
## Prerequisite knowledge



```python
初始状态: 50 tokens, page_size=16

1. cache_unfinished_req 被调用
   req.fill_ids = [t0, t1, ..., t49]
   kv_indices = [kv_0, kv_1, ..., kv_49]

2. 页对齐
   page_aligned_len = 48
   page_aligned_kv_indices = kv_indices[0:48]

3. 插入树
   new_prefix_len = insert([t0...t47], [kv_0...kv_47])
   假设返回 32 (前 2 页已在树中)

4. Free 重复部分
   free(kv_indices[32:48])  # 释放新加入树的部分
   
   内存引用:
   kv_indices[0:32]   - 树引用 + req引用 (旧前缀)
   kv_indices[32:48]  - 树引用 (刚加入)
   kv_indices[48:50]  - req引用 (部分页)

5. 重新匹配
   new_indices = match_prefix([t0...t47])
   返回 [kv_0, ..., kv_47]

6. 更新映射
   req_to_token_pool[req_pool_idx, 32:48] = new_indices[32:48]

7. 记录匹配长度
   req.last_matched_prefix_len = 48

8. 构建完整的 prefix_indices
   req.prefix_indices = [kv_0, ..., kv_47, kv_48, kv_49]
                        ^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^
                           树中 (48)        部分页 (2)

9. 下次调用时
   old_prefix_len = 48  (使用 last_matched_prefix_len)
   新的 new_prefix_len = 64 (假设又填充了一页)
   free(kv_indices[48:64])  # 正确释放，包括之前的部分页
   
```

## ModelRunner
┌─────────────────────────────────────────────────────────────────────┐
│  1. 初始化阶段 (服务器启动)                                             │
└─────────────────────────────────────────────────────────────────────┘
         │
         ├─→ ModelRunner.__init__()
         │       │
         │       └─→ self.initialize()
         │               │
         │               └─→ self.init_attention_backend()  [line 1887]
         │                       │
         │                       └─→ self._get_attention_backend()  [line 1900]
         │                               │
         │                               └─→ self._get_attention_backend_from_str(
         │                                       backend_str="flashinfer"  # 或其他
         │                                   )  [line 1943]
         │                                       │
         │                                       └─→ ATTENTION_BACKENDS[backend_str](self)
         │                                               │
         │                                               ├─→ FlashAttentionBackend(model_runner)
         │                                               ├─→ FlashInferAttnBackend(model_runner)
         │                                               └─→ 其他 backends...
         │
         └─→ self.attn_backend = <FlashAttentionBackend instance>