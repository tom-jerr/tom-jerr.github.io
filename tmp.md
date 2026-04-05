# 图计算接口总结

### 与其他模块交互

#### Context -> Graph

调用：

`llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret)`

作用：

- 把 `ubatch + mctx ` 交给图构建层
- 后续构造 `graph_params`，并调用 `ggml_cgraph * llama_model::build_graph(const llm_graph_params & params)` 构建图
- `llm_graph_result::set_inputs(const llama_ubatch * ubatch)` 会把 `ubatch` 里的输入注册到 `llm_graph_result` 里，供后续 `graph_compute` 调用
- `ggml_status llama_context::graph_compute(ggml_cgraph * gf, bool batched)` 进行图计算

### Memory Context -> Graph

调用：

- `mctx->apply()`： 把 memory context 里 pending 的更新应用到 KV cache 上
- `mctx->get_ubatch()`：获取 ubatch，供图构建和输入注册使用

```cpp
class llama_kv_cache_context{
    bool apply() override;
    const llama_ubatch & get_ubatch() const override;
    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be layed out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    
    // 建图需要的接口，主要是为了构建输入时能正确地把当前 batch 的 KV 索引构建出来，供后续 `cpy_k/cpy_v` 使用
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    
    // 运行时设置输入 KV 索引、mask、pos bucket 的接口，供图计算时调用
    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
};
```

作用：

- 使用 memory/KV 提供的位置、mask、索引、缓存读写能力


### 图计算内部接口

文件：`src/llama-graph.h`


构图接口：

- `llm_graph_context::build_attn_inp_kv`
- `llm_graph_context::build_attn_inp_k`
- `llm_graph_context::build_attn_inp_kv_iswa`
- `llm_graph_context::build_attn`
- `llm_graph_context::build_inp_embd`
- `llm_graph_context::build_inp_pos`
- `llm_graph_context::build_inp_out_ids`

图结果接口：

- `llm_graph_result::reset`
- `llm_graph_result::set_inputs`
- `llm_graph_result::set_outputs`
- `llm_graph_result::can_reuse`
- `llm_graph_result::add_input`
- `llm_graph_result::set_params`


