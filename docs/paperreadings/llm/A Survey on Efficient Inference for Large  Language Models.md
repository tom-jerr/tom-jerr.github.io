## 影响 LLM 推理因素

- computational cost
- memory access cost
- memory usage
  影响这三个 cost 的因素一般可分为下面三个：
- Model Size: 需要 load 模型权重到 GPU，推理需要模型参与
- Attention Operator: 它的 FLOPs 是与输入序列平方成正比的
- Decoding Approach: 在每个解码步骤中，所有模型权重都从 HBM 加载到 GPU chip，导致 memory access cost 较大。此外，KV缓存的大小随着输入长度的增长而增加，可能导致**内存碎片和不规则的内存访问模式**

## 优化手段

### Data-level Optimization

#### Motivation

ICL 以及 CoT 技术导致现在的 prompt 非常长，进而导致计算代价和内存使用按 seq_len 平方级别增加

### Model-level Optimization

### System-level Optimization
