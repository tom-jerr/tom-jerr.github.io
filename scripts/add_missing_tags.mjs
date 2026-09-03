import fs from "node:fs"
import path from "node:path"

const tagsByFile = new Map([
  ["cuda/cuda basic.md", ["CUDA"]],
  ["cuda/cuda performance checklist.md", ["CUDA"]],
  ["cuda/quantization cuda.md", ["CUDA", "LLMInference"]],
  ["llamacpp/server_profile.md", ["LLMInference"]],
  ["llm_inference/deepseek.md", ["LLMInference"]],
  ["llm_inference/deepep_architecture_notes.md", ["LLMInference"]],
  ["llm_inference/llm-compressor框架解析.md", ["LLMInference", "Quantization"]],
  ["llm_inference/longcontext.md", ["LLMInference"]],
  ["llm_inference/optimization_tech.md", ["LLMInference"]],
  ["llm_inference/量化综述.md", ["LLMInference", "Quantization"]],
  ["llm_inference/tools/pip&uv.md", ["Python"]],
  ["llm_inference/tools/python项目构建.md", ["Python"]],
  ["nebula-vsearch/summary.md", ["Database"]],
  ["nebula-vsearch/implements/Create Ann Index.md", ["Database"]],
  ["nebula-vsearch/implements/DDL for vector type.md", ["Database"]],
  ["nebula-vsearch/implements/DML for vector type.md", ["Database"]],
  ["nebula-vsearch/implements/Match for vector property.md", ["Database"]],
  ["nebula-vsearch/implements/wal for vector type.md", ["Database"]],
  ["nebula-vsearch/understanding/8.a kv life.md", ["Database"]],
  ["notes/C++/1-Container.md", ["C++"]],
  ["notes/C++/2-Algorithm.md", ["C++"]],
  ["notes/C++/3-Iterator.md", ["C++"]],
  ["notes/C++/4-Filesystem.md", ["C++"]],
  ["notes/C++/5-View.md", ["C++"]],
  ["notes/C++/6-Span.md", ["C++"]],
  ["notes/c++primer/15_面向对象程序设计.md", ["C++"]],
  ["notes/linux tools/1-grep.md", ["Linux"]],
  ["notes/linux tools/2-sed.md", ["Linux"]],
  ["notes/linux tools/3-awk.md", ["Linux"]],
  ["notes/linux tools/4-find.md", ["Linux"]],
  ["notes/python/Asyncio.md", ["Python"]],
  ["notes/python/Python 中的并行.md", ["Python"]],
  ["notes/python/Python 中的装饰器.md", ["Python"]],
  ["notes/slam/毕设：面向复杂室内环境的无人机自主探索框架.md", ["VINS"]],
  ["notes/slam/Slam基础知识.md", ["VINS"]],
  ["paperreadings/activeslam/active slam.md", ["Paper Notes", "VINS"]],
  ["paperreadings/llm/Prepacking.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/models/deepseekv4.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/myidea/agentkvcache.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/myidea/agentkvcache/CacheSlide.md", ["Paper Notes", "LLMInference"]],
  [
    "paperreadings/llm/myidea/agentkvcache/KV CACHE OPTIMIZATION STRATEGIES FOR SCALABLE  AND EFFICIENT LLM INFERENCE.md",
    ["Paper Notes", "LLMInference"],
  ],
  ["paperreadings/llm/myidea/agentkvcache/KVFlow.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/myidea/agentkvcache/ToolCaching.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/serving/Chunked Prefill.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/serving/distserve.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/serving/PageAttention.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/serving/SGLang.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/speculative decoding/DFLASH.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/speculative decoding/eagle.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/speculative decoding/eagle2.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/speculative decoding/eagle3.md", ["Paper Notes", "LLMInference"]],
  ["paperreadings/llm/speculative decoding/SD in Llama.md", ["Paper Notes", "LLMInference"]],
  [
    "paperreadings/llm/speculative decoding/survey on parallel text gen.md",
    ["Paper Notes", "LLMInference"],
  ],
  ["paperreadings/llm/一致性/batch-invariant kernel.md", ["Paper Notes", "LLMInference"]],
  ["sglang/eplbopt.md", ["LLMInference"]],
  ["summary/2025/November/2025-11-01.md", ["summary"]],
  ["summary/2025/Octorber/2025-10-31.md", ["summary"]],
])

const contentRoot = path.resolve("content")
let updated = 0

for (const [relativePath, tags] of tagsByFile) {
  const filePath = path.join(contentRoot, ...relativePath.split("/"))
  if (!fs.existsSync(filePath)) continue
  const text = fs.readFileSync(filePath, "utf8")
  const eol = text.includes("\r\n") ? "\r\n" : "\n"
  const tagBlock = `tags:${eol}${tags.map((tag) => `  - ${tag}`).join(eol)}${eol}`

  if (text.startsWith("---")) {
    const frontmatterEnd = text.indexOf(`${eol}---`, 3)
    if (frontmatterEnd === -1) {
      throw new Error(`Unclosed frontmatter: ${relativePath}`)
    }

    const frontmatter = text.slice(0, frontmatterEnd)
    if (/^tags\s*:/m.test(frontmatter)) continue

    const next = `${text.slice(0, frontmatterEnd)}${eol}${tagBlock}${text.slice(frontmatterEnd)}`
    fs.writeFileSync(filePath, next)
  } else {
    fs.writeFileSync(filePath, `---${eol}${tagBlock}---${eol}${eol}${text}`)
  }

  updated += 1
}

console.log(`Added tags to ${updated} articles.`)
