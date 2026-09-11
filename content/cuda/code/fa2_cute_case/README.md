# CuTe FlashAttention-2 case-study benchmark

This directory preserves the exact teaching kernel, benchmark harness, raw
results, and figure generator used by the corresponding tom-jerr blog post.

From the `cuda_learn` repository root:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
python benchmarks/fa2_cute_case/bench.py \
  --output benchmarks/fa2_cute_case/results_local
```

Use `--require-ampere` on an Ampere-only run. It accepts SM80, SM86, and SM87
and exits on Ada or another architecture. The benchmark compiles isolated
2-stage and 3-stage shared libraries; it does not alter the working kernel or
the repository CMake configuration.

Generate the article figures from a completed result:

```bash
python benchmarks/fa2_cute_case/figures.py \
  benchmarks/fa2_cute_case/results_local/results.json \
  benchmarks/fa2_cute_case/article_assets
```

The checked-in `results_sm89_clean` data came from an RTX 4060 Laptop GPU
(SM89), so it is an Ada measurement of the Ampere-style instruction path, not
an Ampere hardware result.
