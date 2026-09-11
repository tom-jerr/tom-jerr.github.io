"""Isolated, reproducible CuTe FA2 vs upstream FA2 D64 forward benchmark.

Run from a checkout containing third_party/{cutlass,flash-attention} and src/include.
Never changes the working kernel or the main project's build configuration.
"""
import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import tvm_ffi
from tvm_ffi import libinfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def command(args):
    return subprocess.check_output([str(x) for x in args], text=True).strip()


def build_cute(stages, arch, cuda_home, folder):
    source = (HERE / "kernel_snapshot.cu").read_text()
    assert source.count("constexpr int kDefaultStages = 2;") == 1
    source = source.replace("constexpr int kDefaultStages = 2;",
                            f"constexpr int kDefaultStages = {stages};")
    source = source.replace("cuda_learn.flash_attn_cutlass3",
                            f"fa2_case.forward_s{stages}")
    cu = folder / f"cute_s{stages}.cu"
    cu.write_text(source)
    so = folder / f"cute_s{stages}.so"
    ffi_lib = Path(libinfo.find_libtvm_ffi())
    args = [cuda_home / "bin/nvcc", "-O3", "-std=c++17", "--use_fast_math",
            "--expt-relaxed-constexpr", f"-arch=sm_{arch}", "-Xptxas=-v",
            "--shared", "-Xcompiler=-fPIC", cu, "-o", so,
            f"-I{ROOT / 'third_party/cutlass/include'}",
            f"-I{ROOT / 'src/include'}", f"-I{libinfo.find_include_path()}",
            f"-I{libinfo.find_dlpack_include_path()}", ffi_lib,
            f"-Xlinker=-rpath={ffi_lib.parent}",
            f"-Xlinker=-rpath={cuda_home / 'lib64'}"]
    with (folder / f"compile_s{stages}.log").open("w") as log:
        subprocess.run([str(x) for x in args], check=True, stdout=log, stderr=log)
    # Keep CDLL alive. TVM FFI constructor registers each distinct stage variant.
    lib = ctypes.CDLL(str(so))
    return tvm_ffi.get_global_func(f"fa2_case.forward_s{stages}"), lib, list(map(str, args))


def reference(q, k, v, causal):
    # Full FP32 reference, chunk only query rows to bound temporary memory.
    n = q.shape[2]
    out = torch.empty_like(q, dtype=torch.float32)
    k32, v32 = k.float(), v.float()
    for start in range(0, n, 128):
        end = min(start + 128, n)
        scores = q[:, :, start:end].float() @ k32.transpose(-1, -2) / 8
        if causal:
            mask = torch.arange(n, device=q.device)[None, :] > torch.arange(
                start, end, device=q.device)[:, None]
            scores.masked_fill_(mask, -float("inf"))
        out[:, :, start:end] = torch.softmax(scores, -1) @ v32
    return out


def error(out, ref):
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=2e-3)
    delta = out.float() - ref
    return {"max_abs": delta.abs().max().item(),
            "rmse": delta.square().mean().sqrt().item()}


def graph_for(fn, iterations):
    # All allocations, conversions, module loading and warmup precede capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            fn()
    return graph


def measure(graph, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=HERE / "results")
    ap.add_argument("--cuda-home", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")))
    ap.add_argument("--require-ampere", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=9)
    args = ap.parse_args()
    cc = torch.cuda.get_device_capability()
    if args.require_ampere and cc not in ((8, 0), (8, 6), (8, 7)):
        raise SystemExit(f"Ampere required; actual device is {torch.cuda.get_device_name()} SM{cc[0]}{cc[1]}")
    args.output.mkdir(parents=True, exist_ok=True)
    arch = f"{cc[0]}{cc[1]}"
    os.environ["CUDA_HOME"] = str(args.cuda_home)
    os.environ["MAX_JOBS"] = "2"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260909)
    modules, handles, compile_commands = {}, [], {}
    for stage in (2, 3):
        print(f"Building CuTe stages={stage} sm_{arch}", flush=True)
        fn, handle, build_command = build_cute(stage, arch, args.cuda_home, args.output)
        modules[stage] = fn
        handles.append(handle)
        compile_commands[str(stage)] = build_command
    spec = importlib.util.spec_from_file_location("fa2_baseline", ROOT / "python/cuda_learn/flash_attn_baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    print("Loading upstream FA2 v2.8.3 D64 specializations", flush=True)
    official = baseline._load_module()
    props = torch.cuda.get_device_properties(0)
    meta = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "gpu": props.name, "compute_capability": cc,
            "ampere_hardware": cc in ((8, 0), (8, 6), (8, 7)),
            "sms": props.multi_processor_count, "memory_bytes": props.total_memory,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "tvm_ffi": tvm_ffi.__version__, "nvcc": command([args.cuda_home / "bin/nvcc", "--version"]),
            "driver": command(["nvidia-smi", "--query-gpu=driver_version,name,power.limit,clocks.sm,temperature.gpu", "--format=csv"]),
            "cutlass_commit": command(["git", "-C", ROOT / "third_party/cutlass", "rev-parse", "HEAD"]),
            "fa2_commit": command(["git", "-C", ROOT / "third_party/flash-attention", "rev-parse", "HEAD"]),
            "fa2_dirty": command(["git", "-C", ROOT / "third_party/flash-attention", "status", "--short"]),
            "source_sha256": hashlib.sha256((HERE / "kernel_snapshot.cu").read_bytes()).hexdigest(),
            "compile_commands": compile_commands,
            "iterations_per_graph": args.iterations, "rounds": args.rounds,
            "method": "CUDA Graph repeated forward; preallocated outputs; CUDA events; alternating order; warm same addresses; no cache flush",
            "tolerance": {"atol": 0.002, "rtol": 0.002}, "seed": 20260909,
            "official_binding": "upstream D64 kernels via local thin binding, not public Python API latency"}
    rows, checks = [], []
    cases = [(1, 1, n) for n in (64, 128, 192, 256, 512, 1024, 2048, 4096)]
    cases += [(1, 8, n) for n in (512, 2048, 4096)] + [(4, 8, n) for n in (512, 2048)]
    if args.quick:
        cases = [(1, 1, 64), (1, 1, 192), (1, 1, 512)]
    for b, h, n in cases:
        q, k, v = [torch.randn(b, h, n, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
        for causal in (False, True):
            outputs = {name: torch.empty_like(q) for name in ("cute_s2", "cute_s3", "official_fa2")}
            lse = torch.empty(b, h, n, device="cuda", dtype=torch.float32)
            def cute(stage):
                with tvm_ffi.use_torch_stream():
                    modules[stage](q, k, v, outputs[f"cute_s{stage}"], int(causal))
            funcs = {"cute_s2": lambda: cute(2), "cute_s3": lambda: cute(3),
                     "official_fa2": lambda: official.forward(q, k, v, outputs["official_fa2"], lse, causal)}
            ref = reference(q, k, v, causal)
            errors = {}
            for name, fn in funcs.items():
                fn()
                errors[name] = error(outputs[name], ref)
            checks.append({"shape": [b, h, n, 64], "causal": causal, "errors": errors})
            graphs = {name: graph_for(fn, args.iterations) for name, fn in funcs.items()}
            warm_start = time.perf_counter()
            while time.perf_counter() - warm_start < 0.2:
                for graph in graphs.values():
                    graph.replay()
                torch.cuda.synchronize()
            torch.cuda.synchronize()
            samples = {name: [] for name in funcs}
            for round_id in range(args.rounds):
                names = list(funcs)
                names = names[round_id % 3:] + names[:round_id % 3]
                for name in names:
                    samples[name].append(measure(graphs[name], args.iterations))
            for name, values in samples.items():
                median = statistics.median(values)
                flops = 4 * b * h * n * n * 64 if not causal else 2 * b * h * n * (n + 1) * 64
                rows.append({"batch": b, "heads": h, "n": n, "d": 64, "causal": causal,
                             "implementation": name, "median_us": median, "min_us": min(values),
                             "max_us": max(values), "useful_tflops": flops / (median * 1e6),
                             **errors[name], "samples_us": values})
            print(b, h, n, causal, {x: round(statistics.median(y), 3) for x, y in samples.items()}, flush=True)
            del graphs, ref
            (args.output / "results.json").write_text(json.dumps({"environment": meta, "checks": checks, "results": rows}, indent=2))
    # Probe uniform attention and more concentrated softmax, including pipeline tails.
    for pattern in ("uniform", "scaled"):
        q, k, v = [torch.randn(1, 1, 320, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
        if pattern == "uniform":
            q.zero_()
        else:
            q.mul_(4)
            k.mul_(4)
        for causal in (False, True):
            ref = reference(q, k, v, causal)
            for stage in (2, 3):
                out = torch.empty_like(q)
                with tvm_ffi.use_torch_stream():
                    modules[stage](q, k, v, out, int(causal))
                checks.append({"pattern": pattern, "n": 320, "causal": causal, "stages": stage, **error(out, ref)})
    meta["driver_after"] = command(["nvidia-smi", "--query-gpu=driver_version,name,power.limit,clocks.sm,temperature.gpu", "--format=csv"])
    (args.output / "results.json").write_text(json.dumps({"environment": meta, "checks": checks, "results": rows}, indent=2))
    with (args.output / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[x for x in rows[0] if x != "samples_us"])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "samples_us"} for row in rows)
    print("PASS: all full-row FP32 comparisons and stress checks", flush=True)


if __name__ == "__main__":
    main()
