"""A deterministic E2M1 counterexample, using only Python's standard library.

Run: python content/llm_inference/code/rotation_quantization_probe.py
The scale is a real-valued amax/6: this is NOT an NVFP4/MXFP4 emulator.
"""

import json
import math


def hadamard(x):
    """Normalized Sylvester Hadamard; symmetric and its own inverse."""
    n = len(x)
    if n == 0 or n & (n - 1):
        raise ValueError("length must be a positive power of two")
    out = list(x)
    step = 1
    while step < n:
        for start in range(0, n, 2 * step):
            for j in range(start, start + step):
                a, b = out[j], out[j + step]
                out[j], out[j + step] = a + b, a - b
        step *= 2
    return [v / math.sqrt(n) for v in out]


def quantize_e2m1_real_scale(x):
    if not x or not all(math.isfinite(v) for v in x):
        raise ValueError("input must be nonempty and finite")
    scale = max(map(abs, x)) / 6.0
    if scale == 0:
        return list(x)
    positive = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    # On exact ties, use the E2M1 code with an even low bit.
    def nearest(v):
        index = min(range(8), key=lambda i: (abs(abs(v) / scale - positive[i]), i % 2))
        return math.copysign(positive[index] * scale, v)
    return [nearest(v) for v in x]


def metrics(reference, reconstructed):
    energy = sum(v * v for v in reference)
    error = sum((a - b) ** 2 for a, b in zip(reference, reconstructed))
    return {"squared_error": error, "nmse": error / energy if energy else 0.0}


def main():
    x = [6.0, 1.0, 0.5] + [0.0] * 13
    rotated = hadamard(x)
    restored = hadamard(rotated)
    direct = metrics(x, quantize_e2m1_real_scale(x))
    after_rotation = metrics(x, hadamard(quantize_e2m1_real_scale(rotated)))
    assert max(abs(a - b) for a, b in zip(x, restored)) < 1e-12
    assert math.isclose(sum(v * v for v in x), sum(v * v for v in rotated))
    assert direct["squared_error"] == 0.0
    assert after_rotation["squared_error"] > 0.0
    assert quantize_e2m1_real_scale([0.0] * 16) == [0.0] * 16
    print(json.dumps({
        "format": "E2M1, real amax/6 scale, one block of 16 (educational)",
        "amax_original": max(map(abs, x)),
        "amax_rotated": max(map(abs, rotated)),
        "direct_quantization": direct,
        "rotation_quantization_inverse": after_rotation,
    }, indent=2))


if __name__ == "__main__":
    main()
