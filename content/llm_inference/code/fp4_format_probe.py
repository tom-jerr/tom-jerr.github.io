"""CPU numerical probes for the MXFP4/NVFP4 article (Python 3.10+, stdlib).

This is a scalar-format reference, not a CUDA kernel or bit-exact emulation
of a GPU accumulator. Inputs deliberately avoid extreme exponent ranges.
All scales below are DEQUANTIZATION multipliers. FP4 ties use even codes.
Run: python content/llm_inference/code/fp4_format_probe.py
"""

import json
import math
import random


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3 = tuple(
    mantissa * 2.0**-9 if exponent == 0
    else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    for exponent in range(16)
    for mantissa in range(8)
    if not (exponent == 15 and mantissa == 7)
)


def nearest(value, grid):
    """Round magnitude to nearest, ties to even encoding; saturate at max."""
    code = min(range(len(grid)), key=lambda i: (abs(abs(value) - grid[i]), i % 2))
    return math.copysign(grid[code], value)


def reconstruct(values, scale):
    return [scale * nearest(x / scale, E2M1) for x in values]


def mx(values, mode="safe"):
    amax = max(map(abs, values), default=0)
    if amax == 0:
        return list(values), 1.0
    if mode == "ocp":
        exponent = math.floor(math.log2(amax)) - 2
    elif mode == "safe":
        exponent = math.ceil(math.log2(amax / 6))
    elif mode == "oas":
        exponent = math.ceil(math.log2(amax / 7))
    else:
        raise ValueError(mode)
    scale = 2.0**exponent
    return reconstruct(values, scale), scale


def nv_candidate(values, target, tensor_scale=1.0):
    amax = max(map(abs, values), default=0)
    if amax == 0:
        return list(values), 1.0
    local_scale = nearest(amax / (target * tensor_scale), E4M3)
    if local_scale == 0:
        raise ValueError("Local scale underflow; choose a suitable tensor scale")
    effective_scale = tensor_scale * local_scale
    return reconstruct(values, effective_scale), effective_scale


def sse(a, b):
    return math.fsum((x - y) ** 2 for x, y in zip(a, b))


def main():
    assert len(E4M3) == 127 and max(E4M3) == 448
    assert nearest(0.25, E2M1) == 0 and nearest(3.5, E2M1) == 4
    assert nearest(-0.25, E2M1) == 0 and nearest(7, E2M1) == 6
    rng = random.Random(20260911)
    data = [rng.gauss(0, 1) for _ in range(32)]
    base, _ = mx(data)
    scaled, _ = mx([8 * x for x in data])
    homogeneous_error = max(abs(x - y / 8) for x, y in zip(base, scaled))
    assert homogeneous_error == 0

    # With the same block grouping and finite scales, the 7-boundary policy
    # should not increase squared error over the conservative 6-boundary one.
    checked = 0
    scale_changes = 0
    for _ in range(2000):
        # Vary the block maximum across scale boundaries. A fixed uniform
        # range would concentrate maxima near one value and miss switches.
        amax = 2.0 ** rng.uniform(-6, 6)
        values = [amax] + [amax * rng.uniform(-1, 1) for _ in range(31)]
        safe, safe_scale = mx(values, "safe")
        oas, oas_scale = mx(values, "oas")
        assert sse(values, oas) <= sse(values, safe) + 1e-10
        scale_changes += safe_scale != oas_scale
        checked += 1
    assert scale_changes > 0

    oas_demo = {}
    values = [3.3, 0.2] + [0.0] * 30
    for mode in ("ocp", "safe", "oas"):
        quantized, scale = mx(values, mode)
        oas_demo[mode] = {"scale": scale, "first_two": quantized[:2], "sse": sse(values, quantized)}

    four_six = []
    for first in ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 8.0, 12.0]):
        values = first + [0.0] * 12
        candidates = {}
        for target in (4, 6):
            quantized, scale = nv_candidate(values, target)
            candidates[str(target)] = {"scale": scale, "first_four": quantized[:4], "sse": sse(values, quantized)}
        four_six.append({"first_four": first, "tensor_scale": 1.0, "candidates": candidates})

    # Shared SR randomness: Q(1.2) is 1 or 1.5 with probabilities .6 and .4.
    expected_product_shared = 0.6 * 1.0**2 + 0.4 * 1.5**2
    expected_product_independent = (0.6 * 1.0 + 0.4 * 1.5) ** 2
    assert math.isclose(expected_product_shared, 1.5)
    assert math.isclose(expected_product_independent, 1.44)

    # MBS scales varying along K must multiply partial products before sum.
    partial_products = [1.0, 1.0]
    scale_products = [1.0, 2.0]
    correct = sum(p * s for p, s in zip(partial_products, scale_products))
    wrong = sum(partial_products) * scale_products[0]
    assert correct == 3 and wrong == 2

    # Symmetric square-tile quantization commutes with transpose;
    # independently rowwise-quantized operands generally do not.
    matrix = [[rng.gauss(0, 1) * (1 + row) for _ in range(32)] for row in range(32)]
    transposed = list(map(list, zip(*matrix)))
    rowwise = [mx(row)[0] for row in matrix]
    transpose_first = [mx(row)[0] for row in transposed]
    discrepancy = sse([x for row in zip(*rowwise) for x in row],
                      [x for row in transpose_first for x in row])
    assert discrepancy > 0
    flat = [x for row in matrix for x in row]
    _, common_scale = mx(flat)
    tile = [reconstruct(row, common_scale) for row in matrix]
    tile_t = [reconstruct(row, common_scale) for row in transposed]
    assert list(map(list, zip(*tile))) == tile_t

    print(json.dumps({
        "scope": "CPU scalar reference; no GPU performance measurements",
        "bits_per_element": {"mxfp4": 4 + 8 / 32, "nvfp4": 4 + 8 / 16},
        "oas_example": oas_demo,
        "four_over_six_examples": four_six,
        "power_of_two_rescaling_error": homogeneous_error,
        "oas_vs_safe_blocks_checked": checked,
        "oas_vs_safe_scale_changes": scale_changes,
        "sr_product": {"shared": expected_product_shared, "independent": expected_product_independent},
        "macro_scaling": {"correct": correct, "incorrect_final_scale": wrong},
        "transpose": {"rowwise_difference_sse": discrepancy, "square_tile_consistent": True},
    }, indent=2))


if __name__ == "__main__":
    main()
