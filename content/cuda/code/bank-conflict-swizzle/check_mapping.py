"""Check the article's 8x64, 16-bit shared-memory address model (no GPU)."""

from math import gcd


def swizzle(x):
    return x ^ ((x & 0x1C0) >> 3)


def offset(r, c):
    return 64 * r + (c ^ ((r & 7) << 3))


def words_per_bank(addresses):
    words = [set() for _ in range(32)]
    for address in addresses:
        assert address % 16 == 0
        for k in range(4):
            word = address // 4 + k
            words[word % 32].add(word)
    return [len(group) for group in words]


physical = set()
for r in range(8):
    for c in range(64):
        x = r * 64 + c
        p = offset(r, c)
        assert p == swizzle(x)
        assert swizzle(p) == x
        a = 2 * x
        assert 2 * p == a ^ ((a & 0x380) >> 3)
        physical.add(p)
assert physical == set(range(512))

for q in range(8):
    for r in range(8):
        p = offset(r, 8 * q)
        assert p % 8 == 0
        assert [offset(r, 8 * q + j) for j in range(8)] == list(range(p, p + 8))
    plain = words_per_bank([128 * r + 16 * q for r in range(8)])
    mixed = words_per_bank([2 * offset(r, 8 * q) for r in range(8)])
    padded = words_per_bank([144 * r + 16 * q for r in range(8)])
    assert sorted(plain) == [0] * 28 + [8] * 4
    assert mixed == padded == [1] * 32
    print(f"PASS q={q}: plain=8-way, swizzled=1-way, padded72=1-way")

# One 32-bit word per active lane; s=0 is excluded (broadcast).
for stride in (1, 2, 4, 8, 16, 32, 33):
    counts = [0] * 32
    for lane in range(32):
        counts[(stride * lane) % 32] += 1
    assert max(counts) == gcd(stride, 32)

assert words_per_bank([16 * r for r in range(8)]) == [1] * 32
assert all(swizzle(x) == x for x in range(64))
assert any((130 * r) % 16 != 0 for r in range(8))
print("PASS: coverage, involution, byte/element equivalence, vector continuity, scalar stride and compact/padding counterexamples")
