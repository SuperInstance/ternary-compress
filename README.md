# ternary-compress

Ternary data compression for sparse GPU workloads. Provides run-length encoding (RLE), sparse ternary storage, dictionary compression, and 2-bit trit packing — all designed for data in the balanced ternary alphabet {-1, 0, +1}.

## Why It Matters

Modern quantized neural networks (e.g., BinaryConnect, ternary weight networks) represent weights as {-1, 0, +1}. A model with 100M parameters stored as `int8` consumes 100 MB, but the information content is only ~158 MB ÷ 3.17 ≈ 50 MB at 2 bits/value — and far less after compression, since ternary weights are typically **highly sparse** (60–90% zeros).

This crate provides the compression primitives tailored to ternary data patterns found in:
- Ternary Weight Networks (TWNs) [Li et al., 2016]
- Sparse attention matrices with ternary pruning
- Fleet agent state snapshots (sparse ternary diffs)

Within the **γ + η = C** framework:

| Symbol | Domain |
|--------|--------|
| γ | Input data ∈ {-1, 0, +1}^n |
| η | Compression strategy: RLE vs. sparse vs. dictionary vs. bit-packing |
| C | Information-theoretic bound: $H(X) \leq \log_2(3) \approx 1.585$ bits/symbol |

## How It Works

### Run-Length Encoding (RLE)

Consecutive identical values are encoded as `(value, count)` pairs. For data with long runs (common in pruned neural network layers where entire rows are zero):

$$\text{RLE}(x) = [(v_1, c_1), (v_2, c_2), \ldots, (v_k, c_k)]$$

where $v_i \neq v_{i+1}$ and $\sum c_i = n$.

**Compression ratio**: $\frac{2k}{n}$ (each pair stores one `i8` value + one `usize` count). For data with average run length $\bar{r} = n/k$:

- $\bar{r} > 16$ (64-bit counts): RLE wins
- $\bar{r} < 16$: RLE loses (overhead exceeds savings)

Run counts are capped at 255 to allow single-byte storage in future serialization formats.

**Complexity**: O(n) encode, O(n) decode.

### Sparse Ternary Format

Stores only non-zero values with their indices:

```
Dense:    [0, 1, 0, 0, -1, 0, 0, 0, 1, 0]   (10 bytes as i8)
Sparse:   len=10, indices=[1,4,8], values=[1,-1,1]
```

**Density** $d = \frac{|\text{non-zero}|}{n}$. For $d < \frac{2}{2 + 8}$ (on 8-bit indices), sparse is smaller than dense 2-bit encoding.

**Compression ratio** (sparse vs. dense 2-bit):

$$R = \frac{n \cdot 2}{k \cdot (8 + 2)} = \frac{2n}{10k}$$

where $k$ = non-zero count. Sparse wins when $d < 0.2$ (20% density).

For GPU workloads where 80–90% of weights are zero, this yields **5–10× compression**.

**Complexity**: O(n) encode, O(n) decode.

### Dictionary Compression

Repeated fixed-length patterns are replaced with dictionary indices:

```
Data:     [1,-1,0, 1,-1,0, 1,-1,0]   (9 values, chunk_size=3)
Dict:     {0: [1,-1,0]}
Encoded:  [0, 0, 0]   (3 indices)
```

Dictionary construction is O(n) for fixed chunk size. Encoding is O(n/chunk_size) with O(1) HashMap lookup per chunk.

**Best case**: All chunks identical → 1 dictionary entry + n/chunk_size indices.

### 2-Bit Trit Packing

Each trit is stored in 2 bits, packing **16 trits per `u32`** (or 4 per `u8`):

| Trit | Bit Pattern |
|------|-------------|
| -1   | `11` |
|  0   | `00` |
| +1   | `01` |

```
Pack:   [1, -1, 0, 1, ...] → 0b...01_00_11_01
Unpack: u32 → [1, -1, 0, 1, ...]
```

This gives a constant **4× compression** vs. `int8` storage (2 bits vs. 8 bits per trit). The `10` bit pattern is unused, providing room for a future "extension" marker.

**Complexity**: O(k) for both pack and unpack, where k = number of trits (up to 16 per call).

### Theoretical Bounds

The entropy of a ternary source with probabilities $(p_{-1}, p_0, p_{+1})$ is:

$$H(X) = -\sum_{i} p_i \log_2 p_i$$

For uniform ternary: $H = \log_2 3 \approx 1.585$ bits/symbol. Since we use 2 bits/symbol, the coding efficiency is:

$$\eta = \frac{\log_2 3}{2} \approx 79.2\%$$

For highly skewed distributions (e.g., $p_0 = 0.9, p_{-1} = p_{+1} = 0.05$):

$$H = -0.9\log_2 0.9 - 2 \times 0.05\log_2 0.05 \approx 0.439 + 0.432 = 0.569 \text{ bits/symbol}$$

In this regime, dictionary or Huffman coding can achieve **3.5× compression** beyond 2-bit packing.

## Quick Start

```rust
use ternary_compress::{rle_encode, rle_decode, SparseTernary, TernaryDict, pack_trits, unpack_trits};

// Run-length encoding
let data = vec![1, 1, 1, 0, 0, -1, -1, -1, -1, 0];
let encoded = rle_encode(&data);
assert_eq!(encoded, vec![(1, 3), (0, 2), (-1, 4), (0, 1)]);
assert_eq!(rle_decode(&encoded), data);

// Sparse format (ideal for >80% zeros)
let sparse = SparseTernary::from_dense(&[0, 1, 0, 0, -1, 0, 0, 0, 1, 0]);
assert_eq!(sparse.density(), 0.3);
assert_eq!(sparse.to_dense(), vec![0, 1, 0, 0, -1, 0, 0, 0, 1, 0]);

// Dictionary compression
let mut dict = TernaryDict::new();
let pattern = vec![1, -1, 0];
let data2 = vec![1, -1, 0, 1, -1, 0]; // two identical chunks
let encoded2 = dict.encode(&data2, 3);
assert_eq!(dict.dict_size(), 1); // one unique pattern

// Bit packing: 16 trits → 1 u32
let trits = [1i8, -1, 0, 1, 0, -1, 1, 1, -1, 0, 0, 1, -1, 0, 1, -1];
let packed = pack_trits(&trits);
assert_eq!(unpack_trits(packed), trits);
```

## API

### RLE

| Function | Signature |
|----------|-----------|
| `rle_encode` | `(&[i8]) -> Vec<(i8, usize)>` |
| `rle_decode` | `(&[(i8, usize)]) -> Vec<i8>` |

### `SparseTernary`

| Method | Description |
|--------|-------------|
| `from_dense(&[i8])` | Encode dense array to sparse |
| `to_dense() -> Vec<i8>` | Decode back to dense |
| `density() -> f64` | Fraction of non-zero values |
| `compression_ratio() -> f64` | Dense bits / sparse bits |

### `TernaryDict`

| Method | Description |
|--------|-------------|
| `new()` | Empty dictionary |
| `add(pattern) -> usize` | Add pattern, return index |
| `get(idx) -> Option<&[i8]>` | Look up pattern |
| `encode(data, chunk_size) -> Vec<usize>` | Encode with fixed-size chunks |
| `decode(indices, chunk_size) -> Vec<i8>` | Decode back to flat array |
| `dict_size() -> usize` | Number of unique patterns |

### Bit Packing

| Function | Signature |
|----------|-----------|
| `pack_trits` | `(&[i8]) -> u32` (max 16 trits) |
| `unpack_trits` | `(u32) -> [i8; 16]` |

## Architecture Notes

This crate is optimized for **GPU upload paths**: the sparse format's `indices` and `values` vectors map directly to GPU buffer uploads, avoiding host-side reconstruction. The 2-bit packing format is designed to be decoded on-GPU using simple bit-shift operations in compute shaders.

The dictionary compressor uses a **greedy longest-match** strategy — at each position, it finds the longest dictionary entry that matches. This is O(n · d) where d is dictionary size, but with typical dictionary sizes (< 256 entries), the constant factor is tiny.

The choice of `i8` (not a custom Trit enum) as the canonical type is deliberate: it allows drop-in compatibility with existing ML pipelines that store quantized weights as `int8_t`.

## References

- **Li, F., Zhang, B., & Liu, B.** (2016). "Ternary Weight Networks." *arXiv:1605.04711*. — Ternary quantization for neural networks.
- **Courbariaux, M., Bengio, Y., & David, J.-P.** (2015). "BinaryConnect: Training Deep Neural Networks with binary weights." *NeurIPS*. — Binary/ternary weight quantization.
- **Zehler, P., et al.** (2021). "Revisiting Sparse Ternary Neural Network Weight Compression." — Sparse encoding for ternary weights.
- **Salomon, D.** (2007). *Handbook of Data Compression*. — RLE, dictionary methods, information-theoretic bounds.
- **Shannon, C. E.** (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27. — Entropy bounds.

## License

MIT
