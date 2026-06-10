# ternary-compress

Compression codecs for ternary data sequences — RLE, sparse formats, and dictionary encoding for {-1, 0, +1}.

## Why This Exists

Ternary neural network weights are mostly zeros. A typical ternary model after sign quantization has 60-80% zeros. Standard compression (gzip, LZ4) doesn't understand ternary structure — it sees bytes, not trits. This crate provides compression codecs that exploit ternary-specific patterns: run-length encoding for long sequences of the same value, sparse format for mostly-zero data, and dictionary compression for repeated ternary patterns across weight matrices.

## Architecture

### Compression Formats

- **RLE (Run-Length Encoding)**: `rle_encode` / `rle_decode` — Pairs of (value, count). Efficient for weight matrices with long runs of zeros or identical values.
- **Sparse Ternary**: `SparseTernary` — Stores only non-zero values with their indices. Includes `density()` and `compression_ratio()`.
- **Dictionary Compression**: `TernaryDict` — Builds a dictionary of common patterns, encodes data as dictionary indices. Good for repeated sub-patterns across layers.
- **Trit packing**: `pack_trits` / `unpack_trits` — Low-level 16-trit → u32 packing.

## Usage

```rust
use ternary_compress::*;

// RLE for weight matrices
let weights = vec![0, 0, 0, 0, 1, 0, 0, -1, -1, 0i8];
let rle = rle_encode(&weights);
// [(0, 4), (1, 1), (0, 2), (-1, 2), (0, 1)]
let decoded = rle_decode(&rle);
assert_eq!(weights, decoded);

// Sparse format for mostly-zero data
let dense = vec![0, 0, 1, 0, 0, -1, 0, 0, 0, 1i8];
let sparse = SparseTernary::from_dense(&dense);
println!("Density: {:.1}%", sparse.density() * 100.0);
println!("Compression: {:.1}×", sparse.compression_ratio());

// Dictionary compression
let mut dict = TernaryDict::new();
let data = vec![1, 0, -1, 0, 1, 0, -1, 0]; // repeating pattern
let encoded = dict.encode(&data, 4); // chunk size 4
let decoded = dict.decode(&encoded, 4);
```

## API Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `rle_encode(data)` | `Vec<(i8, usize)>` | Run-length encode ternary data |
| `rle_decode(runs)` | `Vec<i8>` | Decode RLE back to dense |
| `SparseTernary::from_dense(data)` | `SparseTernary` | Create sparse representation |
| `sparse.to_dense()` | `Vec<i8>` | Reconstruct dense array |
| `sparse.density()` | `f64` | Fraction of non-zero values |
| `sparse.compression_ratio()` | `f64` | Dense size / sparse size |
| `TernaryDict::new()` | `TernaryDict` | Create empty dictionary |
| `dict.encode(data, chunk_size)` | `Vec<usize>` | Dictionary-encode data |
| `dict.decode(indices, chunk_size)` | `Vec<i8>` | Decode back to ternary |
| `pack_trits(trits)` | `u32` | Pack 16 trits into one u32 |
| `unpack_trits(packed)` | `[i8; 16]` | Unpack u32 to 16 trits |

## The Deeper Idea

Ternary compression is qualitatively different from binary compression because the **zero value is semantically meaningful**, not just "absent." In binary compression, a zero bit means "off." In ternary compression, zero means "the model chose not to activate this connection" — it's a deliberate decision boundary, not empty space. This means sparse formats must preserve zero positions faithfully, and dictionary compression should treat {+1, 0, -1} triples as atomic units.

## Related Crates

- **ternary-pack** — bit-packing trits into u32 registers
- **ternary-bloom-filter** — ternary Bloom filters for membership testing
- **ternary-sketch** — approximate counting with ternary sketch
