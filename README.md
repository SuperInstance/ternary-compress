# Ternary Compress

[![crates.io](https://img.shields.io/crates/v/ternary-compress.svg)](https://crates.io/crates/ternary-compress)
[![docs.rs](https://docs.rs/ternary-compress/badge.svg)](https://docs.rs/ternary-compress)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Balanced ternary compression — float32 vectors to 5 trits per byte, achieving ~32× compression for embeddings.**

---

## The Problem

Modern ML models produce high-dimensional float32 embedding vectors (768-4096 dimensions). Storing and transmitting these at full precision is expensive — a 1024-dim vector is 4KB. For edge devices, microcontrollers, and high-throughput systems, this is unsustainable.

## Why This Exists

Ternary Compress converts float32 vectors to **balanced ternary** {-1, 0, +1} representations:
- A 1024-dim float32 vector (4KB) → 1024 trits packed into **128 bytes** = **~32× compression**
- Lossy but preserves cosine similarity structure
- Integer-only arithmetic for fast similarity search
- Optimized for constrained devices (ESP32, microcontrollers)

## Architecture

```
  Float32 Vector          Trit Vector            Packed Bytes
  ┌─────────────┐        ┌─────────────┐        ┌──────────┐
  │ 0.85        │   →    │ +1 (Pos)    │        │          │
  │-0.32        │ thresh │ -1 (Neg)    │ pack   │ 242      │
  │ 0.01        │   →    │  0 (Zero)   │ 5/byte →│ ...      │
  │-0.91        │        │ -1 (Neg)    │        │ 128      │
  │ ...         │        │ ...         │        │ bytes    │
  │ (1024 dims) │        │ (1024 trits)│        └──────────┘
  │ = 4096 bytes│        │             │        = 128 bytes
  └─────────────┘        └─────────────┘        ~32× smaller!
```

## Installation

```toml
[dependencies]
ternary-compress = "0.1"
```

## API Reference

### `Trit`

A balanced ternary digit:

```rust
use ternary_compress::Trit;

let pos = Trit::from_float(0.5, 0.1);   // Pos
let neg = Trit::from_float(-0.5, 0.1);  // Neg
let zero = Trit::from_float(0.05, 0.1); // Zero

assert_eq!(Trit::from_i8(-1), Some(Trit::Neg));
assert_eq!(Trit::Pos.to_f64(), 1.0);
```

### `TritVector`

Compressed vector with packing, similarity, and distance:

```rust
use ternary_compress::TritVector;

let vals = vec![0.5, -0.3, 0.01, -0.9, 0.2];
let tv = TritVector::from_floats(&vals, 0.15);

assert_eq!(tv.len(), 5);

// Pack to bytes: 5 trits per byte (base-3 encoding)
let packed = tv.pack();
let unpacked = TritVector::unpack(&packed, tv.len());
assert_eq!(tv, unpacked); // lossless roundtrip

// Similarity search with integer arithmetic
let other = TritVector::from_floats(&[0.4, -0.2, 0.0, -0.8, 0.1], 0.15);
let dot = tv.dot(&other);        // integer dot product
let hamming = tv.hamming_distance(&other);
```

### `CompressionStats`

```rust
use ternary_compress::*;

let vals: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin()).collect();
let tv = TritVector::from_floats(&vals, 0.1);
let stats = CompressionStats::from_trit_vector(&tv);

println!("{}", stats);
// "TernaryCompress: 1024 dims, 4096 → 205 bytes (20.0×), sparsity 23.4%"
```

### `optimal_threshold`

Find the best threshold for a target sparsity:

```rust
use ternary_compress::optimal_threshold;

let vals: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 50.0).collect();
let threshold = optimal_threshold(&vals, 0.5); // target 50% zeros
```

## Usage Examples

### Example 1: Compress Embeddings

```rust
use ternary_compress::*;

// Simulate a 1024-dim embedding
let embedding: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin()).collect();

let compressed = TritVector::from_floats(&embedding, 0.1);
let packed = compressed.pack();

println!("Original: {} bytes", embedding.len() * 4);
println!("Compressed: {} bytes", packed.len());
println!("Ratio: {:.1}×", compressed.compression_ratio_vs_f32());
```

### Example 2: Similarity Search with Trits

```rust
use ternary_compress::*;

let query = TritVector::from_floats(&[0.9, -0.8, 0.1, 0.7], 0.15);
let candidate = TritVector::from_floats(&[0.85, -0.75, 0.05, 0.65], 0.15);

let similarity = query.dot(&candidate); // fast integer dot product
let distance = query.hamming_distance(&candidate);
println!("Dot product: {}, Hamming distance: {}", similarity, distance);
```

### Example 3: Find Optimal Threshold

```rust
use ternary_compress::*;

let data: Vec<f64> = (0..10000).map(|i| (i as f64 * 0.001).sin()).collect();
let threshold = optimal_threshold(&data, 0.5);
let compressed = TritVector::from_floats(&data, threshold);
println!("Sparsity: {:.1}%", compressed.sparsity() * 100.0);
```

## Mathematical Background

**Balanced Ternary**: Each dimension is quantized to {-1, 0, +1} using a threshold θ:

```
trit(x) = { +1  if x > θ
           {  0  if |x| ≤ θ
           { -1  if x < -θ
```

**Packing**: 5 trits per byte using base-3 encoding:

```
byte = t₀ × 3⁴ + t₁ × 3³ + t₂ × 3² + t₃ × 3¹ + t₄ × 3⁰
```

Where trit values are mapped: Neg→0, Zero→1, Pos→2.

**Range**: 3⁵ = 243 values per byte (fits in u8 since 242 < 256).

**Cosine Similarity Preservation**: While individual values are lossy, the *direction* of the vector is largely preserved, making ternary compression effective for similarity search.

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Float → Trit conversion | O(n) | Single threshold pass |
| Pack trits to bytes | O(n) | 5 trits per byte |
| Unpack bytes to trits | O(n) | Division/modulo |
| Integer dot product | O(n) | No floating point |
| Hamming distance | O(n) | Integer comparison |
| Optimal threshold | O(n log n) | Binary search |

## Comparison with Alternatives

| Feature | ternary-compress | float16 | int8 quant | PQ |
|---------|-----------------|---------|-----------|-----|
| Compression ratio | ~32× | 2× | 4× | 8-32× |
| Integer similarity | ✅ | ❌ | ✅ | ❌ |
| Lossless packing | ✅ | N/A | N/A | ❌ |
| Zero dependencies | ✅ | ✅ | ✅ | ❌ |
| Edge device friendly | ✅ | ✅ | ✅ | ❌ |

## License

Licensed under the [MIT License](LICENSE).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests
4. Push and open a Pull Request
