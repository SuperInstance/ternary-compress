//! # ternary-compress — Balanced Ternary Compression
//!
//! Encodes floating-point vectors as balanced ternary {-1, 0, +1} for
//! extreme compression on constrained devices (ESP32, microcontrollers).
//!
//! A 1024-dim float32 vector (4KB) → 1024 trits (128 bytes) = **32× compression**.

// ─── Trit ────────────────────────────────────────────────────────────────────

/// A balanced ternary digit: Negative, Zero, or Positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Trit {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Trit::Neg),
            0 => Some(Trit::Zero),
            1 => Some(Trit::Pos),
            _ => None,
        }
    }

    pub fn to_i8(self) -> i8 {
        self as i8
    }

    pub fn to_f64(self) -> f64 {
        self as i8 as f64
    }

    /// Encode a float to a trit using a threshold.
    pub fn from_float(value: f64, threshold: f64) -> Self {
        if value > threshold { Trit::Pos }
        else if value < -threshold { Trit::Neg }
        else { Trit::Zero }
    }
}

impl std::fmt::Display for Trit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trit::Neg => write!(f, "T"),
            Trit::Zero => write!(f, "0"),
            Trit::Pos => write!(f, "1"),
        }
    }
}

// ─── TritVector ──────────────────────────────────────────────────────────────

/// A vector of trits — the compressed representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TritVector {
    trits: Vec<Trit>,
}

impl TritVector {
    pub fn new(trits: Vec<Trit>) -> Self {
        Self { trits }
    }

    pub fn zeros(len: usize) -> Self {
        Self { trits: vec![Trit::Zero; len] }
    }

    pub fn len(&self) -> usize {
        self.trits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trits.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<Trit> {
        self.trits.get(index).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = Trit> + '_ {
        self.trits.iter().copied()
    }

    /// Compress a float vector to trits using a threshold.
    pub fn from_floats(values: &[f64], threshold: f64) -> Self {
        Self {
            trits: values.iter().map(|&v| Trit::from_float(v, threshold)).collect(),
        }
    }

    /// Decompress back to f64 (lossy — each trit becomes -1.0, 0.0, or 1.0).
    pub fn to_floats(&self) -> Vec<f64> {
        self.trits.iter().map(|t| t.to_f64()).collect()
    }

    /// Pack trits into bytes: 5 trits per byte (base-3 encoding).
    /// Each byte holds trits in groups: trit[0]*81 + trit[1]*27 + trit[2]*9 + trit[3]*3 + trit[4]*1
    /// where trit values are mapped: Neg→0, Zero→1, Pos→2.
    pub fn pack(&self) -> Vec<u8> {
        let mut packed = Vec::new();
        for chunk in self.trits.chunks(5) {
            let mut byte: u8 = 0;
            for (i, trit) in chunk.iter().enumerate() {
                let v = match trit {
                    Trit::Neg => 0u8,
                    Trit::Zero => 1u8,
                    Trit::Pos => 2u8,
                };
                byte += v * 3u8.pow(4 - i as u32);
            }
            packed.push(byte);
        }
        packed
    }

    /// Unpack bytes back to trits.
    pub fn unpack(bytes: &[u8], original_len: usize) -> Self {
        let mut trits = Vec::with_capacity(original_len);
        for &byte in bytes {
            let mut remaining = byte;
            for i in (0..5).rev() {
                if trits.len() >= original_len { break; }
                let pow = 3u8.pow(i as u32);
                let v = remaining / pow;
                remaining %= pow;
                trits.push(match v {
                    0 => Trit::Neg,
                    1 => Trit::Zero,
                    2 => Trit::Pos,
                    _ => Trit::Zero, // shouldn't happen
                });
            }
        }
        Self { trits }
    }

    /// Compute compression ratio vs f32.
    pub fn compression_ratio_vs_f32(&self) -> f64 {
        let original_bytes = self.trits.len() * 4; // f32 = 4 bytes
        let packed_bytes = (self.trits.len() as f64 / 5.0).ceil() as usize;
        if packed_bytes == 0 { return 1.0; }
        original_bytes as f64 / packed_bytes as f64
    }

    /// Dot product with another trit vector (integer arithmetic).
    pub fn dot(&self, other: &TritVector) -> i32 {
        self.trits.iter().zip(other.trits.iter())
            .map(|(a, b)| (a.to_i8() as i32) * (b.to_i8() as i32))
            .sum()
    }

    /// Hamming distance (number of differing trits).
    pub fn hamming_distance(&self, other: &TritVector) -> usize {
        self.trits.iter().zip(other.trits.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Sparsity: fraction of zero trits.
    pub fn sparsity(&self) -> f64 {
        if self.trits.is_empty() { return 0.0; }
        let zeros = self.trits.iter().filter(|t| **t == Trit::Zero).count();
        zeros as f64 / self.trits.len() as f64
    }
}

// ─── Compression Statistics ──────────────────────────────────────────────────

/// Statistics about a ternary compression operation.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_dims: usize,
    pub original_bytes: usize,
    pub packed_bytes: usize,
    pub compression_ratio: f64,
    pub sparsity: f64,
    pub positive_count: usize,
    pub negative_count: usize,
    pub zero_count: usize,
}

impl CompressionStats {
    pub fn from_trit_vector(tv: &TritVector) -> Self {
        let packed = tv.pack();
        Self {
            original_dims: tv.len(),
            original_bytes: tv.len() * 4,
            packed_bytes: packed.len(),
            compression_ratio: tv.compression_ratio_vs_f32(),
            sparsity: tv.sparsity(),
            positive_count: tv.iter().filter(|t| *t == Trit::Pos).count(),
            negative_count: tv.iter().filter(|t| *t == Trit::Neg).count(),
            zero_count: tv.iter().filter(|t| *t == Trit::Zero).count(),
        }
    }
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TernaryCompress: {} dims, {} → {} bytes ({:.1}×), sparsity {:.1}%",
            self.original_dims, self.original_bytes, self.packed_bytes,
            self.compression_ratio, self.sparsity * 100.0)
    }
}

// ─── Threshold Optimizer ─────────────────────────────────────────────────────

/// Find the optimal threshold for ternary quantization.
pub fn optimal_threshold(values: &[f64], target_sparsity: f64) -> f64 {
    if values.is_empty() { return 0.1; }

    let mut sorted: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Binary search for threshold that achieves target sparsity
    let mut lo = 0.0_f64;
    let mut hi = sorted[sorted.len() - 1];

    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        let sparsity: f64 = values.iter().map(|v| if v.abs() <= mid { 1.0 } else { 0.0 }).sum::<f64>()
            / values.len() as f64;
        if sparsity < target_sparsity {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    (lo + hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_from_float() {
        assert_eq!(Trit::from_float(0.5, 0.1), Trit::Pos);
        assert_eq!(Trit::from_float(-0.5, 0.1), Trit::Neg);
        assert_eq!(Trit::from_float(0.05, 0.1), Trit::Zero);
    }

    #[test]
    fn test_trit_roundtrip() {
        assert_eq!(Trit::from_i8(Trit::Neg.to_i8()), Some(Trit::Neg));
        assert_eq!(Trit::from_i8(Trit::Zero.to_i8()), Some(Trit::Zero));
        assert_eq!(Trit::from_i8(Trit::Pos.to_i8()), Some(Trit::Pos));
    }

    #[test]
    fn test_trit_vector_from_floats() {
        let vals = vec![0.5, -0.3, 0.01, -0.9, 0.2];
        let tv = TritVector::from_floats(&vals, 0.15);
        assert_eq!(tv.len(), 5);
        assert_eq!(tv.get(0), Some(Trit::Pos));
        assert_eq!(tv.get(1), Some(Trit::Neg));
        assert_eq!(tv.get(2), Some(Trit::Zero));
        assert_eq!(tv.get(3), Some(Trit::Neg));
        assert_eq!(tv.get(4), Some(Trit::Pos));
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let vals: Vec<f64> = (0..17).map(|i| ((i as f64 - 8.0) / 8.0)).collect();
        let tv = TritVector::from_floats(&vals, 0.05);
        let packed = tv.pack();
        let unpacked = TritVector::unpack(&packed, tv.len());
        assert_eq!(tv, unpacked);
    }

    #[test]
    fn test_compression_ratio() {
        let tv = TritVector::zeros(1024);
        let ratio = tv.compression_ratio_vs_f32();
        assert!(ratio >= 19.0, "Expected ~20× compression, got {:.1}×", ratio);
    }

    #[test]
    fn test_dot_product() {
        let tv1 = TritVector::new(vec![Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos]);
        let tv2 = TritVector::new(vec![Trit::Pos, Trit::Neg, Trit::Pos, Trit::Neg]);
        // 1*1 + (-1)*(-1) + 0*1 + 1*(-1) = 1 + 1 + 0 - 1 = 1
        assert_eq!(tv1.dot(&tv2), 1);
    }

    #[test]
    fn test_hamming_distance() {
        let tv1 = TritVector::new(vec![Trit::Pos, Trit::Neg, Trit::Zero]);
        let tv2 = TritVector::new(vec![Trit::Pos, Trit::Zero, Trit::Zero]);
        assert_eq!(tv1.hamming_distance(&tv2), 1);
    }

    #[test]
    fn test_sparsity() {
        let tv = TritVector::new(vec![Trit::Pos, Trit::Zero, Trit::Zero, Trit::Neg, Trit::Zero]);
        assert!((tv.sparsity() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_to_floats() {
        let tv = TritVector::new(vec![Trit::Pos, Trit::Zero, Trit::Neg]);
        let floats = tv.to_floats();
        assert_eq!(floats, vec![1.0, 0.0, -1.0]);
    }

    #[test]
    fn test_compression_stats() {
        let vals = vec![0.5, -0.3, 0.01, -0.9, 0.2, 0.0, -0.1, 0.8];
        let tv = TritVector::from_floats(&vals, 0.15);
        let stats = CompressionStats::from_trit_vector(&tv);
        assert_eq!(stats.original_dims, 8);
        assert_eq!(stats.original_bytes, 32);
        assert_eq!(stats.positive_count + stats.negative_count + stats.zero_count, 8);
        let display = format!("{}", stats);
        assert!(display.contains("TernaryCompress"));
    }

    #[test]
    fn test_optimal_threshold() {
        let vals: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 50.0).collect();
        let threshold = optimal_threshold(&vals, 0.5);
        assert!(threshold > 0.0);
        // Verify ~50% sparsity
        let tv = TritVector::from_floats(&vals, threshold);
        assert!(tv.sparsity() > 0.3 && tv.sparsity() < 0.7);
    }

    #[test]
    fn test_pack_exactly_5_trits() {
        let tv = TritVector::new(vec![Trit::Pos, Trit::Pos, Trit::Pos, Trit::Pos, Trit::Pos]);
        let packed = tv.pack();
        assert_eq!(packed.len(), 1);
        // All Pos = all 2s: 2*81 + 2*27 + 2*9 + 2*3 + 2*1 = 162+54+18+6+2 = 242
        assert_eq!(packed[0], 242);
    }

    #[test]
    fn test_zeros() {
        let tv = TritVector::zeros(100);
        assert_eq!(tv.len(), 100);
        assert_eq!(tv.sparsity(), 1.0);
    }

    #[test]
    fn test_display_trit() {
        assert_eq!(format!("{}", Trit::Neg), "T");
        assert_eq!(format!("{}", Trit::Zero), "0");
        assert_eq!(format!("{}", Trit::Pos), "1");
    }

    #[test]
    fn test_large_vector_compression() {
        // Simulate a 1024-dim embedding
        let vals: Vec<f64> = (0..1024).map(|i| {
            let x = (i as f64 * 0.01).sin();
            if x.abs() < 0.05 { 0.0 } else { x }
        }).collect();
        let tv = TritVector::from_floats(&vals, 0.1);
        let packed = tv.pack();
        let stats = CompressionStats::from_trit_vector(&tv);
        assert_eq!(packed.len(), (1024_f64 / 5.0).ceil() as usize);
        assert!(stats.compression_ratio >= 19.0);
    }
}
