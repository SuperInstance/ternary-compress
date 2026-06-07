//! # Ternary Compress
//!
//! 2-bit packed ternary vector operations with scalar dot product and Hamming distance.
//!
//! Ternary values: 0 (false/minus), 1 (unknown/null), 2 (true/plus).
//! Each value is stored as 2 bits, packed 4 values per byte.

// ── vector ──────────────────────────────────────────────────────────────────

/// A packed ternary vector using 2 bits per element.
///
/// Values: 0 = minus, 1 = null, 2 = plus.
/// Four values packed per byte: bits [0-1], [2-3], [4-5], [6-7].
#[derive(Debug, Clone, PartialEq)]
pub struct PackedTernaryVector {
    data: Vec<u8>,
    len: usize,
}

impl PackedTernaryVector {
    const VALUES_PER_BYTE: usize = 4;

    /// Create a new vector of `len` elements, all initialized to `default` (0, 1, or 2).
    pub fn new(len: usize, default: u8) -> Self {
        assert!(default <= 2, "default must be 0, 1, or 2");
        let byte_len = len.div_ceil(Self::VALUES_PER_BYTE);
        let packed_default = default | (default << 2) | (default << 4) | (default << 6);
        Self {
            data: vec![packed_default; byte_len],
            len,
        }
    }

    /// Create a zero-initialized vector (all values = 0).
    pub fn zeros(len: usize) -> Self {
        Self::new(len, 0)
    }

    /// Get the value at index `i`. Returns 0, 1, or 2.
    pub fn get(&self, i: usize) -> u8 {
        assert!(i < self.len, "index out of bounds");
        let byte_idx = i / Self::VALUES_PER_BYTE;
        let bit_offset = (i % Self::VALUES_PER_BYTE) * 2;
        (self.data[byte_idx] >> bit_offset) & 0b11
    }

    /// Set the value at index `i` to `val` (0, 1, or 2).
    pub fn set(&mut self, i: usize, val: u8) {
        assert!(i < self.len, "index out of bounds");
        assert!(val <= 2, "value must be 0, 1, or 2");
        let byte_idx = i / Self::VALUES_PER_BYTE;
        let bit_offset = (i % Self::VALUES_PER_BYTE) * 2;
        let mask = 0b11 << bit_offset;
        self.data[byte_idx] = (self.data[byte_idx] & !mask) | ((val & 0b11) << bit_offset);
    }

    /// Number of ternary elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the vector empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw byte data.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Construct from raw byte data and length.
    pub fn from_bytes(data: Vec<u8>, len: usize) -> Self {
        let expected = len.div_ceil(Self::VALUES_PER_BYTE);
        assert_eq!(data.len(), expected, "byte length mismatch");
        Self { data, len }
    }
}

// ── simd (scalar fallback) ──────────────────────────────────────────────────

/// Scalar dot product of two ternary vectors.
///
/// Returns Σ a[i] * b[i] for all elements.
/// Works on any architecture (no SIMD intrinsics, pure scalar).
pub fn dot_product(a: &PackedTernaryVector, b: &PackedTernaryVector) -> i64 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    let mut sum: i64 = 0;
    for i in 0..a.len() {
        sum += (a.get(i) as i64) * (b.get(i) as i64);
    }
    sum
}

/// Scalar multiply-add: result[i] = a[i] * scalar + b[i] (clamped to 2).
pub fn multiply_add(
    a: &PackedTernaryVector,
    scalar: u8,
    b: &PackedTernaryVector,
) -> PackedTernaryVector {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    let mut result = PackedTernaryVector::zeros(a.len());
    for i in 0..a.len() {
        let val = ((a.get(i) as u16) * (scalar as u16) + (b.get(i) as u16)).min(2) as u8;
        result.set(i, val);
    }
    result
}

// ── hamming ─────────────────────────────────────────────────────────────────

/// Compute the ternary Hamming distance between two vectors.
///
/// Returns the number of positions where a[i] != b[i].
pub fn hamming_distance(a: &PackedTernaryVector, b: &PackedTernaryVector) -> usize {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    let mut dist = 0;
    for i in 0..a.len() {
        if a.get(i) != b.get(i) {
            dist += 1;
        }
    }
    dist
}

/// Compute the normalized ternary Hamming distance (0.0 to 1.0).
pub fn normalized_hamming(a: &PackedTernaryVector, b: &PackedTernaryVector) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    hamming_distance(a, b) as f64 / a.len() as f64
}

// ── pack ────────────────────────────────────────────────────────────────────

/// Pack a slice of ternary values (0, 1, 2) into a PackedTernaryVector.
pub fn pack(values: &[u8]) -> PackedTernaryVector {
    let mut vec = PackedTernaryVector::zeros(values.len());
    for (i, &val) in values.iter().enumerate() {
        assert!(val <= 2, "value at index {} must be 0, 1, or 2, got {}", i, val);
        vec.set(i, val);
    }
    vec
}

/// Unpack a PackedTernaryVector into a Vec<u8>.
pub fn unpack(vec: &PackedTernaryVector) -> Vec<u8> {
    (0..vec.len()).map(|i| vec.get(i)).collect()
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_get_roundtrip() {
        let mut v = PackedTernaryVector::zeros(8);
        v.set(0, 2);
        v.set(3, 1);
        v.set(7, 2);
        assert_eq!(v.get(0), 2);
        assert_eq!(v.get(1), 0);
        assert_eq!(v.get(3), 1);
        assert_eq!(v.get(7), 2);
    }

    #[test]
    fn test_set_get_single_element() {
        let mut v = PackedTernaryVector::zeros(1);
        v.set(0, 1);
        assert_eq!(v.get(0), 1);
    }

    #[test]
    fn test_set_get_non_multiple_of_four() {
        let mut v = PackedTernaryVector::zeros(5);
        v.set(4, 2);
        assert_eq!(v.get(4), 2);
        assert_eq!(v.get(0), 0);
    }

    #[test]
    fn test_zeros_all_zero() {
        let v = PackedTernaryVector::zeros(10);
        for i in 0..10 {
            assert_eq!(v.get(i), 0);
        }
    }

    #[test]
    fn test_new_with_default() {
        let v = PackedTernaryVector::new(6, 2);
        for i in 0..6 {
            assert_eq!(v.get(i), 2);
        }
    }

    #[test]
    fn test_dot_product_basic() {
        let mut a = PackedTernaryVector::zeros(4);
        let mut b = PackedTernaryVector::zeros(4);
        // a = [1, 2, 0, 1]
        a.set(0, 1); a.set(1, 2); a.set(3, 1);
        // b = [2, 1, 1, 0]
        b.set(0, 2); b.set(1, 1); b.set(2, 1);
        // dot = 1*2 + 2*1 + 0*1 + 1*0 = 4
        assert_eq!(dot_product(&a, &b), 4);
    }

    #[test]
    fn test_dot_product_zeros() {
        let a = PackedTernaryVector::zeros(5);
        let b = PackedTernaryVector::zeros(5);
        assert_eq!(dot_product(&a, &b), 0);
    }

    #[test]
    fn test_dot_product_all_twos() {
        let a = PackedTernaryVector::new(3, 2);
        let b = PackedTernaryVector::new(3, 2);
        // 2*2 + 2*2 + 2*2 = 12
        assert_eq!(dot_product(&a, &b), 12);
    }

    #[test]
    fn test_hamming_identical() {
        let a = PackedTernaryVector::new(5, 1);
        let b = PackedTernaryVector::new(5, 1);
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_hamming_completely_different() {
        let mut a = PackedTernaryVector::zeros(3);
        let mut b = PackedTernaryVector::zeros(3);
        a.set(0, 0); a.set(1, 0); a.set(2, 0);
        b.set(0, 2); b.set(1, 2); b.set(2, 2);
        assert_eq!(hamming_distance(&a, &b), 3);
    }

    #[test]
    fn test_hamming_partial() {
        let mut a = PackedTernaryVector::zeros(4);
        a.set(0, 0); a.set(1, 1); a.set(2, 2); a.set(3, 0);
        let mut b = PackedTernaryVector::zeros(4);
        b.set(0, 0); b.set(1, 2); b.set(2, 2); b.set(3, 1);
        // differ at indices 1 and 3
        assert_eq!(hamming_distance(&a, &b), 2);
    }

    #[test]
    fn test_normalized_hamming() {
        let mut a = PackedTernaryVector::zeros(4);
        a.set(0, 0); a.set(1, 1); a.set(2, 2); a.set(3, 0);
        let mut b = PackedTernaryVector::zeros(4);
        b.set(0, 0); b.set(1, 2); b.set(2, 2); b.set(3, 1);
        assert_eq!(normalized_hamming(&a, &b), 0.5);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let values = vec![0, 1, 2, 0, 2, 1, 0, 0, 2];
        let packed = pack(&values);
        let unpacked = unpack(&packed);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_pack_unpack_empty() {
        let values: Vec<u8> = vec![];
        let packed = pack(&values);
        assert!(packed.is_empty());
        let unpacked = unpack(&packed);
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_from_bytes_roundtrip() {
        let mut v = PackedTernaryVector::zeros(7);
        v.set(0, 2); v.set(3, 1); v.set(6, 2);
        let bytes = v.as_bytes().to_vec();
        let v2 = PackedTernaryVector::from_bytes(bytes, 7);
        assert_eq!(v2.get(0), 2);
        assert_eq!(v2.get(3), 1);
        assert_eq!(v2.get(6), 2);
    }

    #[test]
    fn test_multiply_add() {
        let mut a = PackedTernaryVector::zeros(3);
        a.set(0, 1); a.set(1, 2); a.set(2, 0);
        let mut b = PackedTernaryVector::zeros(3);
        b.set(0, 1); b.set(1, 0); b.set(2, 2);
        let result = multiply_add(&a, 1, &b);
        // 1*1+1=2, 2*1+0=2, 0*1+2=2
        assert_eq!(result.get(0), 2);
        assert_eq!(result.get(1), 2);
        assert_eq!(result.get(2), 2);
    }

    #[test]
    fn test_vector_len() {
        let v = PackedTernaryVector::zeros(13);
        assert_eq!(v.len(), 13);
    }
}
