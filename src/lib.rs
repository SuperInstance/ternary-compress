//! # ternary-compress
//!
//! Ternary data compression for sparse GPU workloads.
//! Run-length encoding, sparse format, and dictionary compression for {-1,0,+1}.

/// Run-length encode ternary data.
pub fn rle_encode(data: &[i8]) -> Vec<(i8, usize)> {
    if data.is_empty() { return vec![]; }
    let mut runs = vec![];
    let mut current = data[0];
    let mut count = 1usize;
    for &v in &data[1..] {
        if v == current && count < 255 { count += 1; }
        else {
            runs.push((current, count));
            current = v; count = 1;
        }
    }
    runs.push((current, count));
    runs
}

/// Decode RLE ternary data.
pub fn rle_decode(runs: &[(i8, usize)]) -> Vec<i8> {
    runs.iter().flat_map(|&(v, c)| vec![v; c]).collect()
}

/// Sparse ternary format: only store non-zero values with indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseTernary {
    pub len: usize,
    pub indices: Vec<usize>,
    pub values: Vec<i8>,
}

impl SparseTernary {
    pub fn from_dense(data: &[i8]) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, &v) in data.iter().enumerate() {
            if v != 0 {
                indices.push(i);
                values.push(v);
            }
        }
        Self { len: data.len(), indices, values }
    }

    pub fn to_dense(&self) -> Vec<i8> {
        let mut data = vec![0i8; self.len];
        for (&i, &v) in self.indices.iter().zip(&self.values) {
            data[i] = v;
        }
        data
    }

    pub fn density(&self) -> f64 {
        if self.len == 0 { return 0.0; }
        self.values.len() as f64 / self.len as f64
    }

    pub fn compression_ratio(&self) -> f64 {
        if self.len == 0 { return 1.0; }
        // Dense: len * 2 bits. Sparse: indices * usize + values * 2 bits
        let dense_bits = self.len as f64 * 2.0;
        let sparse_bits = self.indices.len() as f64 * 64.0 + self.values.len() as f64 * 2.0;
        dense_bits / sparse_bits
    }
}

/// Dictionary compression: map common ternary patterns to indices.
pub struct TernaryDict {
    entries: Vec<Vec<i8>>,
}

impl TernaryDict {
    pub fn new() -> Self { Self { entries: Vec::new() } }

    pub fn add(&mut self, pattern: Vec<i8>) -> usize {
        if let Some(idx) = self.entries.iter().position(|e| e == &pattern) { return idx; }
        self.entries.push(pattern);
        self.entries.len() - 1
    }

    pub fn get(&self, idx: usize) -> Option<&[i8]> {
        self.entries.get(idx).map(|v| v.as_slice())
    }

    /// Encode data using dictionary (fixed-size chunks).
    pub fn encode(&mut self, data: &[i8], chunk_size: usize) -> Vec<usize> {
        data.chunks(chunk_size).map(|chunk| {
            let padded = if chunk.len() < chunk_size {
                let mut p = chunk.to_vec();
                p.resize(chunk_size, 0);
                p
            } else { chunk.to_vec() };
            self.add(padded)
        }).collect()
    }

    pub fn decode(&self, indices: &[usize], chunk_size: usize) -> Vec<i8> {
        indices.iter().flat_map(|&i| {
            self.get(i).unwrap_or(&[0]).to_vec()
        }).collect()
    }

    pub fn dict_size(&self) -> usize { self.entries.len() }
}

impl Default for TernaryDict {
    fn default() -> Self { Self::new() }
}

/// Pack 16 ternary values into a single u32.
pub fn pack_trits(trits: &[i8]) -> u32 {
    let mut packed = 0u32;
    for (i, &t) in trits.iter().take(16).enumerate() {
        let bits = match t { -1 => 0b11u32, 1 => 0b01, _ => 0b00 };
        packed |= bits << (i * 2);
    }
    packed
}

/// Unpack 16 ternary values from a u32.
pub fn unpack_trits(packed: u32) -> [i8; 16] {
    let mut arr = [0i8; 16];
    for i in 0..16 {
        arr[i] = match (packed >> (i * 2)) & 0b11 {
            0b11 => -1, 0b01 => 1, _ => 0,
        };
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rle_roundtrip() {
        let data = vec![1, 1, 1, 0, 0, -1, -1, -1, -1, 0];
        let encoded = rle_encode(&data);
        assert_eq!(encoded, vec![(1, 3), (0, 2), (-1, 4), (0, 1)]);
        let decoded = rle_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_sparse_roundtrip() {
        let data = vec![0, 1, 0, 0, -1, 0, 0, 0, 1, 0];
        let sparse = SparseTernary::from_dense(&data);
        assert_eq!(sparse.values, vec![1, -1, 1]);
        assert_eq!(sparse.to_dense(), data);
    }

    #[test]
    fn test_sparse_density() {
        let data = vec![1, 0, 0, -1, 0, 0, 0, 0];
        let sparse = SparseTernary::from_dense(&data);
        assert!((sparse.density() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_dict_roundtrip() {
        let mut dict = TernaryDict::new();
        let data = vec![1, -1, 0, 1, -1, 0]; // two identical chunks of 3
        let encoded = dict.encode(&data, 3);
        assert_eq!(dict.dict_size(), 1); // only 1 unique pattern
        let decoded = dict.decode(&encoded, 3);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_pack_unpack() {
        let trits = [1i8, -1, 0, 1, 0, -1, 1, 1, -1, 0, 0, 1, -1, 0, 1, -1];
        let packed = pack_trits(&trits);
        let unpacked = unpack_trits(packed);
        assert_eq!(unpacked, trits);
    }

    #[test]
    fn test_rle_compression_ratio() {
        let data = vec![0; 100]; // all zeros
        let encoded = rle_encode(&data);
        assert_eq!(encoded.len(), 1); // 1 run
    }

    #[test]
    fn test_sparse_all_zeros() {
        let data = vec![0i8; 100];
        let sparse = SparseTernary::from_dense(&data);
        assert!(sparse.values.is_empty());
        assert_eq!(sparse.density(), 0.0);
    }
}
