//! Core tensor operations. All return Result, no panics in hot path.

#[derive(Debug)]
pub enum OpsError {
    ShapeMismatch(String),
    EmptyInput,
}

impl std::fmt::Display for OpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpsError::ShapeMismatch(s) => write!(f, "Shape mismatch: {}", s),
            OpsError::EmptyInput => write!(f, "Empty input"),
        }
    }
}

impl std::error::Error for OpsError {}

pub fn matvec(matrix: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Result<Vec<f32>, OpsError> {
    if matrix.len() != rows * cols {
        return Err(OpsError::ShapeMismatch(format!(
            "matrix len {} != {}x{}",
            matrix.len(),
            rows,
            cols
        )));
    }
    if vec.len() != cols {
        return Err(OpsError::ShapeMismatch(format!(
            "vec len {} != cols {}",
            vec.len(),
            cols
        )));
    }
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0.0f32;
        let base = r * cols;
        for c in 0..cols {
            sum += matrix[base + c] * vec[c];
        }
        out[r] = sum;
    }
    Ok(out)
}

pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, OpsError> {
    if x.is_empty() {
        return Err(OpsError::EmptyInput);
    }
    if x.len() != weight.len() {
        return Err(OpsError::ShapeMismatch(format!(
            "x len {} != weight len {}",
            x.len(),
            weight.len()
        )));
    }
    let n = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    Ok(x.iter()
        .zip(weight.iter())
        .map(|(v, w)| v * inv * w)
        .collect())
}

pub fn softmax(x: &[f32]) -> Result<Vec<f32>, OpsError> {
    if x.is_empty() {
        return Err(OpsError::EmptyInput);
    }
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        return Err(OpsError::ShapeMismatch(
            "softmax sum is zero or non-finite".to_string(),
        ));
    }
    Ok(exps.iter().map(|v| v / sum).collect())
}

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

pub fn apply_rope(
    x: &mut [f32],
    head_dim: usize,
    rotary_dim: usize,
    pos: usize,
    theta: f32,
) -> Result<(), OpsError> {
    if x.len() % head_dim != 0 {
        return Err(OpsError::ShapeMismatch(format!(
            "x len {} not divisible by head_dim {}",
            x.len(),
            head_dim
        )));
    }
    let num_heads = x.len() / head_dim;
    for h in 0..num_heads {
        let base = h * head_dim;
        let mut i = 0;
        while i < rotary_dim {
            let freq = 1.0 / theta.powf((i as f32) / (rotary_dim as f32));
            let angle = (pos as f32) * freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + i];
            let x1 = x[base + i + 1];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + i + 1] = x0 * sin + x1 * cos;
            i += 2;
        }
    }
    Ok(())
}

pub fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

pub fn argmax(x: &[f32]) -> Result<usize, OpsError> {
    if x.is_empty() {
        return Err(OpsError::EmptyInput);
    }
    let mut best_idx = 0;
    let mut best_val = x[0];
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx)
}
