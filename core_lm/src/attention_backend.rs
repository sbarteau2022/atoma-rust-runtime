//! Boundary for server-owned attention memory. Inputs are post-QK-norm/post-RoPE;
//! output is before the output gate and projection. No checkpoint-specific storage here.

pub struct AttentionInput<'a> {
    pub layer: usize,
    pub token_id: u32,
    pub position: usize,
    pub window_start: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub dot_scale: f64,
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
}

impl AttentionInput<'_> {
    pub fn validate(&self) -> Result<(), String> {
        if self.head_dim < 2
            || !self.head_dim.is_multiple_of(2)
            || self.kv_heads == 0
            || self.query_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.query_heads.checked_mul(self.head_dim) != Some(self.query.len())
            || self.kv_heads.checked_mul(self.head_dim) != Some(self.key.len())
            || self.key.len() != self.value.len()
            || self.window_start > self.position
            || !self.dot_scale.is_finite()
            || self.dot_scale <= 0.0
            || !self
                .query
                .iter()
                .chain(self.key)
                .chain(self.value)
                .all(|x| x.is_finite())
        {
            return Err("invalid attention shape, window, scale, or non-finite tensor".into());
        }
        Ok(())
    }
}

pub trait AttentionBackend {
    fn attend(&mut self, input: AttentionInput<'_>) -> Result<Vec<f32>, String>;
}
