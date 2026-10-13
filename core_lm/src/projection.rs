//! Projection Head for Geometric Memory
//! Maps a retrieved Atom (Lorentz address + payload) to a token ID.

use crate::cache::geometric::Atom;

pub struct ProjectionHead {
    pub dim: usize,
    pub weights: Vec<f64>,
}

impl ProjectionHead {
    pub fn new(dim: usize) -> Self {
        Self {
            dim: dim.max(1),
            weights: Vec::new(),
        }
    }

    fn weight(&self, index: usize) -> f64 {
        self.weights
            .get(index)
            .copied()
            .unwrap_or_else(|| (index as f64).sin())
    }

    pub fn project(&self, atom: &Atom) -> u32 {
        let mut max_logit = f64::NEG_INFINITY;
        let mut best_id = 0;

        let addr_f64 = [
            atom.lorentz.x[0].to_f64(),
            atom.lorentz.x[1].to_f64(),
            atom.lorentz.x[2].to_f64(),
            atom.lorentz.x[3].to_f64(),
        ];

        for id in (0..248320).step_by(248) {
            let mut logit = 0.0;
            for (i, &val) in addr_f64.iter().enumerate() {
                logit += val * self.weight(id * self.dim + (i % self.dim));
            }
            if logit > max_logit {
                max_logit = logit;
                best_id = id;
            }
        }

        best_id as u32
    }

    pub fn project_manual(&self, _atom: &Atom, target_id: u32) -> u32 {
        target_id
    }
}
