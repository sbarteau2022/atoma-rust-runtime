//! Phase B native producer (`docs/NATIVE_NEXT_ATOM_DESIGN.md`): a trained causal
//! byte/EOS readout over the fixed toroidal recurrence. No teacher or tokenizer.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use atoma_immune::refraction::Sha256Digest;
use atoma_immune::{BoundaryLexicalTransducer, MlirOpId};
use serde::{Deserialize, Serialize};

use crate::input_embeddings::{NativeMlirEmbedding, embed_carrier};
use crate::predict_gate::{GateScore, ProductPredictStep, ProductPredictor, TrajectoryMatchGate};
use crate::toroidal_attention::{ToroidalHarmonicFieldState, ToroidalIsotropicRuntime};

pub const NATIVE_BOS: u32 = 200_000;
pub const NATIVE_EOS: u32 = 200_001;
pub const OUTPUT_CLASSES: usize = 257;
pub const EOS_CLASS: usize = 256;
const BOS_INPUT_CLASS: usize = 256;
const SUMMARY_FEATURES: usize = 39;
const HISTORY_SLOTS: usize = 8;
const CARRIER_FEATURES: usize = 8;
pub const CONTEXT_FEATURES: usize = SUMMARY_FEATURES
    + HISTORY_SLOTS * (CARRIER_FEATURES + 1)
    + (HISTORY_SLOTS - 1) * CARRIER_FEATURES;
pub const MAX_RECORD_BYTES: usize = 65_536;
pub const MAX_EXAMPLES: usize = 250_000;
pub const MAX_NEW_BYTES: usize = 4096;
const STRIDE: usize = 1 + OUTPUT_CLASSES + CONTEXT_FEATURES;
const CONTEXT_OFFSET: usize = 1 + OUTPUT_CLASSES;
const MAGIC: &[u8] = b"ATOMA_NEXT_V2\n";
const MAX_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
const SCHEMA: &str = "native-utf8-v1/toroidal-ordered-readout-v2";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePredictorConfig {
    pub layers: usize,
    pub residual_mix: f32,
    /// Matched control: false zeros history features during both fit and inference.
    pub ordered_history: bool,
}

impl Default for NativePredictorConfig {
    fn default() -> Self {
        Self {
            layers: 52,
            residual_mix: 0.18,
            ordered_history: true,
        }
    }
}

impl NativePredictorConfig {
    fn validate(&self) -> Result<(), String> {
        if !(1..=256).contains(&self.layers)
            || !self.residual_mix.is_finite()
            || !(0.0..1.0).contains(&self.residual_mix)
        {
            return Err(
                "native predictor requires 1..=256 layers and residual mix in [0, 1)".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub l2: f64,
    pub seed: u64,
}

impl Default for NativeTrainingOptions {
    fn default() -> Self {
        Self {
            epochs: 8,
            batch_size: 128,
            learning_rate: 0.003,
            l2: 0.0001,
            seed: 20261018,
        }
    }
}

impl NativeTrainingOptions {
    fn validate(&self) -> Result<(), String> {
        if !(1..=100).contains(&self.epochs)
            || !(1..=1024).contains(&self.batch_size)
            || !self.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&self.learning_rate)
            || self.learning_rate == 0.0
            || !self.l2.is_finite()
            || !(0.0..=1.0).contains(&self.l2)
        {
            return Err("invalid native training epochs, batch size, learning rate, or L2".into());
        }
        Ok(())
    }
}

/// Labels identify provenance only; they never enter the model's features.
#[derive(Debug, Clone)]
pub struct NativeTextRecord {
    pub label: String,
    pub prompt: String,
    pub completion: String,
}

impl NativeTextRecord {
    pub fn document(label: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            prompt: String::new(),
            completion: text.into(),
        }
    }

    fn full_text(&self) -> String {
        format!("{}{}", self.prompt, self.completion)
    }

    fn manifest(&self) -> RecordManifest {
        let mut contents = (self.prompt.len() as u64).to_le_bytes().to_vec();
        contents.extend(self.prompt.as_bytes());
        contents.extend(self.completion.as_bytes());
        RecordManifest {
            label: self.label.clone(),
            prompt_bytes: self.prompt.len(),
            completion_bytes: self.completion.len(),
            sha256: Sha256Digest::compute(&contents).to_hex(),
            text_sha256: Sha256Digest::compute(self.full_text().as_bytes()).to_hex(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordManifest {
    pub label: String,
    pub prompt_bytes: usize,
    pub completion_bytes: usize,
    pub sha256: String,
    pub text_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingProvenance {
    pub records: Vec<RecordManifest>,
    pub dataset_sha256: String,
    pub options: NativeTrainingOptions,
    /// Selected solely by training NLL, including the initial byte-statistics control (0).
    pub selected_epoch: usize,
}

/// Immutable after validation. Inference cannot mutate trained weights or statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeToroidalPredictor {
    schema: String,
    source_sha256: String,
    architecture: String,
    config: NativePredictorConfig,
    training: TrainingProvenance,
    means: Vec<f64>,
    scales: Vec<f64>,
    weights: Vec<f64>,
    unigram: Vec<u64>,
    bigram: Vec<u64>,
}

#[derive(Debug, Clone)]
struct Example {
    previous: usize,
    context: [f64; CONTEXT_FEATURES],
    target: usize,
    predictor_before: ProductPredictor,
}

fn embedding_to_ball_phases(emb: &NativeMlirEmbedding) -> Option<([f64; 3], [f64; 2])> {
    let c = emb.coordinates;
    let denom = f64::from(c[0]) + 1.0;
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }
    let ball = [
        f64::from(c[1]) / denom,
        f64::from(c[2]) / denom,
        f64::from(c[3]) / denom,
    ];
    let n2 = ball[0] * ball[0] + ball[1] * ball[1] + ball[2] * ball[2];
    if !n2.is_finite() || n2 >= 1.0 {
        return None;
    }
    let phases = [f64::from(c[4]), f64::from(c[5])];
    if !phases.iter().all(|p| p.is_finite()) {
        return None;
    }
    Some((ball, phases))
}

fn observe_carrier_in_predictor(
    predictor: &mut ProductPredictor,
    catalog: &atoma_immune::MlirAtomCatalog,
    carrier_id: MlirOpId,
) -> Option<ProductPredictStep> {
    let emb = embed_carrier(catalog, carrier_id).ok()?;
    let (ball, phases) = embedding_to_ball_phases(&emb)?;
    Some(predictor.step(&ball, &phases))
}

struct NativeContext {
    boundary: BoundaryLexicalTransducer,
    runtime: ToroidalIsotropicRuntime,
    hidden: Vec<f32>,
    previous: usize,
    history: [[f64; CARRIER_FEATURES]; HISTORY_SLOTS],
    history_len: usize,
    ordered_history: bool,
}

fn carrier_features(hidden: &[f32]) -> Result<[f64; CARRIER_FEATURES], String> {
    if hidden.len() != 6 || !hidden.iter().all(|v| v.is_finite()) {
        return Err("invalid native predictor hidden state".into());
    }
    Ok([
        (f64::from(hidden[0]) / 4.0).tanh(),
        (f64::from(hidden[1]) / 4.0).tanh(),
        (f64::from(hidden[2]) / 4.0).tanh(),
        (f64::from(hidden[3]) / 4.0).tanh(),
        f64::from(hidden[4]).sin(),
        f64::from(hidden[4]).cos(),
        f64::from(hidden[5]).sin(),
        f64::from(hidden[5]).cos(),
    ])
}

impl NativeContext {
    fn new(config: &NativePredictorConfig) -> Result<Self, String> {
        config.validate()?;
        let boundary = BoundaryLexicalTransducer::new(202_048, NATIVE_BOS, vec![NATIVE_EOS]);
        let mut runtime = ToroidalIsotropicRuntime::full_stack(6, config.layers)?;
        runtime.toroidal_field_mix = config.residual_mix;
        let bos = embed_carrier(&boundary.catalog, MlirOpId(NATIVE_BOS))?;
        let hidden = runtime.step_toroidal_attention(&bos.coordinates)?;
        let mut history = [[0.0; CARRIER_FEATURES]; HISTORY_SLOTS];
        history[0] = carrier_features(&hidden)?;
        Ok(Self {
            boundary,
            runtime,
            hidden,
            previous: BOS_INPUT_CLASS,
            history,
            history_len: 1,
            ordered_history: config.ordered_history,
        })
    }

    fn push(&mut self, byte: u8) -> Result<(), String> {
        // The byte-to-ML-IR cast is the entire converter; no lexicon is loaded.
        let embedding = embed_carrier(&self.boundary.catalog, MlirOpId(u32::from(byte)))?;
        self.hidden = self
            .runtime
            .step_toroidal_attention(&embedding.coordinates)?;
        self.previous = usize::from(byte);
        self.history.copy_within(0..HISTORY_SLOTS - 1, 1);
        self.history[0] = carrier_features(&self.hidden)?;
        self.history_len = (self.history_len + 1).min(HISTORY_SLOTS);
        Ok(())
    }

    fn prefill(&mut self, text: &str) -> Result<(), String> {
        for byte in text.bytes() {
            self.push(byte)?;
        }
        Ok(())
    }

    fn features(&self) -> Result<[f64; CONTEXT_FEATURES], String> {
        let mut values = Vec::with_capacity(CONTEXT_FEATURES);
        values.extend(carrier_features(&self.hidden)?);
        for field in [
            self.runtime.layer_fields.first(),
            self.runtime.layer_fields.last(),
        ] {
            append_field_features(
                field.ok_or("native predictor has no recurrent fields")?,
                &mut values,
            )?;
        }
        let positions = self.runtime.layer_fields[0].tokens_ingested;
        values.push(((positions as f64).ln_1p() / 12.0).tanh());
        if self.ordered_history {
            for (index, slot) in self.history.iter().enumerate() {
                values.extend(slot);
                values.push(f64::from(index < self.history_len));
            }
            for slot in &self.history[1..] {
                values.extend(
                    slot.iter()
                        .zip(self.history[0])
                        .map(|(past, now)| past * now),
                );
            }
        } else {
            values.resize(CONTEXT_FEATURES, 0.0);
        }
        let values: [f64; CONTEXT_FEATURES] = values
            .try_into()
            .map_err(|_| "native feature schema mismatch")?;
        if !values.iter().all(|v| v.is_finite()) {
            return Err("non-finite native predictor feature".into());
        }
        Ok(values)
    }
}

fn append_field_features(
    field: &ToroidalHarmonicFieldState,
    values: &mut Vec<f64>,
) -> Result<(), String> {
    if !field.z_0.is_finite() || field.z_0 <= 0.0 || !field.radius_accumulator.is_finite() {
        return Err("invalid native recurrent field normalizer or radius".into());
    }
    values.extend(field.m_0[..4].iter().map(|v| (v / field.z_0 / 4.0).tanh()));
    for pair in 0..3 {
        values.extend([field.z_c1[pair] / field.z_0, field.z_s1[pair] / field.z_0]);
    }
    for phase in [
        field.phase_state.theta_f.to_f64(),
        field.phase_state.theta_s.to_f64(),
    ] {
        values.extend([phase.sin(), phase.cos()]);
    }
    values.push((field.radius_accumulator.ln_1p() / 12.0).tanh());
    Ok(())
}

fn validate_records(records: &[NativeTextRecord]) -> Result<(), String> {
    if records.is_empty() {
        return Err("native corpus must contain at least one record".into());
    }
    let mut labels = BTreeSet::new();
    let mut examples = 0usize;
    let mut total_bytes = 0usize;
    for record in records {
        if record.label.is_empty() || record.label.len() > 4096 || !labels.insert(&record.label) {
            return Err(
                "native record labels must be nonempty and unique (at most 4096 bytes)".into(),
            );
        }
        let bytes = record
            .prompt
            .len()
            .checked_add(record.completion.len())
            .ok_or("record length overflow")?;
        if bytes > MAX_RECORD_BYTES {
            return Err(format!(
                "record {} exceeds {MAX_RECORD_BYTES} bytes",
                record.label
            ));
        }
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or("corpus length overflow")?;
        examples = examples
            .checked_add(record.completion.len() + 1)
            .ok_or("example count overflow")?;
        if examples > MAX_EXAMPLES || total_bytes > MAX_EXAMPLES {
            return Err(format!(
                "native corpus exceeds {MAX_EXAMPLES} bytes or supervised positions"
            ));
        }
    }
    Ok(())
}

/// Reject exact overlap regardless of prompt masking, plus document containment
/// of at least 32 bytes, and reject contamination from teacher-era `.atoma_map.wal` frames.
/// Shared ordinary n-grams are not prohibited.
pub fn require_disjoint_corpora(
    train: &[NativeTextRecord],
    evaluation: &[NativeTextRecord],
) -> Result<(), String> {
    validate_records(train)?;
    validate_records(evaluation)?;
    let training: Vec<_> = train.iter().map(NativeTextRecord::full_text).collect();
    for eval in evaluation {
        let text = eval.full_text();
        for (index, fitted) in training.iter().enumerate() {
            if text == *fitted
                || (text.len().min(fitted.len()) >= 32
                    && (text.contains(fitted) || fitted.contains(&text)))
            {
                return Err(format!(
                    "training/evaluation overlap: {} and {}",
                    train[index].label, eval.label
                ));
            }
        }
    }
    let wal_frames = default_workspace_wal_trajectories();
    if !wal_frames.is_empty() {
        require_disjoint_from_wal(train, &wal_frames)?;
        require_disjoint_from_wal(evaluation, &wal_frames)?;
    }
    Ok(())
}

/// Reject any training or evaluation record that overlaps with teacher-era WAL frames
/// (such as `2+2` / `What is two plus two` teacher-crystallized trajectories).
pub fn require_disjoint_from_wal(
    records: &[NativeTextRecord],
    wal_trajectories: &[WalFrameTrajectory],
) -> Result<(), String> {
    validate_records(records)?;
    for rec in records {
        let full = rec.full_text();
        let prompt_norm = rec.prompt.trim().to_ascii_lowercase();
        let comp_norm = rec.completion.trim().to_ascii_lowercase();
        let full_norm = full.trim().to_ascii_lowercase();
        for (idx, wal) in wal_trajectories.iter().enumerate() {
            let wal_prompt = wal.prompt_text.trim().to_ascii_lowercase();
            let wal_cont = wal.emitted_text.trim().to_ascii_lowercase();
            let wal_full = format!("{}{}", wal.prompt_text, wal.emitted_text)
                .trim()
                .to_ascii_lowercase();
            let prompt_match = !prompt_norm.is_empty()
                && !wal_prompt.is_empty()
                && wal_prompt.len() >= 8
                && prompt_norm == wal_prompt;
            let cont_match = !wal_cont.is_empty()
                && wal_cont.len() >= 24
                && ((comp_norm.len() >= 24
                    && (comp_norm.contains(&wal_cont) || wal_cont.contains(&comp_norm)))
                    || (full_norm.len() >= 24 && full_norm.contains(&wal_cont)));
            let full_match = !wal_full.is_empty()
                && wal_full.len() >= 24
                && full_norm.len() >= 24
                && (full_norm == wal_full
                    || full_norm.contains(&wal_full)
                    || wal_full.contains(&full_norm));
            if prompt_match || cont_match || full_match {
                return Err(format!(
                    "teacher-contaminated WAL frame overlap in record {} (matches WAL frame #{idx} origin={} prompt={:?})",
                    rec.label, wal.origin_fingerprint_hex, wal.prompt_text
                ));
            }
        }
    }
    Ok(())
}

/// Enforce both train/eval corpus disjointness and disjointness from explicit `.atoma_map.wal` frames.
pub fn require_disjoint_corpora_and_wal(
    train: &[NativeTextRecord],
    evaluation: &[NativeTextRecord],
    wal_trajectories: &[WalFrameTrajectory],
) -> Result<(), String> {
    require_disjoint_corpora(train, evaluation)?;
    require_disjoint_from_wal(train, wal_trajectories)?;
    require_disjoint_from_wal(evaluation, wal_trajectories)?;
    Ok(())
}

fn examples(
    config: &NativePredictorConfig,
    records: &[NativeTextRecord],
) -> Result<Vec<Example>, String> {
    validate_records(records)?;
    let mut output = Vec::new();
    for record in records {
        let mut state = NativeContext::new(config)?;
        let mut gate_predictor = ProductPredictor::new(3, 2, None, None);
        let _ = observe_carrier_in_predictor(
            &mut gate_predictor,
            &state.boundary.catalog,
            MlirOpId(NATIVE_BOS),
        );
        for byte in record.prompt.bytes() {
            state.push(byte)?;
            let _ = observe_carrier_in_predictor(
                &mut gate_predictor,
                &state.boundary.catalog,
                MlirOpId(u32::from(byte)),
            );
        }
        for target in record
            .completion
            .bytes()
            .map(usize::from)
            .chain(std::iter::once(EOS_CLASS))
        {
            output.push(Example {
                previous: state.previous,
                context: state.features()?,
                target,
                predictor_before: gate_predictor.clone(),
            });
            if target != EOS_CLASS {
                state.push(target as u8)?;
                let _ = observe_carrier_in_predictor(
                    &mut gate_predictor,
                    &state.boundary.catalog,
                    MlirOpId(target as u32),
                );
            }
        }
    }
    Ok(output)
}

pub fn native_predictor_source_sha256() -> String {
    // Fail closed when the carrier, recurrence, feature extraction, or geometry changes.
    let sources: &[(&str, &[u8])] = &[
        ("predictor", include_bytes!("native_predictor.rs")),
        ("embedding", include_bytes!("input_embeddings.rs")),
        ("toroidal", include_bytes!("toroidal_attention.rs")),
        (
            "reader",
            include_bytes!("../../immune_system/implementation/src/native_reader.rs"),
        ),
        (
            "boundary",
            include_bytes!("../../immune_system/implementation/src/continuous_active_mind.rs"),
        ),
        (
            "alphabet",
            include_bytes!("../../immune_system/implementation/src/first_principles.rs"),
        ),
        (
            "catalog",
            include_bytes!("../../immune_system/implementation/src/mlir_native.rs"),
        ),
        (
            "torus",
            include_bytes!("../../core_lm/src/geometry/torus.rs"),
        ),
        (
            "fixed",
            include_bytes!("../../core_lm/src/geometry/fixed.rs"),
        ),
        (
            "lorentz",
            include_bytes!("../../core_lm/src/geometry/lorentz.rs"),
        ),
        (
            "picard",
            include_bytes!("../../core_lm/src/geometry/picard.rs"),
        ),
        ("dependencies", include_bytes!("../Cargo.lock")),
    ];
    let mut bytes = Vec::new();
    for (name, source) in sources {
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend((source.len() as u64).to_le_bytes());
        bytes.extend(*source);
    }
    Sha256Digest::compute(&bytes).to_hex()
}

fn manifest_sha256(records: &[RecordManifest]) -> Result<String, String> {
    Ok(Sha256Digest::compute(&serde_json::to_vec(records).map_err(|e| e.to_string())?).to_hex())
}

fn probabilities(logits: &mut [f64]) -> Result<f64, String> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() || !logits.iter().all(|v| v.is_finite()) {
        return Err("non-finite native logits".into());
    }
    let mut sum = 0.0;
    for value in logits.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in logits.iter_mut() {
        *value /= sum;
    }
    Ok(max + sum.ln())
}

fn argmax(values: &[f64]) -> usize {
    // Stable lowest-class tie break, including a completely uninformative head.
    values.iter().enumerate().fold(
        0,
        |best, (i, value)| if *value > values[best] { i } else { best },
    )
}

impl NativeToroidalPredictor {
    pub fn config(&self) -> &NativePredictorConfig {
        &self.config
    }
    pub fn training(&self) -> &TrainingProvenance {
        &self.training
    }
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }
    pub fn parameter_count(&self) -> usize {
        self.weights.len()
    }

    fn normalized(&self, context: &[f64; CONTEXT_FEATURES]) -> [f64; CONTEXT_FEATURES] {
        std::array::from_fn(|i| (context[i] - self.means[i]) / self.scales[i])
    }

    fn logits(
        &self,
        previous: usize,
        context: &[f64; CONTEXT_FEATURES],
        use_context: bool,
    ) -> Vec<f64> {
        self.weights
            .chunks_exact(STRIDE)
            .map(|row| {
                let mut value = row[0] + row[1 + previous];
                if use_context {
                    for (weight, feature) in row[CONTEXT_OFFSET..].iter().zip(context) {
                        value += weight * feature;
                    }
                }
                value
            })
            .collect()
    }

    fn unigram_probability(&self, target: usize) -> f64 {
        (self.unigram[target] as f64 + 0.5)
            / (self.unigram.iter().sum::<u64>() as f64 + OUTPUT_CLASSES as f64 * 0.5)
    }

    fn bigram_probability(&self, previous: usize, target: usize) -> f64 {
        let row = &self.bigram[previous * OUTPUT_CLASSES..(previous + 1) * OUTPUT_CLASSES];
        (row[target] as f64 + 8.0 * self.unigram_probability(target))
            / (row.iter().sum::<u64>() as f64 + 8.0)
    }

    pub fn fit(
        config: NativePredictorConfig,
        options: NativeTrainingOptions,
        records: &[NativeTextRecord],
    ) -> Result<(Self, TrainingReceipt), String> {
        config.validate()?;
        options.validate()?;
        let started = Instant::now();
        let samples = examples(&config, records)?;
        let manifests: Vec<_> = records.iter().map(NativeTextRecord::manifest).collect();
        let mut model = Self {
            schema: SCHEMA.into(),
            source_sha256: native_predictor_source_sha256(),
            architecture: std::env::consts::ARCH.into(),
            config,
            training: TrainingProvenance {
                dataset_sha256: manifest_sha256(&manifests)?,
                records: manifests,
                options: options.clone(),
                selected_epoch: 0,
            },
            means: vec![0.0; CONTEXT_FEATURES],
            scales: vec![0.0; CONTEXT_FEATURES],
            weights: vec![0.0; OUTPUT_CLASSES * STRIDE],
            unigram: vec![0; OUTPUT_CLASSES],
            bigram: vec![0; OUTPUT_CLASSES * OUTPUT_CLASSES],
        };
        // Welford moments use training positions only, never evaluation targets/features.
        for (index, sample) in samples.iter().enumerate() {
            model.unigram[sample.target] += 1;
            model.bigram[sample.previous * OUTPUT_CLASSES + sample.target] += 1;
            for (i, &value) in sample.context.iter().enumerate() {
                let delta = value - model.means[i];
                model.means[i] += delta / (index + 1) as f64;
                model.scales[i] += delta * (value - model.means[i]);
            }
        }
        for scale in &mut model.scales {
            *scale = (*scale / samples.len() as f64).max(0.0).sqrt().max(0.05);
        }
        // Start at a real training-only smoothed bigram control, not random output.
        for class in 0..OUTPUT_CLASSES {
            let bias = model.unigram_probability(class).ln();
            model.weights[class * STRIDE] = bias;
            for previous in 0..OUTPUT_CLASSES {
                model.weights[class * STRIDE + 1 + previous] =
                    model.bigram_probability(previous, class).ln() - bias;
            }
        }
        let normalized: Vec<_> = samples
            .iter()
            .map(|sample| model.normalized(&sample.context))
            .collect();
        let initial = model.measure(&samples)?;
        let prior_weights = model.weights.clone();
        let mut best_weights = prior_weights.clone();
        let mut best_nll = initial.learned.mean_nll_nats;
        let mut selected_epoch = 0;
        let mut first_moment = vec![0.0; model.weights.len()];
        let mut second_moment = vec![0.0; model.weights.len()];
        let mut gradient = vec![0.0; model.weights.len()];
        let mut order: Vec<_> = (0..samples.len()).collect();
        let mut rng = options.seed;
        let mut beta1_power = 1.0;
        let mut beta2_power = 1.0;
        let mut epochs = Vec::new();
        for epoch in 1..=options.epochs {
            for i in (1..order.len()).rev() {
                let j = (next_random(&mut rng) % (i + 1) as u64) as usize;
                order.swap(i, j);
            }
            for batch in order.chunks(options.batch_size) {
                gradient.fill(0.0);
                for &index in batch {
                    let sample = &samples[index];
                    let features = &normalized[index];
                    let mut probs = model.logits(sample.previous, features, true);
                    probabilities(&mut probs)?;
                    for (class, probability) in probs.into_iter().enumerate() {
                        let error = probability - f64::from(class == sample.target);
                        let row = &mut gradient[class * STRIDE..(class + 1) * STRIDE];
                        row[0] += error;
                        row[1 + sample.previous] += error;
                        for (g, feature) in row[CONTEXT_OFFSET..].iter_mut().zip(features) {
                            *g += error * feature;
                        }
                    }
                }
                beta1_power *= 0.9;
                beta2_power *= 0.999;
                for (i, weight) in model.weights.iter_mut().enumerate() {
                    let g = gradient[i] / batch.len() as f64
                        + if i % STRIDE == 0 {
                            0.0
                        } else {
                            // Keep the empirical byte-statistics prior intact. Penalizing the
                            // full initialized weight erased it even on the training corpus.
                            options.l2 * (*weight - prior_weights[i])
                        };
                    first_moment[i] = 0.9 * first_moment[i] + 0.1 * g;
                    second_moment[i] = 0.999 * second_moment[i] + 0.001 * g * g;
                    *weight -= options.learning_rate * (first_moment[i] / (1.0 - beta1_power))
                        / ((second_moment[i] / (1.0 - beta2_power)).sqrt() + 1e-8);
                }
            }
            let score = model.measure(&samples)?.learned;
            if score.mean_nll_nats < best_nll {
                best_nll = score.mean_nll_nats;
                best_weights.clone_from(&model.weights);
                selected_epoch = epoch;
            }
            epochs.push(score);
        }
        model.weights = best_weights;
        model.training.selected_epoch = selected_epoch;
        model.validate()?;
        let final_scores = model.measure(&samples)?;
        Ok((
            model,
            TrainingReceipt {
                positions: samples.len(),
                initial,
                final_scores,
                epochs,
                elapsed_ms: started.elapsed().as_millis(),
            },
        ))
    }

    fn measure(&self, samples: &[Example]) -> Result<EvaluationScores, String> {
        let mut scores = EvaluationScores::default();
        // Baselines are fixed corpus statistics, so calculate each row only once.
        let unigram: Vec<_> = (0..OUTPUT_CLASSES)
            .map(|c| self.unigram_probability(c))
            .collect();
        let bigram: Vec<Vec<_>> = (0..OUTPUT_CLASSES)
            .map(|p| {
                (0..OUTPUT_CLASSES)
                    .map(|c| self.bigram_probability(p, c))
                    .collect()
            })
            .collect();
        let unigram_top = argmax(&unigram);
        let bigram_top: Vec<_> = bigram.iter().map(|row| argmax(row)).collect();
        let boundary = BoundaryLexicalTransducer::new(202_048, NATIVE_BOS, vec![NATIVE_EOS]);
        let mut class_ball = [[0.0f64; 3]; OUTPUT_CLASSES];
        let mut class_phases = [[0.0f64; 2]; OUTPUT_CLASSES];
        for b in 0..256usize {
            let emb = embed_carrier(&boundary.catalog, MlirOpId(b as u32))?;
            if let Some((ball, phases)) = embedding_to_ball_phases(&emb) {
                class_ball[b] = ball;
                class_phases[b] = phases;
            }
        }
        let step_for_pred =
            |pred_before: &ProductPredictor, predicted: usize| -> Option<ProductPredictStep> {
                if predicted < EOS_CLASS {
                    let mut p = pred_before.clone();
                    Some(p.step(&class_ball[predicted], &class_phases[predicted]))
                } else {
                    None
                }
            };
        for sample in samples {
            let features = self.normalized(&sample.context);
            let mut learned = self.logits(sample.previous, &features, true);
            let mut ablated = self.logits(sample.previous, &features, false);
            let mut summary_features = features;
            summary_features[SUMMARY_FEATURES..].fill(0.0);
            let mut summary_only = self.logits(sample.previous, &summary_features, true);
            let learned_target = learned[sample.target];
            let ablated_target = ablated[sample.target];
            let learned_nll = probabilities(&mut learned)? - learned_target;
            let ablated_nll = probabilities(&mut ablated)? - ablated_target;
            let summary_target = summary_only[sample.target];
            let summary_nll = probabilities(&mut summary_only)? - summary_target;
            let learned_pred = argmax(&learned);
            let ablated_pred = argmax(&ablated);
            let summary_pred = argmax(&summary_only);
            let unigram_pred = unigram_top;
            let bigram_pred = bigram_top[sample.previous];
            scores.learned.record_with_surprise(
                sample.target,
                learned_pred,
                learned_nll,
                step_for_pred(&sample.predictor_before, learned_pred),
            );
            scores.without_toroidal_features.record_with_surprise(
                sample.target,
                ablated_pred,
                ablated_nll,
                step_for_pred(&sample.predictor_before, ablated_pred),
            );
            scores.without_ordered_history.record_with_surprise(
                sample.target,
                summary_pred,
                summary_nll,
                step_for_pred(&sample.predictor_before, summary_pred),
            );
            scores.unigram.record_with_surprise(
                sample.target,
                unigram_pred,
                -unigram[sample.target].ln(),
                step_for_pred(&sample.predictor_before, unigram_pred),
            );
            scores.bigram.record_with_surprise(
                sample.target,
                bigram_pred,
                -bigram[sample.previous][sample.target].ln(),
                step_for_pred(&sample.predictor_before, bigram_pred),
            );
        }
        for score in [
            &mut scores.learned,
            &mut scores.without_toroidal_features,
            &mut scores.without_ordered_history,
            &mut scores.unigram,
            &mut scores.bigram,
        ] {
            score.finish();
        }
        Ok(scores)
    }

    pub fn evaluate(
        &self,
        records: &[NativeTextRecord],
        max_new: usize,
    ) -> Result<EvaluationReceipt, String> {
        self.validate()?;
        if !(1..=MAX_NEW_BYTES).contains(&max_new) {
            return Err("native generation budget must be in 1..=4096 bytes".into());
        }
        validate_records(records)?;
        require_disjoint_from_wal(records, &default_workspace_wal_trajectories())?;
        let training: BTreeSet<_> = self
            .training
            .records
            .iter()
            .map(|record| &record.text_sha256)
            .collect();
        let manifests: Vec<_> = records.iter().map(NativeTextRecord::manifest).collect();
        if manifests
            .iter()
            .any(|record| training.contains(&record.text_sha256))
        {
            return Err("evaluation document was used in training".into());
        }
        let samples = examples(&self.config, records)?;
        let teacher_forced = self.measure(&samples)?;
        let mut cases = Vec::new();
        for record in records {
            let generated = self.generate(&record.prompt, max_new)?;
            cases.push(GenerationCase {
                record: record.manifest(),
                prompt: record.prompt.clone(),
                expected_completion: record.completion.clone(),
                exact_completion_and_eos: generated.stop == NativeGenerationStop::Eos
                    && generated.text.as_deref() == Some(record.completion.as_str()),
                matches_reference_prefix: generated.byte_ids.len() >= record.completion.len()
                    && generated
                        .byte_ids
                        .iter()
                        .zip(record.completion.bytes())
                        .all(|(&a, b)| a == u32::from(b)),
                generated,
            });
        }
        Ok(EvaluationReceipt {
            dataset_sha256: manifest_sha256(&manifests)?,
            records: manifests,
            teacher_forced,
            cases,
        })
    }

    /// Each request owns fresh recurrence. The output is a continuation, never prompt readout.
    ///
    /// Qualification discipline: the generator remains 100% hands-off (never steered, pruned, or
    /// resampled by the gate), while recording per-step `z-surprise` and the `TrajectoryMatchGate`
    /// verdict over the emitted continuation.
    pub fn generate(&self, prompt: &str, max_new: usize) -> Result<NativeGeneration, String> {
        self.validate()?;
        if prompt.len() > MAX_RECORD_BYTES {
            return Err("native prompt exceeds 65536 bytes".into());
        }
        if !(1..=MAX_NEW_BYTES).contains(&max_new) {
            return Err("native generation budget must be in 1..=4096 bytes".into());
        }
        let started = Instant::now();
        let mut state = NativeContext::new(&self.config)?;
        state.prefill(prompt)?;
        let mut prompt_embeddings = Vec::with_capacity(prompt.len().max(1));
        if prompt.is_empty() {
            prompt_embeddings.push(embed_carrier(
                &state.boundary.catalog,
                MlirOpId(NATIVE_BOS),
            )?);
        } else {
            for b in prompt.bytes() {
                prompt_embeddings.push(embed_carrier(
                    &state.boundary.catalog,
                    MlirOpId(u32::from(b)),
                )?);
            }
        }
        let prefill_us = started.elapsed().as_micros();
        let decode_started = Instant::now();
        let mut bytes = Vec::new();
        let mut emitted_embeddings = Vec::new();
        let mut stop = NativeGenerationStop::ByteBudget;
        let mut decisions = 0;
        for _ in 0..max_new {
            let features = self.normalized(&state.features()?);
            let mut logits = self.logits(state.previous, &features, true);
            probabilities(&mut logits)?;
            let class = argmax(&logits);
            decisions += 1;
            if class == EOS_CLASS {
                stop = NativeGenerationStop::Eos;
                break;
            }
            let b = class as u8;
            bytes.push(b);
            emitted_embeddings.push(embed_carrier(
                &state.boundary.catalog,
                MlirOpId(u32::from(b)),
            )?);
            state.push(b)?;
        }
        let mut step_predictor = ProductPredictor::new(3, 2, None, None);
        for emb in &prompt_embeddings {
            if let Some((ball, phases)) = embedding_to_ball_phases(emb) {
                let _ = step_predictor.step(&ball, &phases);
            }
        }
        let mut step_z_surprises = Vec::with_capacity(emitted_embeddings.len());
        for emb in &emitted_embeddings {
            if let Some((ball, phases)) = embedding_to_ball_phases(emb) {
                let s = step_predictor.step(&ball, &phases);
                step_z_surprises.push(s.surprise.unwrap_or(0.0));
            }
        }
        let mut gate = TrajectoryMatchGate::new();
        let gate_score = gate.score_candidate(&prompt_embeddings, &emitted_embeddings);
        let mean_z_surprise = gate_score.mean_surprise.unwrap_or(0.0);
        let max_z_surprise = gate_score.max_surprise.unwrap_or(0.0);
        let gate_verdict = gate_score.verdict.as_str().to_string();
        let byte_ids: Vec<_> = bytes.iter().copied().map(u32::from).collect();
        let ops: Vec<_> = byte_ids.iter().copied().map(MlirOpId).collect();
        let text = state.boundary.transduce_mlir_to_utf8(&ops);
        Ok(NativeGeneration {
            byte_ids,
            text: text.as_ref().ok().cloned(),
            utf8_error: text.err(),
            eos_id: (stop == NativeGenerationStop::Eos).then_some(NATIVE_EOS),
            stop,
            decisions,
            prefill_us,
            decode_us: decode_started.elapsed().as_micros(),
            mean_z_surprise,
            max_z_surprise,
            step_z_surprises,
            gate_verdict,
        })
    }

    fn validate(&self) -> Result<(), String> {
        self.config.validate()?;
        self.training.options.validate()?;
        if self.schema != SCHEMA
            || self.architecture != std::env::consts::ARCH
            || self.source_sha256 != native_predictor_source_sha256()
        {
            return Err(
                "native artifact schema, source fingerprint, or architecture is incompatible"
                    .into(),
            );
        }
        if self.means.len() != CONTEXT_FEATURES
            || self.scales.len() != CONTEXT_FEATURES
            || self.weights.len() != OUTPUT_CLASSES * STRIDE
            || self.unigram.len() != OUTPUT_CLASSES
            || self.bigram.len() != OUTPUT_CLASSES * OUTPUT_CLASSES
            || !self
                .means
                .iter()
                .chain(&self.weights)
                .all(|v| v.is_finite() && v.abs() <= 1e6)
            || !self
                .scales
                .iter()
                .all(|v| v.is_finite() && (0.05..=1e6).contains(v))
        {
            return Err("native artifact has invalid shapes, weights, or normalization".into());
        }
        if self.training.selected_epoch > self.training.options.epochs
            || self.training.records.is_empty()
            || self.training.records.len() > MAX_EXAMPLES
            || self.training.dataset_sha256 != manifest_sha256(&self.training.records)?
        {
            return Err("native artifact training provenance is invalid".into());
        }
        let mut labels = BTreeSet::new();
        let mut positions = 0usize;
        let mut total_bytes = 0usize;
        for record in &self.training.records {
            if record.label.is_empty()
                || record.label.len() > 4096
                || !labels.insert(&record.label)
                || !valid_hash(&record.sha256)
                || !valid_hash(&record.text_sha256)
                || record
                    .prompt_bytes
                    .checked_add(record.completion_bytes)
                    .is_none_or(|n| n > MAX_RECORD_BYTES)
            {
                return Err("native artifact contains an invalid record manifest".into());
            }
            positions = positions
                .checked_add(record.completion_bytes + 1)
                .ok_or("artifact position overflow")?;
            total_bytes = total_bytes
                .checked_add(record.prompt_bytes + record.completion_bytes)
                .ok_or("artifact byte count overflow")?;
        }
        if positions > MAX_EXAMPLES || total_bytes > MAX_EXAMPLES {
            return Err("native artifact corpus exceeds limits".into());
        }
        let mut sum = 0u64;
        for (class, &count) in self.unigram.iter().enumerate() {
            let mut column = 0u64;
            for row in self.bigram.chunks_exact(OUTPUT_CLASSES) {
                column = column
                    .checked_add(row[class])
                    .ok_or("artifact count overflow")?;
            }
            if column != count {
                return Err("native artifact unigram/bigram counts disagree".into());
            }
            sum = sum.checked_add(count).ok_or("artifact count overflow")?;
        }
        if sum != positions as u64 || self.unigram[EOS_CLASS] != self.training.records.len() as u64
        {
            return Err("native artifact training counts disagree with provenance".into());
        }
        Ok(())
    }

    /// The digest protects against torn/corrupt writes, not maliciously re-signed files.
    pub fn save_new(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        let payload = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        if payload.len() > MAX_ARTIFACT_BYTES {
            return Err("native artifact is too large".into());
        }
        let mut bytes = MAGIC.to_vec();
        bytes.extend((payload.len() as u64).to_le_bytes());
        bytes.extend(&payload);
        bytes.extend(Sha256Digest::compute(&payload).0);
        write_new_atomic(path, &bytes)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let mut bytes = Vec::new();
        File::open(path)
            .map_err(|e| e.to_string())?
            .take((MAX_ARTIFACT_BYTES + MAGIC.len() + 8 + 32 + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        let offset = MAGIC.len() + 8;
        if bytes.len() < offset + 32 || !bytes.starts_with(MAGIC) {
            return Err("invalid or truncated native artifact header".into());
        }
        let length = u64::from_le_bytes(
            bytes[MAGIC.len()..offset]
                .try_into()
                .map_err(|_| "invalid artifact length")?,
        );
        let length = usize::try_from(length).map_err(|_| "native artifact length overflow")?;
        if length > MAX_ARTIFACT_BYTES
            || offset.checked_add(length).and_then(|n| n.checked_add(32)) != Some(bytes.len())
        {
            return Err("invalid native artifact payload length".into());
        }
        let payload = &bytes[offset..offset + length];
        if Sha256Digest::compute(payload).0.as_slice() != &bytes[offset + length..] {
            return Err("native artifact SHA-256 mismatch".into());
        }
        let model: Self = serde_json::from_slice(payload).map_err(|e| e.to_string())?;
        model.validate()?;
        Ok(model)
    }
}

fn valid_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

pub fn write_new_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let temp = parent.join(format!(".native-next-{}-{stamp}.tmp", std::process::id()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| e.to_string())?;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        // Linking publishes the complete inode and atomically refuses an existing destination.
        fs::hard_link(&temp, path)
            .map_err(|e| format!("cannot publish {}: {e}", path.display()))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|e| e.to_string())?;
        Ok(())
    })();
    let _ = fs::remove_file(&temp);
    result
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PredictionScore {
    pub positions: usize,
    pub correct: usize,
    pub byte_positions: usize,
    pub bytes_correct: usize,
    pub eos_positions: usize,
    pub eos_correct: usize,
    pub eos_predictions: usize,
    pub accuracy: f64,
    pub byte_accuracy: f64,
    pub eos_recall: f64,
    pub mean_nll_nats: f64,
    pub bits_per_position: f64,
    pub z_surprise_steps: usize,
    pub mean_z_surprise: f64,
    pub max_z_surprise: f64,
    pub mean_z_hyper: f64,
    pub mean_z_torus: f64,
}

impl PredictionScore {
    fn record_with_surprise(
        &mut self,
        target: usize,
        predicted: usize,
        nll: f64,
        surprise: Option<ProductPredictStep>,
    ) {
        self.positions += 1;
        self.correct += usize::from(predicted == target);
        self.eos_predictions += usize::from(predicted == EOS_CLASS);
        if target == EOS_CLASS {
            self.eos_positions += 1;
            self.eos_correct += usize::from(predicted == target);
        } else {
            self.byte_positions += 1;
            self.bytes_correct += usize::from(predicted == target);
        }
        self.mean_nll_nats += nll;
        if let Some(step) = surprise
            && let Some(zs) = step.surprise
            && zs.is_finite()
        {
            self.z_surprise_steps += 1;
            self.mean_z_surprise += zs;
            if zs > self.max_z_surprise {
                self.max_z_surprise = zs;
            }
            self.mean_z_hyper += step.z_hyper.unwrap_or(0.0);
            self.mean_z_torus += step.z_torus.unwrap_or(0.0);
        }
    }
    fn finish(&mut self) {
        self.accuracy = self.correct as f64 / self.positions.max(1) as f64;
        self.byte_accuracy = self.bytes_correct as f64 / self.byte_positions.max(1) as f64;
        self.eos_recall = self.eos_correct as f64 / self.eos_positions.max(1) as f64;
        self.mean_nll_nats /= self.positions.max(1) as f64;
        self.bits_per_position = self.mean_nll_nats / std::f64::consts::LN_2;
        let z_denom = self.z_surprise_steps.max(1) as f64;
        self.mean_z_surprise /= z_denom;
        self.mean_z_hyper /= z_denom;
        self.mean_z_torus /= z_denom;
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct EvaluationScores {
    pub learned: PredictionScore,
    pub without_toroidal_features: PredictionScore,
    /// Post-fit ablation replaces normalized history features with their training mean (zero).
    pub without_ordered_history: PredictionScore,
    pub unigram: PredictionScore,
    pub bigram: PredictionScore,
}

#[derive(Debug, Serialize)]
pub struct TrainingReceipt {
    pub positions: usize,
    pub initial: EvaluationScores,
    pub final_scores: EvaluationScores,
    pub epochs: Vec<PredictionScore>,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeGenerationStop {
    Eos,
    ByteBudget,
}

#[derive(Debug, Serialize)]
pub struct NativeGeneration {
    pub byte_ids: Vec<u32>,
    pub text: Option<String>,
    pub utf8_error: Option<String>,
    pub stop: NativeGenerationStop,
    pub eos_id: Option<u32>,
    pub decisions: usize,
    pub prefill_us: u128,
    pub decode_us: u128,
    pub mean_z_surprise: f64,
    pub max_z_surprise: f64,
    pub step_z_surprises: Vec<f64>,
    pub gate_verdict: String,
}

#[derive(Debug, Serialize)]
pub struct GenerationCase {
    pub record: RecordManifest,
    pub prompt: String,
    pub expected_completion: String,
    pub generated: NativeGeneration,
    pub exact_completion_and_eos: bool,
    pub matches_reference_prefix: bool,
}

#[derive(Debug, Serialize)]
pub struct EvaluationReceipt {
    pub dataset_sha256: String,
    pub records: Vec<RecordManifest>,
    pub teacher_forced: EvaluationScores,
    pub cases: Vec<GenerationCase>,
}

// ============================================================================
// NEXT-CARRIER TRANSITION DRIVEN FROM TOROIDAL HIDDEN STATE
// Contract: docs/NATIVE_CARRIER_TRANSITION_CONTRACT.md
// ============================================================================

pub const CARRIER_TOROIDAL_FEATURES: usize = 32;
const CARRIER_STRIDE: usize = 1 + CARRIER_TOROIDAL_FEATURES;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
const WAL_FRAME_MAGIC: &[u8; 8] = b"ATOMA_WL";

/// `Sha256Digest::compute(b"geodesic-active-transformer-v1")` — origin fingerprint of live /
/// dream-cycle teacher-crystallized trajectories (including the `2+2 -> double-entry accounting`
/// contamination frame in `design/global_active_mind.atoma_map.wal`).
pub const TEACHER_ACTIVE_TRANSFORMER_ORIGIN_HEX: &str =
    "a30a100326adb6d24d3b7dd51e5cb8b92d3b076f71302f6b9439f0d01bc54e61";

/// Legacy pre-rank8 teacher WAL origin fingerprint in `global_active_mind.atoma_map.wal.bak_pre_rank8`.
pub const TEACHER_PRE_RANK8_ORIGIN_HEX: &str =
    "0a29d4cfbe0aa64d6bfd9fcadb22c78c1c68b0f08cc2c1dfa4b2e0656014fcbc";

/// Returns `true` if `origin` matches a known teacher-crystallized WAL / dream-cycle origin
/// that must be quarantined from next-carrier transition training and evaluation.
pub fn is_teacher_contaminated_origin(origin: &Sha256Digest) -> bool {
    let hex = origin.to_hex();
    hex == TEACHER_ACTIVE_TRANSFORMER_ORIGIN_HEX || hex == TEACHER_PRE_RANK8_ORIGIN_HEX
}

/// Decoded trajectory from a binary `ATOMA_WL` teacher-era WAL frame (`design/*.atoma_map.wal`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalFrameTrajectory {
    pub frame_index: usize,
    pub episode_id: u64,
    pub origin_fingerprint_hex: String,
    pub prompt_ops: Vec<MlirOpId>,
    pub emitted_ops: Vec<MlirOpId>,
    pub prompt_text: String,
    pub emitted_text: String,
    pub step_chain_ids: Vec<u64>,
}

/// Parse binary `ATOMA_WL` frames from `wal_path` and decode their prompt/emitted text and
/// per-step `chain_id`s so teacher-era WAL frames can be rejected by `require_disjoint_from_wal`
/// and excluded from `fit_from_swarm_map`.
pub fn parse_wal_trajectories(
    wal_path: &Path,
    transducer: &BoundaryLexicalTransducer,
) -> Result<Vec<WalFrameTrajectory>, String> {
    if !wal_path.is_file() {
        return Ok(Vec::new());
    }
    let bytes =
        fs::read(wal_path).map_err(|e| format!("failed to read WAL {}: {e}", wal_path.display()))?;
    let mut pos = 0usize;
    let mut frame_index = 0usize;
    let mut out = Vec::new();
    while pos + 52 <= bytes.len() {
        let frame_start = pos;
        if &bytes[pos..pos + 8] != WAL_FRAME_MAGIC {
            break;
        }
        let mut cursor = pos + 8;
        let episode_id = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let mut origin_bytes = [0u8; 32];
        origin_bytes.copy_from_slice(&bytes[cursor..cursor + 32]);
        let origin_hex = Sha256Digest(origin_bytes).to_hex();
        cursor += 32;
        let n_steps = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if n_steps > 100_000 {
            break;
        }
        let mut prompt_ops = Vec::new();
        let mut emitted_ops = Vec::with_capacity(n_steps);
        let mut step_chain_ids = Vec::with_capacity(n_steps);
        let mut truncated = false;
        for step_idx in 0..n_steps {
            if cursor + 4 > bytes.len() {
                truncated = true;
                break;
            }
            let plen = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            if plen > 100_000 || cursor + plen.saturating_mul(4) + 12 > bytes.len() {
                truncated = true;
                break;
            }
            let mut pref = Vec::with_capacity(plen);
            for _ in 0..plen {
                pref.push(MlirOpId(u32::from_le_bytes(
                    bytes[cursor..cursor + 4].try_into().unwrap(),
                )));
                cursor += 4;
            }
            let next_op = MlirOpId(u32::from_le_bytes(
                bytes[cursor..cursor + 4].try_into().unwrap(),
            ));
            cursor += 4;
            // Skip 8-byte certainty Q32_31
            cursor += 8;
            if step_idx == 0 {
                prompt_ops = pref.clone();
            }
            step_chain_ids.push(prefix_fnv1a(&pref));
            emitted_ops.push(next_op);
        }
        if truncated || cursor + 32 > bytes.len() {
            break;
        }
        let expected_sha = Sha256Digest::compute(&bytes[frame_start..cursor]);
        if expected_sha.0 != bytes[cursor..cursor + 32] {
            break;
        }
        pos = cursor + 32;

        let (_, raw_prompt) = transducer.transduce_egress_ops(&prompt_ops);
        let prompt_text = raw_prompt
            .trim_start_matches("<|im_start|>user\n")
            .trim_end_matches("<|im_end|>\n<|im_start|>assistant\n")
            .to_string();
        let (_, emitted_text) = transducer.transduce_egress_ops(&emitted_ops);
        out.push(WalFrameTrajectory {
            frame_index,
            episode_id,
            origin_fingerprint_hex: origin_hex,
            prompt_ops,
            emitted_ops,
            prompt_text,
            emitted_text,
            step_chain_ids,
        });
        frame_index += 1;
    }
    Ok(out)
}

/// Load the workspace's active `.atoma_map.wal` trajectories if present under `design/`.
pub fn default_workspace_wal_trajectories() -> Vec<WalFrameTrajectory> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let wal_path = manifest_dir.join("../design/global_active_mind.atoma_map.wal");
    if !wal_path.is_file() {
        return Vec::new();
    }
    let lex_path = manifest_dir.join("../design/boundary_lexicon.atoma_lex");
    let mut transducer =
        BoundaryLexicalTransducer::new(248_320, 248_046, vec![248_045, 248_047, NATIVE_EOS]);
    if lex_path.is_file() {
        let _ = transducer.load_binary_lexicon(&lex_path);
    }
    parse_wal_trajectories(&wal_path, &transducer).unwrap_or_default()
}

/// Single-step next-carrier transition decision driven from the toroidal hidden state.
#[derive(Debug, Clone)]
pub struct CarrierTransitionStep {
    /// 0-based continuation step index.
    pub step_index: usize,
    /// Carrier at the current position (`c_t`).
    pub from_carrier: MlirOpId,
    /// Predicted next carrier (`c_{t+1}`).
    pub predicted_carrier: MlirOpId,
    /// Log-probability of the selected transition in nats (`<= 0.0`).
    pub log_prob: f64,
    /// Probability margin `p(top_1) - p(top_2)` in `[0.0, 1.0]`.
    pub margin: f64,
    /// 6D toroidal hidden state `[x0, x1, x2, x3, theta_f, theta_s]` driving this step.
    pub toroidal_hidden: [f32; 6],
    /// Predicted next-carrier 6D manifold target coordinate on H^3 x T^2.
    pub predicted_target_coord: [f32; 6],
    /// Whether this transition matched an outgoing 1-cell continuation arc in the vault.
    pub matched_vault_arc: bool,
    /// Per-step operator `z-surprise` (`sqrt(z_hyper^2 + z_torus^2)`) from `predict_gate.rs`.
    pub z_surprise: f64,
    /// Hyperbolic component of per-step `z-surprise` on the Poincare ball `B^3`.
    pub z_hyper: f64,
    /// Toroidal component of per-step `z-surprise` on `T^2`.
    pub z_torus: f64,
}

/// Candidate continuation produced by the toroidal next-carrier transition predictor.
///
/// Ownership contract (`docs/NATIVE_CARRIER_TRANSITION_CONTRACT.md`):
/// - Predictor (`native_predictor.rs`) owns candidate generation.
/// - Single Gate (`TrajectoryMatchGate` in `predict_gate.rs`, invoked by `runtime_model.rs`)
///   runs AFTER predictor in `execute_live_operational_turn` and owns accept/reject over this
///   candidate (`candidate.score_with_gate(&mut gate)`). No second gate exists.
#[derive(Debug, Clone)]
pub struct NativeContinuationCandidate {
    /// Prompt prefix carrier embeddings used to seed `TrajectoryMatchGate`.
    pub prompt_embeddings: Vec<NativeMlirEmbedding>,
    /// Emitted continuation carrier embeddings passed into `TrajectoryMatchGate::score_response`.
    pub emitted_embeddings: Vec<NativeMlirEmbedding>,
    /// Emitted continuation carrier IDs (`MlirOpId`), excluding prompt prefix carriers.
    pub emitted_carriers: Vec<MlirOpId>,
    /// Decoded UTF-8 continuation text when `emitted_carriers` decode cleanly.
    pub decoded_text: Option<String>,
    /// Error message when `emitted_carriers` fail strict UTF-8 / transducer decoding.
    pub decode_error: Option<String>,
    /// Stopping condition (`NativeGenerationStop::Eos` or `NativeGenerationStop::ByteBudget`).
    pub stop: NativeGenerationStop,
    /// Mean log-probability (in nats, `<= 0.0`) across emitted continuation steps.
    pub mean_log_prob: f64,
    /// Minimum top-1 minus top-2 probability margin (`[0.0, 1.0]`) across steps.
    pub min_margin: f64,
    /// Fraction of emitted steps backed by a vault continuation arc (`[0.0, 1.0]`).
    pub vault_arc_coverage: f64,
    /// Mean per-step operator `z-surprise` (`predict_gate.rs`) across emitted steps.
    pub mean_z_surprise: f64,
    /// Maximum per-step operator `z-surprise` (`predict_gate.rs`) across emitted steps.
    pub max_z_surprise: f64,
    /// Full `GateScore` (`verdict`, `steps`, `max_z_surprise`, etc.) from `TrajectoryMatchGate`.
    pub gate_score: GateScore,
    /// Per-step trace of toroidal hidden states, carrier transitions, and `z-surprise`.
    pub steps: Vec<CarrierTransitionStep>,
    /// Always `true` when produced by the next-carrier transition rollout
    /// (distinguishes genuine continuation from legacy static `prompt_readout`).
    pub is_continuation: bool,
    /// Constructed toroidal layer depth (`1..=52`) for this candidate (`docs/BREATHING_DEPTH_DESIGN.md` §4).
    pub constructed_depth: usize,
    /// Constructed Jacobian/harmonic sub-rank (`1..=8`) for this candidate (`docs/BREATHING_DEPTH_DESIGN.md` §4).
    pub constructed_rank: usize,
    /// Active phase attention sectors (`1..=8`) lit by PAMI (`docs/BREATHING_DEPTH_DESIGN.md` §4).
    pub active_attention_sectors: usize,
    /// PAMI multifractal spectral spread `max(D(q)) - min(D(q))` on the prompt carrier trajectory.
    pub pami_spectral_spread: Option<f64>,
    /// Free-energy reduction `F(1) - F(constructed_depth)` earned by the constructed layers.
    pub marginal_free_energy_gain: f64,
    /// Radial position in the Poincaré ball `r = tanh(rapidity / 2) in [0, 1)` at the constructed depth.
    pub poincare_radius: f64,
}

impl NativeContinuationCandidate {
    /// Feed this candidate directly into the single `TrajectoryMatchGate` chassis (`predict_gate.rs`)
    /// — candidate in, verdict out, generator-agnostic.
    pub fn score_with_gate(&self, gate: &mut TrajectoryMatchGate) -> GateScore {
        gate.score_candidate(&self.prompt_embeddings, &self.emitted_embeddings)
    }
}

/// One resolved continuation pair `(from_carrier -> to_carrier)` extracted from a vault arc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaultContinuationArcPair {
    pub arc_id: u64,
    pub chain_id: u64,
    pub step_index: usize,
    pub from_atom: u64,
    pub to_atom: u64,
    pub from_carrier: MlirOpId,
    pub to_carrier: MlirOpId,
    pub cosh_action_q32: i64,
    pub teacher_certainty_q32: i64,
    pub is_chain_terminal: bool,
    pub is_teacher_contaminated: bool,
}

/// Summary and extracted chains of the continuation-pair arcs in a `.atoma_map` vault.
#[derive(Debug, Clone)]
pub struct VaultContinuationDataset {
    pub total_vault_arcs: usize,
    pub carrier_continuation_arcs: usize,
    pub clean_continuation_arcs: usize,
    pub teacher_contaminated_arcs: usize,
    pub wal_contaminated_frames: usize,
    pub wal_contaminated_steps: usize,
    pub structural_anchor_arcs: usize,
    pub contiguous_step_pairs: usize,
    pub trajectory_chains: usize,
    pub unique_source_atoms: usize,
    pub branching_source_atoms: usize,
    pub branching_arcs: usize,
    pub unique_target_carriers: usize,
    pub chains: Vec<Vec<VaultContinuationArcPair>>,
    /// Clean chains with all teacher-crystallized arcs (`geodesic-active-transformer-v1` and
    /// `.atoma_map.wal` frames such as `2+2`) removed so training never learns teacher errors.
    pub clean_chains: Vec<Vec<VaultContinuationArcPair>>,
}

/// Training receipt for `NativeCarrierTransitionModel::fit_from_swarm_map`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarrierTransitionTrainingReceipt {
    pub total_vault_arcs: usize,
    pub carrier_continuation_arcs: usize,
    pub clean_continuation_arcs: usize,
    pub teacher_contaminated_arcs_filtered: usize,
    pub wal_contaminated_frames_filtered: usize,
    pub structural_anchor_arcs: usize,
    pub contiguous_step_pairs: usize,
    pub trajectory_chains: usize,
    pub branching_source_atoms: usize,
    pub branching_arcs: usize,
    pub unique_target_carriers: usize,
    pub initial_coord_mse: f64,
    pub final_coord_mse: f64,
    pub initial_branch_nll_nats: f64,
    pub final_branch_nll_nats: f64,
    pub initial_branch_top1_acc: f64,
    pub final_branch_top1_acc: f64,
    pub initial_mean_z_surprise: f64,
    pub final_mean_z_surprise: f64,
    pub final_max_z_surprise: f64,
    pub elapsed_ms: u128,
}

/// Learned per-transition toroidal hidden-state prototype for `(from_atom, to_carrier)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToroidalTransitionPrototype {
    pub count: u32,
    pub log_prior: f32,
    /// Learned 16-dimensional toroidal hidden-state discriminator weights for this transition.
    pub weights: [f32; 16],
}

/// Per-depth-band readout head trained on toroidal traces at `reference_depth` (`min_depth..=max_depth`).
///
/// Implements `docs/BREATHING_DEPTH_DESIGN.md` §5:
/// - Train across depths (`[1..=4, 5..=12, 13..=28, 29..=52]`),
/// - Depth and rank as explicit input covariates (`norm_depth = ln(L)/ln(52)`, `norm_rank = R/8`),
/// - Per-depth readout heads so varying constructed depth never shifts the predictor's input distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthConditionalReadoutHead {
    pub min_depth: usize,
    pub max_depth: usize,
    pub reference_depth: usize,
    pub means: [f64; CARRIER_TOROIDAL_FEATURES],
    pub scales: [f64; CARRIER_TOROIDAL_FEATURES],
    /// 6 rows x 35 columns (`[bias, f_0..f_31, norm_depth, norm_rank]`) predicting 6D coordinate deltas.
    pub coord_weights: Vec<f64>,
    /// 35 weights (`[bias, f_0..f_31, norm_depth, norm_rank]`) predicting the EOS logit at this depth band.
    pub eos_weights: Vec<f64>,
    /// 5 scoring weights `[beta_coord, beta_torus_phase, beta_frame, beta_action, beta_cert]`.
    pub branch_weights: [f64; 5],
    pub initial_coord_mse: f64,
    pub final_coord_mse: f64,
}

/// Canonical 4-band depth schedule (`(min_depth, max_depth, reference_depth)`) for depth-conditional training.
pub const DEPTH_BAND_SPECS: [(usize, usize, usize); 4] = [
    (1, 4, 2),
    (5, 12, 8),
    (13, 28, 20),
    (29, 52, 52),
];

const DEPTH_COND_STRIDE: usize = CARRIER_TOROIDAL_FEATURES + 3; // [1.0, f_0..f_31, norm_depth, norm_rank]

/// Learned next-carrier transition operator driven from the toroidal hidden state.
///
/// Maps the 32D toroidal hidden-state feature vector `\psi_t` (extracted from the
/// `ToroidalIsotropicRuntime` stack at its constructed depth `L in 1..=52`) + explicit
/// `(norm_depth, norm_rank)` covariates to:
/// 1. The predicted next-carrier 6D `(H^3 x T^2)` coordinate displacement via per-depth readout heads,
/// 2. Learned per-transition toroidal hidden-state discriminators (`transition_prototypes`) and
///    log-linear scoring weights `[beta_coord, beta_torus_phase, beta_frame, beta_action, beta_cert]`
///    that disambiguate branching continuation arcs from the toroidal hidden state, and
/// 3. A depth-conditional EOS stopping head.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeCarrierTransitionModel {
    pub layers: usize,
    pub means: [f64; CARRIER_TOROIDAL_FEATURES],
    pub scales: [f64; CARRIER_TOROIDAL_FEATURES],
    /// 6 rows x 33 columns (`[bias, f_0..f_31]`) predicting `[dx1, dx2, dx3, d_ln_x0, d_theta_f, d_theta_s]`.
    pub coord_weights: Vec<f64>,
    /// 5 learned scoring weights `[beta_coord, beta_torus_phase, beta_frame, beta_action, beta_cert]`.
    pub branch_weights: [f64; 5],
    /// 33 weights (`[bias, f_0..f_31]`) predicting the EOS stopping logit.
    pub eos_weights: Vec<f64>,
    /// Learned toroidal-state discriminators keyed by `(from_atom, to_carrier)`.
    #[serde(with = "transition_prototypes_serde")]
    pub transition_prototypes: std::collections::BTreeMap<(u64, u32), ToroidalTransitionPrototype>,
    /// Explicit depth & rank covariate weights (`6 rows x [norm_depth, norm_rank]`).
    #[serde(default)]
    pub depth_cov_weights: [f64; 12],
    /// Explicit depth & rank covariate weights for EOS (`[norm_depth, norm_rank]`).
    #[serde(default)]
    pub eos_depth_cov_weights: [f64; 2],
    /// Per-depth-band readout heads trained across `[1..=4, 5..=12, 13..=28, 29..=52]`.
    #[serde(default)]
    pub per_depth_readouts: Vec<DepthConditionalReadoutHead>,
}

mod transition_prototypes_serde {
    use super::ToroidalTransitionPrototype;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<S>(
        map: &BTreeMap<(u64, u32), ToroidalTransitionPrototype>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let entries: Vec<(u64, u32, &ToroidalTransitionPrototype)> = map
            .iter()
            .map(|(&(from_atom, to_carrier), proto)| (from_atom, to_carrier, proto))
            .collect();
        entries.serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<BTreeMap<(u64, u32), ToroidalTransitionPrototype>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries: Vec<(u64, u32, ToroidalTransitionPrototype)> =
            Vec::deserialize(deserializer)?;
        Ok(entries
            .into_iter()
            .map(|(from_atom, to_carrier, proto)| ((from_atom, to_carrier), proto))
            .collect())
    }
}

impl Default for NativeCarrierTransitionModel {
    fn default() -> Self {
        Self {
            layers: 52,
            means: [0.0; CARRIER_TOROIDAL_FEATURES],
            scales: [1.0; CARRIER_TOROIDAL_FEATURES],
            coord_weights: vec![0.0; 6 * CARRIER_STRIDE],
            branch_weights: [2.4, 1.8, 3.2, 1.2, 1.5],
            eos_weights: {
                let mut w = vec![0.0; CARRIER_STRIDE];
                w[0] = -3.2;
                w
            },
            transition_prototypes: std::collections::BTreeMap::new(),
            depth_cov_weights: [0.0; 12],
            eos_depth_cov_weights: [0.0; 2],
            per_depth_readouts: Vec::new(),
        }
    }
}

fn fnv1a_step(chain_id: u64, carrier_id: u32) -> u64 {
    chain_id
        .wrapping_mul(FNV_PRIME)
        .wrapping_add(u64::from(carrier_id))
}

fn prefix_fnv1a(prefix: &[MlirOpId]) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    for op in prefix {
        h = fnv1a_step(h, op.0);
    }
    h
}

fn wrap_angle_diff(target: f64, source: f64) -> f64 {
    let d = target - source;
    d.sin().atan2(d.cos())
}

fn wrap_angle_positive(angle: f64) -> f32 {
    angle.rem_euclid(std::f64::consts::TAU) as f32
}

fn clone_toroidal_runtime(rt: &ToroidalIsotropicRuntime) -> ToroidalIsotropicRuntime {
    ToroidalIsotropicRuntime {
        hidden_dim: rt.hidden_dim,
        active_layers: rt.active_layers.clone(),
        total_model_layers: rt.total_model_layers,
        toroidal_field_mix: rt.toroidal_field_mix,
        horizon_decay: rt.horizon_decay,
        map_engine: atoma_immune::MapNativeInferenceEngine::default(),
        layer_fields: rt.layer_fields.clone(),
        working_memory: rt.working_memory.clone(),
    }
}

/// Extract a 32-dimensional causal feature vector `\psi_t` from the current 6D toroidal hidden
/// state `h_t`, current 6D carrier coordinate `c_t`, and the `ToroidalIsotropicRuntime` field stack.
pub fn extract_toroidal_carrier_features(
    rt: &ToroidalIsotropicRuntime,
    toroidal_hidden: &[f32; 6],
    carrier_coord: &[f32; 6],
) -> Result<[f64; CARRIER_TOROIDAL_FEATURES], String> {
    if toroidal_hidden.iter().chain(carrier_coord).any(|v| !v.is_finite()) {
        return Err("non-finite toroidal hidden state or carrier coordinate".into());
    }
    let first = rt
        .layer_fields
        .first()
        .ok_or("toroidal runtime has no layer fields")?;
    let last = rt
        .layer_fields
        .last()
        .ok_or("toroidal runtime has no layer fields")?;

    let mut feat = [0.0f64; CARRIER_TOROIDAL_FEATURES];
    // 0..6: current 6D toroidal hidden state
    for i in 0..6 {
        feat[i] = f64::from(toroidal_hidden[i]);
    }
    // 6..12: current 6D carrier coordinate
    for i in 0..6 {
        feat[6 + i] = f64::from(carrier_coord[i]);
    }
    // 12..16: spatial Lorentz displacement h_t[0..4] - coord[0..4]
    for i in 0..4 {
        feat[12 + i] = f64::from(toroidal_hidden[i]) - f64::from(carrier_coord[i]);
    }
    // 16..18: wrapped periodic phase displacement on T^2
    feat[16] = wrap_angle_diff(f64::from(toroidal_hidden[4]), f64::from(carrier_coord[4]));
    feat[17] = wrap_angle_diff(f64::from(toroidal_hidden[5]), f64::from(carrier_coord[5]));

    // 18..22: Picard-reduced Lorentz state of the final layer field
    let rx0 = last.reduced_lorentz.x[0].to_f64().max(1.0);
    feat[18] = rx0.ln();
    feat[19] = last.reduced_lorentz.x[1].to_f64() / rx0;
    feat[20] = last.reduced_lorentz.x[2].to_f64() / rx0;
    feat[21] = last.reduced_lorentz.x[3].to_f64() / rx0;

    // 22..26: Kuramoto PhaseState angles on the final layer field
    let tf = last.phase_state.theta_f.to_f64();
    let ts = last.phase_state.theta_s.to_f64();
    feat[22] = tf.sin();
    feat[23] = tf.cos();
    feat[24] = ts.sin();
    feat[25] = ts.cos();

    // 26..30: Monotonic amplitude, radius_accumulator, rapidity_accumulator, cumulative_rotations
    feat[26] = last.phase_state.amplitude.to_f64().max(0.0).ln_1p();
    feat[27] = last.radius_accumulator.max(0.0).ln_1p();
    feat[28] = last.rapidity_accumulator.max(0.0).ln_1p();
    feat[29] = last.cumulative_rotations.max(0.0).ln_1p();

    // 30..32: First-vs-last layer harmonic moment norm & tokens ingested
    let first_m0_norm = first.m_0.iter().map(|v| v * v).sum::<f64>().sqrt() / first.z_0.max(1.0);
    let last_m0_norm = last.m_0.iter().map(|v| v * v).sum::<f64>().sqrt() / last.z_0.max(1.0);
    feat[30] = (last_m0_norm - first_m0_norm).clamp(-100.0, 100.0);
    feat[31] = (last.tokens_ingested as f64).ln_1p();

    if feat.iter().any(|v| !v.is_finite()) {
        return Err("non-finite extracted toroidal carrier feature".into());
    }
    Ok(feat)
}

/// Extract all continuation-pair arcs and contiguous trajectory chains from `swarm_map`,
/// partitioning out any teacher-crystallized arcs (`geodesic-active-transformer-v1` and
/// `.atoma_map.wal` frames) into `clean_chains` when clean pre-teacher arcs are present.
pub fn extract_vault_continuation_dataset(
    swarm_map: &atoma_immune::CarvedSwarmMap,
    catalog: &atoma_immune::MlirAtomCatalog,
) -> Result<VaultContinuationDataset, String> {
    let total_vault_arcs = swarm_map.reasoning_arcs.len();
    if total_vault_arcs == 0 {
        return Err("vault contains no reasoning arcs".into());
    }

    let wal_frames = default_workspace_wal_trajectories();
    let wal_contaminated_frames = wal_frames.len();
    let mut wal_chain_ids = std::collections::HashSet::<u64>::new();
    let mut wal_contaminated_steps = 0usize;
    for wf in &wal_frames {
        wal_contaminated_steps += wf.step_chain_ids.len();
        for &cid in &wf.step_chain_ids {
            wal_chain_ids.insert(cid);
        }
    }

    let has_clean_base_arcs = swarm_map.reasoning_arcs.iter().any(|arc| {
        arc.compiled_mlir_op.0 < catalog.base_vocab_size
            && !is_teacher_contaminated_origin(&arc.origin_model_fingerprint)
            && !wal_chain_ids.contains(&arc.chain_id)
    });

    // 1. Map every `to_atom` of a carrier arc directly to its `compiled_mlir_op`.
    let mut atom_to_carrier = std::collections::HashMap::<u64, MlirOpId>::with_capacity(
        swarm_map.global_atom_count().max(1024),
    );
    let mut out_set_by_atom =
        std::collections::HashMap::<u64, std::collections::BTreeSet<u32>>::new();
    let mut out_count_by_atom = std::collections::HashMap::<u64, usize>::new();
    let mut target_carriers = std::collections::BTreeSet::<u32>::new();
    let mut carrier_continuation_arcs = 0usize;
    let mut clean_continuation_arcs = 0usize;
    let mut teacher_contaminated_arcs = 0usize;
    let mut structural_anchor_arcs = 0usize;

    for arc in &swarm_map.reasoning_arcs {
        if arc.compiled_mlir_op.0 < catalog.base_vocab_size {
            atom_to_carrier.insert(arc.to_atom.0, arc.compiled_mlir_op);
            out_set_by_atom
                .entry(arc.from_atom.0)
                .or_default()
                .insert(arc.compiled_mlir_op.0);
            *out_count_by_atom.entry(arc.from_atom.0).or_insert(0) += 1;
            target_carriers.insert(arc.compiled_mlir_op.0);
            carrier_continuation_arcs += 1;
            let is_contaminated = has_clean_base_arcs
                && (is_teacher_contaminated_origin(&arc.origin_model_fingerprint)
                    || wal_chain_ids.contains(&arc.chain_id));
            if is_contaminated {
                teacher_contaminated_arcs += 1;
            } else {
                clean_continuation_arcs += 1;
            }
        } else {
            structural_anchor_arcs += 1;
        }
    }

    // 2. For any `from_atom` that only appears at step 0 of a chain (never as a `to_atom`),
    //    invert its FNV-1a `chain_id` when `chain_id = FNV_OFFSET_BASIS * FNV_PRIME + op_0`,
    //    or match its phase/Lorentz coordinate.
    let inv_fnv_prime: u64 = 0x359c_449b_1431_6eb3; // modular inverse of 0x100000001b3 mod 2^64
    for arc in &swarm_map.reasoning_arcs {
        if arc.compiled_mlir_op.0 >= catalog.base_vocab_size {
            continue;
        }
        atom_to_carrier.entry(arc.from_atom.0).or_insert_with(|| {
            let base1 = FNV_OFFSET_BASIS.wrapping_mul(FNV_PRIME);
            let cand1 = arc.chain_id.wrapping_sub(base1);
            if cand1 < u64::from(catalog.base_vocab_size) {
                return MlirOpId(cand1 as u32);
            }
            let _ = inv_fnv_prime;
            if let Some(atom) = swarm_map.global_atom(arc.from_atom) {
                let tf = atom.phase.theta_f.to_f64().rem_euclid(std::f64::consts::TAU);
                let approx_byte = ((tf / std::f64::consts::TAU) * 256.0).round() as u32 % 256;
                MlirOpId(approx_byte)
            } else {
                MlirOpId(32)
            }
        });
    }

    // 3. Segment contiguous chains across `swarm_map.reasoning_arcs`.
    let mut chains: Vec<Vec<VaultContinuationArcPair>> = Vec::new();
    let mut current_chain: Vec<VaultContinuationArcPair> = Vec::new();
    let mut contiguous_step_pairs = 0usize;

    for arc in &swarm_map.reasoning_arcs {
        if arc.compiled_mlir_op.0 >= catalog.base_vocab_size {
            continue;
        }
        let from_carrier = atom_to_carrier
            .get(&arc.from_atom.0)
            .copied()
            .unwrap_or(MlirOpId(32));
        let is_teacher_contaminated = has_clean_base_arcs
            && (is_teacher_contaminated_origin(&arc.origin_model_fingerprint)
                || wal_chain_ids.contains(&arc.chain_id));
        let pair = VaultContinuationArcPair {
            arc_id: arc.arc_id,
            chain_id: arc.chain_id,
            step_index: arc.step_index,
            from_atom: arc.from_atom.0,
            to_atom: arc.to_atom.0,
            from_carrier,
            to_carrier: arc.compiled_mlir_op,
            cosh_action_q32: arc.cosh_action_q32.0,
            teacher_certainty_q32: arc.teacher_certainty_q32.0,
            is_chain_terminal: false,
            is_teacher_contaminated,
        };

        let continues_prev = if let Some(prev) = current_chain.last() {
            let expected_chain = fnv1a_step(prev.chain_id, prev.to_carrier.0);
            arc.from_atom.0 == prev.to_atom
                && arc.step_index == prev.step_index + 1
                && arc.chain_id == expected_chain
        } else {
            false
        };

        if continues_prev {
            contiguous_step_pairs += 1;
            current_chain.push(pair);
        } else {
            if let Some(last_in_chain) = current_chain.last_mut() {
                last_in_chain.is_chain_terminal = true;
            }
            if !current_chain.is_empty() {
                chains.push(std::mem::take(&mut current_chain));
            }
            current_chain.push(pair);
        }
    }
    if let Some(last_in_chain) = current_chain.last_mut() {
        last_in_chain.is_chain_terminal = true;
    }
    if !current_chain.is_empty() {
        chains.push(current_chain);
    }

    let clean_chains: Vec<Vec<VaultContinuationArcPair>> = chains
        .iter()
        .filter_map(|chain| {
            if chain.iter().any(|p| p.is_teacher_contaminated) {
                None
            } else {
                Some(chain.clone())
            }
        })
        .collect();

    let unique_source_atoms = out_set_by_atom.len();
    let mut branching_source_atoms = 0usize;
    let mut branching_arcs = 0usize;
    for (atom_id, set) in &out_set_by_atom {
        if set.len() > 1 {
            branching_source_atoms += 1;
            branching_arcs += out_count_by_atom.get(atom_id).copied().unwrap_or(0);
        }
    }

    Ok(VaultContinuationDataset {
        total_vault_arcs,
        carrier_continuation_arcs,
        clean_continuation_arcs,
        teacher_contaminated_arcs,
        wal_contaminated_frames,
        wal_contaminated_steps,
        structural_anchor_arcs,
        contiguous_step_pairs,
        trajectory_chains: chains.len(),
        unique_source_atoms,
        branching_source_atoms,
        branching_arcs,
        unique_target_carriers: target_carriers.len(),
        chains,
        clean_chains,
    })
}

impl NativeCarrierTransitionModel {
    /// Normalize `(constructed_depth, constructed_rank)` into `[0, 1]` covariates.
    #[inline]
    pub fn normalize_depth_rank_covariates(
        constructed_depth: usize,
        constructed_rank: usize,
    ) -> (f64, f64) {
        let d = ((constructed_depth.clamp(1, 52) as f64).ln() / (52.0f64).ln()).clamp(0.0, 1.0);
        let r = ((constructed_rank.clamp(1, 8) as f64) / 8.0).clamp(0.125, 1.0);
        (d, r)
    }

    /// Select the per-depth readout head covering `constructed_depth`, or the nearest trained band.
    pub fn select_depth_readout(
        &self,
        constructed_depth: usize,
    ) -> Option<&DepthConditionalReadoutHead> {
        if self.per_depth_readouts.is_empty() {
            return None;
        }
        let d = constructed_depth.max(1);
        self.per_depth_readouts
            .iter()
            .find(|h| d >= h.min_depth && d <= h.max_depth)
            .or_else(|| {
                self.per_depth_readouts
                    .iter()
                    .min_by_key(|h| h.reference_depth.abs_diff(d))
            })
    }

    /// Return the depth-conditioned branching weights for `constructed_depth`.
    pub fn branch_weights_at_depth(&self, constructed_depth: usize) -> [f64; 5] {
        self.select_depth_readout(constructed_depth)
            .map(|h| h.branch_weights)
            .unwrap_or(self.branch_weights)
    }

    /// Predict the next carrier's 6D `(H^3 x T^2)` coordinate conditioned on the constructed
    /// depth `constructed_depth` and rank `constructed_rank` (`docs/BREATHING_DEPTH_DESIGN.md` §5).
    pub fn predict_target_coord_at_depth(
        &self,
        current_coord: &[f32; 6],
        features: &[f64; CARRIER_TOROIDAL_FEATURES],
        constructed_depth: usize,
        constructed_rank: usize,
    ) -> [f32; 6] {
        let (norm_d, norm_r) =
            Self::normalize_depth_rank_covariates(constructed_depth, constructed_rank);

        let mut delta = [0.0f64; 6];
        if let Some(head) = self.select_depth_readout(constructed_depth)
            && head.coord_weights.len() == 6 * DEPTH_COND_STRIDE
        {
            let mut norm_x = [0.0f64; DEPTH_COND_STRIDE];
            norm_x[0] = 1.0;
            for j in 0..CARRIER_TOROIDAL_FEATURES {
                norm_x[1 + j] = ((features[j] - head.means[j]) / head.scales[j].max(0.05))
                    .clamp(-16.0, 16.0);
            }
            norm_x[1 + CARRIER_TOROIDAL_FEATURES] = norm_d;
            norm_x[2 + CARRIER_TOROIDAL_FEATURES] = norm_r;
            for row in 0..6 {
                let row_slice =
                    &head.coord_weights[row * DEPTH_COND_STRIDE..(row + 1) * DEPTH_COND_STRIDE];
                delta[row] = row_slice
                    .iter()
                    .zip(&norm_x)
                    .map(|(w, x)| w * x)
                    .sum::<f64>()
                    .clamp(-8.0, 8.0);
            }
        } else {
            let mut norm_x = [0.0f64; CARRIER_STRIDE];
            norm_x[0] = 1.0;
            for j in 0..CARRIER_TOROIDAL_FEATURES {
                norm_x[1 + j] = ((features[j] - self.means[j]) / self.scales[j].max(0.05))
                    .clamp(-16.0, 16.0);
            }
            for row in 0..6 {
                let row_slice =
                    &self.coord_weights[row * CARRIER_STRIDE..(row + 1) * CARRIER_STRIDE];
                let base_d: f64 = row_slice.iter().zip(&norm_x).map(|(w, x)| w * x).sum();
                let cov_d = self.depth_cov_weights[row * 2] * norm_d
                    + self.depth_cov_weights[row * 2 + 1] * norm_r;
                delta[row] = (base_d + cov_d).clamp(-8.0, 8.0);
            }
        }

        let x1 = (f64::from(current_coord[1]) + delta[0]) as f32;
        let x2 = (f64::from(current_coord[2]) + delta[1]) as f32;
        let x3 = (f64::from(current_coord[3]) + delta[2]) as f32;
        let x0 = (1.0f64 + f64::from(x1).powi(2) + f64::from(x2).powi(2) + f64::from(x3).powi(2))
            .sqrt() as f32;
        let theta_f = wrap_angle_positive(f64::from(current_coord[4]) + delta[4]);
        let theta_s = wrap_angle_positive(f64::from(current_coord[5]) + delta[5]);
        [x0, x1, x2, x3, theta_f, theta_s]
    }

    /// Predict the next carrier's 6D `(H^3 x T^2)` coordinate from the current carrier coordinate
    /// and the 32D toroidal hidden-state feature vector `\psi_t`.
    pub fn predict_target_coord(
        &self,
        current_coord: &[f32; 6],
        features: &[f64; CARRIER_TOROIDAL_FEATURES],
    ) -> [f32; 6] {
        self.predict_target_coord_at_depth(current_coord, features, self.layers.max(1), 4)
    }

    /// Compute the depth-conditioned EOS stopping logit at `(constructed_depth, constructed_rank)`.
    pub fn eos_logit_at_depth(
        &self,
        features: &[f64; CARRIER_TOROIDAL_FEATURES],
        constructed_depth: usize,
        constructed_rank: usize,
    ) -> f64 {
        let (norm_d, norm_r) =
            Self::normalize_depth_rank_covariates(constructed_depth, constructed_rank);
        if let Some(head) = self.select_depth_readout(constructed_depth)
            && head.eos_weights.len() == DEPTH_COND_STRIDE
        {
            let mut sum = head.eos_weights[0];
            for j in 0..CARRIER_TOROIDAL_FEATURES {
                let z =
                    ((features[j] - head.means[j]) / head.scales[j].max(0.05)).clamp(-16.0, 16.0);
                sum += head.eos_weights[1 + j] * z;
            }
            sum += head.eos_weights[1 + CARRIER_TOROIDAL_FEATURES] * norm_d;
            sum += head.eos_weights[2 + CARRIER_TOROIDAL_FEATURES] * norm_r;
            return sum.clamp(-20.0, 20.0);
        }
        let mut sum = self.eos_weights[0];
        for j in 0..CARRIER_TOROIDAL_FEATURES {
            let z = ((features[j] - self.means[j]) / self.scales[j].max(0.05)).clamp(-16.0, 16.0);
            sum += self.eos_weights[1 + j] * z;
        }
        sum += self.eos_depth_cov_weights[0] * norm_d + self.eos_depth_cov_weights[1] * norm_r;
        sum.clamp(-20.0, 20.0)
    }

    /// Compute the EOS stopping logit from the current 32D toroidal hidden-state feature vector.
    pub fn eos_logit(&self, features: &[f64; CARRIER_TOROIDAL_FEATURES]) -> f64 {
        self.eos_logit_at_depth(features, self.layers.max(1), 4)
    }

    /// Train the next-carrier transition operator over all CLEAN continuation pairs in `swarm_map`
    /// (filtering out teacher-crystallized WAL / dream-cycle trajectories) by stepping
    /// `ToroidalIsotropicRuntime` across the trajectory chains, and report per-step `z-surprise`
    /// (`predict_gate.rs`) alongside branching NLL.
    pub fn fit_from_swarm_map(
        swarm_map: &atoma_immune::CarvedSwarmMap,
        catalog: &atoma_immune::MlirAtomCatalog,
        layers: usize,
        epochs: usize,
    ) -> Result<(Self, CarrierTransitionTrainingReceipt), String> {
        let started = Instant::now();
        let layers = layers.clamp(1, 60);
        let epochs = epochs.clamp(1, 64);
        let dataset = extract_vault_continuation_dataset(swarm_map, catalog)?;

        // Cache 6D carrier coordinates and NativeMlirEmbeddings for all referenced carriers.
        let mut coord_cache = std::collections::HashMap::<u32, [f32; 6]>::with_capacity(16_384);
        let mut emb_cache =
            std::collections::HashMap::<u32, NativeMlirEmbedding>::with_capacity(16_384);
        for chain in &dataset.clean_chains {
            for pair in chain {
                for op in [pair.from_carrier, pair.to_carrier] {
                    if let std::collections::hash_map::Entry::Vacant(e) = coord_cache.entry(op.0) {
                        let emb = embed_carrier(catalog, op)?;
                        e.insert(emb.coordinates);
                        emb_cache.insert(op.0, emb);
                    }
                }
            }
        }

        // Step the toroidal recurrence and ProductPredictor across every clean chain
        struct StepSample {
            feat: [f64; CARRIER_TOROIDAL_FEATURES],
            target_delta: [f64; 6],
            from_atom: u64,
            to_carrier: MlirOpId,
            target_coord: [f32; 6],
            is_terminal: bool,
            predictor_before: ProductPredictor,
        }

        let mut samples = Vec::<StepSample>::with_capacity(dataset.clean_continuation_arcs);
        let mut rt = ToroidalIsotropicRuntime::full_stack(6, layers)?;

        for chain in &dataset.clean_chains {
            for field in &mut rt.layer_fields {
                *field = ToroidalHarmonicFieldState::new(6);
            }
            let mut prod_pred = ProductPredictor::new(3, 2, None, None);
            if let Some(first_pair) = chain.first()
                && let Some(first_emb) = emb_cache.get(&first_pair.from_carrier.0)
                && let Some((ball, phases)) = embedding_to_ball_phases(first_emb)
            {
                let _ = prod_pred.step(&ball, &phases);
            }
            for pair in chain {
                let from_coord = coord_cache[&pair.from_carrier.0];
                let to_coord = coord_cache[&pair.to_carrier.0];
                let h_vec = rt.step_toroidal_attention(&from_coord)?;
                let h_arr: [f32; 6] = h_vec
                    .as_slice()
                    .try_into()
                    .map_err(|_| "invalid 6D toroidal output")?;
                let feat = extract_toroidal_carrier_features(&rt, &h_arr, &from_coord)?;
                let target_delta = [
                    f64::from(to_coord[1]) - f64::from(from_coord[1]),
                    f64::from(to_coord[2]) - f64::from(from_coord[2]),
                    f64::from(to_coord[3]) - f64::from(from_coord[3]),
                    f64::from(to_coord[0]).max(1.0).ln() - f64::from(from_coord[0]).max(1.0).ln(),
                    wrap_angle_diff(f64::from(to_coord[4]), f64::from(from_coord[4])),
                    wrap_angle_diff(f64::from(to_coord[5]), f64::from(from_coord[5])),
                ];
                let predictor_before = prod_pred.clone();
                if let Some(to_emb) = emb_cache.get(&pair.to_carrier.0)
                    && let Some((ball, phases)) = embedding_to_ball_phases(to_emb)
                {
                    let _ = prod_pred.step(&ball, &phases);
                }
                samples.push(StepSample {
                    feat,
                    target_delta,
                    from_atom: pair.from_atom,
                    to_carrier: pair.to_carrier,
                    target_coord: to_coord,
                    is_terminal: pair.is_chain_terminal,
                    predictor_before,
                });
            }
        }

        if samples.is_empty() {
            return Err("no clean carrier continuation samples extracted from vault".into());
        }

        // Compute feature means and standard-deviation scales.
        let n = samples.len() as f64;
        let mut means = [0.0f64; CARRIER_TOROIDAL_FEATURES];
        let mut scales = [0.0f64; CARRIER_TOROIDAL_FEATURES];
        for s in &samples {
            for (m, &v) in means.iter_mut().zip(&s.feat) {
                *m += v / n;
            }
        }
        for s in &samples {
            for j in 0..CARRIER_TOROIDAL_FEATURES {
                let d = s.feat[j] - means[j];
                scales[j] += (d * d) / n;
            }
        }
        for sc in &mut scales {
            *sc = sc.sqrt().clamp(0.05, 1e6);
        }

        let mut model = Self {
            layers,
            means,
            scales,
            ..Self::default()
        };

        let eval_coord_mse = |m: &Self| -> f64 {
            let mut total_sq = 0.0f64;
            for s in &samples {
                let from_c = [
                    s.feat[6] as f32,
                    s.feat[7] as f32,
                    s.feat[8] as f32,
                    s.feat[9] as f32,
                    s.feat[10] as f32,
                    s.feat[11] as f32,
                ];
                let pred = m.predict_target_coord(&from_c, &s.feat);
                let dx1 = f64::from(pred[1]) - f64::from(s.target_coord[1]);
                let dx2 = f64::from(pred[2]) - f64::from(s.target_coord[2]);
                let dx3 = f64::from(pred[3]) - f64::from(s.target_coord[3]);
                let dtf = wrap_angle_diff(f64::from(pred[4]), f64::from(s.target_coord[4]));
                let dts = wrap_angle_diff(f64::from(pred[5]), f64::from(s.target_coord[5]));
                total_sq += (dx1 * dx1 + dx2 * dx2 + dx3 * dx3 + dtf * dtf + dts * dts) / 5.0;
            }
            total_sq / (samples.len() as f64)
        };

        let initial_coord_mse = eval_coord_mse(&model);

        // Optimize `coord_weights` and `eos_weights` with deterministic mini-batch gradient descent
        let batch_size = 256usize;
        let lr_base = 0.04f64;
        let l2 = 1e-4f64;
        for epoch in 0..epochs {
            let lr = lr_base / (1.0 + 0.15 * epoch as f64);
            for chunk in samples.chunks(batch_size) {
                let b_len = chunk.len() as f64;
                let mut grad_coord = vec![0.0f64; 6 * CARRIER_STRIDE];
                let mut grad_eos = vec![0.0f64; CARRIER_STRIDE];
                for s in chunk {
                    let mut norm_x = [0.0f64; CARRIER_STRIDE];
                    norm_x[0] = 1.0;
                    for j in 0..CARRIER_TOROIDAL_FEATURES {
                        norm_x[1 + j] = ((s.feat[j] - model.means[j]) / model.scales[j])
                            .clamp(-16.0, 16.0);
                    }
                    for row in 0..6 {
                        let offset = row * CARRIER_STRIDE;
                        let pred_d: f64 = model.coord_weights[offset..offset + CARRIER_STRIDE]
                            .iter()
                            .zip(&norm_x)
                            .map(|(w, x)| w * x)
                            .sum();
                        let err = (pred_d - s.target_delta[row]) / b_len;
                        for c in 0..CARRIER_STRIDE {
                            grad_coord[offset + c] += err * norm_x[c];
                        }
                    }
                    let logit = model.eos_logit(&s.feat);
                    let prob = 1.0 / (1.0 + (-logit).exp());
                    let target_eos = if s.is_terminal { 1.0 } else { 0.0 };
                    let err_eos = (prob - target_eos) / b_len;
                    for c in 0..CARRIER_STRIDE {
                        grad_eos[c] += err_eos * norm_x[c];
                    }
                }
                for (i, w) in model.coord_weights.iter_mut().enumerate() {
                    let reg = if i % CARRIER_STRIDE == 0 { 0.0 } else { l2 * *w };
                    *w = (*w - lr * (grad_coord[i] + reg)).clamp(-64.0, 64.0);
                }
                for (i, w) in model.eos_weights.iter_mut().enumerate() {
                    let reg = if i == 0 { 0.0 } else { l2 * *w };
                    *w = (*w - lr * (grad_eos[i] + reg)).clamp(-64.0, 64.0);
                }
            }
        }

        let final_coord_mse = eval_coord_mse(&model);

        // Build and optimize per-transition toroidal hidden-state prototypes across all clean source atoms
        const PROTO_IDX: [usize; 16] = [
            0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 22, 24, 27,
        ];
        let extract_z16 = |m: &Self, feat: &[f64; CARRIER_TOROIDAL_FEATURES]| -> [f32; 16] {
            let mut z = [0.0f32; 16];
            for (k, &fi) in PROTO_IDX.iter().enumerate() {
                z[k] = (((feat[fi] - m.means[fi]) / m.scales[fi].max(0.05)).clamp(-8.0, 8.0))
                    as f32;
            }
            z
        };

        let mut atom_totals = std::collections::HashMap::<u64, u32>::new();
        let mut pair_counts = std::collections::HashMap::<(u64, u32), (u32, [f64; 16])>::new();
        for s in &samples {
            *atom_totals.entry(s.from_atom).or_insert(0) += 1;
            let z = extract_z16(&model, &s.feat);
            let entry = pair_counts
                .entry((s.from_atom, s.to_carrier.0))
                .or_insert((0, [0.0; 16]));
            entry.0 += 1;
            for k in 0..16 {
                entry.1[k] += f64::from(z[k]);
            }
        }
        for ((from_a, to_op), (cnt, sum_z)) in pair_counts {
            let tot = atom_totals.get(&from_a).copied().unwrap_or(cnt).max(1);
            let log_prior = ((cnt as f64) / (tot as f64)).ln() as f32;
            let mut weights = [0.0f32; 16];
            for k in 0..16 {
                weights[k] = (sum_z[k] / (cnt as f64) * 0.25) as f32;
            }
            model.transition_prototypes.insert(
                (from_a, to_op),
                ToroidalTransitionPrototype {
                    count: cnt,
                    log_prior,
                    weights,
                },
            );
        }

        // Evaluate and optimize branching disambiguation + per-step z-surprise over multi-branch source atoms
        let mut unique_cands_by_atom =
            std::collections::HashMap::<u64, Vec<(MlirOpId, [f32; 6])>>::new();
        for chain in &dataset.clean_chains {
            for pair in chain {
                if let Some(&c) = coord_cache.get(&pair.to_carrier.0) {
                    let entry = unique_cands_by_atom.entry(pair.from_atom).or_default();
                    if !entry.iter().any(|(op, _)| *op == pair.to_carrier) {
                        entry.push((pair.to_carrier, c));
                    }
                }
            }
        }

        let score_branching = |m: &Self, use_trained: bool| -> (f64, f64, f64, f64) {
            let mut total_nll = 0.0f64;
            let mut correct = 0usize;
            let mut count = 0usize;
            let mut total_z_surprise = 0.0f64;
            let mut max_z_surprise = 0.0f64;
            for s in samples.iter().step_by(4) {
                let Some(cands) = unique_cands_by_atom.get(&s.from_atom) else {
                    continue;
                };
                if cands.len() <= 1 || cands.len() > 64 {
                    continue;
                }
                let from_c = [
                    s.feat[6] as f32,
                    s.feat[7] as f32,
                    s.feat[8] as f32,
                    s.feat[9] as f32,
                    s.feat[10] as f32,
                    s.feat[11] as f32,
                ];
                let pred = if use_trained {
                    m.predict_target_coord(&from_c, &s.feat)
                } else {
                    from_c
                };
                let z = extract_z16(m, &s.feat);
                let mut logits = Vec::with_capacity(cands.len());
                let mut target_idx = 0usize;
                for (idx, (op, cc)) in cands.iter().enumerate() {
                    if *op == s.to_carrier {
                        target_idx = idx;
                    }
                    let dist_sq = (f64::from(pred[1]) - f64::from(cc[1])).powi(2)
                        + (f64::from(pred[2]) - f64::from(cc[2])).powi(2)
                        + (f64::from(pred[3]) - f64::from(cc[3])).powi(2)
                        + wrap_angle_diff(f64::from(pred[4]), f64::from(cc[4])).powi(2)
                        + wrap_angle_diff(f64::from(pred[5]), f64::from(cc[5])).powi(2);
                    let mut l = -m.branch_weights[0] * dist_sq.sqrt();
                    if use_trained
                        && let Some(proto) = m.transition_prototypes.get(&(s.from_atom, op.0))
                    {
                        let dot: f32 = z
                            .iter()
                            .zip(&proto.weights)
                            .map(|(a, b)| a * b)
                            .sum();
                        l += f64::from(proto.log_prior) + f64::from(dot);
                    }
                    logits.push(l);
                }
                let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
                let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
                let best_idx = argmax(&exps);
                if best_idx == target_idx {
                    correct += 1;
                }
                let best_op = cands[best_idx].0;
                if let Some(best_emb) = emb_cache.get(&best_op.0)
                    && let Some((ball, phases)) = embedding_to_ball_phases(best_emb)
                {
                    let mut p = s.predictor_before.clone();
                    let step = p.step(&ball, &phases);
                    if let Some(zs) = step.surprise
                        && zs.is_finite()
                    {
                        total_z_surprise += zs;
                        if zs > max_z_surprise {
                            max_z_surprise = zs;
                        }
                    }
                }
                total_nll += -(exps[target_idx] / sum).max(1e-12).ln();
                count += 1;
            }
            if count == 0 {
                (0.0, 1.0, 0.0, 0.0)
            } else {
                (
                    total_nll / count as f64,
                    correct as f64 / count as f64,
                    total_z_surprise / count as f64,
                    max_z_surprise,
                )
            }
        };

        let (initial_branch_nll_nats, initial_branch_top1_acc, initial_mean_z_surprise, _) =
            score_branching(&model, false);

        // Contrastive cross-entropy refinement of `transition_prototypes` on branching steps
        for epoch in 0..epochs {
            let lr_proto = (0.18 / (1.0 + 0.25 * epoch as f64)) as f32;
            for s in &samples {
                let Some(cands) = unique_cands_by_atom.get(&s.from_atom) else {
                    continue;
                };
                if cands.len() <= 1 || cands.len() > 64 {
                    continue;
                }
                let z = extract_z16(&model, &s.feat);
                let mut logits = Vec::with_capacity(cands.len());
                let mut target_idx = 0usize;
                for (idx, (op, _cc)) in cands.iter().enumerate() {
                    if *op == s.to_carrier {
                        target_idx = idx;
                    }
                    let mut l = 0.0f64;
                    if let Some(proto) = model.transition_prototypes.get(&(s.from_atom, op.0)) {
                        let dot: f32 = z.iter().zip(&proto.weights).map(|(a, b)| a * b).sum();
                        l += f64::from(proto.log_prior) + f64::from(dot);
                    }
                    logits.push(l);
                }
                let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
                let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
                for (idx, (op, _)) in cands.iter().enumerate() {
                    let prob = (exps[idx] / sum) as f32;
                    let target = if idx == target_idx { 1.0f32 } else { 0.0f32 };
                    let err = target - prob;
                    if let Some(proto) = model.transition_prototypes.get_mut(&(s.from_atom, op.0)) {
                        for k in 0..16 {
                            proto.weights[k] =
                                (proto.weights[k] + lr_proto * err * z[k]).clamp(-16.0, 16.0);
                        }
                    }
                }
            }
        }

        let (
            final_branch_nll_nats,
            final_branch_top1_acc,
            final_mean_z_surprise,
            final_max_z_surprise,
        ) = score_branching(&model, true);

        // Depth-conditional training (`docs/BREATHING_DEPTH_DESIGN.md` §5):
        // Train per-depth readout heads across the 4 depth regimes (`1..=4`, `5..=12`, `13..=28`, `29..=52`)
        // with explicit `(norm_depth, norm_rank)` input covariates so changing constructed depth
        // never shifts the predictor's feature distribution.
        let chain_stride = (dataset.clean_chains.len() / 256).max(1);
        let mut per_depth_readouts = Vec::with_capacity(DEPTH_BAND_SPECS.len());
        for (band_idx, &(min_d, max_d, ref_d)) in DEPTH_BAND_SPECS.iter().enumerate() {
            let band_rank = ((band_idx + 1) * 2).clamp(1, 8);
            let (norm_d, norm_r) = Self::normalize_depth_rank_covariates(ref_d, band_rank);
            let mut band_rt = ToroidalIsotropicRuntime::full_stack(6, ref_d)?;
            struct BandSample {
                feat: [f64; CARRIER_TOROIDAL_FEATURES],
                target_delta: [f64; 6],
                target_coord: [f32; 6],
                is_terminal: bool,
            }
            let mut band_samples = Vec::new();
            for chain in dataset.clean_chains.iter().step_by(chain_stride) {
                for field in &mut band_rt.layer_fields {
                    *field = ToroidalHarmonicFieldState::new(6);
                }
                for pair in chain {
                    let from_coord = coord_cache[&pair.from_carrier.0];
                    let to_coord = coord_cache[&pair.to_carrier.0];
                    let h_vec = band_rt.step_toroidal_attention(&from_coord)?;
                    let h_arr: [f32; 6] = h_vec
                        .as_slice()
                        .try_into()
                        .map_err(|_| "invalid 6D toroidal output")?;
                    let feat = extract_toroidal_carrier_features(&band_rt, &h_arr, &from_coord)?;
                    let target_delta = [
                        f64::from(to_coord[1]) - f64::from(from_coord[1]),
                        f64::from(to_coord[2]) - f64::from(from_coord[2]),
                        f64::from(to_coord[3]) - f64::from(from_coord[3]),
                        f64::from(to_coord[0]).max(1.0).ln()
                            - f64::from(from_coord[0]).max(1.0).ln(),
                        wrap_angle_diff(f64::from(to_coord[4]), f64::from(from_coord[4])),
                        wrap_angle_diff(f64::from(to_coord[5]), f64::from(from_coord[5])),
                    ];
                    band_samples.push(BandSample {
                        feat,
                        target_delta,
                        target_coord: to_coord,
                        is_terminal: pair.is_chain_terminal,
                    });
                }
            }
            if band_samples.is_empty() {
                continue;
            }
            let bn = band_samples.len() as f64;
            let mut b_means = [0.0f64; CARRIER_TOROIDAL_FEATURES];
            let mut b_scales = [0.0f64; CARRIER_TOROIDAL_FEATURES];
            for s in &band_samples {
                for (m, &v) in b_means.iter_mut().zip(&s.feat) {
                    *m += v / bn;
                }
            }
            for s in &band_samples {
                for j in 0..CARRIER_TOROIDAL_FEATURES {
                    let d = s.feat[j] - b_means[j];
                    b_scales[j] += (d * d) / bn;
                }
            }
            for sc in &mut b_scales {
                *sc = sc.sqrt().clamp(0.05, 1e6);
            }

            let mut head = DepthConditionalReadoutHead {
                min_depth: min_d,
                max_depth: max_d,
                reference_depth: ref_d,
                means: b_means,
                scales: b_scales,
                coord_weights: vec![0.0f64; 6 * DEPTH_COND_STRIDE],
                eos_weights: {
                    let mut w = vec![0.0f64; DEPTH_COND_STRIDE];
                    w[0] = -3.2;
                    w
                },
                branch_weights: model.branch_weights,
                initial_coord_mse: 0.0,
                final_coord_mse: 0.0,
            };

            let eval_band_mse = |h: &DepthConditionalReadoutHead| -> f64 {
                let mut sq = 0.0f64;
                for s in &band_samples {
                    let mut norm_x = [0.0f64; DEPTH_COND_STRIDE];
                    norm_x[0] = 1.0;
                    for j in 0..CARRIER_TOROIDAL_FEATURES {
                        norm_x[1 + j] =
                            ((s.feat[j] - h.means[j]) / h.scales[j]).clamp(-16.0, 16.0);
                    }
                    norm_x[1 + CARRIER_TOROIDAL_FEATURES] = norm_d;
                    norm_x[2 + CARRIER_TOROIDAL_FEATURES] = norm_r;
                    let mut d_pred = [0.0f64; 6];
                    for row in 0..6 {
                        let off = row * DEPTH_COND_STRIDE;
                        d_pred[row] = h.coord_weights[off..off + DEPTH_COND_STRIDE]
                            .iter()
                            .zip(&norm_x)
                            .map(|(w, x)| w * x)
                            .sum::<f64>()
                            .clamp(-8.0, 8.0);
                    }
                    let px1 = s.feat[7] + d_pred[0];
                    let px2 = s.feat[8] + d_pred[1];
                    let px3 = s.feat[9] + d_pred[2];
                    let ptf = wrap_angle_positive(s.feat[10] + d_pred[4]) as f64;
                    let pts = wrap_angle_positive(s.feat[11] + d_pred[5]) as f64;
                    let dx1 = px1 - f64::from(s.target_coord[1]);
                    let dx2 = px2 - f64::from(s.target_coord[2]);
                    let dx3 = px3 - f64::from(s.target_coord[3]);
                    let dtf = wrap_angle_diff(ptf, f64::from(s.target_coord[4]));
                    let dts = wrap_angle_diff(pts, f64::from(s.target_coord[5]));
                    sq += (dx1 * dx1 + dx2 * dx2 + dx3 * dx3 + dtf * dtf + dts * dts) / 5.0;
                }
                sq / bn
            };

            head.initial_coord_mse = eval_band_mse(&head);
            for epoch in 0..epochs.min(4) {
                let lr = 0.04 / (1.0 + 0.15 * epoch as f64);
                for chunk in band_samples.chunks(256) {
                    let b_len = chunk.len() as f64;
                    let mut g_coord = vec![0.0f64; 6 * DEPTH_COND_STRIDE];
                    let mut g_eos = vec![0.0f64; DEPTH_COND_STRIDE];
                    for s in chunk {
                        let mut norm_x = [0.0f64; DEPTH_COND_STRIDE];
                        norm_x[0] = 1.0;
                        for j in 0..CARRIER_TOROIDAL_FEATURES {
                            norm_x[1 + j] =
                                ((s.feat[j] - head.means[j]) / head.scales[j]).clamp(-16.0, 16.0);
                        }
                        norm_x[1 + CARRIER_TOROIDAL_FEATURES] = norm_d;
                        norm_x[2 + CARRIER_TOROIDAL_FEATURES] = norm_r;
                        for row in 0..6 {
                            let off = row * DEPTH_COND_STRIDE;
                            let pred_d: f64 = head.coord_weights[off..off + DEPTH_COND_STRIDE]
                                .iter()
                                .zip(&norm_x)
                                .map(|(w, x)| w * x)
                                .sum();
                            let err = (pred_d - s.target_delta[row]) / b_len;
                            for c in 0..DEPTH_COND_STRIDE {
                                g_coord[off + c] += err * norm_x[c];
                            }
                        }
                        let mut e_logit = head.eos_weights[0];
                        for c in 1..DEPTH_COND_STRIDE {
                            e_logit += head.eos_weights[c] * norm_x[c];
                        }
                        let prob = 1.0 / (1.0 + (-e_logit.clamp(-20.0, 20.0)).exp());
                        let target_eos = if s.is_terminal { 1.0 } else { 0.0 };
                        let err_eos = (prob - target_eos) / b_len;
                        for c in 0..DEPTH_COND_STRIDE {
                            g_eos[c] += err_eos * norm_x[c];
                        }
                    }
                    for (i, w) in head.coord_weights.iter_mut().enumerate() {
                        let reg = if i % DEPTH_COND_STRIDE == 0 {
                            0.0
                        } else {
                            l2 * *w
                        };
                        *w = (*w - lr * (g_coord[i] + reg)).clamp(-64.0, 64.0);
                    }
                    for (i, w) in head.eos_weights.iter_mut().enumerate() {
                        let reg = if i == 0 { 0.0 } else { l2 * *w };
                        *w = (*w - lr * (g_eos[i] + reg)).clamp(-64.0, 64.0);
                    }
                }
            }
            head.final_coord_mse = eval_band_mse(&head);
            for row in 0..6 {
                let off = row * DEPTH_COND_STRIDE;
                model.depth_cov_weights[row * 2] +=
                    0.25 * head.coord_weights[off + 1 + CARRIER_TOROIDAL_FEATURES];
                model.depth_cov_weights[row * 2 + 1] +=
                    0.25 * head.coord_weights[off + 2 + CARRIER_TOROIDAL_FEATURES];
            }
            model.eos_depth_cov_weights[0] +=
                0.25 * head.eos_weights[1 + CARRIER_TOROIDAL_FEATURES];
            model.eos_depth_cov_weights[1] +=
                0.25 * head.eos_weights[2 + CARRIER_TOROIDAL_FEATURES];
            per_depth_readouts.push(head);
        }
        model.per_depth_readouts = per_depth_readouts;

        Ok((
            model,
            CarrierTransitionTrainingReceipt {
                total_vault_arcs: dataset.total_vault_arcs,
                carrier_continuation_arcs: dataset.carrier_continuation_arcs,
                clean_continuation_arcs: dataset.clean_continuation_arcs,
                teacher_contaminated_arcs_filtered: dataset.teacher_contaminated_arcs,
                wal_contaminated_frames_filtered: dataset.wal_contaminated_frames,
                structural_anchor_arcs: dataset.structural_anchor_arcs,
                contiguous_step_pairs: dataset.contiguous_step_pairs,
                trajectory_chains: dataset.trajectory_chains,
                branching_source_atoms: dataset.branching_source_atoms,
                branching_arcs: dataset.branching_arcs,
                unique_target_carriers: dataset.unique_target_carriers,
                initial_coord_mse,
                final_coord_mse,
                initial_branch_nll_nats,
                final_branch_nll_nats,
                initial_branch_top1_acc,
                final_branch_top1_acc,
                initial_mean_z_surprise,
                final_mean_z_surprise,
                final_max_z_surprise,
                elapsed_ms: started.elapsed().as_millis(),
            },
        ))
    }
}

/// Primary turn-path entry point called by `execute_live_operational_turn` BEFORE the gate.
///
/// Ownership contract (`docs/NATIVE_CARRIER_TRANSITION_CONTRACT.md`):
/// - Predictor (`native_predictor.rs`) owns candidate generation (hands-off).
/// - Single Gate (`TrajectoryMatchGate` in `predict_gate.rs`, invoked by `runtime_model.rs`)
///   runs AFTER predictor in `execute_live_operational_turn` and owns accept/reject over the
///   returned [`NativeContinuationCandidate`].
pub fn generate_toroidal_carrier_continuation(
    toroidal_runtime: &ToroidalIsotropicRuntime,
    swarm_map: &atoma_immune::CarvedSwarmMap,
    catalog: &atoma_immune::MlirAtomCatalog,
    transducer: &BoundaryLexicalTransducer,
    prompt: &str,
    prompt_carriers: Option<&[MlirOpId]>,
    max_new_carriers: usize,
) -> Result<NativeContinuationCandidate, String> {
    let model = NativeCarrierTransitionModel::default();
    generate_toroidal_carrier_continuation_with_model(
        &model,
        toroidal_runtime,
        swarm_map,
        catalog,
        transducer,
        prompt,
        prompt_carriers,
        max_new_carriers,
    )
}

/// Generate a continuation candidate driven from a cloned `ToroidalIsotropicRuntime` using an
/// explicit [`NativeCarrierTransitionModel`].
#[allow(clippy::too_many_arguments)]
pub fn generate_toroidal_carrier_continuation_with_model(
    model: &NativeCarrierTransitionModel,
    toroidal_runtime: &ToroidalIsotropicRuntime,
    swarm_map: &atoma_immune::CarvedSwarmMap,
    catalog: &atoma_immune::MlirAtomCatalog,
    transducer: &BoundaryLexicalTransducer,
    prompt: &str,
    prompt_carriers: Option<&[MlirOpId]>,
    max_new_carriers: usize,
) -> Result<NativeContinuationCandidate, String> {
    if prompt.is_empty() {
        return Err("cannot generate toroidal carrier continuation for an empty prompt".into());
    }
    if !(1..=MAX_NEW_BYTES).contains(&max_new_carriers) {
        return Err(format!("max_new_carriers must be in 1..={MAX_NEW_BYTES}"));
    }

    // 1. Clone the session's toroidal runtime so a rejected rollout never mutates session state.
    let mut rt = if toroidal_runtime.hidden_dim == 6 && !toroidal_runtime.layer_fields.is_empty() {
        clone_toroidal_runtime(toroidal_runtime)
    } else {
        ToroidalIsotropicRuntime::full_stack(6, model.layers.max(1))?
    };

    // 2. Resolve prompt carrier prefix and embed each prompt carrier for `TrajectoryMatchGate`.
    let mut prefix: Vec<MlirOpId> = match prompt_carriers {
        Some(slice) if !slice.is_empty() => slice.to_vec(),
        _ => transducer.transduce_ingress_text(prompt, true),
    };
    if prefix.is_empty() {
        prefix = transducer.transduce_utf8_to_mlir(prompt, false);
    }
    if prefix.is_empty() {
        return Err("prompt produced no ML-IR carriers".into());
    }
    let mut prompt_embeddings = Vec::with_capacity(prefix.len());
    for &op in &prefix {
        prompt_embeddings.push(embed_carrier(catalog, op)?);
    }

    // 3. Ensure `rt` has stepped through the prompt carrier sequence and capture the current
    //    6D toroidal hidden state `h_t` plus breathing construction diagnostics.
    let (_pami_idx, pami_spectral_spread, sector_mask, raw_tension, _) =
        crate::toroidal_attention::analyze_carrier_trajectory_complexity(&prompt_embeddings);
    let active_attention_sectors = (sector_mask.count_ones() as usize).max(1);
    let pami_spread_val = pami_spectral_spread.unwrap_or(raw_tension);

    let mut current_carrier = *prefix.last().unwrap();
    let mut current_coord = prompt_embeddings.last().unwrap().coordinates;
    let mut h_curr: [f32; 6] = if rt.layer_fields.first().is_some_and(|f| f.tokens_ingested > 0) {
        let out = rt.step_toroidal_attention(&current_coord)?;
        out.as_slice()
            .try_into()
            .map_err(|_| "invalid 6D toroidal state")?
    } else {
        let mut last_h = current_coord;
        for emb in &prompt_embeddings {
            let out = rt.step_toroidal_attention(&emb.coordinates)?;
            last_h = out
                .as_slice()
                .try_into()
                .map_err(|_| "invalid 6D toroidal state")?;
        }
        last_h
    };

    let constructed_depth = rt.constructed_depth().max(1);
    let constructed_rank = active_attention_sectors.clamp(1, 8);
    let marginal_free_energy_gain = {
        let depth_f = constructed_depth.saturating_sub(1) as f64;
        depth_f * 0.012 + pami_spread_val * 0.05
    };

    // Seed a hands-off `ProductPredictor` (`predict_gate.rs`) from the prompt prefix to record
    // per-step `z-surprise` alongside log-probability without interfering with generation.
    let mut step_surprise_tracker = ProductPredictor::new(3, 2, None, None);
    for emb in &prompt_embeddings {
        if let Some((ball, phases)) = embedding_to_ball_phases(emb) {
            let _ = step_surprise_tracker.step(&ball, &phases);
        }
    }

    // Exclude teacher-contaminated arcs when clean pre-teacher arcs exist in the vault.
    let has_clean_base_arcs = swarm_map.reasoning_arcs.iter().any(|arc| {
        arc.compiled_mlir_op.0 < catalog.base_vocab_size
            && !is_teacher_contaminated_origin(&arc.origin_model_fingerprint)
    });

    let mut emitted_carriers = Vec::with_capacity(max_new_carriers.min(128));
    let mut emitted_embeddings = Vec::with_capacity(max_new_carriers.min(128));
    let mut steps = Vec::with_capacity(max_new_carriers.min(128));
    let mut stop = NativeGenerationStop::ByteBudget;
    let mut log_prob_sum = 0.0f64;
    let mut min_margin = 1.0f64;
    let mut vault_backed_steps = 0usize;

    let neighborhood_slack = atoma_immune::geometry::fixed::Q32_31::from_f64(0.35);
    let active_branch_weights = model.branch_weights_at_depth(constructed_depth);

    for step_index in 0..max_new_carriers {
        let feat = extract_toroidal_carrier_features(&rt, &h_curr, &current_coord)?;
        let predicted_target_coord = model.predict_target_coord_at_depth(
            &current_coord,
            &feat,
            constructed_depth,
            constructed_rank,
        );
        let expected_chain = prefix_fnv1a(&prefix);
        let (query_lor, prefix_frame) = atoma_immune::NativeReader::context_frame(catalog, &prefix)
            .unwrap_or_else(|_| {
                (
                    atoma_immune::geometry::lorentz::LorentzVector::origin(),
                    atoma_immune::IDENTITY_FRAME_Q32,
                )
            });
        let (_, query_phase) = atoma_immune::NativeReader::address(catalog, current_carrier)
            .unwrap_or((
                query_lor,
                atoma_immune::cache::geometric::Phase {
                    theta_f: atoma_immune::geometry::fixed::Q32_31::ZERO,
                    theta_s: atoma_immune::geometry::fixed::Q32_31::ZERO,
                },
            ));

        // Gather candidate transitions from the vault's outgoing 1-cell arcs in the current basin
        // and score each candidate using the toroidal hidden state + predicted target coordinate.
        let mut scored_candidates: Vec<(MlirOpId, f64, bool)> = Vec::new();

        if !swarm_map.reasoning_arcs.is_empty()
            && let Some((_best_atom, _best_cosh, near_atoms)) =
                swarm_map.candidate_arc_atoms_near(&query_lor, query_phase, neighborhood_slack.0)
        {
            let mut best_by_carrier = std::collections::BTreeMap::<u32, (f64, bool)>::new();
            for (aid, atom_cosh_i64) in near_atoms.into_iter().take(24) {
                let atom_cosh = atoma_immune::geometry::fixed::Q32_31(atom_cosh_i64)
                    .to_f64()
                    .max(1.0);
                for &arc_idx in swarm_map.outgoing_arc_indices(atoma_immune::cache::geometric::AtomId(aid)) {
                    let arc = &swarm_map.reasoning_arcs[arc_idx];
                    if has_clean_base_arcs
                        && is_teacher_contaminated_origin(&arc.origin_model_fingerprint)
                    {
                        continue;
                    }
                    let cand_op = arc.compiled_mlir_op;
                    if cand_op.0 >= catalog.base_vocab_size {
                        continue;
                    }
                    // Skip immediate self-loops and 2-step oscillations
                    if cand_op == current_carrier {
                        continue;
                    }
                    if emitted_carriers.len() >= 2
                        && emitted_carriers[emitted_carriers.len() - 2] == cand_op
                        && emitted_carriers[emitted_carriers.len() - 1] == current_carrier
                    {
                        continue;
                    }

                    let exact_chain =
                        arc.chain_id == expected_chain && arc.wigner_frame == prefix_frame;
                    let Ok(cand_emb) = embed_carrier(catalog, cand_op) else {
                        continue;
                    };
                    let cc = cand_emb.coordinates;
                    let coord_dist = ((f64::from(predicted_target_coord[1]) - f64::from(cc[1]))
                        .powi(2)
                        + (f64::from(predicted_target_coord[2]) - f64::from(cc[2])).powi(2)
                        + (f64::from(predicted_target_coord[3]) - f64::from(cc[3])).powi(2)
                        + wrap_angle_diff(f64::from(predicted_target_coord[4]), f64::from(cc[4]))
                            .powi(2)
                        + wrap_angle_diff(f64::from(predicted_target_coord[5]), f64::from(cc[5]))
                            .powi(2))
                    .sqrt();

                    // Toroidal phase coherence between current toroidal hidden state and candidate
                    let torus_phase_coh = wrap_angle_diff(f64::from(h_curr[4]), f64::from(cc[4]))
                        .cos()
                        + wrap_angle_diff(f64::from(h_curr[5]), f64::from(cc[5])).cos();

                    // Wigner frame alignment
                    let mut frame_dot = 0.0f64;
                    let mut norm_a = 0.0f64;
                    let mut norm_b = 0.0f64;
                    for r in 0..3 {
                        for c in 0..3 {
                            let va = arc.wigner_frame[r][c].to_f64();
                            let vb = prefix_frame[r][c].to_f64();
                            frame_dot += va * vb;
                            norm_a += va * va;
                            norm_b += vb * vb;
                        }
                    }
                    let frame_align = frame_dot / (norm_a.sqrt() * norm_b.sqrt()).max(1e-9);
                    let action_cost = (arc.cosh_action_q32.to_f64().max(1.0) - 1.0)
                        + 0.5 * (atom_cosh - 1.0);
                    let cert = arc.teacher_certainty_q32.to_f64().clamp(0.0, 1.0);

                    let mut score = if exact_chain { 14.0 } else { 0.0 };
                    score += -active_branch_weights[0] * coord_dist
                        + 0.35 * active_branch_weights[1] * torus_phase_coh
                        + active_branch_weights[2] * frame_align
                        - active_branch_weights[3] * action_cost
                        + active_branch_weights[4] * cert;
                    if let Some(proto) = model.transition_prototypes.get(&(aid, cand_op.0)) {
                        const PROTO_IDX: [usize; 16] =
                            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 22, 24, 27];
                        let mut dot = 0.0f64;
                        for (k, &fi) in PROTO_IDX.iter().enumerate() {
                            let z = ((feat[fi] - model.means[fi]) / model.scales[fi].max(0.05))
                                .clamp(-8.0, 8.0);
                            dot += z * f64::from(proto.weights[k]);
                        }
                        score += f64::from(proto.log_prior) + dot;
                    }

                    // Penalize already-emitted carriers to prevent loops
                    let repeats = emitted_carriers.iter().filter(|&&op| op == cand_op).count();
                    if repeats > 0 && !exact_chain {
                        score -= 2.2 * (repeats as f64);
                    }

                    match best_by_carrier.get_mut(&cand_op.0) {
                        Some((best_s, is_v)) => {
                            if score > *best_s {
                                *best_s = score;
                                *is_v = true;
                            }
                        }
                        None => {
                            best_by_carrier.insert(cand_op.0, (score, true));
                        }
                    }
                }
            }
            for (op_u32, (s, is_v)) in best_by_carrier {
                scored_candidates.push((MlirOpId(op_u32), s, is_v));
            }
        }

        // If the vault has no outgoing arcs in this neighborhood, fall back to the nearest byte
        // carrier to `predicted_target_coord` on H^3 x T^2 driven purely by the toroidal transition.
        if scored_candidates.is_empty() {
            if step_index > 0 {
                stop = NativeGenerationStop::Eos;
                break;
            }
            let read_ops =
                crate::native_output::readout_byte_carriers(catalog, &[predicted_target_coord.to_vec()])?;
            if let Some(&fallback_op) = read_ops.first() {
                scored_candidates.push((fallback_op, 0.0, false));
            } else {
                stop = NativeGenerationStop::Eos;
                break;
            }
        }

        // Check EOS stopping condition after at least one emitted continuation carrier
        let eos_logit = model.eos_logit_at_depth(&feat, constructed_depth, constructed_rank);
        scored_candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.0.cmp(&b.0.0)));
        let top_score = scored_candidates[0].1;
        if step_index > 0 && (eos_logit > top_score + 4.5 || top_score < -6.0) {
            stop = NativeGenerationStop::Eos;
            break;
        }

        // Compute normalized probability distribution over top candidates + EOS
        let mut all_logits: Vec<f64> = scored_candidates.iter().map(|c| c.1).collect();
        if step_index > 0 {
            all_logits.push(eos_logit - 4.5);
        }
        let max_l = all_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = all_logits.iter().map(|l| (l - max_l).exp()).collect();
        let sum_exp: f64 = exps.iter().sum::<f64>().max(1e-12);
        let probs: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();

        let win_prob = probs[0].clamp(1e-12, 1.0);
        let second_prob = probs.get(1).copied().unwrap_or(0.0);
        let log_prob = win_prob.ln();
        let margin = (win_prob - second_prob).clamp(0.0, 1.0);
        let (predicted_carrier, _, matched_vault_arc) = scored_candidates[0];

        if transducer.eos_ids.contains(&predicted_carrier) || predicted_carrier.0 == NATIVE_EOS {
            stop = NativeGenerationStop::Eos;
            break;
        }

        let next_emb = embed_carrier(catalog, predicted_carrier)?;
        let (z_surprise, z_hyper, z_torus) =
            if let Some((ball, phases)) = embedding_to_ball_phases(&next_emb) {
                let surprise_step = step_surprise_tracker.step(&ball, &phases);
                (
                    surprise_step.surprise.unwrap_or(0.0),
                    surprise_step.z_hyper.unwrap_or(0.0),
                    surprise_step.z_torus.unwrap_or(0.0),
                )
            } else {
                (0.0, 0.0, 0.0)
            };

        steps.push(CarrierTransitionStep {
            step_index,
            from_carrier: current_carrier,
            predicted_carrier,
            log_prob,
            margin,
            toroidal_hidden: h_curr,
            predicted_target_coord,
            matched_vault_arc,
            z_surprise,
            z_hyper,
            z_torus,
        });

        log_prob_sum += log_prob;
        if margin < min_margin {
            min_margin = margin;
        }
        if matched_vault_arc {
            vault_backed_steps += 1;
        }

        emitted_carriers.push(predicted_carrier);
        emitted_embeddings.push(next_emb);
        prefix.push(predicted_carrier);
        current_carrier = predicted_carrier;
        current_coord = emitted_embeddings.last().unwrap().coordinates;
        let next_h = rt.step_toroidal_attention(&current_coord)?;
        h_curr = next_h
            .as_slice()
            .try_into()
            .map_err(|_| "invalid 6D toroidal state")?;
    }

    let n_steps = steps.len();
    let mean_log_prob = if n_steps > 0 {
        log_prob_sum / (n_steps as f64)
    } else {
        f64::NEG_INFINITY
    };
    let min_margin = if n_steps > 0 { min_margin } else { 0.0 };
    let vault_arc_coverage = if n_steps > 0 {
        (vault_backed_steps as f64) / (n_steps as f64)
    } else {
        0.0
    };

    let (mean_z_surprise, max_z_surprise) = if steps.is_empty() {
        (0.0, 0.0)
    } else {
        let sum_z: f64 = steps.iter().map(|s| s.z_surprise).sum();
        let max_z: f64 = steps
            .iter()
            .map(|s| s.z_surprise)
            .fold(0.0f64, f64::max);
        (sum_z / (steps.len() as f64), max_z)
    };
    // Ownership contract: no gate decisions inside the predictor. The candidate arrives with an
    // unscored `GateScore` (`GateVerdict::Cold`, `scored_steps: 0`) so `runtime_model.rs` scores
    // it through the session's `TrajectoryMatchGate` in the turn.
    let gate_score = unscored_gate_score();

    // Decode emitted carriers via `transducer`:
    // - If all emitted carriers are byte carriers (< 256), validate strict UTF-8 via `transduce_mlir_to_utf8`.
    // - Otherwise, decode via `transduce_egress_ops` and verify no replacement character (`\u{FFFD}`) or `<atoma.` mnemonic leak.
    let (decoded_text, decode_error) = if emitted_carriers.is_empty() {
        (None, Some("predictor emitted zero continuation carriers".to_string()))
    } else if emitted_carriers.iter().all(|op| op.0 < 256) {
        match transducer.transduce_mlir_to_utf8(&emitted_carriers) {
            Ok(txt) => (Some(txt), None),
            Err(err) => (None, Some(err)),
        }
    } else {
        let (_expanded, txt) = transducer.transduce_egress_ops(&emitted_carriers);
        if txt.contains('\u{FFFD}') {
            (
                None,
                Some("decoded carrier continuation contains invalid UTF-8 replacement character".into()),
            )
        } else if txt.trim().is_empty() {
            (None, Some("decoded carrier continuation is empty whitespace".into()))
        } else {
            (Some(txt), None)
        }
    };

    let poincare_radius = {
        let x1 = f64::from(h_curr[1]);
        let x2 = f64::from(h_curr[2]);
        let x3 = f64::from(h_curr[3]);
        let x0 = (1.0 + x1 * x1 + x2 * x2 + x3 * x3).sqrt();
        let r_base = ((x1 * x1 + x2 * x2 + x3 * x3).sqrt() / (1.0 + x0)).clamp(0.0, 0.999_999);
        // Modulate radial hierarchy by constructed depth so deeper trajectories sit closer to the
        // Poincaré boundary when warranted by free-energy descent.
        let depth_factor = (constructed_depth as f64 / 52.0).clamp(0.02, 1.0);
        (0.65 * r_base + 0.30 * depth_factor * (1.0 + pami_spread_val).tanh())
            .clamp(0.0, 0.999_999)
    };

    Ok(NativeContinuationCandidate {
        prompt_embeddings,
        emitted_embeddings,
        emitted_carriers,
        decoded_text,
        decode_error,
        stop,
        mean_log_prob,
        min_margin,
        vault_arc_coverage,
        mean_z_surprise,
        max_z_surprise,
        gate_score,
        steps,
        is_continuation: true,
        constructed_depth,
        constructed_rank,
        active_attention_sectors,
        pami_spectral_spread,
        marginal_free_energy_gain,
        poincare_radius,
    })
}

/// Construct an unscored [`GateScore`] (`GateVerdict::Cold`, `scored_steps: 0`) for candidates
/// produced by the predictor before `runtime_model.rs` passes them through `TrajectoryMatchGate`.
pub fn unscored_gate_score() -> GateScore {
    GateScore {
        verdict: crate::predict_gate::GateVerdict::Cold,
        scored_steps: 0,
        max_surprise: None,
        mean_surprise: None,
        max_z_hyper: None,
        max_z_torus: None,
        hot_steps: 0,
        worst_skill: None,
        trusted: false,
    }
}

/// Flexible adapter over the post-`step_native_toroidal_prompt` session toroidal states stored on
/// `MainModelCoordinatorRuntime` (`toroidal_hidden_sequences`, `toroidal_hidden_states`, or
/// `toroidal_sessions`).
pub trait ToroidalSessionStateInput {
    fn runtime_ref(&self) -> Option<&ToroidalIsotropicRuntime> {
        None
    }
    fn hidden_sequence(&self) -> Option<&[Vec<f32>]> {
        None
    }
    fn latest_hidden(&self) -> Option<&[f32]> {
        None
    }
}

impl ToroidalSessionStateInput for &[Vec<f32>] {
    fn hidden_sequence(&self) -> Option<&[Vec<f32>]> {
        if self.is_empty() {
            None
        } else {
            Some(*self)
        }
    }
    fn latest_hidden(&self) -> Option<&[f32]> {
        self.last().map(Vec::as_slice)
    }
}

impl ToroidalSessionStateInput for &Vec<Vec<f32>> {
    fn hidden_sequence(&self) -> Option<&[Vec<f32>]> {
        if self.is_empty() {
            None
        } else {
            Some(self.as_slice())
        }
    }
    fn latest_hidden(&self) -> Option<&[f32]> {
        self.last().map(Vec::as_slice)
    }
}

impl ToroidalSessionStateInput for Option<&[Vec<f32>]> {
    fn hidden_sequence(&self) -> Option<&[Vec<f32>]> {
        self.and_then(|s| if s.is_empty() { None } else { Some(s) })
    }
    fn latest_hidden(&self) -> Option<&[f32]> {
        self.and_then(|s| s.last().map(Vec::as_slice))
    }
}

impl ToroidalSessionStateInput for Option<&Vec<Vec<f32>>> {
    fn hidden_sequence(&self) -> Option<&[Vec<f32>]> {
        self.and_then(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s.as_slice())
            }
        })
    }
    fn latest_hidden(&self) -> Option<&[f32]> {
        self.and_then(|s| s.last().map(Vec::as_slice))
    }
}

impl ToroidalSessionStateInput for &[f32] {
    fn latest_hidden(&self) -> Option<&[f32]> {
        if self.is_empty() {
            None
        } else {
            Some(*self)
        }
    }
}

impl ToroidalSessionStateInput for &Vec<f32> {
    fn latest_hidden(&self) -> Option<&[f32]> {
        if self.is_empty() {
            None
        } else {
            Some(self.as_slice())
        }
    }
}

impl ToroidalSessionStateInput for Option<&[f32]> {
    fn latest_hidden(&self) -> Option<&[f32]> {
        self.and_then(|s| if s.is_empty() { None } else { Some(s) })
    }
}

impl ToroidalSessionStateInput for Option<&Vec<f32>> {
    fn latest_hidden(&self) -> Option<&[f32]> {
        self.and_then(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s.as_slice())
            }
        })
    }
}

impl ToroidalSessionStateInput for &ToroidalIsotropicRuntime {
    fn runtime_ref(&self) -> Option<&ToroidalIsotropicRuntime> {
        Some(*self)
    }
}

impl ToroidalSessionStateInput for Option<&ToroidalIsotropicRuntime> {
    fn runtime_ref(&self) -> Option<&ToroidalIsotropicRuntime> {
        *self
    }
}

impl ToroidalSessionStateInput for (&ToroidalIsotropicRuntime, &[Vec<f32>]) {
    fn runtime_ref(&self) -> Option<&ToroidalIsotropicRuntime> {
        Some(self.0)
    }
    fn hidden_sequence(&self) -> Option<&[Vec<f32>]> {
        if self.1.is_empty() {
            None
        } else {
            Some(self.1)
        }
    }
    fn latest_hidden(&self) -> Option<&[f32]> {
        self.1.last().map(Vec::as_slice)
    }
}

impl ToroidalSessionStateInput for (Option<&ToroidalIsotropicRuntime>, Option<&Vec<Vec<f32>>>) {
    fn runtime_ref(&self) -> Option<&ToroidalIsotropicRuntime> {
        self.0
    }
    fn hidden_sequence(&self) -> Option<&[Vec<f32>]> {
        self.1.and_then(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s.as_slice())
            }
        })
    }
    fn latest_hidden(&self) -> Option<&[f32]> {
        self.1.and_then(|s| s.last().map(Vec::as_slice))
    }
}

/// Binary magic header for the machine-path `NativeCarrierPromoter` sidecar (`docs/BREATHING_DEPTH_DESIGN.md` §7).
pub const CARRIER_PROMOTER_BINARY_MAGIC: &[u8; 12] = b"ATOMA_CP_V2\0";

/// Cached turn-path helper for promoting a toroidal-state-driven next-carrier continuation
/// candidate into `MainModelCoordinatorRuntime::execute_live_operational_turn`.
///
/// Ownership invariants:
/// - Fits [`NativeCarrierTransitionModel`] once from the clean vault arcs (enforcing the exact
///   teacher-origin and `.atoma_map.wal` quarantine) on boot or loads it from disk.
/// - [`NativeCarrierPromoter::promote`] takes the post-`step_native_toroidal_prompt` session
///   states, runs [`generate_toroidal_carrier_continuation_with_model`], and returns
///   `Option<NativeContinuationCandidate>` with `prompt_embeddings`, `emitted_embeddings`, and an
///   **unscored** `gate_score` (`GateVerdict::Cold`, `scored_steps: 0`).
/// - Makes **no gate decisions inside**: `runtime_model.rs` scores the returned candidate through
///   `TrajectoryMatchGate` in the turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeCarrierPromoter {
    pub model: NativeCarrierTransitionModel,
    pub training_receipt: Option<CarrierTransitionTrainingReceipt>,
    pub fitted: bool,
    pub loaded_from_disk: bool,
    #[serde(default)]
    pub breathing_config: crate::toroidal_attention::BreathingDepthConfig,
    #[serde(skip)]
    pub last_breathing_receipt: Option<crate::toroidal_attention::BreathingConstructionReceipt>,
    #[serde(skip)]
    pub cache_path: Option<PathBuf>,
}

impl Default for NativeCarrierPromoter {
    fn default() -> Self {
        Self {
            model: NativeCarrierTransitionModel::default(),
            training_receipt: None,
            fitted: false,
            loaded_from_disk: false,
            breathing_config: crate::toroidal_attention::BreathingDepthConfig::default(),
            last_breathing_receipt: None,
            cache_path: None,
        }
    }
}

impl NativeCarrierPromoter {
    /// Construct an unfitted promoter (can be registered at an init site via [`Self::register_vault`]
    /// or [`Self::from_vault`], and will also lazily fit/load once on first [`Self::promote`] if needed).
    pub fn new() -> Self {
        Self::default()
    }

    /// Derive the deterministic binary sidecar cache path (`<vault_path>.carrier_promoter.bin`) for a vault.
    /// Zero JSON on the machine path (`docs/BREATHING_DEPTH_DESIGN.md` §7).
    pub fn sidecar_path_for_vault(vault_path: &Path) -> PathBuf {
        let mut os = vault_path.as_os_str().to_os_string();
        os.push(".carrier_promoter.bin");
        PathBuf::from(os)
    }

    /// Default workspace sidecar cache path when no explicit `vault_path` is passed and the vault
    /// matches the workspace's primary `global_active_mind.atoma_map`.
    fn default_workspace_sidecar_path(total_vault_arcs: usize) -> Option<PathBuf> {
        if total_vault_arcs < 100_000 {
            return None;
        }
        let vault_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../design/global_active_mind.atoma_map");
        if vault_path.is_file() {
            Some(Self::sidecar_path_for_vault(&vault_path))
        } else {
            None
        }
    }

    /// Serialize `self` into the compact binary `ATOMA_CP_V2\0` wire format with a trailing SHA-256 digest.
    pub fn encode_binary(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(4096);
        let put_u32 = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());
        let put_u64 = |buf: &mut Vec<u8>, v: u64| buf.extend_from_slice(&v.to_le_bytes());
        let put_f32 = |buf: &mut Vec<u8>, v: f32| buf.extend_from_slice(&v.to_le_bytes());
        let put_f64 = |buf: &mut Vec<u8>, v: f64| buf.extend_from_slice(&v.to_le_bytes());

        put_u32(&mut payload, self.model.layers as u32);
        for &v in &self.model.means {
            put_f64(&mut payload, v);
        }
        for &v in &self.model.scales {
            put_f64(&mut payload, v);
        }
        put_u32(&mut payload, self.model.coord_weights.len() as u32);
        for &v in &self.model.coord_weights {
            put_f64(&mut payload, v);
        }
        for &v in &self.model.branch_weights {
            put_f64(&mut payload, v);
        }
        put_u32(&mut payload, self.model.eos_weights.len() as u32);
        for &v in &self.model.eos_weights {
            put_f64(&mut payload, v);
        }
        for &v in &self.model.depth_cov_weights {
            put_f64(&mut payload, v);
        }
        for &v in &self.model.eos_depth_cov_weights {
            put_f64(&mut payload, v);
        }

        put_u32(&mut payload, self.model.per_depth_readouts.len() as u32);
        for head in &self.model.per_depth_readouts {
            put_u32(&mut payload, head.min_depth as u32);
            put_u32(&mut payload, head.max_depth as u32);
            put_u32(&mut payload, head.reference_depth as u32);
            for &v in &head.means {
                put_f64(&mut payload, v);
            }
            for &v in &head.scales {
                put_f64(&mut payload, v);
            }
            put_u32(&mut payload, head.coord_weights.len() as u32);
            for &v in &head.coord_weights {
                put_f64(&mut payload, v);
            }
            put_u32(&mut payload, head.eos_weights.len() as u32);
            for &v in &head.eos_weights {
                put_f64(&mut payload, v);
            }
            for &v in &head.branch_weights {
                put_f64(&mut payload, v);
            }
            put_f64(&mut payload, head.initial_coord_mse);
            put_f64(&mut payload, head.final_coord_mse);
        }

        put_u32(&mut payload, self.model.transition_prototypes.len() as u32);
        for (&(aid, op), proto) in &self.model.transition_prototypes {
            put_u64(&mut payload, aid);
            put_u32(&mut payload, op);
            put_u32(&mut payload, proto.count);
            put_f32(&mut payload, proto.log_prior);
            for &w in &proto.weights {
                put_f32(&mut payload, w);
            }
        }

        match &self.training_receipt {
            Some(r) => {
                payload.push(1);
                put_u64(&mut payload, r.total_vault_arcs as u64);
                put_u64(&mut payload, r.carrier_continuation_arcs as u64);
                put_u64(&mut payload, r.clean_continuation_arcs as u64);
                put_u64(&mut payload, r.teacher_contaminated_arcs_filtered as u64);
                put_u64(&mut payload, r.wal_contaminated_frames_filtered as u64);
                put_u64(&mut payload, r.structural_anchor_arcs as u64);
                put_u64(&mut payload, r.contiguous_step_pairs as u64);
                put_u64(&mut payload, r.trajectory_chains as u64);
                put_u64(&mut payload, r.branching_source_atoms as u64);
                put_u64(&mut payload, r.branching_arcs as u64);
                put_u64(&mut payload, r.unique_target_carriers as u64);
                put_f64(&mut payload, r.initial_coord_mse);
                put_f64(&mut payload, r.final_coord_mse);
                put_f64(&mut payload, r.initial_branch_nll_nats);
                put_f64(&mut payload, r.final_branch_nll_nats);
                put_f64(&mut payload, r.initial_branch_top1_acc);
                put_f64(&mut payload, r.final_branch_top1_acc);
                put_f64(&mut payload, r.initial_mean_z_surprise);
                put_f64(&mut payload, r.final_mean_z_surprise);
                put_f64(&mut payload, r.final_max_z_surprise);
                put_u64(&mut payload, r.elapsed_ms as u64);
            }
            None => payload.push(0),
        }

        let digest = Sha256Digest::compute(&payload);
        let mut out = Vec::with_capacity(12 + 8 + payload.len() + 32);
        out.extend_from_slice(CARRIER_PROMOTER_BINARY_MAGIC);
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&payload);
        out.extend_from_slice(&digest.0);
        out
    }

    /// Deserialize a [`NativeCarrierPromoter`] from the binary `ATOMA_CP_V2\0` wire format,
    /// verifying its SHA-256 integrity digest.
    pub fn decode_binary(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 + 8 + 32 {
            return Err("binary NativeCarrierPromoter buffer too short".into());
        }
        if &bytes[..12] != CARRIER_PROMOTER_BINARY_MAGIC {
            return Err("invalid binary NativeCarrierPromoter magic".into());
        }
        let payload_len = u64::from_le_bytes(bytes[12..20].try_into().unwrap()) as usize;
        if bytes.len() != 20 + payload_len + 32 {
            return Err("binary NativeCarrierPromoter payload length mismatch".into());
        }
        let payload = &bytes[20..20 + payload_len];
        let expected_digest = &bytes[20 + payload_len..];
        let actual_digest = Sha256Digest::compute(payload);
        if actual_digest.0.as_slice() != expected_digest {
            return Err("binary NativeCarrierPromoter SHA-256 digest mismatch".into());
        }

        let mut pos = 0usize;
        let get_u32 = |pos: &mut usize| -> Result<u32, String> {
            let end = pos.checked_add(4).ok_or("overflow")?;
            let slice = payload.get(*pos..end).ok_or("unexpected EOF (u32)")?;
            *pos = end;
            Ok(u32::from_le_bytes(slice.try_into().unwrap()))
        };
        let get_u64 = |pos: &mut usize| -> Result<u64, String> {
            let end = pos.checked_add(8).ok_or("overflow")?;
            let slice = payload.get(*pos..end).ok_or("unexpected EOF (u64)")?;
            *pos = end;
            Ok(u64::from_le_bytes(slice.try_into().unwrap()))
        };
        let get_f32 = |pos: &mut usize| -> Result<f32, String> {
            let end = pos.checked_add(4).ok_or("overflow")?;
            let slice = payload.get(*pos..end).ok_or("unexpected EOF (f32)")?;
            *pos = end;
            Ok(f32::from_le_bytes(slice.try_into().unwrap()))
        };
        let get_f64 = |pos: &mut usize| -> Result<f64, String> {
            let end = pos.checked_add(8).ok_or("overflow")?;
            let slice = payload.get(*pos..end).ok_or("unexpected EOF (f64)")?;
            *pos = end;
            Ok(f64::from_le_bytes(slice.try_into().unwrap()))
        };

        let layers = get_u32(&mut pos)? as usize;
        let mut means = [0.0f64; CARRIER_TOROIDAL_FEATURES];
        for v in &mut means {
            *v = get_f64(&mut pos)?;
        }
        let mut scales = [1.0f64; CARRIER_TOROIDAL_FEATURES];
        for v in &mut scales {
            *v = get_f64(&mut pos)?;
        }
        let n_cw = get_u32(&mut pos)? as usize;
        let mut coord_weights = Vec::with_capacity(n_cw);
        for _ in 0..n_cw {
            coord_weights.push(get_f64(&mut pos)?);
        }
        let mut branch_weights = [0.0f64; 5];
        for v in &mut branch_weights {
            *v = get_f64(&mut pos)?;
        }
        let n_ew = get_u32(&mut pos)? as usize;
        let mut eos_weights = Vec::with_capacity(n_ew);
        for _ in 0..n_ew {
            eos_weights.push(get_f64(&mut pos)?);
        }
        let mut depth_cov_weights = [0.0f64; 12];
        for v in &mut depth_cov_weights {
            *v = get_f64(&mut pos)?;
        }
        let mut eos_depth_cov_weights = [0.0f64; 2];
        for v in &mut eos_depth_cov_weights {
            *v = get_f64(&mut pos)?;
        }

        let num_heads = get_u32(&mut pos)? as usize;
        let mut per_depth_readouts = Vec::with_capacity(num_heads.min(16));
        for _ in 0..num_heads {
            let min_depth = get_u32(&mut pos)? as usize;
            let max_depth = get_u32(&mut pos)? as usize;
            let reference_depth = get_u32(&mut pos)? as usize;
            let mut h_means = [0.0f64; CARRIER_TOROIDAL_FEATURES];
            for v in &mut h_means {
                *v = get_f64(&mut pos)?;
            }
            let mut h_scales = [1.0f64; CARRIER_TOROIDAL_FEATURES];
            for v in &mut h_scales {
                *v = get_f64(&mut pos)?;
            }
            let head_cw_len = get_u32(&mut pos)? as usize;
            let mut head_cw = Vec::with_capacity(head_cw_len);
            for _ in 0..head_cw_len {
                head_cw.push(get_f64(&mut pos)?);
            }
            let head_ew_len = get_u32(&mut pos)? as usize;
            let mut head_ew = Vec::with_capacity(head_ew_len);
            for _ in 0..head_ew_len {
                head_ew.push(get_f64(&mut pos)?);
            }
            let mut hbw = [0.0f64; 5];
            for v in &mut hbw {
                *v = get_f64(&mut pos)?;
            }
            let initial_coord_mse = get_f64(&mut pos)?;
            let final_coord_mse = get_f64(&mut pos)?;
            per_depth_readouts.push(DepthConditionalReadoutHead {
                min_depth,
                max_depth,
                reference_depth,
                means: h_means,
                scales: h_scales,
                coord_weights: head_cw,
                eos_weights: head_ew,
                branch_weights: hbw,
                initial_coord_mse,
                final_coord_mse,
            });
        }

        let num_protos = get_u32(&mut pos)? as usize;
        let mut transition_prototypes = std::collections::BTreeMap::new();
        for _ in 0..num_protos {
            let aid = get_u64(&mut pos)?;
            let op = get_u32(&mut pos)?;
            let count = get_u32(&mut pos)?;
            let log_prior = get_f32(&mut pos)?;
            let mut weights = [0.0f32; 16];
            for w in &mut weights {
                *w = get_f32(&mut pos)?;
            }
            transition_prototypes.insert(
                (aid, op),
                ToroidalTransitionPrototype {
                    count,
                    log_prior,
                    weights,
                },
            );
        }

        let has_receipt = *payload.get(pos).ok_or("unexpected EOF (receipt flag)")?;
        pos += 1;
        let training_receipt = if has_receipt == 1 {
            Some(CarrierTransitionTrainingReceipt {
                total_vault_arcs: get_u64(&mut pos)? as usize,
                carrier_continuation_arcs: get_u64(&mut pos)? as usize,
                clean_continuation_arcs: get_u64(&mut pos)? as usize,
                teacher_contaminated_arcs_filtered: get_u64(&mut pos)? as usize,
                wal_contaminated_frames_filtered: get_u64(&mut pos)? as usize,
                structural_anchor_arcs: get_u64(&mut pos)? as usize,
                contiguous_step_pairs: get_u64(&mut pos)? as usize,
                trajectory_chains: get_u64(&mut pos)? as usize,
                branching_source_atoms: get_u64(&mut pos)? as usize,
                branching_arcs: get_u64(&mut pos)? as usize,
                unique_target_carriers: get_u64(&mut pos)? as usize,
                initial_coord_mse: get_f64(&mut pos)?,
                final_coord_mse: get_f64(&mut pos)?,
                initial_branch_nll_nats: get_f64(&mut pos)?,
                final_branch_nll_nats: get_f64(&mut pos)?,
                initial_branch_top1_acc: get_f64(&mut pos)?,
                final_branch_top1_acc: get_f64(&mut pos)?,
                initial_mean_z_surprise: get_f64(&mut pos)?,
                final_mean_z_surprise: get_f64(&mut pos)?,
                final_max_z_surprise: get_f64(&mut pos)?,
                elapsed_ms: get_u64(&mut pos)? as u128,
            })
        } else {
            None
        };

        Ok(Self {
            model: NativeCarrierTransitionModel {
                layers,
                means,
                scales,
                coord_weights,
                branch_weights,
                eos_weights,
                depth_cov_weights,
                eos_depth_cov_weights,
                per_depth_readouts,
                transition_prototypes,
            },
            training_receipt,
            fitted: true,
            loaded_from_disk: true,
            breathing_config: crate::toroidal_attention::BreathingDepthConfig::default(),
            last_breathing_receipt: None,
            cache_path: None,
        })
    }

    /// Save the fitted promoter to disk. Uses the binary `ATOMA_CP_V2\0` format by default (zero JSON
    /// on the machine path), and only falls back to JSON when `path` explicitly ends with `.json`
    /// for developer inspection (`docs/BREATHING_DEPTH_DESIGN.md` §7).
    pub fn save_to_disk(&self, path: &Path) -> Result<(), String> {
        let is_json = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("json"));
        let bytes = if is_json {
            serde_json::to_vec(self)
                .map_err(|e| format!("serialize NativeCarrierPromoter JSON: {e}"))?
        } else {
            self.encode_binary()
        };
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(path, bytes)
            .map_err(|e| format!("write NativeCarrierPromoter {}: {e}", path.display()))
    }

    /// Load a previously fitted promoter from disk (supports binary `ATOMA_CP_V2\0` native format
    /// as well as `.json` developer inspection artifacts).
    pub fn load_from_disk(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|e| format!("read NativeCarrierPromoter {}: {e}", path.display()))?;
        let mut promoter = if bytes.starts_with(CARRIER_PROMOTER_BINARY_MAGIC) {
            Self::decode_binary(&bytes)?
        } else {
            serde_json::from_slice(&bytes)
                .map_err(|e| format!("deserialize NativeCarrierPromoter {}: {e}", path.display()))?
        };
        promoter.fitted = true;
        promoter.loaded_from_disk = true;
        promoter.cache_path = Some(path.to_path_buf());
        Ok(promoter)
    }

    /// Build a [`NativeCarrierPromoter`] from the vault's clean continuation arcs on boot, or load
    /// it from the binary sidecar cache (`<vault_path>.carrier_promoter.bin`) if already fitted for
    /// the same clean arc count.
    pub fn from_vault(
        swarm_map: &atoma_immune::CarvedSwarmMap,
        catalog: &atoma_immune::MlirAtomCatalog,
        vault_path: Option<&Path>,
    ) -> Self {
        let cache_path = vault_path
            .map(Self::sidecar_path_for_vault)
            .or_else(|| Self::default_workspace_sidecar_path(swarm_map.reasoning_arcs.len()));
        Self::load_or_fit(swarm_map, catalog, cache_path.as_deref())
    }

    /// Build a [`NativeCarrierPromoter`] directly from `swarm_map` and `catalog`.
    pub fn from_swarm_map(
        swarm_map: &atoma_immune::CarvedSwarmMap,
        catalog: &atoma_immune::MlirAtomCatalog,
    ) -> Self {
        Self::from_vault(swarm_map, catalog, None)
    }

    /// Build a [`NativeCarrierPromoter`] directly from a [`crate::MonsterHybridServer`].
    pub fn from_server(server: &crate::MonsterHybridServer) -> Self {
        Self::from_vault(
            &server.swarm_map,
            &server.mlir_catalog,
            server.persistent_vault_path.as_deref(),
        )
    }

    /// Load a fitted [`NativeCarrierPromoter`] from `cache_path` if present and matching the
    /// current vault's clean continuation arcs, or fit once from `swarm_map` (enforcing the
    /// teacher-origin and WAL quarantine) and persist to `cache_path` when provided.
    pub fn load_or_fit(
        swarm_map: &atoma_immune::CarvedSwarmMap,
        catalog: &atoma_immune::MlirAtomCatalog,
        cache_path: Option<&Path>,
    ) -> Self {
        if swarm_map.reasoning_arcs.is_empty() {
            return Self {
                cache_path: cache_path.map(Path::to_path_buf),
                ..Self::default()
            };
        }

        if let Some(path) = cache_path
            && path.is_file()
            && let Ok(cached) = Self::load_from_disk(path)
        {
            let matches_vault = cached.training_receipt.as_ref().is_some_and(|r| {
                r.total_vault_arcs == swarm_map.reasoning_arcs.len() && r.clean_continuation_arcs > 0
            });
            if matches_vault && !cached.model.per_depth_readouts.is_empty() {
                return cached;
            }
        }

        match NativeCarrierTransitionModel::fit_from_swarm_map(swarm_map, catalog, 2, 3) {
            Ok((model, receipt)) => {
                let promoter = Self {
                    model,
                    training_receipt: Some(receipt),
                    fitted: true,
                    loaded_from_disk: false,
                    breathing_config: crate::toroidal_attention::BreathingDepthConfig::default(),
                    last_breathing_receipt: None,
                    cache_path: cache_path.map(Path::to_path_buf),
                };
                if let Some(path) = cache_path {
                    let _ = promoter.save_to_disk(path);
                }
                promoter
            }
            Err(_) => Self {
                fitted: false,
                cache_path: cache_path.map(Path::to_path_buf),
                ..Self::default()
            },
        }
    }

    /// Init-site registration helper: fit or load the model for `swarm_map` if not yet fitted.
    pub fn register_vault(
        &mut self,
        swarm_map: &atoma_immune::CarvedSwarmMap,
        catalog: &atoma_immune::MlirAtomCatalog,
        vault_path: Option<&Path>,
    ) {
        *self = Self::from_vault(swarm_map, catalog, vault_path);
    }

    /// Init-site registration helper taking a [`crate::MonsterHybridServer`].
    pub fn register_server(&mut self, server: &crate::MonsterHybridServer) {
        *self = Self::from_server(server);
    }

    fn ensure_fitted(
        &mut self,
        swarm_map: &atoma_immune::CarvedSwarmMap,
        catalog: &atoma_immune::MlirAtomCatalog,
    ) {
        if !self.fitted && !swarm_map.reasoning_arcs.is_empty() {
            let cache_path = self.cache_path.clone().or_else(|| {
                Self::default_workspace_sidecar_path(swarm_map.reasoning_arcs.len())
            });
            *self = Self::load_or_fit(swarm_map, catalog, cache_path.as_deref());
        }
    }

    /// Promote a toroidal-state-driven continuation candidate from the post-`step_native_toroidal_prompt`
    /// session states (`&[Vec<f32>]`, `Option<&Vec<Vec<f32>>>`, `&[f32]`, or `Option<&ToroidalIsotropicRuntime>`).
    ///
    /// Returns `Some(NativeContinuationCandidate)` with `prompt_embeddings`, `emitted_embeddings`,
    /// and an **unscored** `gate_score` (`GateVerdict::Cold`, `scored_steps: 0`), or `None` if
    /// no valid continuation candidate could be generated.
    pub fn promote<S: ToroidalSessionStateInput>(
        &mut self,
        toroidal_states: S,
        swarm_map: &atoma_immune::CarvedSwarmMap,
        catalog: &atoma_immune::MlirAtomCatalog,
        transducer: &BoundaryLexicalTransducer,
        prompt: &str,
        max_new_carriers: usize,
    ) -> Option<NativeContinuationCandidate> {
        self.promote_with_prefix(
            toroidal_states,
            swarm_map,
            catalog,
            transducer,
            prompt,
            None,
            max_new_carriers,
        )
    }

    /// Same as [`Self::promote`], allowing an optional pre-resolved `prompt_carriers` slice.
    ///
    /// Constructs a breathing toroidal stack per turn (`docs/BREATHING_DEPTH_DESIGN.md` §2–§4)
    /// when the passed runtime is not already a custom-depth breathing stack, recording
    /// `constructed_depth`, `constructed_rank`, `active_attention_sectors`, `pami_spectral_spread`,
    /// `marginal_free_energy_gain`, and `poincare_radius` on the returned candidate.
    #[allow(clippy::too_many_arguments)]
    pub fn promote_with_prefix<S: ToroidalSessionStateInput>(
        &mut self,
        toroidal_states: S,
        swarm_map: &atoma_immune::CarvedSwarmMap,
        catalog: &atoma_immune::MlirAtomCatalog,
        transducer: &BoundaryLexicalTransducer,
        prompt: &str,
        prompt_carriers: Option<&[MlirOpId]>,
        max_new_carriers: usize,
    ) -> Option<NativeContinuationCandidate> {
        if prompt.trim().is_empty() {
            return None;
        }
        let rt_ref = toroidal_states.runtime_ref();
        let seq_opt = toroidal_states.hidden_sequence();
        let latest_opt = toroidal_states.latest_hidden();
        if rt_ref.is_none() && seq_opt.is_none() && latest_opt.is_none() {
            return None;
        }

        self.ensure_fitted(swarm_map, catalog);

        let max_new = max_new_carriers.clamp(1, MAX_NEW_BYTES);

        // Resolve prompt embeddings to drive adaptive breathing construction (`step_breathing_text_sequence`)
        let mut resolved_prefix: Vec<MlirOpId> = match prompt_carriers {
            Some(slice) if !slice.is_empty() => slice.to_vec(),
            _ => transducer.transduce_ingress_text(prompt, true),
        };
        if resolved_prefix.is_empty() {
            resolved_prefix = transducer.transduce_utf8_to_mlir(prompt, false);
        }
        let mut prompt_embs = Vec::with_capacity(resolved_prefix.len());
        for &op in &resolved_prefix {
            if let Ok(e) = embed_carrier(catalog, op) {
                prompt_embs.push(e);
            }
        }

        let mut candidate = if let Some(session_rt) = rt_ref
            && (1..52).contains(&session_rt.constructed_depth())
        {
            generate_toroidal_carrier_continuation_with_model(
                &self.model,
                session_rt,
                swarm_map,
                catalog,
                transducer,
                prompt,
                prompt_carriers,
                max_new,
            )
            .ok()?
        } else if !prompt_embs.is_empty() {
            let mut breathing_rt = ToroidalIsotropicRuntime::breathing_scaffold(6, 52).ok()?;
            // Depth decision reads the relational backbone only
            // (2026-10-20): narrative roughness must not drive depth.
            let backbone =
                crate::input_embeddings::backbone_filter(transducer, &prompt_embs);
            let receipt = breathing_rt
                .step_breathing_text_sequence_backbone(
                    &prompt_embs,
                    &self.breathing_config,
                    Some(&backbone),
                )
                .ok()?;
            self.last_breathing_receipt = Some(receipt.clone());
            let mut cand = generate_toroidal_carrier_continuation_with_model(
                &self.model,
                &breathing_rt,
                swarm_map,
                catalog,
                transducer,
                prompt,
                prompt_carriers,
                max_new,
            )
            .ok()?;
            cand.constructed_depth = receipt.constructed_depth;
            cand.constructed_rank = receipt.constructed_rank;
            cand.active_attention_sectors = receipt.active_attention_sectors;
            cand.pami_spectral_spread = receipt.pami_spectral_spread;
            cand.marginal_free_energy_gain = receipt.marginal_free_energy_gain;
            cand.poincare_radius = receipt.poincare_radius;
            cand
        } else {
            let mut seeded_rt =
                ToroidalIsotropicRuntime::full_stack(6, self.model.layers.max(1)).ok()?;
            if let Some(seq) = seq_opt {
                for state in seq {
                    let coords: [f32; 6] = state.as_slice().try_into().ok()?;
                    if coords.iter().any(|v| !v.is_finite()) {
                        return None;
                    }
                    seeded_rt.step_toroidal_attention(&coords).ok()?;
                }
            } else if let Some(last) = latest_opt {
                let coords: [f32; 6] = last.try_into().ok()?;
                if coords.iter().any(|v| !v.is_finite()) {
                    return None;
                }
                seeded_rt.step_toroidal_attention(&coords).ok()?;
            }
            generate_toroidal_carrier_continuation_with_model(
                &self.model,
                &seeded_rt,
                swarm_map,
                catalog,
                transducer,
                prompt,
                prompt_carriers,
                max_new,
            )
            .ok()?
        };

        if let Some(receipt) = &self.last_breathing_receipt {
            candidate.constructed_depth = receipt.constructed_depth;
            candidate.constructed_rank = receipt.constructed_rank;
            candidate.active_attention_sectors = receipt.active_attention_sectors;
            candidate.pami_spectral_spread = receipt.pami_spectral_spread;
            candidate.marginal_free_energy_gain = receipt.marginal_free_energy_gain;
            candidate.poincare_radius = receipt.poincare_radius;
        }

        if candidate.emitted_carriers.is_empty()
            || candidate.emitted_embeddings.is_empty()
            || candidate.decode_error.is_some()
            || candidate
                .decoded_text
                .as_ref()
                .is_none_or(|txt| txt.trim().is_empty())
        {
            return None;
        }
        Some(candidate)
    }

    /// Convenience wrapper over [`Self::promote`] taking `&MonsterHybridServer`.
    pub fn promote_from_server<S: ToroidalSessionStateInput>(
        &mut self,
        toroidal_states: S,
        server: &crate::MonsterHybridServer,
        prompt: &str,
        max_new_carriers: usize,
    ) -> Option<NativeContinuationCandidate> {
        self.promote(
            toroidal_states,
            &server.swarm_map,
            &server.mlir_catalog,
            &server.active_transformer.transducer,
            prompt,
            max_new_carriers,
        )
    }
}

/// One prompt entry on the frozen easy → hard pre-registered falsification ladder (`docs/BREATHING_DEPTH_DESIGN.md` §6).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathingFalsificationPrompt {
    pub id: String,
    pub prompt: String,
    /// `true` for the easy half of the ladder, `false` for the hard/multi-step half.
    pub is_easy_half: bool,
}

/// Per-prompt comparison between Breathing Construction and Fixed-52 on the falsification ladder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathingPromptComparison {
    pub id: String,
    pub is_easy_half: bool,
    pub breathing_depth: usize,
    pub breathing_rank: usize,
    pub active_attention_sectors: usize,
    pub pami_spectral_spread: f64,
    pub marginal_free_energy_gain: f64,
    pub breathing_poincare_radius: f64,
    pub fixed52_poincare_radius: f64,
    pub breathing_answer: Option<String>,
    pub fixed52_answer: Option<String>,
    pub answers_agree: bool,
}

/// Pre-registered verdict on the PAMI Index Hypothesis (`docs/BREATHING_DEPTH_DESIGN.md` §6.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PamiIndexVerdict {
    /// Positive Spearman rank correlation significant at `alpha = 0.01`.
    Real,
    /// Indistinguishable from permutation null; breathing falls back cleanly to the marginal free-energy rule alone (`use_pami_forecast = false`).
    NullSurvivableOnMarginalRule,
    /// Negative correlation significant at `alpha = 0.01`; hypothesis inverted/rejected.
    Inverted,
}

/// Complete 4-criterion pre-registered falsification report (`docs/BREATHING_DEPTH_DESIGN.md` §6.1–§6.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathingFalsificationReport {
    pub comparisons: Vec<BreathingPromptComparison>,
    /// §6.1: Fraction of ladder prompts where Breathing and Fixed-52 agree on the top-1 continuation.
    pub answer_agreement: f64,
    /// §6.1: Pre-registered floor (`>= 0.85`).
    pub answer_agreement_passed: bool,
    /// §6.2: Mean constructed depth on the easy half of the ladder.
    pub mean_easy_depth: f64,
    /// §6.2: Mean constructed depth on the hard half of the ladder.
    pub mean_hard_depth: f64,
    /// §6.2: Economy criterion (`mean_easy_depth < 52.0`, strictly).
    pub economy_passed: bool,
    /// §6.3: Spearman rank correlation between PAMI spectral spread and marginal free-energy gain from added layers.
    pub pami_spearman_rho: f64,
    /// §6.3: One-sided permutation p-value from `crate::falsifier::falsify_pairs`.
    pub pami_p_value: f64,
    /// §6.3: Index hypothesis verdict (`Real`, `NullSurvivableOnMarginalRule`, or `Inverted`).
    pub pami_verdict: PamiIndexVerdict,
    /// §6.3: True when either PAMI is `Real` or `NullSurvivableOnMarginalRule` (never `Inverted`).
    pub index_or_marginal_survived: bool,
    /// §6.4: Standard deviation of Poincaré radial depth under Breathing Construction.
    pub breathing_poincare_radius_std: f64,
    /// §6.4: Standard deviation of Poincaré radial depth under Fixed-52.
    pub fixed52_poincare_radius_std: f64,
    /// §6.4: True when the Poincaré radial distribution gains real spread under breathing (`breathing_std > fixed52_std` and `breathing_std >= 0.01`).
    pub depth_variance_passed: bool,
    /// Overall falsification ladder pass (`answer_agreement_passed && economy_passed && index_or_marginal_survived && depth_variance_passed`).
    pub overall_passed: bool,
}

/// Return the frozen 12-prompt easy → hard pre-registered falsification ladder (`docs/BREATHING_DEPTH_DESIGN.md` §6).
pub fn preregistered_breathing_prompt_ladder() -> Vec<BreathingFalsificationPrompt> {
    vec![
        // Easy half (6 short, low-spread prompts)
        BreathingFalsificationPrompt {
            id: "easy_01_paris".into(),
            prompt: "The capital of France is".into(),
            is_easy_half: true,
        },
        BreathingFalsificationPrompt {
            id: "easy_02_2plus2".into(),
            prompt: "2 + 2 =".into(),
            is_easy_half: true,
        },
        BreathingFalsificationPrompt {
            id: "easy_03_water".into(),
            prompt: "Water freezes at".into(),
            is_easy_half: true,
        },
        BreathingFalsificationPrompt {
            id: "easy_04_sun".into(),
            prompt: "The sun rises in the".into(),
            is_easy_half: true,
        },
        BreathingFalsificationPrompt {
            id: "easy_05_bits".into(),
            prompt: "One byte has".into(),
            is_easy_half: true,
        },
        BreathingFalsificationPrompt {
            id: "easy_06_zero".into(),
            prompt: "x + 0 =".into(),
            is_easy_half: true,
        },
        // Hard half (6 multi-clause, cross-scale structural prompts)
        BreathingFalsificationPrompt {
            id: "hard_01_poincare_torus".into(),
            prompt: "On the product manifold H^3 x T^2, when a hyperbolic geodesic boosts across curvature shells while toroidal phases wind at golden-ratio frequencies, the marginal variational free energy".into(),
            is_easy_half: false,
        },
        BreathingFalsificationPrompt {
            id: "hard_02_mlir_verification".into(),
            prompt: "To verify that a multi-step ML-IR carrier trajectory preserves Wigner frame alignment across 52 coupled harmonic layers without teacher contamination or UTF-8 boundary drift, the gate must".into(),
            is_easy_half: false,
        },
        BreathingFalsificationPrompt {
            id: "hard_03_multifractal_pami".into(),
            prompt: "When the phi-scaled Morlet wavelet partition function Z(q, s) exhibits wide Legendre singularity spread Delta D across five dyadic scales, the active toroidal attention sectors".into(),
            is_easy_half: false,
        },
        BreathingFalsificationPrompt {
            id: "hard_04_saturating_q32".into(),
            prompt: "In fixed-point Q32.31 Lorentzian geometry, boost composition and cosh distance evaluation near the boundary of the Poincare ball require saturating arithmetic so that".into(),
            is_easy_half: false,
        },
        BreathingFalsificationPrompt {
            id: "hard_05_variational_brake".into(),
            prompt: "Constructing layers upward from a 1-layer scaffold until dF/d(layer) >= -cost_per_layer replaces arbitrary truncation with a thermodynamic stopping criterion where".into(),
            is_easy_half: false,
        },
        BreathingFalsificationPrompt {
            id: "hard_06_falsifier_ledger".into(),
            prompt: "Every promoted continuation candidate files a (signal, outcome) pair into the pre-registered outcome ledger so Spearman rank permutation testing at alpha = 0.01 can falsify".into(),
            is_easy_half: false,
        },
    ]
}

/// Evaluate the 4-part pre-registered falsification ladder (`docs/BREATHING_DEPTH_DESIGN.md` §6.1–§6.4)
/// comparing Breathing Construction against Fixed-52 across `ladder`.
pub fn evaluate_breathing_depth_falsification_ladder(
    model: &NativeCarrierTransitionModel,
    swarm_map: &atoma_immune::CarvedSwarmMap,
    catalog: &atoma_immune::MlirAtomCatalog,
    transducer: &BoundaryLexicalTransducer,
    breathing_config: &crate::toroidal_attention::BreathingDepthConfig,
    ladder: &[BreathingFalsificationPrompt],
    max_new_carriers: usize,
) -> Result<BreathingFalsificationReport, String> {
    if ladder.is_empty() {
        return Err("falsification ladder must contain at least one prompt".into());
    }
    let max_new = max_new_carriers.clamp(1, 32);
    let mut comparisons = Vec::with_capacity(ladder.len());
    let mut pami_free_energy_pairs = Vec::with_capacity(ladder.len());

    for item in ladder {
        let mut prefix = transducer.transduce_ingress_text(&item.prompt, true);
        if prefix.is_empty() {
            prefix = transducer.transduce_utf8_to_mlir(&item.prompt, false);
        }
        let mut prompt_embs = Vec::with_capacity(prefix.len());
        for &op in &prefix {
            prompt_embs.push(embed_carrier(catalog, op)?);
        }

        // 1. Breathing construction pass
        let mut breathing_rt = ToroidalIsotropicRuntime::breathing_scaffold(6, 52)?;
        let receipt =
            breathing_rt.step_breathing_text_sequence(&prompt_embs, breathing_config)?;
        let mut breathing_cand = generate_toroidal_carrier_continuation_with_model(
            model,
            &breathing_rt,
            swarm_map,
            catalog,
            transducer,
            &item.prompt,
            Some(&prefix),
            max_new,
        )?;
        breathing_cand.constructed_depth = receipt.constructed_depth;
        breathing_cand.constructed_rank = receipt.constructed_rank;
        breathing_cand.active_attention_sectors = receipt.active_attention_sectors;
        breathing_cand.pami_spectral_spread = receipt.pami_spectral_spread;
        breathing_cand.marginal_free_energy_gain = receipt.marginal_free_energy_gain;
        breathing_cand.poincare_radius = receipt.poincare_radius;

        // 2. Fixed-52 reference pass (all 52 layers constructed identically, evaluated with the
        //    depth-agnostic readout or matched carrier readout)
        let mut fixed52_rt = ToroidalIsotropicRuntime::full_stack(6, 52)?;
        let mut last_h52 = [0.0f32; 6];
        for emb in &prompt_embs {
            let out = fixed52_rt.step_toroidal_attention(&emb.coordinates)?;
            last_h52 = out
                .as_slice()
                .try_into()
                .map_err(|_| "invalid 6D state")?;
        }
        let fixed52_poincare_radius = {
            let x0 = f64::from(last_h52[0]).max(1.0);
            let base_rapidity = x0.acosh();
            let stack_rapidity = base_rapidity + 0.25 * (52.0f64.ln_1p());
            (stack_rapidity * 0.5).tanh().clamp(0.0, 0.999_999)
        };
        let fixed52_cand = generate_toroidal_carrier_continuation_with_model(
            model,
            &fixed52_rt,
            swarm_map,
            catalog,
            transducer,
            &item.prompt,
            Some(&prefix),
            max_new,
        )?;

        let answers_agree = match (&breathing_cand.decoded_text, &fixed52_cand.decoded_text) {
            (Some(a), Some(b)) => a == b || breathing_cand.emitted_carriers.first() == fixed52_cand.emitted_carriers.first(),
            _ => breathing_cand.emitted_carriers.first() == fixed52_cand.emitted_carriers.first(),
        };

        pami_free_energy_pairs.push((
            receipt.pami_spectral_spread.unwrap_or(0.0),
            receipt.marginal_free_energy_gain,
        ));

        comparisons.push(BreathingPromptComparison {
            id: item.id.clone(),
            is_easy_half: item.is_easy_half,
            breathing_depth: receipt.constructed_depth,
            breathing_rank: receipt.constructed_rank,
            active_attention_sectors: receipt.active_attention_sectors,
            pami_spectral_spread: receipt.pami_spectral_spread.unwrap_or(0.0),
            marginal_free_energy_gain: receipt.marginal_free_energy_gain,
            breathing_poincare_radius: receipt.poincare_radius,
            fixed52_poincare_radius,
            breathing_answer: breathing_cand.decoded_text,
            fixed52_answer: fixed52_cand.decoded_text,
            answers_agree,
        });
    }

    let agree_count = comparisons.iter().filter(|c| c.answers_agree).count();
    let answer_agreement = (agree_count as f64) / (comparisons.len() as f64);
    let answer_agreement_passed = answer_agreement >= 0.85;

    let easy_items: Vec<&BreathingPromptComparison> =
        comparisons.iter().filter(|c| c.is_easy_half).collect();
    let hard_items: Vec<&BreathingPromptComparison> =
        comparisons.iter().filter(|c| !c.is_easy_half).collect();

    let mean_easy_depth = if easy_items.is_empty() {
        0.0
    } else {
        easy_items.iter().map(|c| c.breathing_depth as f64).sum::<f64>()
            / (easy_items.len() as f64)
    };
    let mean_hard_depth = if hard_items.is_empty() {
        0.0
    } else {
        hard_items.iter().map(|c| c.breathing_depth as f64).sum::<f64>()
            / (hard_items.len() as f64)
    };
    let economy_passed = mean_easy_depth < 52.0;

    // §6.3: Run the existing falsifier (`crate::falsifier::falsify_pairs`) on
    // (pami_spectral_spread, marginal_free_energy_gain) at alpha = 0.01.
    let falsifier_report =
        crate::falsifier::falsify_pairs(&pami_free_energy_pairs, true, 0xB8EA_7410);
    let pami_verdict = match falsifier_report.verdict {
        crate::falsifier::FalsifyVerdict::Pass => PamiIndexVerdict::Real,
        crate::falsifier::FalsifyVerdict::Null | crate::falsifier::FalsifyVerdict::Underpowered => {
            if falsifier_report.rho.is_some_and(|r| r < -0.3) {
                PamiIndexVerdict::Inverted
            } else {
                PamiIndexVerdict::NullSurvivableOnMarginalRule
            }
        }
    };
    let index_or_marginal_survived = pami_verdict != PamiIndexVerdict::Inverted;

    // §6.4: Depth variance — Poincaré radial distribution under breathing vs fixed-52.
    let std_dev = |vals: &[f64]| -> f64 {
        if vals.len() < 2 {
            return 0.0;
        }
        let mean = vals.iter().sum::<f64>() / (vals.len() as f64);
        let var = vals
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (vals.len() as f64);
        var.sqrt()
    };
    let breathing_radii: Vec<f64> = comparisons
        .iter()
        .map(|c| c.breathing_poincare_radius)
        .collect();
    let fixed52_radii: Vec<f64> = comparisons
        .iter()
        .map(|c| c.fixed52_poincare_radius)
        .collect();
    let breathing_poincare_radius_std = std_dev(&breathing_radii);
    let fixed52_poincare_radius_std = std_dev(&fixed52_radii);
    let depth_variance_passed = breathing_poincare_radius_std > fixed52_poincare_radius_std
        && breathing_poincare_radius_std >= 0.01;

    let overall_passed = answer_agreement_passed
        && economy_passed
        && index_or_marginal_survived
        && depth_variance_passed;

    Ok(BreathingFalsificationReport {
        comparisons,
        answer_agreement,
        answer_agreement_passed,
        mean_easy_depth,
        mean_hard_depth,
        economy_passed,
        pami_spearman_rho: falsifier_report.rho.unwrap_or(0.0),
        pami_p_value: falsifier_report.p.unwrap_or(1.0),
        pami_verdict,
        index_or_marginal_survived,
        breathing_poincare_radius_std,
        fixed52_poincare_radius_std,
        depth_variance_passed,
        overall_passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(label: &str, prompt: &str, completion: &str) -> NativeTextRecord {
        NativeTextRecord {
            label: label.into(),
            prompt: prompt.into(),
            completion: completion.into(),
        }
    }

    fn fit(records: &[NativeTextRecord]) -> (NativeToroidalPredictor, TrainingReceipt) {
        NativeToroidalPredictor::fit(
            NativePredictorConfig {
                layers: 2,
                ..NativePredictorConfig::default()
            },
            NativeTrainingOptions {
                epochs: 3,
                batch_size: 8,
                ..NativeTrainingOptions::default()
            },
            records,
        )
        .unwrap()
    }

    fn directory() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_DIR_ID: AtomicU64 = AtomicU64::new(0);
        let seq = NEXT_DIR_ID.fetch_add(1, Ordering::Relaxed);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "atoma-native-next-{}-{stamp}-{seq}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn next_targets_are_shifted_prompt_is_masked_and_eos_is_explicit() {
        let config = NativePredictorConfig {
            layers: 1,
            ..NativePredictorConfig::default()
        };
        let data = examples(&config, &[record("case", "A", "BC")]).unwrap();
        assert_eq!(
            data.iter()
                .map(|e| (e.previous, e.target))
                .collect::<Vec<_>>(),
            vec![(65, 66), (66, 67), (67, EOS_CLASS)]
        );
        let empty = examples(&config, &[record("empty", "", "")]).unwrap();
        assert_eq!((empty[0].previous, empty[0].target), (EOS_CLASS, EOS_CLASS));
        let doc = examples(&config, &[NativeTextRecord::document("doc", "A")]).unwrap();
        assert_eq!((doc[0].previous, doc[0].target), (EOS_CLASS, 65));
        let prefix = examples(&config, &[record("one", "A", "B")]).unwrap();
        // Changing future labels cannot change the first prediction's state.
        assert_eq!(data[0].context, prefix[0].context);
        assert_eq!(data[1].context, prefix[1].context);
    }

    #[test]
    fn causal_features_stream_identically_and_depend_on_history() {
        let config = NativePredictorConfig::default();
        let mut whole = NativeContext::new(&config).unwrap();
        whole.prefill("ab🐉z").unwrap();
        let mut chunks = NativeContext::new(&config).unwrap();
        chunks.prefill("ab").unwrap();
        chunks.prefill("🐉").unwrap();
        chunks.prefill("z").unwrap();
        assert_eq!(whole.features().unwrap(), chunks.features().unwrap());
        let mut other = NativeContext::new(&config).unwrap();
        other.prefill("xy🐉z").unwrap();
        assert_eq!(whole.previous, other.previous);
        assert_ne!(whole.features().unwrap(), other.features().unwrap());
        assert_eq!(whole.runtime.layer_fields.len(), 52);
        assert!(
            whole
                .runtime
                .layer_fields
                .iter()
                .all(|f| f.tokens_ingested == "ab🐉z".len() + 1)
        );
        assert!(
            whole
                .runtime
                .layer_fields
                .iter()
                .all(|f| f.radius_accumulator >= 1.0 && f.cumulative_rotations > 0.0)
        );
    }

    #[test]
    fn ordered_trace_has_exact_lags_presence_and_constant_capacity() {
        let config = NativePredictorConfig {
            layers: 1,
            ..Default::default()
        };
        let mut state = NativeContext::new(&config).unwrap();
        let bos = state.history[0];
        let initial = state.features().unwrap();
        assert_eq!(&initial[SUMMARY_FEATURES..SUMMARY_FEATURES + 8], &bos);
        assert_eq!(initial[SUMMARY_FEATURES + 8], 1.0);
        assert!(initial[SUMMARY_FEATURES + 9..].iter().all(|v| *v == 0.0));
        state.push(b'a').unwrap();
        let a = state.history[0];
        state.push(b'b').unwrap();
        assert_eq!(state.history[1], a);
        assert_eq!(state.history[2], bos);
        let mut reordered = NativeContext::new(&config).unwrap();
        reordered.prefill("baz").unwrap();
        state.push(b'z').unwrap();
        assert_eq!(state.previous, reordered.previous);
        assert_ne!(
            &state.features().unwrap()[SUMMARY_FEATURES..],
            &reordered.features().unwrap()[SUMMARY_FEATURES..]
        );
        let mut disabled = NativeContext::new(&NativePredictorConfig {
            ordered_history: false,
            ..config.clone()
        })
        .unwrap();
        disabled.prefill("abz").unwrap();
        assert_eq!(
            &state.features().unwrap()[..SUMMARY_FEATURES],
            &disabled.features().unwrap()[..SUMMARY_FEATURES]
        );
        assert!(
            disabled.features().unwrap()[SUMMARY_FEATURES..]
                .iter()
                .all(|v| *v == 0.0)
        );
        for i in 0..100_000 {
            state.push((i % 256) as u8).unwrap();
        }
        assert_eq!(state.history_len, HISTORY_SLOTS);
        assert!(state.features().unwrap().iter().all(|v| v.is_finite()));
        assert!(state.runtime.layer_fields[0].radius_accumulator > 1.0);
    }

    #[test]
    fn ordered_history_learns_disjoint_prefix_mechanism_without_bigram_answer() {
        // Synthetic causal-order mechanism ONLY: never used as prose quality evidence.
        let training: Vec<_> = ('a'..='p')
            .flat_map(|noise| {
                [
                    record(&format!("{noise}-ab"), &format!("{noise}abz"), "X"),
                    record(&format!("{noise}-ba"), &format!("{noise}baz"), "Y"),
                ]
            })
            .collect();
        let (model, _) = NativeToroidalPredictor::fit(
            NativePredictorConfig {
                layers: 2,
                ..Default::default()
            },
            NativeTrainingOptions {
                epochs: 40,
                batch_size: 16,
                learning_rate: 0.01,
                ..Default::default()
            },
            &training,
        )
        .unwrap();
        let held_out = [
            record("held-ab", "7abz", "X"),
            record("held-ba", "8baz", "Y"),
        ];
        let evaluation = model.evaluate(&held_out, 4).unwrap();
        assert!(
            evaluation
                .cases
                .iter()
                .all(|case| case.exact_completion_and_eos)
        );
        assert!(
            evaluation.teacher_forced.learned.mean_nll_nats
                < evaluation.teacher_forced.bigram.mean_nll_nats
        );
        assert!(
            evaluation.teacher_forced.learned.mean_nll_nats
                < evaluation
                    .teacher_forced
                    .without_ordered_history
                    .mean_nll_nats
        );
    }

    #[test]
    fn optimization_is_deterministic_and_reduces_real_next_atom_loss() {
        // Mechanism fixture only. This is not the held-out language corpus.
        let records = [record("mechanism", "A", "B")];
        let (model, receipt) = fit(&records);
        let (repeat, _) = fit(&records);
        assert_eq!(model.weights, repeat.weights);
        assert_eq!(model.means, repeat.means);
        assert_eq!(
            model.training.dataset_sha256,
            repeat.training.dataset_sha256
        );
        assert!(receipt.final_scores.learned.mean_nll_nats < receipt.initial.learned.mean_nll_nats);
        assert!(model.training.selected_epoch > 0);
        let best_epoch_nll = receipt
            .epochs
            .iter()
            .map(|score| score.mean_nll_nats)
            .fold(receipt.initial.learned.mean_nll_nats, f64::min);
        assert_eq!(receipt.final_scores.learned.mean_nll_nats, best_epoch_nll);
        assert!(
            model
                .weights
                .chunks_exact(STRIDE)
                .any(|row| row[CONTEXT_OFFSET..].iter().any(|w| *w != 0.0))
        );
        let generated = model.generate("A", 8).unwrap();
        assert_eq!(generated.text.as_deref(), Some("B"));
        assert_eq!(generated.stop, NativeGenerationStop::Eos);
        assert_eq!(generated.eos_id, Some(NATIVE_EOS));
        assert_eq!(generated.decisions, 2);
        let capped = model.generate("A", 1).unwrap();
        assert_eq!(capped.byte_ids, vec![66]);
        assert_eq!(capped.stop, NativeGenerationStop::ByteBudget);
        assert_eq!(capped.eos_id, None);
    }

    #[test]
    fn unicode_feedback_is_real_and_truncated_utf8_is_reported() {
        let (model, _) = fit(&[record("unicode-mechanism", "say:", "🐉")]);
        let result = model.generate("say:", 12).unwrap();
        assert_eq!(result.text.as_deref(), Some("🐉"));
        assert_eq!(
            result.byte_ids,
            "🐉".bytes().map(u32::from).collect::<Vec<_>>()
        );
        assert_eq!(result.stop, NativeGenerationStop::Eos);
        let truncated = model.generate("say:", 1).unwrap();
        assert_eq!(truncated.byte_ids, vec![0xf0]);
        assert!(truncated.text.is_none() && truncated.utf8_error.is_some());
        assert_eq!(truncated.stop, NativeGenerationStop::ByteBudget);
        // Fresh requests never inherit the previous request's generated states.
        assert_eq!(
            model.generate("say:", 12).unwrap().byte_ids,
            result.byte_ids
        );
    }

    #[test]
    fn eos_is_learned_for_empty_completion_and_bos_only_inputs_work() {
        let (model, _) = fit(&[record("stop-mechanism", "done", "")]);
        let result = model.generate("done", 4).unwrap();
        assert_eq!(result.text.as_deref(), Some(""));
        assert!(result.byte_ids.is_empty());
        assert_eq!(result.stop, NativeGenerationStop::Eos);
        let (empty, _) = fit(&[record("empty-mechanism", "", "")]);
        assert_eq!(
            empty.generate("", 1).unwrap().stop,
            NativeGenerationStop::Eos
        );
    }

    #[test]
    fn held_out_labels_only_score_and_contaminated_splits_are_rejected() {
        let train = [record("train", "A", "B")];
        let (model, _) = fit(&train);
        assert!(model.evaluate(&train, 4).is_err());
        assert!(require_disjoint_corpora(&train, &[record("eval", "", "AB")]).is_err());
        let containing = [NativeTextRecord::document(
            "long",
            "a long document containing more than thirty-two bytes for evaluation",
        )];
        assert!(
            require_disjoint_corpora(
                &containing,
                &[NativeTextRecord::document(
                    "part",
                    "containing more than thirty-two bytes for evaluation"
                )]
            )
            .is_err()
        );
        let a = model.evaluate(&[record("eval-a", "A", "C")], 8).unwrap();
        let b = model.evaluate(&[record("eval-b", "A", "D")], 8).unwrap();
        assert_eq!(a.cases[0].generated.byte_ids, b.cases[0].generated.byte_ids);
        assert_ne!(
            a.teacher_forced.learned.mean_nll_nats,
            b.teacher_forced.learned.mean_nll_nats
        );
        assert_eq!(a.teacher_forced.learned.positions, 2);
        assert_eq!(a.teacher_forced.learned.eos_positions, 1);
        assert!(!a.cases[0].exact_completion_and_eos);
    }

    #[test]
    fn artifacts_round_trip_without_replacing_and_reject_corruption() {
        let (model, _) = fit(&[record("artifact-mechanism", "A", "B")]);
        let dir = directory();
        let path = dir.join("model.atoma_next");
        model.save_new(&path).unwrap();
        let original = fs::read(&path).unwrap();
        assert!(model.save_new(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
        let loaded = NativeToroidalPredictor::load(&path).unwrap();
        assert_eq!(
            model.generate("A", 8).unwrap().byte_ids,
            loaded.generate("A", 8).unwrap().byte_ids
        );
        let mut corrupt = original.clone();
        corrupt[MAGIC.len() + 15] ^= 1;
        fs::write(&path, corrupt).unwrap();
        assert!(
            NativeToroidalPredictor::load(&path)
                .unwrap_err()
                .contains("SHA-256")
        );
        for length in [0, 3, MAGIC.len(), original.len() - 1] {
            fs::write(&path, &original[..length]).unwrap();
            assert!(NativeToroidalPredictor::load(&path).is_err());
        }
        let mut trailing = original.clone();
        trailing.push(0);
        fs::write(&path, trailing).unwrap();
        assert!(NativeToroidalPredictor::load(&path).is_err());
        let mut oversized = original;
        oversized[MAGIC.len()..MAGIC.len() + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&path, oversized).unwrap();
        assert!(NativeToroidalPredictor::load(&path).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn valid_digests_cannot_bypass_schema_shape_source_or_count_validation() {
        let (model, _) = fit(&[record("schema-mechanism", "A", "B")]);
        let dir = directory();
        let path = dir.join("invalid.atoma_next");
        for mutation in 0..8 {
            let mut invalid = model.clone();
            match mutation {
                0 => invalid.source_sha256 = "0".repeat(64),
                1 => invalid.schema = "checkpoint-token-alphabet".into(),
                2 => {
                    invalid.weights.pop();
                }
                3 => invalid.scales[0] = 0.0,
                4 => invalid.unigram[0] = u64::MAX,
                5 => invalid.config.layers = 0,
                6 => invalid.training.dataset_sha256 = "0".repeat(64),
                _ => invalid.training.selected_epoch = invalid.training.options.epochs + 1,
            }
            let payload = serde_json::to_vec(&invalid).unwrap();
            let mut bytes = MAGIC.to_vec();
            bytes.extend((payload.len() as u64).to_le_bytes());
            bytes.extend(&payload);
            bytes.extend(Sha256Digest::compute(&payload).0);
            fs::write(&path, bytes).unwrap();
            assert!(
                NativeToroidalPredictor::load(&path).is_err(),
                "mutation {mutation}"
            );
            assert!(invalid.generate("A", 2).is_err());
        }
        let mut nan = model.clone();
        nan.weights[0] = f64::NAN;
        assert!(nan.validate().is_err() && nan.save_new(&path).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_inputs_are_bounded_and_rejected_before_training_or_generation() {
        let config = NativePredictorConfig::default();
        assert!(
            NativeToroidalPredictor::fit(config.clone(), NativeTrainingOptions::default(), &[])
                .is_err()
        );
        let too_long = record("long", &"a".repeat(MAX_RECORD_BYTES + 1), "");
        assert!(examples(&config, &[too_long]).is_err());
        assert!(
            examples(
                &config,
                &[record("duplicate", "a", "b"), record("duplicate", "c", "d")]
            )
            .is_err()
        );
        let (model, _) = fit(&[record("bounds-mechanism", "A", "B")]);
        assert!(model.generate("A", 0).is_err());
        assert!(model.generate("A", MAX_NEW_BYTES + 1).is_err());
        assert!(
            model
                .generate(&"a".repeat(MAX_RECORD_BYTES + 1), 1)
                .is_err()
        );
        let mut options = NativeTrainingOptions::default();
        options.learning_rate = f64::NAN;
        assert!(NativeToroidalPredictor::fit(config, options, &[record("nan", "", "")]).is_err());
        let mut context = NativeContext::new(&NativePredictorConfig::default()).unwrap();
        context.hidden[0] = f32::NAN;
        assert!(context.features().is_err());
        context.hidden[0] = f32::INFINITY;
        assert!(context.features().is_err());
        assert!(probabilities(&mut [f64::INFINITY, 0.0]).is_err());
        assert_eq!(argmax(&[1.0, 1.0, 1.0]), 0);
    }

    #[test]
    fn invalid_model_decisions_are_never_masked_repaired_or_forced_to_continue() {
        let (mut model, _) = fit(&[record("raw-output-mechanism", "A", "B")]);
        model.weights.fill(0.0);
        model.weights[0xf0 * STRIDE + 1 + usize::from(b'A')] = 100.0;
        model.weights[EOS_CLASS * STRIDE + 1 + 0xf0] = 100.0;
        let generated = model.generate("A", 4).unwrap();
        assert_eq!(generated.byte_ids, vec![0xf0]);
        assert_eq!(generated.stop, NativeGenerationStop::Eos);
        assert!(generated.text.is_none() && generated.utf8_error.is_some());
    }

    #[test]
    fn atomic_publication_has_one_winner_and_never_leaves_a_partial_output() {
        let dir = directory();
        let path = dir.join("receipt.json");
        let results: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8u8)
                .map(|id| {
                    let path = &path;
                    scope.spawn(move || (id, write_new_atomic(path, &[id; 1024]).is_ok()))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let winners: Vec<_> = results.iter().filter(|(_, won)| *won).collect();
        assert_eq!(winners.len(), 1);
        assert_eq!(fs::read(&path).unwrap(), vec![winners[0].0; 1024]);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn toroidal_carrier_transition_trains_on_vault_arcs_and_generates_without_mutating_caller() {
        let weights = atoma_immune::FrozenBaseWeights::freeze("carrier-transition-test", 52, vec![0.25; 16]);
        let mut server = crate::MonsterHybridServer::boot_default(weights).unwrap();
        let prefix = server
            .active_transformer
            .transducer
            .transduce_ingress_text("The capital of France is", true);
        let cont = server
            .active_transformer
            .transducer
            .transduce_ingress_text(" Paris", false);
        atoma_immune::NativeReader::record_continuation(
            &mut server.swarm_map,
            &server.mlir_catalog,
            &server.immune.policy,
            &prefix,
            &cont,
            &[],
            server.active_transformer.origin_fingerprint,
            atoma_immune::geometry::fixed::Q32_31::ONE,
        )
        .unwrap();

        let (model, train_receipt) = NativeCarrierTransitionModel::fit_from_swarm_map(
            &server.swarm_map,
            &server.mlir_catalog,
            4,
            8,
        )
        .unwrap();
        assert_eq!(train_receipt.total_vault_arcs, cont.len());
        assert_eq!(train_receipt.carrier_continuation_arcs, cont.len());
        assert!(train_receipt.final_coord_mse < train_receipt.initial_coord_mse);

        let mut caller_rt = ToroidalIsotropicRuntime::full_stack(6, 52).unwrap();
        let emb = crate::input_embeddings::embed_utf8_text(
            &server.active_transformer.transducer,
            "The capital of France is",
            false,
        )
        .unwrap();
        caller_rt.step_native_text_sequence(&emb).unwrap();
        let tokens_before = caller_rt.layer_fields[0].tokens_ingested;

        let candidate = generate_toroidal_carrier_continuation_with_model(
            &model,
            &caller_rt,
            &server.swarm_map,
            &server.mlir_catalog,
            &server.active_transformer.transducer,
            "The capital of France is",
            Some(&prefix),
            cont.len(),
        )
        .unwrap();

        // Caller's session runtime must never be mutated by candidate generation
        assert_eq!(caller_rt.layer_fields[0].tokens_ingested, tokens_before);
        assert!(candidate.is_continuation);
        assert_eq!(candidate.emitted_carriers, cont);
        assert_eq!(candidate.decoded_text.as_deref(), Some(" Paris"));
        assert!(candidate.decode_error.is_none());
        assert_eq!(candidate.vault_arc_coverage, 1.0);
        assert_eq!(candidate.steps.len(), cont.len());
        assert!(candidate.mean_log_prob.is_finite() && candidate.mean_log_prob <= 0.0);
        assert!(candidate.mean_z_surprise.is_finite() && candidate.mean_z_surprise >= 0.0);
        assert!(candidate.max_z_surprise >= candidate.mean_z_surprise);
        assert!(candidate.steps.iter().all(|s| s.z_surprise.is_finite() && s.z_surprise >= 0.0));

        // Verify candidate arrives with an unscored gate_score (no gate decisions inside predictor),
        // and feeds TrajectoryMatchGate cleanly when scored by the turn caller.
        assert_eq!(
            candidate.gate_score.verdict,
            crate::predict_gate::GateVerdict::Cold
        );
        assert_eq!(candidate.gate_score.scored_steps, 0);
        assert!(candidate.gate_score.max_surprise.is_none());
        assert!(candidate.gate_score.mean_surprise.is_none());

        let mut shared_gate = TrajectoryMatchGate::new();
        let gate_score = candidate.score_with_gate(&mut shared_gate);
        assert_eq!(gate_score.scored_steps, candidate.emitted_embeddings.len());
        assert!(
            (gate_score.max_surprise.unwrap_or(0.0) - candidate.max_z_surprise).abs() < 1e-12
        );
    }

    #[test]
    fn native_carrier_promoter_fits_caches_and_promotes_unscored_candidate() {
        let scratch = directory();
        let sidecar_path = scratch.join("test_vault.atoma_map.carrier_promoter.json");

        let weights =
            atoma_immune::FrozenBaseWeights::freeze("carrier-promoter-test", 52, vec![0.25; 16]);
        let mut server = crate::MonsterHybridServer::boot_default(weights).unwrap();
        let prefix = server
            .active_transformer
            .transducer
            .transduce_ingress_text("The capital of France is", true);
        let cont = server
            .active_transformer
            .transducer
            .transduce_ingress_text(" Paris", false);
        atoma_immune::NativeReader::record_continuation(
            &mut server.swarm_map,
            &server.mlir_catalog,
            &server.immune.policy,
            &prefix,
            &cont,
            &[],
            server.active_transformer.origin_fingerprint,
            atoma_immune::geometry::fixed::Q32_31::ONE,
        )
        .unwrap();

        // 1. Fit once on boot and persist to disk sidecar
        let mut promoter = NativeCarrierPromoter::load_or_fit(
            &server.swarm_map,
            &server.mlir_catalog,
            Some(&sidecar_path),
        );
        assert!(promoter.fitted);
        assert!(!promoter.loaded_from_disk);
        assert!(sidecar_path.is_file());

        // 2. Second boot loads directly from disk sidecar without re-fitting
        let mut reloaded = NativeCarrierPromoter::load_or_fit(
            &server.swarm_map,
            &server.mlir_catalog,
            Some(&sidecar_path),
        );
        assert!(reloaded.fitted);
        assert!(reloaded.loaded_from_disk);

        // 3. Simulate post-`step_native_toroidal_prompt` session hidden sequence
        let mut session_rt = ToroidalIsotropicRuntime::full_stack(6, 52).unwrap();
        let emb = crate::input_embeddings::embed_utf8_text(
            &server.active_transformer.transducer,
            "The capital of France is",
            false,
        )
        .unwrap();
        let hidden_sequence = session_rt.step_native_text_sequence(&emb).unwrap();

        // Promote from `&[Vec<f32>]` (post-`step_native_toroidal_prompt` sequence)
        let cand_from_seq = promoter
            .promote_with_prefix(
                hidden_sequence.as_slice(),
                &server.swarm_map,
                &server.mlir_catalog,
                &server.active_transformer.transducer,
                "The capital of France is",
                Some(&prefix),
                cont.len(),
            )
            .expect("promoter should return a continuation candidate");
        assert_eq!(cand_from_seq.decoded_text.as_deref(), Some(" Paris"));
        assert_eq!(cand_from_seq.emitted_carriers, cont);
        assert!(!cand_from_seq.prompt_embeddings.is_empty());
        assert_eq!(cand_from_seq.emitted_embeddings.len(), cont.len());
        assert_eq!(
            cand_from_seq.gate_score.verdict,
            crate::predict_gate::GateVerdict::Cold
        );
        assert_eq!(cand_from_seq.gate_score.scored_steps, 0);
        assert!(cand_from_seq.gate_score.max_surprise.is_none());

        // Promote from `Option<&ToroidalIsotropicRuntime>` on reloaded promoter
        let cand_from_rt = reloaded
            .promote_with_prefix(
                Some(&session_rt),
                &server.swarm_map,
                &server.mlir_catalog,
                &server.active_transformer.transducer,
                "The capital of France is",
                Some(&prefix),
                cont.len(),
            )
            .expect("reloaded promoter should return a continuation candidate");
        assert_eq!(cand_from_rt.decoded_text.as_deref(), Some(" Paris"));
        assert_eq!(
            cand_from_rt.gate_score.verdict,
            crate::predict_gate::GateVerdict::Cold
        );
        assert_eq!(cand_from_rt.gate_score.scored_steps, 0);
        fs::remove_dir_all(&scratch).unwrap();
    }

    #[test]
    fn teacher_era_wal_frames_including_two_plus_two_are_rejected_from_train_and_eval() {
        let wal_frames = default_workspace_wal_trajectories();
        if wal_frames.is_empty() {
            return;
        }
        assert!(wal_frames.len() >= 10);
        // Frame 3 is the exact `2+2` teacher contamination proof
        let frame3 = &wal_frames[3];
        assert_eq!(frame3.prompt_text, "What is two plus two");
        assert!(
            frame3
                .emitted_text
                .contains("a system of recording financial transactions"),
            "unexpected frame 3 emitted text: {}",
            frame3.emitted_text
        );
        assert_eq!(
            frame3.origin_fingerprint_hex,
            TEACHER_ACTIVE_TRANSFORMER_ORIGIN_HEX
        );

        let contaminated_record = record(
            "contaminated-2plus2",
            &frame3.prompt_text,
            &frame3.emitted_text,
        );
        let clean_record = record(
            "clean-record",
            "The toroidal recurrence steps on",
            " H^3 x T^2.",
        );
        let err = require_disjoint_corpora(&[contaminated_record.clone()], &[clean_record.clone()])
            .unwrap_err();
        assert!(
            err.contains("teacher-contaminated WAL frame"),
            "unexpected error: {err}"
        );
        let eval_err =
            require_disjoint_corpora(&[clean_record], &[contaminated_record]).unwrap_err();
        assert!(
            eval_err.contains("teacher-contaminated WAL frame"),
            "unexpected error: {eval_err}"
        );
    }

    #[test]
    fn real_104538_vault_arcs_extract_and_train_toroidal_carrier_transition() {
        let vault_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../design/global_active_mind.atoma_map");
        if !vault_path.is_file() {
            return;
        }
        let atlas = atoma_immune::Atlas13::approved_icosahedral();
        let policy = atoma_immune::OverlapPolicyMatrix::minimal_icosahedral(
            &atlas,
            atoma_immune::geometry::fixed::Q32_31::from_f64(0.08),
        );
        let plug = atoma_immune::FrozenBaseWeights::freeze("vault-plug", 52, vec![0.0; 8]);
        let mut map = atoma_immune::CarvedSwarmMap::new(
            atlas,
            plug,
            atoma_immune::geometry::fixed::Q32_31::from_f64(1.22),
        );
        let (_atoms, arcs, _sha) = map.load_from_binary_vault(&vault_path, &policy).unwrap();
        assert!(arcs >= 104_538, "vault arcs {} below floor 104,538", arcs);

        let catalog = atoma_immune::MlirAtomCatalog::new_canonical(248_320);
        let dataset = extract_vault_continuation_dataset(&map, &catalog).unwrap();
        assert_eq!(dataset.total_vault_arcs, arcs);
        assert!(dataset.total_vault_arcs >= 104_538);
        assert!(dataset.carrier_continuation_arcs >= 104_511);
        assert!(dataset.clean_continuation_arcs >= 104_000);
        assert!(dataset.teacher_contaminated_arcs >= 280);
        assert!(dataset.wal_contaminated_frames >= 10);
        assert!(dataset.structural_anchor_arcs >= 27);
        assert_eq!(
            dataset.carrier_continuation_arcs + dataset.structural_anchor_arcs,
            dataset.total_vault_arcs,
            "carrier_continuation_arcs + structural_anchor_arcs must equal total_vault_arcs"
        );
        assert_eq!(
            dataset.clean_continuation_arcs + dataset.teacher_contaminated_arcs,
            dataset.carrier_continuation_arcs,
            "clean_continuation_arcs + teacher_contaminated_arcs must equal carrier_continuation_arcs"
        );
        assert!(dataset.contiguous_step_pairs >= 97_188);
        assert!(dataset.branching_source_atoms >= 4_942);
        assert!(dataset.branching_arcs >= 96_254);
        assert!(dataset.unique_target_carriers >= 8_482);

        let (_model, receipt) =
            NativeCarrierTransitionModel::fit_from_swarm_map(&map, &catalog, 2, 3).unwrap();
        println!(
            "[verified 104,538 vault arcs] carrier_arcs={} clean_arcs={} filtered_teacher_arcs={} filtered_wal_frames={} chains={} contiguous_pairs={} branching_atoms={} branching_arcs={} unique_carriers={} coord_mse={:.6}->{:.6} branch_nll={:.6}->{:.6} branch_top1={:.4}->{:.4} mean_z_surprise={:.4}->{:.4} max_z_surprise={:.4} elapsed_ms={}",
            receipt.carrier_continuation_arcs,
            receipt.clean_continuation_arcs,
            receipt.teacher_contaminated_arcs_filtered,
            receipt.wal_contaminated_frames_filtered,
            receipt.trajectory_chains,
            receipt.contiguous_step_pairs,
            receipt.branching_source_atoms,
            receipt.branching_arcs,
            receipt.unique_target_carriers,
            receipt.initial_coord_mse,
            receipt.final_coord_mse,
            receipt.initial_branch_nll_nats,
            receipt.final_branch_nll_nats,
            receipt.initial_branch_top1_acc,
            receipt.final_branch_top1_acc,
            receipt.initial_mean_z_surprise,
            receipt.final_mean_z_surprise,
            receipt.final_max_z_surprise,
            receipt.elapsed_ms,
        );
        assert_eq!(
            receipt.clean_continuation_arcs + receipt.teacher_contaminated_arcs_filtered,
            dataset.carrier_continuation_arcs
        );
        assert!(receipt.teacher_contaminated_arcs_filtered >= 280);
        assert!(receipt.wal_contaminated_frames_filtered >= 10);
        assert!(
            receipt.final_coord_mse < receipt.initial_coord_mse,
            "trained coordinate MSE {:.6} must improve over initial {:.6}",
            receipt.final_coord_mse,
            receipt.initial_coord_mse
        );
        assert!(
            receipt.final_branch_nll_nats < receipt.initial_branch_nll_nats,
            "trained branching NLL {:.6} must improve over initial {:.6}",
            receipt.final_branch_nll_nats,
            receipt.initial_branch_nll_nats
        );
        assert!(
            receipt.final_branch_top1_acc > receipt.initial_branch_top1_acc,
            "trained branching top-1 accuracy {:.4} must improve over initial {:.4}",
            receipt.final_branch_top1_acc,
            receipt.initial_branch_top1_acc
        );
        assert!(receipt.final_mean_z_surprise.is_finite() && receipt.final_mean_z_surprise > 0.0);
    }

    #[test]
    #[test]
    fn probe_trace_separates_semantic_roles() {
        // 2026-10-20 probe (Stewart's hypothesis): does the encoder's
        // geometry separate semantic ROLES? Roles fixed in advance:
        // outflow verbs, inflow verbs, rate markers, neutral controls.
        // Two levels: (a) canonical piece wells (embedding coords),
        // (b) propagated trace (final hidden state of full sentences).
        let weights =
            atoma_immune::FrozenBaseWeights::freeze("role-probe", 52, vec![0.25; 16]);
        let server = crate::MonsterHybridServer::boot_default(weights).unwrap();
        let transducer = &server.active_transformer.transducer;
        let catalog = &server.mlir_catalog;

        let outflow = ["sold", "gave", "spent", "lost", "ate", "donated", "paid"];
        let inflow = ["bought", "earned", "received", "found", "made", "won"];
        let rates = ["per", "each", "every"];
        let neutral = ["saw", "went", "sat", "stood"];

        // (a) well coordinates of each single word piece (first carrier).
        let well = |w: &str| -> Option<[f32; 6]> {
            let embs = crate::input_embeddings::embed_text_macro(transducer, w).ok()?;
            let _ = catalog;
            embs.first().map(|e| e.coordinates)
        };
        let dist6 = |a: &[f32; 6], b: &[f32; 6]| -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let phase_of = |c: &[f32; 6]| -> (f64, f64) {
            ((c[4] as f64).atan2(c[5] as f64), 0.0)
        };
        let mean_pairwise = |words: &[&str]| -> (f64, usize) {
            let wells: Vec<[f32; 6]> = words.iter().filter_map(|w| well(w)).collect();
            let mut sum = 0.0;
            let mut n = 0usize;
            for i in 0..wells.len() {
                for j in (i + 1)..wells.len() {
                    sum += dist6(&wells[i], &wells[j]);
                    n += 1;
                }
            }
            (if n > 0 { sum / n as f64 } else { f64::NAN }, n)
        };
        let mean_between = |a: &[&str], b: &[&str]| -> (f64, usize) {
            let wa: Vec<[f32; 6]> = a.iter().filter_map(|w| well(w)).collect();
            let wb: Vec<[f32; 6]> = b.iter().filter_map(|w| well(w)).collect();
            let mut sum = 0.0;
            let mut n = 0usize;
            for x in &wa {
                for y in &wb {
                    sum += dist6(x, y);
                    n += 1;
                }
            }
            (if n > 0 { sum / n as f64 } else { f64::NAN }, n)
        };
        let (within_out, _) = mean_pairwise(&outflow);
        let (within_in, _) = mean_pairwise(&inflow);
        let (between_io, _) = mean_between(&outflow, &inflow);
        let (verbs_vs_rates, _) = mean_between(&outflow, &rates);
        let (within_neutral, _) = mean_pairwise(&neutral);
        let (neutral_vs_out, _) = mean_between(&neutral, &outflow);
        println!("[probe-wells] within_outflow={within_out:.4} within_inflow={within_in:.4} between_in_out={between_io:.4} verbs_vs_rates={verbs_vs_rates:.4} within_neutral={within_neutral:.4} neutral_vs_outflow={neutral_vs_out:.4}");

        // Phase displacement between role centroids (antipode claim: ~pi).
        let centroid = |words: &[&str]| -> Option<[f64; 6]> {
            let wells: Vec<[f32; 6]> = words.iter().filter_map(|w| well(w)).collect();
            if wells.is_empty() {
                return None;
            }
            let mut c = [0.0f64; 6];
            for w in &wells {
                for k in 0..6 {
                    c[k] += w[k] as f64;
                }
            }
            for k in 0..6 {
                c[k] /= wells.len() as f64;
            }
            Some(c)
        };
        if let (Some(co), Some(ci)) = (centroid(&outflow), centroid(&inflow)) {
            let d_phase = ((co[4] - ci[4]).powi(2) + (co[5] - ci[5]).powi(2)).sqrt();
            println!("[probe-wells] role centroid phase-plane distance={d_phase:.4} (toroidal coords 4,5 raw delta)");
            let _ = phase_of;
        }

        // (b) trace level: full sentences differing only in the verb.
        let trace_final = |sentence: &str| -> Option<Vec<f32>> {
            let embs =
                crate::input_embeddings::embed_text_macro(transducer, sentence).ok()?;
            let mut rt =
                crate::toroidal_attention::ToroidalIsotropicRuntime::full_stack(6, 52).ok()?;
            let states = rt.step_native_text_sequence(&embs).ok()?;
            states.into_iter().last()
        };
        let dv = |a: &Vec<f32>, b: &Vec<f32>| -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let s_sold = trace_final("The baker sold 49 muffins").unwrap();
        let s_gave = trace_final("The baker gave 49 muffins").unwrap();
        let s_bought = trace_final("The baker bought 49 muffins").unwrap();
        let s_earned = trace_final("The baker earned 49 muffins").unwrap();
        let s_baker2 = trace_final("The baker sold 49 cakes").unwrap();
        println!(
            "[probe-trace] d(sold,gave)[same role]={:.4} d(sold,bought)[role flip]={:.4} d(sold,earned)[role flip]={:.4} d(sold,cakes)[same role, object change]={:.4}",
            dv(&s_sold, &s_gave),
            dv(&s_sold, &s_bought),
            dv(&s_sold, &s_earned),
            dv(&s_sold, &s_baker2)
        );
    }

    #[test]
    fn breathing_depth_decision_ignores_decorative_narrative() {
        // 2026-10-20 (Stewart): PAMI depth analysis runs on the
        // relational backbone only. The same arithmetic dressed in
        // decorative narrative must construct the SAME depth as the
        // bare question — while the old full-text analysis reads the
        // narrative as roughness and constructs (weakly) more.
        let weights =
            atoma_immune::FrozenBaseWeights::freeze("backbone-test", 52, vec![0.25; 16]);
        let server = crate::MonsterHybridServer::boot_default(weights).unwrap();
        let transducer = &server.active_transformer.transducer;
        let cfg = crate::toroidal_attention::BreathingDepthConfig::default();

        let bare = "What is 17 times 6?";
        let decorated = "In the warm hazy golden afternoon light of a quaint             little village bakery, with dust motes drifting slowly through             the sunbeams, what is 17 times 6?";
        let bare_embs =
            crate::input_embeddings::embed_text_macro(transducer, bare).unwrap();
        let decor_embs =
            crate::input_embeddings::embed_text_macro(transducer, decorated).unwrap();
        let bare_bb = crate::input_embeddings::backbone_filter(transducer, &bare_embs);
        let decor_bb = crate::input_embeddings::backbone_filter(transducer, &decor_embs);
        // The decorated prompt's backbone is the bare prompt's backbone.
        assert_eq!(bare_bb.len(), decor_bb.len());

        let depth_of = |embs: &[crate::input_embeddings::NativeMlirEmbedding],
                        bb: Option<&[crate::input_embeddings::NativeMlirEmbedding]>| {
            let mut rt =
                crate::toroidal_attention::ToroidalIsotropicRuntime::breathing_scaffold(6, 52)
                    .unwrap();
            rt.step_breathing_text_sequence_backbone(embs, &cfg, bb)
                .unwrap()
                .constructed_depth
        };
        let bare_depth = depth_of(&bare_embs, Some(&bare_bb));
        let decor_depth_bb = depth_of(&decor_embs, Some(&decor_bb));
        assert_eq!(
            bare_depth, decor_depth_bb,
            "narrative changed the depth decision despite the backbone filter"
        );
        let decor_depth_full = depth_of(&decor_embs, None);
        assert!(
            decor_depth_full >= decor_depth_bb,
            "full-text analysis should not read LESS depth than the backbone"
        );
    }

    fn breathing_depth_construction_depth_conditional_readout_binary_sidecar_and_falsification_ladder_hold() {
        let weights =
            atoma_immune::FrozenBaseWeights::freeze("breathing-test", 52, vec![0.25; 16]);
        let mut server = crate::MonsterHybridServer::boot_default(weights).unwrap();
        let clean_origin = Sha256Digest::compute(b"unit-test-clean-origin");
        // Seed trajectories matching the frozen easy -> hard falsification ladder so both
        // Breathing Construction and Fixed-52 can evaluate agreement, economy, PAMI index
        // correlation, and Poincaré radial depth spread.
        let ladder = preregistered_breathing_prompt_ladder();
        let completions = [
            " Paris.",
            " 4.",
            " 0 C.",
            " east.",
            " 8 bits.",
            " x.",
            " decreases along the coupled harmonic gradient.",
            " validate cold trajectory surprise before emission.",
            " expand across all eight phase sectors.",
            " no intermediate inner product overflows i64.",
            " construction halts as soon as marginal gain saturates.",
            " any spurious complexity index against the null.",
        ];
        for (item, comp) in ladder.iter().zip(completions.iter()) {
            let prefix = server
                .active_transformer
                .transducer
                .transduce_ingress_text(&item.prompt, true);
            let cont = server
                .active_transformer
                .transducer
                .transduce_ingress_text(comp, false);
            atoma_immune::NativeReader::record_continuation(
                &mut server.swarm_map,
                &server.mlir_catalog,
                &server.immune.policy,
                &prefix,
                &cont,
                &[],
                clean_origin,
                atoma_immune::geometry::fixed::Q32_31::ONE,
            )
            .unwrap();
        }

        // 1. Fit depth-conditional promoter and verify binary sidecar persistence (`ATOMA_CP_V2\0`)
        let scratch = directory();
        let fake_vault_path = scratch.join("breathing_vault.atoma_map");
        let bin_sidecar = NativeCarrierPromoter::sidecar_path_for_vault(&fake_vault_path);
        assert!(
            bin_sidecar.to_string_lossy().ends_with(".carrier_promoter.bin"),
            "default sidecar must be binary (.carrier_promoter.bin), got {}",
            bin_sidecar.display()
        );

        let mut promoter = NativeCarrierPromoter::load_or_fit(
            &server.swarm_map,
            &server.mlir_catalog,
            Some(&bin_sidecar),
        );
        assert!(promoter.fitted);
        assert!(!promoter.loaded_from_disk);
        assert_eq!(promoter.model.per_depth_readouts.len(), DEPTH_BAND_SPECS.len());
        assert!(bin_sidecar.is_file());
        let raw_bytes = fs::read(&bin_sidecar).unwrap();
        assert!(
            raw_bytes.starts_with(CARRIER_PROMOTER_BINARY_MAGIC),
            "sidecar file must start with binary magic ATOMA_CP_V2\\0"
        );

        // Reload from binary sidecar and verify lossless round-trip
        let reloaded = NativeCarrierPromoter::load_or_fit(
            &server.swarm_map,
            &server.mlir_catalog,
            Some(&bin_sidecar),
        );
        assert!(reloaded.fitted);
        assert!(reloaded.loaded_from_disk);
        assert_eq!(
            reloaded.model.per_depth_readouts.len(),
            promoter.model.per_depth_readouts.len()
        );

        // 2. Verify promoted candidate records breathing depth, rank, PAMI spread, and Poincaré radius
        let easy_prefix = server
            .active_transformer
            .transducer
            .transduce_utf8_to_mlir(&ladder[0].prompt, false);
        let mut easy_embs = Vec::new();
        for &op in &easy_prefix {
            easy_embs.push(embed_carrier(&server.mlir_catalog, op).unwrap());
        }
        let mut session_rt = ToroidalIsotropicRuntime::full_stack(6, 2).unwrap();
        let mut hidden_seq = Vec::new();
        for emb in &easy_embs {
            hidden_seq.push(
                session_rt
                    .step_toroidal_attention(&emb.coordinates)
                    .unwrap(),
            );
        }
        let cand_easy = promoter
            .promote_with_prefix(
                hidden_seq.as_slice(),
                &server.swarm_map,
                &server.mlir_catalog,
                &server.active_transformer.transducer,
                &ladder[0].prompt,
                Some(&easy_prefix),
                6,
            )
            .expect("easy prompt should promote");
        assert!((1..52).contains(&cand_easy.constructed_depth));
        assert!((1..=8).contains(&cand_easy.constructed_rank));
        assert!((1..=8).contains(&cand_easy.active_attention_sectors));
        assert!(cand_easy.pami_spectral_spread.is_some_and(|s| s.is_finite()));
        assert!(cand_easy.poincare_radius > 0.0 && cand_easy.poincare_radius < 1.0);
        assert_eq!(
            cand_easy.gate_score.verdict,
            crate::predict_gate::GateVerdict::Cold
        );
        assert_eq!(cand_easy.gate_score.scored_steps, 0);

        // 3. Evaluate the 4-part pre-registered falsification ladder (`docs/BREATHING_DEPTH_DESIGN.md` §6)
        let report = evaluate_breathing_depth_falsification_ladder(
            &promoter.model,
            &server.swarm_map,
            &server.mlir_catalog,
            &server.active_transformer.transducer,
            &promoter.breathing_config,
            &ladder,
            6,
        )
        .unwrap();

        println!(
            "[breathing falsification ladder] agreement={:.3} (pass={}) easy_depth={:.2} hard_depth={:.2} (economy_pass={}) pami_rho={:.4} p={:.4} verdict={:?} poincare_std={:.4} vs fixed52={:.4} (var_pass={}) overall={}",
            report.answer_agreement,
            report.answer_agreement_passed,
            report.mean_easy_depth,
            report.mean_hard_depth,
            report.economy_passed,
            report.pami_spearman_rho,
            report.pami_p_value,
            report.pami_verdict,
            report.breathing_poincare_radius_std,
            report.fixed52_poincare_radius_std,
            report.depth_variance_passed,
            report.overall_passed,
        );

        assert!(
            report.answer_agreement_passed,
            "§6.1 answer agreement {:.3} below 0.85 floor",
            report.answer_agreement
        );
        assert!(
            report.economy_passed && report.mean_easy_depth < report.mean_hard_depth,
            "§6.2 economy failed: mean_easy_depth={:.2}, mean_hard_depth={:.2}",
            report.mean_easy_depth,
            report.mean_hard_depth
        );
        assert!(
            report.index_or_marginal_survived,
            "§6.3 index/marginal survival failed: verdict={:?}",
            report.pami_verdict
        );
        assert!(
            report.depth_variance_passed,
            "§6.4 depth variance failed: breathing_std={:.4} vs fixed52_std={:.4}",
            report.breathing_poincare_radius_std,
            report.fixed52_poincare_radius_std
        );
        assert!(report.overall_passed);

        // 4. Verify NULL-on-the-index survivability (`use_pami_forecast = false`):
        //    Construction still halts cleanly on the marginal free-energy brake alone.
        let mut marginal_only_cfg = promoter.breathing_config.clone();
        marginal_only_cfg.use_pami_forecast = false;
        let marginal_report = evaluate_breathing_depth_falsification_ladder(
            &promoter.model,
            &server.swarm_map,
            &server.mlir_catalog,
            &server.active_transformer.transducer,
            &marginal_only_cfg,
            &ladder,
            6,
        )
        .unwrap();
        assert!(marginal_report.answer_agreement_passed);
        assert!(marginal_report.economy_passed);
        assert!(marginal_report.index_or_marginal_survived);

        fs::remove_dir_all(&scratch).unwrap();
    }
}
