//! Phase B native producer (`docs/NATIVE_NEXT_ATOM_DESIGN.md`): a trained causal
//! byte/EOS readout over the fixed toroidal recurrence. No teacher or tokenizer.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use atoma_immune::refraction::Sha256Digest;
use atoma_immune::{BoundaryLexicalTransducer, MlirOpId};
use serde::{Deserialize, Serialize};

use crate::input_embeddings::embed_carrier;
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
/// of at least 32 bytes. Shared ordinary n-grams are not prohibited.
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
        state.prefill(&record.prompt)?;
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
            });
            if target != EOS_CLASS {
                state.push(target as u8)?;
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
            scores
                .learned
                .record(sample.target, argmax(&learned), learned_nll);
            scores
                .without_toroidal_features
                .record(sample.target, argmax(&ablated), ablated_nll);
            scores.without_ordered_history.record(
                sample.target,
                argmax(&summary_only),
                summary_nll,
            );
            scores.unigram.record(
                sample.target,
                argmax(&unigram),
                -unigram[sample.target].ln(),
            );
            scores.bigram.record(
                sample.target,
                argmax(&bigram[sample.previous]),
                -bigram[sample.previous][sample.target].ln(),
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
        let prefill_us = started.elapsed().as_micros();
        let decode_started = Instant::now();
        let mut bytes = Vec::new();
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
            bytes.push(class as u8);
            state.push(class as u8)?;
        }
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
}

impl PredictionScore {
    fn record(&mut self, target: usize, predicted: usize, nll: f64) {
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
    }
    fn finish(&mut self) {
        self.accuracy = self.correct as f64 / self.positions.max(1) as f64;
        self.byte_accuracy = self.bytes_correct as f64 / self.byte_positions.max(1) as f64;
        self.eos_recall = self.eos_correct as f64 / self.eos_positions.max(1) as f64;
        self.mean_nll_nats /= self.positions.max(1) as f64;
        self.bits_per_position = self.mean_nll_nats / std::f64::consts::LN_2;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("atoma-native-next-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
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
}
