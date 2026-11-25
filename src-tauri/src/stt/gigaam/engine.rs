use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::{Tensor, TensorElementType, ValueType};

use super::features::GigaAmFeatures;
use super::model::{
    GigaAmConfig, GigaAmModel, SentencePieceTokenizer, ENCODER_HIDDEN, PREDICTOR_HIDDEN,
};

const MAX_SYMBOLS_PER_FRAME: usize = 3;

#[derive(Debug, Clone)]
pub struct GigaAmRecognition {
    pub text: String,
    pub confidence: f32,
}

pub struct GigaAmEngine {
    inner: Mutex<EngineInner>,
    features: GigaAmFeatures,
    tokenizer: SentencePieceTokenizer,
    config: GigaAmConfig,
}

struct EngineInner {
    encoder: Session,
    decoder: Session,
    joint: Session,
}

impl GigaAmEngine {
    pub fn load(model: GigaAmModel) -> Result<Self, String> {
        crate::ort_runtime::ensure_loaded()?;
        for warning in &model.warnings {
            log::warn!("{warning}");
        }
        let encoder = load_session(&model.encoder_path)?;
        let decoder = load_session(&model.decoder_path)?;
        let joint = load_session(&model.joint_path)?;
        validate_session(
            &encoder,
            &[
                TensorSpec::new("audio_signal", TensorElementType::Float32, &[-1, 64, -1]),
                TensorSpec::new("length", TensorElementType::Int64, &[-1]),
            ],
            &[
                TensorSpec::new("encoded", TensorElementType::Float32, &[-1, 768, -1]),
                TensorSpec::new("encoded_len", TensorElementType::Int32, &[-1]),
            ],
        )?;
        validate_session(
            &decoder,
            &[
                TensorSpec::new("x", TensorElementType::Int64, &[-1, 1]),
                TensorSpec::new("hi", TensorElementType::Float32, &[1, -1, 320]),
                TensorSpec::new("ci", TensorElementType::Float32, &[1, -1, 320]),
            ],
            &[
                TensorSpec::new("dec", TensorElementType::Float32, &[-1, 1, 320]),
                TensorSpec::new("ho", TensorElementType::Float32, &[1, -1, 320]),
                TensorSpec::new("co", TensorElementType::Float32, &[1, -1, 320]),
            ],
        )?;
        validate_session(
            &joint,
            &[
                TensorSpec::new("enc", TensorElementType::Float32, &[-1, 768, 1]),
                TensorSpec::new("dec", TensorElementType::Float32, &[-1, 320, 1]),
            ],
            &[TensorSpec::new(
                "joint",
                TensorElementType::Float32,
                &[-1, 1, 1, model.config.num_classes as i64],
            )],
        )?;
        Ok(Self {
            inner: Mutex::new(EngineInner {
                encoder,
                decoder,
                joint,
            }),
            features: GigaAmFeatures::new(),
            tokenizer: model.tokenizer,
            config: model.config,
        })
    }

    pub fn recognize(&self, samples: &[f32]) -> Result<GigaAmRecognition, String> {
        let features = self.features.extract(samples)?;
        let frames = features.shape()[2];
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| "GigaAM engine lock poisoned")?;
        let audio =
            Tensor::from_array(features).map_err(|e| format!("GigaAM feature tensor: {e}"))?;
        let length = Tensor::from_array(Array1::from_vec(vec![frames as i64]))
            .map_err(|e| format!("GigaAM length tensor: {e}"))?;
        let encoded_index = output_index(&inner.encoder, "encoded")?;
        let length_index = output_index(&inner.encoder, "encoded_len")?;
        let (encoded, encoded_time, encoded_frames) = {
            let encoder_outputs = inner
                .encoder
                .run(ort::inputs![
                    "audio_signal" => audio,
                    "length" => length,
                ])
                .map_err(|e| format!("GigaAM encoder inference: {e}"))?;
            let (encoded_shape, encoded) = encoder_outputs[encoded_index]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("GigaAM encoder output: {e}"))?;
            let encoded_time = validate_encoder_shape(&encoded_shape, encoded.len())?;
            let (length_shape, encoded_len) = encoder_outputs[length_index]
                .try_extract_tensor::<i32>()
                .map_err(|e| format!("GigaAM encoder length: {e}"))?;
            let encoded_frames = validate_encoder_length(&length_shape, encoded_len, encoded_time)?;
            (encoded.to_vec(), encoded_time, encoded_frames)
        };

        let blank = self.tokenizer.blank_id();
        let mut last_label = blank;
        let mut hidden = Array3::<f32>::zeros((self.config.predictor_layers, 1, PREDICTOR_HIDDEN));
        let mut cell = Array3::<f32>::zeros((self.config.predictor_layers, 1, PREDICTOR_HIDDEN));
        let mut tokens = Vec::new();
        let mut confidence_sum = 0.0f32;
        let mut confidence_steps = 0usize;

        for t in 0..encoded_frames {
            for _ in 0..MAX_SYMBOLS_PER_FRAME {
                let label = Tensor::from_array(
                    Array2::from_shape_vec((1, 1), vec![last_label as i64]).unwrap(),
                )
                .map_err(|e| format!("GigaAM label tensor: {e}"))?;
                let hidden_tensor = Tensor::from_array(hidden.clone())
                    .map_err(|e| format!("GigaAM hidden tensor: {e}"))?;
                let cell_tensor = Tensor::from_array(cell.clone())
                    .map_err(|e| format!("GigaAM cell tensor: {e}"))?;
                let dec_index = output_index(&inner.decoder, "dec")?;
                let hidden_index = output_index(&inner.decoder, "ho")?;
                let cell_index = output_index(&inner.decoder, "co")?;
                let (decoder_frame, next_hidden, next_cell) = {
                    let decoder_outputs = inner
                        .decoder
                        .run(ort::inputs![
                            "x" => label,
                            "hi" => hidden_tensor,
                            "ci" => cell_tensor,
                        ])
                        .map_err(|e| format!("GigaAM predictor inference: {e}"))?;
                    let (dec_shape, dec) = decoder_outputs[dec_index]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| format!("GigaAM predictor output: {e}"))?;
                    let decoder_frame = predictor_frame(&dec_shape, dec)?;
                    let next_hidden = extract_state(
                        &decoder_outputs[hidden_index],
                        "hidden",
                        self.config.predictor_layers,
                    )?;
                    let next_cell = extract_state(
                        &decoder_outputs[cell_index],
                        "cell",
                        self.config.predictor_layers,
                    )?;
                    (decoder_frame, next_hidden, next_cell)
                };
                let encoder_frame_values =
                    encoder_frame_values(&encoded, ENCODER_HIDDEN, encoded_time, t);
                let encoder_frame =
                    Array3::from_shape_vec((1, ENCODER_HIDDEN, 1), encoder_frame_values)
                        .map_err(|e| format!("GigaAM encoder frame reshape: {e}"))?;
                let encoder_tensor = Tensor::from_array(encoder_frame)
                    .map_err(|e| format!("GigaAM joint encoder tensor: {e}"))?;
                let decoder_tensor = Tensor::from_array(decoder_frame)
                    .map_err(|e| format!("GigaAM joint predictor tensor: {e}"))?;
                let joint_index = output_index(&inner.joint, "joint")?;
                let joint_outputs = inner
                    .joint
                    .run(ort::inputs![
                        "enc" => encoder_tensor,
                        "dec" => decoder_tensor,
                    ])
                    .map_err(|e| format!("GigaAM joint inference: {e}"))?;
                let (joint_shape, logits) = joint_outputs[joint_index]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("GigaAM joint output: {e}"))?;
                let logits = joint_logits(&joint_shape, logits, self.config.num_classes)?;
                let (token, log_probability) = argmax(logits);
                confidence_sum += log_probability.exp();
                confidence_steps += 1;
                if token == blank {
                    break;
                }
                tokens.push(token);
                last_label = token;
                hidden = next_hidden;
                cell = next_cell;
            }
        }

        Ok(GigaAmRecognition {
            text: self.tokenizer.decode(&tokens),
            confidence: mean_confidence(confidence_sum, confidence_steps),
        })
    }
}

fn load_session(path: &Path) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("GigaAM session builder: {e}"))?
        .with_intra_threads(4)
        .map_err(|e| format!("GigaAM thread configuration: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("Cannot load ONNX model {}: {e}", path.display()))
}

struct TensorSpec<'a> {
    name: &'static str,
    element_type: TensorElementType,
    shape: &'a [i64],
}

impl<'a> TensorSpec<'a> {
    fn new(name: &'static str, element_type: TensorElementType, shape: &'a [i64]) -> Self {
        Self {
            name,
            element_type,
            shape,
        }
    }
}

fn validate_session(
    session: &Session,
    inputs: &[TensorSpec<'_>],
    outputs: &[TensorSpec<'_>],
) -> Result<(), String> {
    for spec in inputs {
        let outlet = session
            .inputs()
            .iter()
            .find(|input| input.name() == spec.name)
            .ok_or_else(|| format!("GigaAM ONNX model is missing input '{}'", spec.name))?;
        validate_outlet(outlet, spec)?;
    }
    for spec in outputs {
        let outlet = session
            .outputs()
            .iter()
            .find(|output| output.name() == spec.name)
            .ok_or_else(|| format!("GigaAM ONNX model is missing output '{}'", spec.name))?;
        validate_outlet(outlet, spec)?;
    }
    Ok(())
}

fn validate_outlet(outlet: &ort::value::Outlet, spec: &TensorSpec<'_>) -> Result<(), String> {
    let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
        return Err(format!(
            "GigaAM ONNX {} must be a tensor, got {:?}",
            spec.name,
            outlet.dtype()
        ));
    };
    if ty != &spec.element_type {
        return Err(format!(
            "GigaAM ONNX {} has type {:?}; expected {:?}",
            spec.name, ty, spec.element_type
        ));
    }
    if shape.len() != spec.shape.len() {
        return Err(format!(
            "GigaAM ONNX {} has shape {:?}; expected rank {}",
            spec.name,
            shape,
            spec.shape.len()
        ));
    }
    for (dimension, expected) in shape.iter().zip(spec.shape) {
        let matches = if *expected == -1 {
            *dimension == -1 || *dimension > 0
        } else {
            dimension == expected
        };
        if !matches {
            return Err(format!(
                "GigaAM ONNX {} has shape {:?}; expected {:?}",
                spec.name, shape, spec.shape
            ));
        }
    }
    Ok(())
}

fn output_index(session: &Session, name: &str) -> Result<usize, String> {
    session
        .outputs()
        .iter()
        .position(|output| output.name() == name)
        .ok_or_else(|| format!("GigaAM ONNX model is missing output '{name}'"))
}

pub(super) fn validate_encoder_shape(shape: &[i64], data_len: usize) -> Result<usize, String> {
    if shape.len() != 3 || shape[0] != 1 || shape[1] != ENCODER_HIDDEN as i64 || shape[2] <= 0 {
        return Err(format!(
            "GigaAM encoder returned unexpected shape {shape:?}"
        ));
    }
    let encoded_time = usize::try_from(shape[2])
        .map_err(|_| "GigaAM encoder time dimension is too large".to_string())?;
    if ENCODER_HIDDEN.checked_mul(encoded_time) != Some(data_len) {
        return Err(format!(
            "GigaAM encoder data does not match shape {shape:?}"
        ));
    }
    Ok(encoded_time)
}

pub(super) fn validate_encoder_length(
    length_shape: &[i64],
    encoded_len: &[i32],
    encoded_time: usize,
) -> Result<usize, String> {
    if length_shape != [1] || encoded_len.len() != 1 || encoded_len[0] <= 0 {
        return Err(format!(
            "GigaAM encoder returned invalid length {encoded_len:?} with shape {length_shape:?}"
        ));
    }
    let encoded_len = usize::try_from(encoded_len[0])
        .map_err(|_| "GigaAM encoder length is negative".to_string())?;
    if encoded_len > encoded_time {
        return Err(format!(
            "GigaAM encoder length {encoded_len} exceeds output time {encoded_time}"
        ));
    }
    Ok(encoded_len)
}

pub(super) fn predictor_frame(shape: &[i64], values: &[f32]) -> Result<Array3<f32>, String> {
    if shape != [1, 1, PREDICTOR_HIDDEN as i64]
        || values.len() != PREDICTOR_HIDDEN
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "GigaAM predictor returned unexpected shape or non-finite values {shape:?}"
        ));
    }
    Array3::from_shape_vec((1, PREDICTOR_HIDDEN, 1), values.to_vec())
        .map_err(|e| format!("GigaAM predictor reshape: {e}"))
}

pub(super) fn joint_logits<'a>(
    shape: &[i64],
    values: &'a [f32],
    num_classes: usize,
) -> Result<&'a [f32], String> {
    let classes =
        i64::try_from(num_classes).map_err(|_| "GigaAM class count is too large".to_string())?;
    if shape != [1, 1, 1, classes]
        || values.len() != num_classes
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "GigaAM joint returned unexpected shape or non-finite values {shape:?}"
        ));
    }
    Ok(values)
}

fn extract_state(
    value: &ort::value::Value,
    name: &str,
    layers: usize,
) -> Result<Array3<f32>, String> {
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("GigaAM {name} state: {e}"))?;
    if shape.len() != 3
        || shape[0] != layers as i64
        || shape[1] != 1
        || shape[2] != PREDICTOR_HIDDEN as i64
        || data.len() != layers * PREDICTOR_HIDDEN
        || data.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "GigaAM {name} state has unexpected shape or non-finite values {shape:?}"
        ));
    }
    Array3::from_shape_vec((layers, 1, PREDICTOR_HIDDEN), data.to_vec())
        .map_err(|e| format!("GigaAM {name} state reshape: {e}"))
}

pub(super) fn encoder_frame_values(
    encoded: &[f32],
    channels: usize,
    encoded_time: usize,
    frame: usize,
) -> Vec<f32> {
    (0..channels)
        .map(|channel| encoded[channel * encoded_time + frame])
        .collect()
}

pub(super) fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, value)| (index, *value))
        .unwrap_or((0, 0.0))
}

pub(super) fn mean_confidence(sum: f32, steps: usize) -> f32 {
    if steps == 0 {
        0.0
    } else {
        (sum / steps as f32).clamp(0.0, 1.0)
    }
}
