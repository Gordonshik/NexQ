use md5::Digest;
use serde::Serialize;
use serde_yaml::Value;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const MODEL_NAME: &str = "v3_e2e_rnnt";
pub const SAMPLE_RATE: u32 = 16_000;
pub const FEATURES: usize = 64;
pub const WIN_LENGTH: usize = 320;
pub const HOP_LENGTH: usize = 160;
pub const ENCODER_HIDDEN: usize = 768;
pub const PREDICTOR_HIDDEN: usize = 320;

#[derive(Debug, Clone, Serialize)]
pub struct GigaAmConfig {
    pub model_name: String,
    pub sample_rate: u32,
    pub features: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub encoder_hidden: usize,
    pub predictor_hidden: usize,
    pub predictor_layers: usize,
    pub num_classes: usize,
}

#[derive(Debug, Clone)]
pub struct GigaAmModel {
    pub dir: PathBuf,
    pub config_path: PathBuf,
    pub encoder_path: PathBuf,
    pub decoder_path: PathBuf,
    pub joint_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub config: GigaAmConfig,
    pub tokenizer: SentencePieceTokenizer,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct Piece {
    text: String,
    kind: u64,
}

#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    pieces: Vec<Piece>,
}

impl SentencePieceTokenizer {
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn blank_id(&self) -> usize {
        self.pieces.len()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut text = String::new();
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(piece) = self.pieces.get(id) else {
                continue;
            };
            match piece.kind {
                1 | 4 => {
                    if !bytes.is_empty() {
                        text.push_str(&String::from_utf8_lossy(&bytes));
                        bytes.clear();
                    }
                    text.push_str(&piece.text.replace('▁', " "));
                }
                6 => {
                    if let Some(hex) = piece
                        .text
                        .strip_prefix("<0x")
                        .and_then(|s| s.strip_suffix('>'))
                    {
                        if let Ok(byte) = u8::from_str_radix(hex, 16) {
                            bytes.push(byte);
                        }
                    }
                }
                _ => {}
            }
        }
        if !bytes.is_empty() {
            text.push_str(&String::from_utf8_lossy(&bytes));
        }
        text.trim().to_owned()
    }
}

pub fn default_model_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .map(|base| base.join("GigaPisar").join("model"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub fn resolve_model_dir(selected: Option<&Path>) -> Result<PathBuf, String> {
    let Some(selected) = selected else {
        return default_model_dir()
            .ok_or_else(|| "Choose the GigaAM model folder in Settings".to_string());
    };
    if selected.as_os_str().is_empty() {
        return Err("GigaAM model folder is empty".to_string());
    }
    if !selected.is_dir() {
        return Err(format!(
            "GigaAM folder does not exist: {}",
            selected.display()
        ));
    }
    let nested = selected.join("model");
    if nested.is_dir() {
        Ok(nested)
    } else {
        Ok(selected.to_path_buf())
    }
}

pub fn load_model(dir: &Path) -> Result<GigaAmModel, String> {
    let config_path = dir.join(format!("{MODEL_NAME}.yaml"));
    let encoder_path = dir.join(format!("{MODEL_NAME}_encoder.onnx"));
    let decoder_path = dir.join(format!("{MODEL_NAME}_decoder.onnx"));
    let joint_path = dir.join(format!("{MODEL_NAME}_joint.onnx"));
    let tokenizer_path = dir.join(format!("{MODEL_NAME}_tokenizer.model"));
    for path in [
        &config_path,
        &encoder_path,
        &decoder_path,
        &joint_path,
        &tokenizer_path,
    ] {
        if !path.is_file() {
            return Err(format!("GigaAM model is missing {}", path.display()));
        }
    }

    let yaml = fs::read_to_string(&config_path)
        .map_err(|e| format!("Cannot read {}: {e}", config_path.display()))?;
    let value: Value =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Invalid GigaAM YAML: {e}"))?;
    let config = parse_config(&value)?;
    let warnings = integrity_warnings(&value, &encoder_path, &tokenizer_path);
    let tokenizer = parse_sentencepiece(
        &fs::read(&tokenizer_path)
            .map_err(|e| format!("Cannot read {}: {e}", tokenizer_path.display()))?,
    )?;
    let expected_classes = tokenizer
        .len()
        .checked_add(1)
        .ok_or("GigaAM vocabulary is too large".to_string())?;
    if config.num_classes != expected_classes {
        return Err(format!(
            "GigaAM vocabulary mismatch: YAML declares {} classes, tokenizer has {} pieces (expected {})",
            config.num_classes,
            tokenizer.len(),
            expected_classes
        ));
    }

    Ok(GigaAmModel {
        dir: dir.to_path_buf(),
        config_path,
        encoder_path,
        decoder_path,
        joint_path,
        tokenizer_path,
        config,
        tokenizer,
        warnings,
    })
}

pub(super) fn parse_config(value: &Value) -> Result<GigaAmConfig, String> {
    let model_name = string_at(&value, "/model_name")?;
    if model_name != MODEL_NAME {
        return Err(format!(
            "Unsupported GigaAM model '{model_name}'; expected {MODEL_NAME}"
        ));
    }
    if string_at(&value, "/model_class")? != "rnnt" {
        return Err("GigaAM model must use the RNNT architecture".to_string());
    }
    let sample_rate = u32_at(&value, "/sample_rate")?;
    if sample_rate != SAMPLE_RATE {
        return Err(format!(
            "GigaAM must use 16 kHz audio; YAML declares {sample_rate} Hz"
        ));
    }
    let features = usize_at(&value, "/preprocessor/features")?;
    let win_length = usize_at(&value, "/preprocessor/win_length")?;
    let hop_length = usize_at(&value, "/preprocessor/hop_length")?;
    let n_fft = usize_at(&value, "/preprocessor/n_fft")?;
    if (features, win_length, hop_length, n_fft) != (FEATURES, WIN_LENGTH, HOP_LENGTH, WIN_LENGTH) {
        return Err(
            "GigaAM preprocessing must use 64 mel bins, a 320-sample window and a 160-sample hop"
                .to_string(),
        );
    }
    if string_at(&value, "/preprocessor/mel_scale")? != "htk" {
        return Err("GigaAM preprocessing must use the HTK mel scale".to_string());
    }
    if !value_at(&value, "/preprocessor/mel_norm").map_or(true, Value::is_null) {
        return Err("GigaAM preprocessing must not normalize mel filters".to_string());
    }
    if bool_at(&value, "/preprocessor/center")? {
        return Err("GigaAM preprocessing must disable STFT centering".to_string());
    }

    let encoder_hidden = usize_at(&value, "/encoder/d_model")?;
    let predictor_hidden = usize_at(&value, "/head/decoder/pred_hidden")?;
    let predictor_layers = usize_at(&value, "/head/decoder/pred_rnn_layers")?;
    let num_classes = usize_at(&value, "/head/decoder/num_classes")?;
    let joint_classes = usize_at(&value, "/head/joint/num_classes")?;
    let joint_encoder_hidden = usize_at(&value, "/head/joint/enc_hidden")?;
    let joint_predictor_hidden = usize_at(&value, "/head/joint/pred_hidden")?;
    if encoder_hidden != ENCODER_HIDDEN
        || joint_encoder_hidden != ENCODER_HIDDEN
        || predictor_hidden != PREDICTOR_HIDDEN
        || joint_predictor_hidden != PREDICTOR_HIDDEN
        || predictor_layers != 1
    {
        return Err(
            "GigaAM v3 encoder/predictor dimensions do not match the supported RNNT contract"
                .to_string(),
        );
    }
    if joint_classes != num_classes {
        return Err("GigaAM decoder and joint class counts differ".to_string());
    }
    if num_classes == 0 || i64::try_from(num_classes).is_err() {
        return Err("GigaAM class count is outside the supported ONNX range".to_string());
    }

    Ok(GigaAmConfig {
        model_name,
        sample_rate,
        features,
        win_length,
        hop_length,
        n_fft,
        encoder_hidden,
        predictor_hidden,
        predictor_layers,
        num_classes,
    })
}

fn value_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a Value> {
    pointer
        .strip_prefix('/')?
        .split('/')
        .try_fold(value, |current, segment| match current {
            Value::Mapping(map) => map.get(segment),
            Value::Sequence(sequence) => segment
                .parse::<usize>()
                .ok()
                .and_then(|index| sequence.get(index)),
            _ => None,
        })
}

fn string_at(value: &Value, pointer: &str) -> Result<String, String> {
    value_at(value, pointer)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("GigaAM YAML is missing string field {pointer}"))
}

fn u64_at(value: &Value, pointer: &str) -> Result<u64, String> {
    value_at(value, pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("GigaAM YAML is missing integer field {pointer}"))
}

fn u32_at(value: &Value, pointer: &str) -> Result<u32, String> {
    u32::try_from(u64_at(value, pointer)?)
        .map_err(|_| format!("GigaAM YAML integer field {pointer} is too large"))
}

fn usize_at(value: &Value, pointer: &str) -> Result<usize, String> {
    usize::try_from(u64_at(value, pointer)?)
        .map_err(|_| format!("GigaAM YAML integer field {pointer} is too large"))
}

fn bool_at(value: &Value, pointer: &str) -> Result<bool, String> {
    value_at(value, pointer)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("GigaAM YAML is missing boolean field {pointer}"))
}

pub(super) fn integrity_warnings(
    value: &Value,
    encoder_path: &Path,
    tokenizer_path: &Path,
) -> Vec<String> {
    let mut warnings = Vec::new();
    reference_hash_warning(
        value,
        "/hashes/model",
        "encoder",
        encoder_path,
        &mut warnings,
    );
    reference_hash_warning(
        value,
        "/hashes/tokenizer",
        "tokenizer",
        tokenizer_path,
        &mut warnings,
    );
    warnings
}

fn reference_hash_warning(
    value: &Value,
    pointer: &str,
    label: &str,
    path: &Path,
    warnings: &mut Vec<String>,
) {
    let Some(expected) = value_at(value, pointer) else {
        return;
    };
    let Some(expected) = expected.as_str().map(str::trim) else {
        warnings.push(format!("GigaAM {label} reference hash is not a string."));
        return;
    };
    if expected.len() != 32 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        warnings.push(format!("GigaAM {label} reference hash is malformed."));
        return;
    }
    match file_md5(path) {
        Ok(actual) if actual.eq_ignore_ascii_case(expected) => {}
        Ok(_) => warnings.push(format!(
            "GigaAM {label} differs from the reference MD5. Locally exported models are supported, but this file is not identical to the reference release."
        )),
        Err(error) => warnings.push(format!(
            "Cannot verify the GigaAM {label} reference hash: {error}"
        )),
    }
}

fn file_md5(path: &Path) -> Result<String, String> {
    let mut file =
        fs::File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut hasher = md5::Md5::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn parse_sentencepiece(data: &[u8]) -> Result<SentencePieceTokenizer, String> {
    let mut pieces = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let key = read_varint(data, &mut pos)?;
        let field = key >> 3;
        let wire = key & 7;
        if field == 1 && wire == 2 {
            let len = read_varint(data, &mut pos)? as usize;
            let end = pos
                .checked_add(len)
                .ok_or("SentencePiece length overflow")?;
            if end > data.len() {
                return Err("SentencePiece piece extends past end of file".to_string());
            }
            pieces.push(parse_piece(&data[pos..end])?);
            pos = end;
        } else {
            skip_field(data, &mut pos, wire)?;
        }
    }
    if pieces.is_empty() {
        return Err("SentencePiece model contains no vocabulary pieces".to_string());
    }
    Ok(SentencePieceTokenizer { pieces })
}

fn parse_piece(data: &[u8]) -> Result<Piece, String> {
    let mut text = None;
    let mut kind = 1;
    let mut pos = 0;
    while pos < data.len() {
        let key = read_varint(data, &mut pos)?;
        let field = key >> 3;
        let wire = key & 7;
        match (field, wire) {
            (1, 2) => {
                let len = read_varint(data, &mut pos)? as usize;
                let end = pos
                    .checked_add(len)
                    .ok_or("SentencePiece string length overflow")?;
                if end > data.len() {
                    return Err("SentencePiece string extends past end of piece".to_string());
                }
                text = Some(
                    std::str::from_utf8(&data[pos..end])
                        .map_err(|e| format!("SentencePiece piece is not UTF-8: {e}"))?
                        .to_string(),
                );
                pos = end;
            }
            (3, 0) => kind = read_varint(data, &mut pos)?,
            _ => skip_field(data, &mut pos, wire)?,
        }
    }
    Ok(Piece {
        text: text.ok_or("SentencePiece piece has no text")?,
        kind,
    })
}

fn read_varint(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *data
            .get(*pos)
            .ok_or("SentencePiece protobuf ends inside a varint")?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err("SentencePiece protobuf varint is too long".to_string())
}

fn skip_field(data: &[u8], pos: &mut usize, wire: u64) -> Result<(), String> {
    match wire {
        0 => {
            read_varint(data, pos)?;
        }
        1 => {
            *pos = pos.checked_add(8).ok_or("SentencePiece field overflow")?;
            if *pos > data.len() {
                return Err("SentencePiece fixed64 field extends past end of file".to_string());
            }
        }
        2 => {
            let len = read_varint(data, pos)? as usize;
            *pos = pos.checked_add(len).ok_or("SentencePiece field overflow")?;
            if *pos > data.len() {
                return Err("SentencePiece bytes field extends past end of file".to_string());
            }
        }
        5 => {
            *pos = pos.checked_add(4).ok_or("SentencePiece field overflow")?;
            if *pos > data.len() {
                return Err("SentencePiece fixed32 field extends past end of file".to_string());
            }
        }
        _ => {
            return Err(format!(
                "Unsupported SentencePiece protobuf wire type {wire}"
            ))
        }
    }
    Ok(())
}
