use super::model::{
    integrity_warnings, load_model, parse_config, parse_sentencepiece, resolve_model_dir,
};
use md5::Digest;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("nexq-gigaam-test-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn yaml(sample_rate: u32) -> String {
    format!(
        r#"
model_class: rnnt
sample_rate: {sample_rate}
model_name: v3_e2e_rnnt
preprocessor:
  features: 64
  win_length: 320
  hop_length: 160
  mel_scale: htk
  n_fft: 320
  mel_norm: null
  center: false
encoder:
  d_model: 768
head:
  decoder:
    pred_hidden: 320
    pred_rnn_layers: 1
    num_classes: 6
  joint:
    enc_hidden: 768
    pred_hidden: 320
    num_classes: 6
"#
    )
}

fn varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

fn bytes_field(field: u64, value: &[u8]) -> Vec<u8> {
    let mut out = varint((field << 3) | 2);
    out.extend(varint(value.len() as u64));
    out.extend(value);
    out
}

fn sentencepiece_model() -> Vec<u8> {
    let pieces = [
        (1, "▁Привет", 1),
        (1, ",", 1),
        (1, "▁мир", 1),
        (1, "!", 1),
        (3, "</s>", 1),
    ];
    let mut model = Vec::new();
    for (kind, piece, _) in pieces {
        let mut message = bytes_field(1, piece.as_bytes());
        if kind != 1 {
            message.extend(varint((3 << 3) | 0));
            message.extend(varint(kind));
        }
        model.extend(bytes_field(1, &message));
    }
    model
}

fn write_model(root: &Path, sample_rate: u32) -> PathBuf {
    let model = root.join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("v3_e2e_rnnt.yaml"), yaml(sample_rate)).unwrap();
    for suffix in ["encoder.onnx", "decoder.onnx", "joint.onnx"] {
        std::fs::write(
            model.join(format!("v3_e2e_rnnt_{suffix}")),
            b"not an onnx model",
        )
        .unwrap();
    }
    std::fs::write(
        model.join("v3_e2e_rnnt_tokenizer.model"),
        sentencepiece_model(),
    )
    .unwrap();
    model
}

#[test]
fn resolves_both_the_installer_root_and_the_model_directory() {
    let temp = TempDir::new();
    let model = write_model(&temp.0, 16_000);

    assert_eq!(resolve_model_dir(Some(&temp.0)).unwrap(), model);
    assert_eq!(resolve_model_dir(Some(&model)).unwrap(), model);
}

#[test]
fn rejects_a_folder_with_a_missing_model_file() {
    let temp = TempDir::new();
    let model = write_model(&temp.0, 16_000);
    std::fs::remove_file(model.join("v3_e2e_rnnt_joint.onnx")).unwrap();

    let error = load_model(&model).unwrap_err();
    assert!(error.contains("v3_e2e_rnnt_joint.onnx"), "{error}");
}

#[test]
fn rejects_an_incompatible_sample_rate() {
    let temp = TempDir::new();
    let model = write_model(&temp.0, 8_000);

    let error = load_model(&model).unwrap_err();
    assert!(error.contains("16 kHz"), "{error}");
}

#[test]
fn sentencepiece_pieces_decode_to_text_and_expose_the_blank_id() {
    let tokenizer = parse_sentencepiece(&sentencepiece_model()).unwrap();
    assert_eq!(tokenizer.len(), 5);
    assert_eq!(tokenizer.blank_id(), 5);
    assert_eq!(tokenizer.decode(&[0, 1, 2, 3]), "Привет, мир!");
}

#[test]
fn malformed_sentencepiece_data_is_rejected() {
    assert!(parse_sentencepiece(&[0x80]).is_err());
}

#[test]
fn oversized_yaml_integers_are_rejected_instead_of_truncated() {
    let yaml = yaml(16_000).replacen("sample_rate: 16000", "sample_rate: 18446744073709551615", 1);
    let value = serde_yaml::from_str(&yaml).unwrap();
    let error = parse_config(&value).unwrap_err();
    assert!(error.contains("sample_rate"), "{error}");
}

#[test]
fn reference_hash_mismatches_are_reported_without_rejecting_local_exports() {
    let temp = TempDir::new();
    let encoder = temp.0.join("encoder.onnx");
    let tokenizer = temp.0.join("tokenizer.model");
    std::fs::write(&encoder, b"locally exported encoder").unwrap();
    std::fs::write(&tokenizer, b"unchanged tokenizer").unwrap();
    let tokenizer_hash = format!("{:x}", md5::Md5::digest(b"unchanged tokenizer"));
    let yaml = format!(
        "hashes:\n  model: 00000000000000000000000000000000\n  tokenizer: {tokenizer_hash}\n"
    );
    let value = serde_yaml::from_str(&yaml).unwrap();

    let warnings = integrity_warnings(&value, &encoder, &tokenizer);
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("encoder"), "{}", warnings[0]);
    assert!(warnings[0].contains("reference"), "{}", warnings[0]);
}

#[test]
fn missing_reference_hashes_do_not_produce_warnings() {
    let temp = TempDir::new();
    let value = serde_yaml::from_str("model_name: v3_e2e_rnnt\n").unwrap();
    assert!(integrity_warnings(
        &value,
        &temp.0.join("encoder.onnx"),
        &temp.0.join("tokenizer.model")
    )
    .is_empty());
}
