use super::features::GigaAmFeatures;

#[test]
fn a_320_sample_frame_produces_one_64_bin_log_mel_frame() {
    let features = GigaAmFeatures::new();
    let output = features.extract(&vec![0.0; 320]).unwrap();
    assert_eq!(output.shape(), &[1, 64, 1]);
    let expected = (1e-9f32).ln();
    for value in output.iter() {
        assert!((value - expected).abs() < 1e-5, "{value}");
    }
}

#[test]
fn frames_follow_the_uncentered_160_sample_hop() {
    let features = GigaAmFeatures::new();
    let output = features.extract(&vec![0.0; 1_600]).unwrap();
    assert_eq!(output.shape(), &[1, 64, 9]);
}

#[test]
fn a_low_tone_activates_lower_mel_bins_than_a_high_tone() {
    let features = GigaAmFeatures::new();
    let low: Vec<f32> = (0..3_200)
        .map(|i| (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 16_000.0).sin())
        .collect();
    let high: Vec<f32> = (0..3_200)
        .map(|i| (2.0 * std::f32::consts::PI * 7_000.0 * i as f32 / 16_000.0).sin())
        .collect();
    let low_output = features.extract(&low).unwrap();
    let high_output = features.extract(&high).unwrap();
    let low_peak = (0..64)
        .max_by(|&a, &b| low_output[[0, a, 8]].total_cmp(&low_output[[0, b, 8]]))
        .unwrap();
    let high_peak = (0..64)
        .max_by(|&a, &b| high_output[[0, a, 8]].total_cmp(&high_output[[0, b, 8]]))
        .unwrap();
    assert!(low_peak < high_peak, "low={low_peak}, high={high_peak}");
}
