use super::engine::{
    argmax, encoder_frame_values, joint_logits, mean_confidence, predictor_frame,
    validate_encoder_length, validate_encoder_shape,
};

#[test]
fn argmax_uses_the_largest_logit() {
    assert_eq!(argmax(&[0.1, 2.0, 1.0]), (1, 2.0));
    assert_eq!(argmax(&[-3.0, -1.0, -2.0]), (1, -1.0));
}

#[test]
fn confidence_is_the_average_selected_probability() {
    assert_eq!(mean_confidence(1.8, 2), 0.9);
    assert_eq!(mean_confidence(0.0, 0), 0.0);
}

#[test]
fn encoder_frames_keep_channels_separate() {
    let encoded = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
    assert_eq!(encoder_frame_values(&encoded, 2, 3, 1), vec![1.0, 11.0]);
}

#[test]
fn encoder_length_uses_int32_and_must_fit_the_output() {
    assert_eq!(validate_encoder_length(&[1], &[3], 4), Ok(3));
    assert!(validate_encoder_length(&[1], &[0], 4).is_err());
    assert!(validate_encoder_length(&[1], &[-1], 4).is_err());
    assert!(validate_encoder_length(&[1], &[5], 4).is_err());
    assert!(validate_encoder_length(&[1, 1], &[3], 4).is_err());
    assert!(validate_encoder_length(&[1], &[], 4).is_err());
}

#[test]
fn encoder_shape_and_data_length_are_validated_before_indexing() {
    assert_eq!(validate_encoder_shape(&[1, 768, 4], 3_072), Ok(4));
    assert!(validate_encoder_shape(&[1, 768, 0], 0).is_err());
    assert!(validate_encoder_shape(&[2, 768, 4], 6_144).is_err());
    assert!(validate_encoder_shape(&[1, 767, 4], 3_068).is_err());
    assert!(validate_encoder_shape(&[1, 768, 4], 3_071).is_err());
}

#[test]
fn predictor_output_is_transposed_from_batch_time_hidden_layout() {
    let values = vec![1.0; 320];
    let frame = predictor_frame(&[1, 1, 320], &values).unwrap();
    assert_eq!(frame.shape(), &[1, 320, 1]);
    assert_eq!(frame.iter().copied().collect::<Vec<_>>(), values);
    assert!(predictor_frame(&[1, 320, 1], &values).is_err());
    assert!(predictor_frame(&[1, 1, 320], &[1.0, 2.0]).is_err());
}

#[test]
fn joint_output_has_one_finite_vector_of_classes() {
    let values = [0.0, -1.0, -2.0];
    assert_eq!(joint_logits(&[1, 1, 1, 3], &values, 3).unwrap(), values);
    assert!(joint_logits(&[1, 3, 1, 1], &values, 3).is_err());
    assert!(joint_logits(&[1, 1, 1, 2], &values, 3).is_err());
    assert!(joint_logits(&[1, 1, 1, 3], &[0.0, f32::NAN, -2.0], 3).is_err());
    assert!(joint_logits(&[1, 1, 1, 3], &[0.0, f32::INFINITY, -2.0], 3).is_err());
}
