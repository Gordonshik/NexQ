use super::segmenter::{Segment, Segmenter};

const RATE: usize = 16_000;

fn speech(ms: usize) -> Vec<i16> {
    vec![1_000; ms * RATE / 1_000]
}

fn silence(ms: usize) -> Vec<i16> {
    vec![0; ms * RATE / 1_000]
}

#[test]
fn a_pause_flushes_once_the_segment_has_at_least_one_second() {
    let mut segmenter = Segmenter::new();
    assert!(segmenter.push(&speech(1_000), 100).is_empty());
    let segments = segmenter.push(&silence(800), 1_100);
    assert_eq!(segments.len(), 1);
    let segment = &segments[0];
    assert_eq!(segment.start_ms, 100);
    assert_eq!(segment.samples.len(), 1_800 * RATE / 1_000);
}

#[test]
fn a_short_pause_is_retained_for_the_idle_flush() {
    let mut segmenter = Segmenter::new();
    assert!(segmenter.push(&speech(100), 0).is_empty());
    assert!(segmenter.push(&silence(800), 100).is_empty());
    assert_eq!(segmenter.flush_idle().len(), 1);
}

#[test]
fn continuing_silence_does_not_make_a_short_segment_pause_eligible() {
    let mut segmenter = Segmenter::new();
    assert!(segmenter.push(&speech(100), 0).is_empty());
    assert!(segmenter.push(&silence(800), 100).is_empty());
    assert!(segmenter.push(&silence(800), 900).is_empty());
    assert_eq!(segmenter.flush_idle().len(), 1);
}

#[test]
fn silence_never_produces_a_segment() {
    let mut segmenter = Segmenter::new();
    assert!(segmenter.push(&silence(2_000), 0).is_empty());
    assert!(segmenter.flush_idle().is_empty());
    assert!(segmenter.finish().is_empty());
}

#[test]
fn the_maximum_segment_length_flushes_without_waiting_for_a_pause() {
    let mut segmenter = Segmenter::new();
    let segments = segmenter.push(&speech(20_000), 25);
    assert_eq!(segments.len(), 1);
    let segment = &segments[0];
    assert_eq!(segment.start_ms, 25);
    assert_eq!(segment.samples.len(), 20_000 * RATE / 1_000);
}

#[test]
fn oversized_chunks_are_split_into_all_maximum_length_segments() {
    let mut segmenter = Segmenter::new();
    let segments = segmenter.push(&speech(60_000), 1_000);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].start_ms, 1_000);
    assert_eq!(segments[1].start_ms, 21_000);
    assert_eq!(segments[2].start_ms, 41_000);
    assert_eq!(segments[0].samples.len(), 20_000 * RATE / 1_000);
    assert_eq!(segments[1].samples.len(), 20_000 * RATE / 1_000);
    assert_eq!(segments[2].samples.len(), 20_000 * RATE / 1_000);
}

#[test]
fn trailing_silence_inside_a_speech_chunk_triggers_the_pause_flush() {
    let mut segmenter = Segmenter::new();
    let mut pcm = speech(1_000);
    pcm.extend(silence(400));
    pcm.extend(speech(400));
    pcm.extend(silence(800));
    let segments = segmenter.push(&pcm, 50);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_ms, 50);
    assert_eq!(segments[0].samples.len(), 2_600 * RATE / 1_000);
    assert_eq!(segmenter.push(&speech(200), 2_650).len(), 0);
    let final_segments = segmenter.finish();
    assert_eq!(final_segments.len(), 1);
    assert_eq!(final_segments[0].start_ms, 2_650);
}

#[test]
fn full_scale_negative_samples_are_treated_as_speech_without_overflow() {
    let mut segmenter = Segmenter::new();
    let segments = segmenter.push(&[i16::MIN], 0);
    assert!(segments.is_empty());
    assert_eq!(segmenter.finish().len(), 0);
    let mut segmenter = Segmenter::new();
    let segments = segmenter.push(&vec![i16::MIN; RATE], 0);
    assert_eq!(segments.len(), 0);
    assert_eq!(segmenter.finish().len(), 1);
}

#[test]
fn zero_pcm_is_silent_even_when_vad_reports_speech() {
    let mut segmenter = Segmenter::new();
    assert!(segmenter.push(&vec![0; RATE], 0).is_empty());
    assert!(segmenter.flush_idle().is_empty());
    assert!(segmenter.finish().is_empty());
}

#[test]
fn stopping_flushes_a_short_non_silent_segment() {
    let mut segmenter = Segmenter::new();
    assert!(segmenter.push(&speech(200), 50).is_empty());
    let segments: Vec<Segment> = segmenter.finish();
    assert_eq!(segments.len(), 1);
    let segment = &segments[0];
    assert_eq!(segment.start_ms, 50);
    assert_eq!(segment.samples.len(), 200 * RATE / 1_000);
}
