use crate::stt::gigaam::model::SAMPLE_RATE;

const PAUSE_SAMPLES: usize = SAMPLE_RATE as usize * 800 / 1_000;
const MIN_PAUSE_SEGMENT_SAMPLES: usize = SAMPLE_RATE as usize;
const MAX_SEGMENT_SAMPLES: usize = SAMPLE_RATE as usize * 20;
const SILENCE_PEAK: i16 = 32;

#[derive(Debug)]
pub struct Segment {
    pub samples: Vec<f32>,
    pub start_ms: u64,
}

#[derive(Debug, Default)]
pub struct Segmenter {
    samples: Vec<f32>,
    start_ms: u64,
    silence_samples: usize,
    has_speech: bool,
}

impl Segmenter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, pcm: &[i16], timestamp_ms: u64) -> Vec<Segment> {
        let mut segments = Vec::new();
        for (index, sample) in pcm.iter().enumerate() {
            if self.samples.is_empty() {
                self.start_ms = timestamp_ms + index as u64 * 1_000 / SAMPLE_RATE as u64;
            }
            if i32::from(*sample).abs() > i32::from(SILENCE_PEAK) {
                self.has_speech = true;
                self.silence_samples = 0;
            } else {
                self.silence_samples += 1;
            }
            self.samples.push(*sample as f32 / 32_768.0);

            if self.samples.len() >= MAX_SEGMENT_SAMPLES {
                if let Some(segment) = self.take() {
                    segments.push(segment);
                }
            } else if self.silence_samples == PAUSE_SAMPLES {
                if self.has_speech && self.samples.len() >= MIN_PAUSE_SEGMENT_SAMPLES {
                    if let Some(segment) = self.take() {
                        segments.push(segment);
                    }
                } else if !self.has_speech {
                    self.clear();
                }
            }
        }
        segments
    }

    pub fn flush_idle(&mut self) -> Vec<Segment> {
        self.take().into_iter().collect()
    }

    pub fn finish(mut self) -> Vec<Segment> {
        self.take().into_iter().collect()
    }

    fn take(&mut self) -> Option<Segment> {
        if !self.has_speech || self.samples.len() < crate::stt::gigaam::model::WIN_LENGTH {
            self.clear();
            return None;
        }
        let segment = Segment {
            samples: std::mem::take(&mut self.samples),
            start_ms: self.start_ms,
        };
        self.silence_samples = 0;
        self.has_speech = false;
        Some(segment)
    }

    fn clear(&mut self) {
        self.samples.clear();
        self.silence_samples = 0;
        self.has_speech = false;
    }
}
