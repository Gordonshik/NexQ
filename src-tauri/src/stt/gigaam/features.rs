use ndarray::Array3;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::sync::Arc;

use super::model::{FEATURES, HOP_LENGTH, SAMPLE_RATE, WIN_LENGTH};

pub struct GigaAmFeatures {
    window: Vec<f32>,
    filterbank: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl GigaAmFeatures {
    pub fn new() -> Self {
        let window = (0..WIN_LENGTH)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / WIN_LENGTH as f32).cos())
            .collect();
        Self {
            window,
            filterbank: build_filterbank(),
            fft: FftPlanner::new().plan_fft_forward(WIN_LENGTH),
        }
    }

    pub fn extract(&self, samples: &[f32]) -> Result<Array3<f32>, String> {
        if samples.len() < WIN_LENGTH {
            return Err(format!(
                "GigaAM needs at least {WIN_LENGTH} samples, got {}",
                samples.len()
            ));
        }
        let frames = (samples.len() - WIN_LENGTH) / HOP_LENGTH + 1;
        let fft_bins = WIN_LENGTH / 2 + 1;
        let mut output = Array3::zeros((1, FEATURES, frames));
        let mut spectrum = vec![Complex::new(0.0f32, 0.0); WIN_LENGTH];

        for t in 0..frames {
            let start = t * HOP_LENGTH;
            for (i, bin) in spectrum.iter_mut().enumerate() {
                *bin = Complex::new(samples[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut spectrum);
            for mel in 0..FEATURES {
                let mut energy = 0.0f32;
                let row = mel * fft_bins;
                for bin in 0..fft_bins {
                    energy += self.filterbank[row + bin] * spectrum[bin].norm_sqr();
                }
                output[[0, mel, t]] = energy.clamp(1e-9, 1e9).ln();
            }
        }
        Ok(output)
    }
}

fn build_filterbank() -> Vec<f32> {
    let fft_bins = WIN_LENGTH / 2 + 1;
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(SAMPLE_RATE as f32 / 2.0);
    let points: Vec<f32> = (0..FEATURES + 2)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (FEATURES + 1) as f32)
        .map(mel_to_hz)
        .map(|hz| hz * WIN_LENGTH as f32 / SAMPLE_RATE as f32)
        .collect();
    let mut bank = vec![0.0f32; FEATURES * fft_bins];
    for mel in 0..FEATURES {
        let left = points[mel];
        let center = points[mel + 1];
        let right = points[mel + 2];
        for bin in 0..fft_bins {
            let value = if bin as f32 >= left && bin as f32 <= center && center > left {
                (bin as f32 - left) / (center - left)
            } else if bin as f32 > center && bin as f32 <= right && right > center {
                (right - bin as f32) / (right - center)
            } else {
                0.0
            };
            bank[mel * fft_bins + bin] = value;
        }
    }
    bank
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}
