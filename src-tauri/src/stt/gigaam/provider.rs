use std::sync::mpsc::{self as sync_mpsc, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tauri::AppHandle;
use tokio::sync::mpsc::{error::TrySendError, Sender as TokioSender};

use crate::audio::AudioChunk;
use crate::stt::provider::{STTProvider, STTProviderType, TranscriptResult};

use super::engine::{GigaAmEngine, GigaAmRecognition};
use super::model::SAMPLE_RATE;
use super::segmenter::{Segment, Segmenter};

const IDLE_FLUSH: Duration = Duration::from_millis(800);
const WORKER_POLL: Duration = Duration::from_millis(100);
const AUDIO_QUEUE_CAPACITY: usize = 16;
const RESULT_SEND_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug)]
pub(super) enum AudioMessage {
    Samples { pcm: Vec<i16>, timestamp_ms: u64 },
}

#[derive(Debug)]
pub(super) enum WorkerEvent {
    Samples(AudioMessage),
    Idle,
    Closed,
}

pub struct GigaAmProvider {
    engine: Arc<GigaAmEngine>,
    party: String,
    language: String,
    is_streaming: bool,
    audio_tx: Option<SyncSender<AudioMessage>>,
    stop_tx: Option<sync_mpsc::Sender<()>>,
    inference_thread: Option<JoinHandle<()>>,
    completion_rx: Option<Mutex<Receiver<Result<(), String>>>>,
    failure: Option<String>,
    app_handle: Option<AppHandle>,
}

impl GigaAmProvider {
    pub fn new(engine: Arc<GigaAmEngine>, party: impl Into<String>) -> Self {
        Self {
            engine,
            party: party.into(),
            language: "ru-RU".to_string(),
            is_streaming: false,
            audio_tx: None,
            stop_tx: None,
            inference_thread: None,
            completion_rx: None,
            failure: None,
            app_handle: None,
        }
    }

    pub fn set_app_handle(&mut self, app: AppHandle) {
        self.app_handle = Some(app);
    }

    fn check_worker(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(error) = &self.failure {
            return Err(error.clone().into());
        }
        let Some(rx) = self.completion_rx.as_ref() else {
            return Ok(());
        };
        let rx = rx
            .lock()
            .map_err(|_| "GigaAM worker completion lock poisoned")?;
        match rx.try_recv() {
            Ok(Err(error)) => {
                self.failure = Some(error.clone());
                Err(error.into())
            }
            Ok(Ok(())) => {
                let error = "GigaAM audio worker stopped unexpectedly".to_string();
                self.failure = Some(error.clone());
                Err(error.into())
            }
            Err(sync_mpsc::TryRecvError::Empty) => Ok(()),
            Err(sync_mpsc::TryRecvError::Disconnected) => {
                let error = "GigaAM audio worker exited without reporting completion".to_string();
                self.failure = Some(error.clone());
                Err(error.into())
            }
        }
    }
}

#[async_trait]
impl STTProvider for GigaAmProvider {
    fn provider_name(&self) -> &str {
        "GigaAM v3 RNNT (Local)"
    }

    fn provider_type(&self) -> STTProviderType {
        STTProviderType::GigaAM
    }

    async fn start_stream(
        &mut self,
        result_tx: TokioSender<TranscriptResult>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.is_streaming {
            return Err("GigaAM stream already active".into());
        }
        let (audio_tx, audio_rx) = sync_mpsc::sync_channel(AUDIO_QUEUE_CAPACITY);
        let (stop_tx, stop_rx) = sync_mpsc::channel();
        let (completion_tx, completion_rx) = sync_mpsc::channel();
        let engine = Arc::clone(&self.engine);
        let party = self.party.clone();
        let language = self.language.clone();
        let app_handle = self.app_handle.clone();
        let thread = std::thread::Builder::new()
            .name(format!("gigaam-stt-{}", self.party.to_lowercase()))
            .spawn(move || {
                let outcome = worker_main(
                    engine, audio_rx, stop_rx, result_tx, party, language, app_handle,
                );
                let _ = completion_tx.send(outcome);
            })?;
        self.audio_tx = Some(audio_tx);
        self.stop_tx = Some(stop_tx);
        self.inference_thread = Some(thread);
        self.completion_rx = Some(Mutex::new(completion_rx));
        self.failure = None;
        self.is_streaming = true;
        log::info!("GigaAM stream started for {}", self.party);
        Ok(())
    }

    async fn feed_audio(
        &mut self,
        chunk: AudioChunk,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.is_streaming || chunk.pcm_data.is_empty() {
            return Ok(());
        }
        self.check_worker()?;
        let Some(tx) = self.audio_tx.as_ref() else {
            return Err("GigaAM audio worker is unavailable".into());
        };
        let tx = tx.clone();
        match tokio::task::spawn_blocking(move || {
            tx.send(AudioMessage::Samples {
                pcm: chunk.pcm_data,
                timestamp_ms: chunk.timestamp_ms,
            })
        })
        .await
        {
            Ok(Ok(())) => {
                self.check_worker()?;
                Ok(())
            }
            Ok(Err(_)) => match self.check_worker() {
                Err(error) => Err(error),
                Ok(()) => Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "GigaAM audio worker stopped",
                ))),
            },
            Err(error) => Err(Box::new(error)),
        }
    }

    async fn stop_stream(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.is_streaming {
            return Ok(());
        }
        let mut failure = self.failure.take();
        self.audio_tx.take();
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(thread) = self.inference_thread.take() {
            match tokio::task::spawn_blocking(move || thread.join()).await {
                Ok(Err(_)) => {
                    if failure.is_none() {
                        failure = Some(format!(
                            "GigaAM inference thread panicked for {}",
                            self.party
                        ));
                    }
                }
                Err(error) => {
                    if failure.is_none() {
                        failure = Some(format!(
                            "GigaAM inference thread join failed for {}: {error}",
                            self.party
                        ));
                    }
                }
                Ok(Ok(())) => {}
            }
        }
        if let Some(rx) = self.completion_rx.take() {
            let rx = rx.into_inner().unwrap_or_else(|error| error.into_inner());
            match rx.try_recv() {
                Ok(Err(error)) => {
                    if failure.is_none() {
                        failure = Some(error);
                    }
                }
                Ok(Ok(())) => {}
                Err(sync_mpsc::TryRecvError::Disconnected) => {
                    if failure.is_none() {
                        failure = Some(format!(
                            "GigaAM audio worker exited without reporting completion for {}",
                            self.party
                        ));
                    }
                }
                Err(sync_mpsc::TryRecvError::Empty) => {
                    if failure.is_none() {
                        failure = Some(format!(
                            "GigaAM audio worker did not report completion for {}",
                            self.party
                        ));
                    }
                }
            }
        }
        self.is_streaming = false;
        if let Some(error) = failure {
            let message = format!("GigaAM stream failed for {}: {error}", self.party);
            log::error!("{message}");
            if let Some(app) = &self.app_handle {
                crate::stt::emit_stt_debug(app, "error", "stt", &message);
            }
            return Err(message.into());
        }
        log::info!("GigaAM stream stopped for {}", self.party);
        Ok(())
    }

    async fn test_connection(&self) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        Ok(true)
    }

    fn set_language(&mut self, language: &str) {
        match validate_language(language) {
            Ok(language) => self.language = language,
            Err(error) => {
                log::warn!("{error}; using ru-RU");
                self.language = "ru-RU".to_string();
            }
        }
    }
}

impl Drop for GigaAmProvider {
    fn drop(&mut self) {
        self.audio_tx.take();
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
    }
}

pub fn validate_language(language: &str) -> Result<String, String> {
    let language = language.trim();
    if language.eq_ignore_ascii_case("auto") {
        return Err("GigaAM supports Russian only; Auto-detect is not supported".to_string());
    }
    if matches!(language.to_ascii_lowercase().as_str(), "ru" | "ru-ru") {
        Ok("ru-RU".to_string())
    } else {
        Err("GigaAM supports Russian only; select ru-RU".to_string())
    }
}

fn worker_main(
    engine: Arc<GigaAmEngine>,
    audio_rx: Receiver<AudioMessage>,
    stop_rx: Receiver<()>,
    result_tx: TokioSender<TranscriptResult>,
    party: String,
    language: String,
    app_handle: Option<AppHandle>,
) -> Result<(), String> {
    let mut segmenter = Segmenter::new();
    let mut last_audio = Instant::now();
    let mut stopping = false;
    loop {
        match next_worker_event(&audio_rx, &stop_rx, &mut stopping) {
            WorkerEvent::Samples(AudioMessage::Samples { pcm, timestamp_ms }) => {
                last_audio = Instant::now();
                emit_segments(
                    &engine,
                    segmenter.push(&pcm, timestamp_ms),
                    &result_tx,
                    &stop_rx,
                    &party,
                    &language,
                    app_handle.as_ref(),
                )?;
            }
            WorkerEvent::Idle => {
                if last_audio.elapsed() >= IDLE_FLUSH {
                    emit_segments(
                        &engine,
                        segmenter.flush_idle(),
                        &result_tx,
                        &stop_rx,
                        &party,
                        &language,
                        app_handle.as_ref(),
                    )?;
                }
            }
            WorkerEvent::Closed => break,
        }
    }
    emit_segments(
        &engine,
        segmenter.finish(),
        &result_tx,
        &stop_rx,
        &party,
        &language,
        app_handle.as_ref(),
    )
}

pub(super) fn next_worker_event(
    audio_rx: &Receiver<AudioMessage>,
    stop_rx: &Receiver<()>,
    stopping: &mut bool,
) -> WorkerEvent {
    if !*stopping && stop_requested(stop_rx) {
        *stopping = true;
    }
    if *stopping {
        return match audio_rx.recv() {
            Ok(samples) => WorkerEvent::Samples(samples),
            Err(_) => WorkerEvent::Closed,
        };
    }
    match audio_rx.recv_timeout(WORKER_POLL) {
        Ok(samples) => WorkerEvent::Samples(samples),
        Err(sync_mpsc::RecvTimeoutError::Timeout) => WorkerEvent::Idle,
        Err(sync_mpsc::RecvTimeoutError::Disconnected) => WorkerEvent::Closed,
    }
}

pub(super) fn stop_requested(stop_rx: &Receiver<()>) -> bool {
    matches!(
        stop_rx.try_recv(),
        Ok(()) | Err(sync_mpsc::TryRecvError::Disconnected)
    )
}

fn emit_segments(
    engine: &GigaAmEngine,
    segments: Vec<Segment>,
    result_tx: &TokioSender<TranscriptResult>,
    stop_rx: &Receiver<()>,
    party: &str,
    language: &str,
    app_handle: Option<&AppHandle>,
) -> Result<(), String> {
    for segment in segments {
        emit_segment(
            engine, segment, result_tx, stop_rx, party, language, app_handle,
        )?;
    }
    Ok(())
}

fn emit_segment(
    engine: &GigaAmEngine,
    segment: Segment,
    result_tx: &TokioSender<TranscriptResult>,
    stop_rx: &Receiver<()>,
    party: &str,
    language: &str,
    app_handle: Option<&AppHandle>,
) -> Result<(), String> {
    let duration_ms = segment.samples.len() as u64 * 1_000 / SAMPLE_RATE as u64;
    let recognition = match engine.recognize(&segment.samples) {
        Ok(recognition) => recognition,
        Err(error) => {
            let message = format!("GigaAM recognition failed for {party}: {error}");
            log::error!("{message}");
            if let Some(app) = app_handle {
                crate::stt::emit_stt_debug(app, "error", "stt", &message);
            }
            return Err(message);
        }
    };
    emit_recognition(
        recognition,
        segment.start_ms,
        duration_ms,
        result_tx,
        stop_rx,
        party,
        language,
    )
}

pub(super) fn emit_recognition(
    recognition: GigaAmRecognition,
    timestamp_ms: u64,
    duration_ms: u64,
    result_tx: &TokioSender<TranscriptResult>,
    stop_rx: &Receiver<()>,
    party: &str,
    language: &str,
) -> Result<(), String> {
    if recognition.text.trim().is_empty() {
        return Ok(());
    }
    let mut pending = TranscriptResult {
        text: recognition.text,
        is_final: true,
        confidence: recognition.confidence,
        timestamp_ms,
        speaker: Some(party.to_string()),
        language: Some(language.to_string()),
        segment_id: None,
        duration_ms: Some(duration_ms),
        speaker_tag: None,
        turn: None,
    };
    let deadline = Instant::now() + RESULT_SEND_TIMEOUT;
    loop {
        match result_tx.try_send(pending) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Closed(_)) => {
                return Err(format!("GigaAM transcript receiver closed for {party}"));
            }
            Err(TrySendError::Full(result)) => {
                if stop_requested(stop_rx) {
                    return Err(format!(
                        "GigaAM transcript delivery stopped for {party} while the output queue was full"
                    ));
                }
                if Instant::now() >= deadline {
                    return Err(format!(
                        "GigaAM transcript delivery timed out for {party} while the output queue was full"
                    ));
                }
                pending = result;
                std::thread::sleep(WORKER_POLL);
            }
        }
    }
}
