use std::sync::mpsc;

use super::engine::GigaAmRecognition;
use super::provider::{
    emit_recognition, next_worker_event, stop_requested, validate_language, AudioMessage,
    WorkerEvent,
};

#[test]
fn russian_locales_are_normalized() {
    assert_eq!(validate_language("ru").unwrap(), "ru-RU");
    assert_eq!(validate_language("ru-RU").unwrap(), "ru-RU");
    assert_eq!(validate_language(" RU-ru ").unwrap(), "ru-RU");
}

#[test]
fn auto_detect_is_refused() {
    let error = validate_language("auto").unwrap_err();
    assert!(error.contains("Russian"));
    assert!(error.contains("Auto-detect"));
}

#[test]
fn non_russian_languages_are_refused() {
    assert!(validate_language("en-US").is_err());
    assert!(validate_language("").is_err());
}

#[test]
fn malformed_russian_language_tags_are_refused() {
    assert!(validate_language("rubbish").is_err());
    assert!(validate_language("ru-XX").is_err());
    assert!(validate_language("ru-RU-extra").is_err());
}

#[test]
fn shutdown_signal_is_visible_without_draining_audio() {
    let (audio_tx, audio_rx) = mpsc::sync_channel(1);
    let (stop_tx, stop_rx) = mpsc::channel();
    audio_tx.send(()).unwrap();
    assert!(!stop_requested(&stop_rx));
    stop_tx.send(()).unwrap();
    assert!(stop_requested(&stop_rx));
    assert_eq!(audio_rx.try_recv().unwrap(), ());
}

#[test]
fn dropped_provider_also_requests_shutdown() {
    let (stop_tx, stop_rx) = mpsc::channel();
    drop(stop_tx);
    assert!(stop_requested(&stop_rx));
}

#[test]
fn shutdown_drains_every_accepted_audio_message_before_closing() {
    let (audio_tx, audio_rx) = mpsc::sync_channel(2);
    let (stop_tx, stop_rx) = mpsc::channel();
    for timestamp_ms in [10, 20] {
        audio_tx
            .send(AudioMessage::Samples {
                pcm: vec![1_000],
                timestamp_ms,
            })
            .unwrap();
    }
    stop_tx.send(()).unwrap();
    drop(stop_tx);
    drop(audio_tx);

    let mut stopping = false;
    for expected in [10, 20] {
        match next_worker_event(&audio_rx, &stop_rx, &mut stopping) {
            WorkerEvent::Samples(AudioMessage::Samples { timestamp_ms, .. }) => {
                assert_eq!(timestamp_ms, expected);
            }
            event => panic!("expected queued audio, got {event:?}"),
        }
    }
    assert!(stopping);
    assert!(matches!(
        next_worker_event(&audio_rx, &stop_rx, &mut stopping),
        WorkerEvent::Closed
    ));
}

#[test]
fn transcript_delivery_reports_a_closed_consumer() {
    let (result_tx, result_rx) = tokio::sync::mpsc::channel(1);
    drop(result_rx);
    let (_stop_tx, stop_rx) = mpsc::channel();

    let error = emit_recognition(
        GigaAmRecognition {
            text: "привет".to_string(),
            confidence: 0.9,
        },
        10,
        1_000,
        &result_tx,
        &stop_rx,
        "You",
        "ru-RU",
    )
    .unwrap_err();

    assert!(error.contains("receiver closed"));
}

#[test]
fn shutdown_does_not_wait_indefinitely_for_a_blocked_transcript_consumer() {
    let (result_tx, _result_rx) = tokio::sync::mpsc::channel(1);
    let (stop_tx, stop_rx) = mpsc::channel();
    let recognition = GigaAmRecognition {
        text: "привет".to_string(),
        confidence: 0.9,
    };
    emit_recognition(
        recognition.clone(),
        10,
        1_000,
        &result_tx,
        &stop_rx,
        "You",
        "ru-RU",
    )
    .unwrap();
    stop_tx.send(()).unwrap();
    drop(stop_tx);

    let error =
        emit_recognition(recognition, 20, 1_000, &result_tx, &stop_rx, "You", "ru-RU").unwrap_err();

    assert!(error.contains("stopped"));
}

#[test]
fn disconnecting_the_producer_finishes_the_buffered_segment() {
    let (audio_tx, audio_rx) = mpsc::sync_channel(1);
    let (_stop_tx, stop_rx) = mpsc::channel();
    audio_tx
        .send(AudioMessage::Samples {
            pcm: vec![1_000],
            timestamp_ms: 10,
        })
        .unwrap();
    drop(audio_tx);

    let mut stopping = false;
    assert!(matches!(
        next_worker_event(&audio_rx, &stop_rx, &mut stopping),
        WorkerEvent::Samples(_)
    ));
    assert!(matches!(
        next_worker_event(&audio_rx, &stop_rx, &mut stopping),
        WorkerEvent::Closed
    ));
}
