use super::manager::WeakCache;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

fn key(name: &str) -> PathBuf {
    PathBuf::from(name)
}

#[test]
fn leases_for_the_same_model_share_one_engine() {
    let cache = WeakCache::default();
    let loads = AtomicUsize::new(0);
    let first = cache
        .get_or_load(key("model"), || {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(()))
        })
        .unwrap();
    let second = cache
        .get_or_load(key("model"), || {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(()))
        })
        .unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(loads.load(Ordering::SeqCst), 1);
}

#[test]
fn dropping_the_last_lease_unloads_the_engine() {
    let cache = WeakCache::default();
    let loads = AtomicUsize::new(0);
    let first = cache
        .get_or_load(key("model"), || {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(()))
        })
        .unwrap();
    drop(first);
    let second = cache
        .get_or_load(key("model"), || {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(()))
        })
        .unwrap();
    assert_eq!(loads.load(Ordering::SeqCst), 2);
    drop(second);
}

#[test]
fn different_model_directories_do_not_share_an_engine() {
    let cache = WeakCache::default();
    let first = cache.get_or_load(key("one"), || Ok(Arc::new(()))).unwrap();
    let second = cache.get_or_load(key("two"), || Ok(Arc::new(()))).unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
}

#[test]
fn a_failed_load_is_not_cached() {
    let cache: WeakCache<()> = WeakCache::default();
    assert!(cache
        .get_or_load(key("model"), || Err("load failed".to_string()))
        .is_err());
    assert!(cache.get_or_load(key("model"), || Ok(Arc::new(()))).is_ok());
}

#[test]
fn stale_weak_entries_are_harmless() {
    let cache = WeakCache::default();
    cache
        .entries
        .lock()
        .unwrap()
        .insert(key("model"), Weak::new());
    assert!(cache.get_or_load(key("model"), || Ok(Arc::new(()))).is_ok());
}
