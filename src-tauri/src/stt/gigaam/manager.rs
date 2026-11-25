use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

use serde::Serialize;

use super::engine::GigaAmEngine;
use super::model::{load_model, resolve_model_dir, GigaAmModel, SAMPLE_RATE};

#[derive(Debug, Clone, Serialize)]
pub struct GigaAmModelInfo {
    pub model_dir: String,
    pub model_name: String,
    pub sample_rate: u32,
    pub features: usize,
    pub vocabulary_size: usize,
    pub num_classes: usize,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Default)]
pub struct GigaAmManager {
    cache: WeakCache<GigaAmEngine>,
}

impl GigaAmManager {
    pub fn lease(&self, selected: Option<&Path>) -> Result<Arc<GigaAmEngine>, String> {
        let dir = canonical_model_dir(selected)?;
        self.cache.get_or_load(dir.clone(), || {
            let model = load_model(&dir)?;
            GigaAmEngine::load(model).map(Arc::new)
        })
    }

    pub fn inspect(&self, selected: Option<&Path>) -> Result<GigaAmModelInfo, String> {
        let dir = canonical_model_dir(selected)?;
        let model = load_model(&dir)?;
        Ok(model_info(&dir, &model))
    }

    pub fn validate(&self, selected: Option<&Path>) -> Result<GigaAmModelInfo, String> {
        let dir = canonical_model_dir(selected)?;
        let model = load_model(&dir)?;
        let info = model_info(&dir, &model);
        let engine = GigaAmEngine::load(model)?;
        engine.recognize(&vec![0.0; SAMPLE_RATE as usize])?;
        Ok(info)
    }
}

fn model_info(dir: &Path, model: &GigaAmModel) -> GigaAmModelInfo {
    GigaAmModelInfo {
        model_dir: dir.display().to_string(),
        model_name: model.config.model_name.clone(),
        sample_rate: model.config.sample_rate,
        features: model.config.features,
        vocabulary_size: model.tokenizer.len(),
        num_classes: model.config.num_classes,
        warnings: model.warnings.clone(),
    }
}

pub(super) struct WeakCache<T> {
    pub(super) entries: Mutex<HashMap<PathBuf, Weak<T>>>,
}

impl<T> Default for WeakCache<T> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl<T> WeakCache<T> {
    pub(super) fn get_or_load<F>(&self, key: PathBuf, load: F) -> Result<Arc<T>, String>
    where
        F: FnOnce() -> Result<Arc<T>, String>,
    {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| "GigaAM engine cache lock poisoned".to_string())?;
        if let Some(engine) = entries.get(&key).and_then(Weak::upgrade) {
            return Ok(engine);
        }
        entries.retain(|_, weak| weak.strong_count() > 0);
        let engine = load()?;
        entries.insert(key, Arc::downgrade(&engine));
        Ok(engine)
    }
}

fn canonical_model_dir(selected: Option<&Path>) -> Result<PathBuf, String> {
    let dir = resolve_model_dir(selected)?;
    dir.canonicalize()
        .map_err(|e| format!("Cannot resolve GigaAM folder {}: {e}", dir.display()))
}
