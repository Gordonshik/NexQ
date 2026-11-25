pub mod engine;
pub mod features;
pub mod manager;
pub mod model;
pub mod provider;
pub mod segmenter;

pub use manager::{GigaAmManager, GigaAmModelInfo};

#[cfg(test)]
mod engine_tests;
#[cfg(test)]
mod feature_tests;
#[cfg(test)]
mod manager_tests;
#[cfg(test)]
mod model_tests;
#[cfg(test)]
mod provider_tests;
#[cfg(test)]
mod segment_tests;
