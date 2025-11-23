/// Housing experiment module for Neural Additive Models (NAM)
///
/// This module implements a NAM experiment on the California housing dataset.
/// NAM is an interpretable ML model that learns individual shape functions for
/// each feature and combines them additively.
pub mod data;
pub mod nam_experiment;
pub mod training;
pub mod visualization;

// Re-export main experiment function for convenience
pub use nam_experiment::run_housing_nam_experiment;
