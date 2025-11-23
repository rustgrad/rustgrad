use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::ReaderBuilder;
use ndarray::{Array, Array1, Array2, IxDyn};
use serde::Deserialize;

use crate::data::labeled_dataset::LabeledTensorSample;
use crate::dimensions::{Rank1, S};
use crate::tensor::Tensor;

/// The number of numeric features in the California housing dataset
/// (excluding 'ocean_proximity' which is categorical)
pub const NUM_FEATURES: usize = 8;

/// Feature names in the same order as they appear in the feature vectors
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
];

/// Minimum standard deviation threshold to avoid division by zero
const MIN_STD_DEV: f32 = 1e-6;

/// A single record from the California housing dataset
#[derive(Debug, Deserialize)]
pub struct HousingRecord {
    longitude: f32,
    latitude: f32,
    housing_median_age: f32,
    total_rooms: f32,
    total_bedrooms: f32,
    population: f32,
    households: f32,
    median_income: f32,
    median_house_value: f32,
    ocean_proximity: String,
}

impl HousingRecord {
    /// Extracts numeric features as a vector in the canonical order
    fn to_feature_vec(&self) -> Vec<f32> {
        vec![
            self.longitude,
            self.latitude,
            self.housing_median_age,
            self.total_rooms,
            self.total_bedrooms,
            self.population,
            self.households,
            self.median_income,
        ]
    }
}

/// Calculates column-wise mean values for imputation of missing data
///
/// # Arguments
/// * `records` - All housing records from the dataset
///
/// # Returns
/// Vector of mean values, one per feature column
fn calculate_column_statistics(records: &[HousingRecord]) -> Vec<f32> {
    let mut column_sums = vec![0.0; NUM_FEATURES];
    let mut column_counts = vec![0; NUM_FEATURES];

    // Accumulate sums for each feature, skipping NaN values
    for record in records {
        let feature_values = record.to_feature_vec();
        for (i, &value) in feature_values.iter().enumerate() {
            if !value.is_nan() {
                column_sums[i] += value;
                column_counts[i] += 1;
            }
        }
    }

    // Calculate means for each column
    column_sums
        .iter()
        .zip(column_counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
        .collect()
}

/// Imputes missing values and extracts features and targets
///
/// # Arguments
/// * `records` - All housing records from the dataset
/// * `column_means` - Mean values for each feature (used for imputation)
///
/// # Returns
/// Tuple of (features matrix, targets vector)
fn impute_and_extract_data(
    records: Vec<HousingRecord>,
    column_means: &[f32],
) -> Result<(Array2<f32>, Array1<f32>), Box<dyn Error>> {
    let mut features = Vec::new();
    let mut targets = Vec::new();

    for record in records {
        let mut feature_row = record.to_feature_vec();

        // Replace NaN values with column means
        for i in 0..NUM_FEATURES {
            if feature_row[i].is_nan() {
                feature_row[i] = column_means[i];
            }
        }

        features.push(feature_row);
        targets.push(record.median_house_value);
    }

    let n_samples = features.len();
    let x = Array2::from_shape_vec(
        (n_samples, NUM_FEATURES),
        features.into_iter().flatten().collect(),
    )?;
    let y = Array1::from_shape_vec(n_samples, targets)?;

    Ok((x, y))
}

/// Loads the California housing dataset from a CSV file
///
/// This function reads the CSV, handles missing values by imputing with
/// column means, and returns features and targets as ndarray structures.
///
/// # Arguments
/// * `path` - Path to the housing.csv file
///
/// # Returns
/// Tuple of (features matrix [N x 8], targets vector [N])
pub fn load_housing_dataset(
    path: impl AsRef<Path>,
) -> Result<(Array2<f32>, Array1<f32>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(File::open(path)?);

    // Read all records
    let records: Vec<HousingRecord> = rdr.deserialize().filter_map(|result| result.ok()).collect();

    // Calculate column means for imputation
    let column_means = calculate_column_statistics(&records);

    // Impute missing values and extract features/targets
    impute_and_extract_data(records, &column_means)
}

/// Normalizes features to have zero mean and unit variance (z-score normalization)
///
/// # Arguments
/// * `features` - Mutable reference to features matrix (modified in-place)
///
/// # Returns
/// Tuple of (means vector, standard deviations vector) for each feature
pub fn normalize_features(features: &mut Array2<f32>) -> (Vec<f32>, Vec<f32>) {
    let mut means = Vec::with_capacity(NUM_FEATURES);
    let mut stds = Vec::with_capacity(NUM_FEATURES);

    for i in 0..NUM_FEATURES {
        let col = features.column(i);

        // Calculate mean
        let sum: f32 = col.iter().sum();
        let mean = sum / col.len() as f32;
        means.push(mean);

        // Calculate standard deviation
        let sum_squared_deviations: f32 = col.iter().map(|&val| (val - mean).powi(2)).sum();
        let std_dev = (sum_squared_deviations / col.len() as f32).sqrt();
        stds.push(if std_dev < MIN_STD_DEV { 1.0 } else { std_dev });

        // Normalize the column
        for val in features.column_mut(i) {
            *val = (*val - mean) / stds[i];
        }
    }

    (means, stds)
}

/// Normalizes targets to have zero mean and unit variance (z-score normalization)
///
/// # Arguments
/// * `targets` - Mutable reference to targets vector (modified in-place)
///
/// # Returns
/// Tuple of (mean, standard deviation) for the targets
pub fn normalize_targets(targets: &mut Array1<f32>) -> (f32, f32) {
    let n_samples = targets.len();

    // Calculate mean
    let sum: f32 = targets.sum();
    let mean = sum / n_samples as f32;

    // Calculate standard deviation
    let sum_squared_deviations: f32 = targets.iter().map(|&val| (val - mean).powi(2)).sum();
    let std_dev = (sum_squared_deviations / n_samples as f32).sqrt();
    let std_dev = if std_dev < MIN_STD_DEV { 1.0 } else { std_dev };

    // Normalize the targets
    for val in targets.iter_mut() {
        *val = (*val - mean) / std_dev;
    }

    (mean, std_dev)
}

/// Prepares tensor data for the NAM model
///
/// Converts ndarray structures into Tensor format required by the model.
///
/// # Arguments
/// * `features` - Features matrix [N x 8]
/// * `targets` - Targets vector [N]
///
/// # Returns
/// Vector of labeled tensor samples for the data loader
pub fn prepare_tensor_data(
    features: Array2<f32>,
    targets: Array1<f32>,
) -> Vec<LabeledTensorSample<Rank1<S<NUM_FEATURES>>, Rank1<S<1>>>> {
    let n_samples = features.shape()[0];
    let mut labeled_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let feature_row = features.row(i).to_vec();
        let feature_array = Array::from_shape_vec(IxDyn(&[NUM_FEATURES]), feature_row).unwrap();
        let feature_tensor = Tensor::new(feature_array);

        let target_array = Array::from_shape_vec(IxDyn(&[1]), vec![targets[i]]).unwrap();
        let target_tensor = Tensor::new(target_array);

        labeled_data.push(LabeledTensorSample {
            input: feature_tensor,
            label: target_tensor,
        });
    }

    labeled_data
}
