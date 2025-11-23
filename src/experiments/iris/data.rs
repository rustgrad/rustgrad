use std::error::Error;
use std::fs::File;
use std::path::Path;

use csv::ReaderBuilder;
use ndarray::{Array, Array1, Array2, IxDyn};
use serde::Deserialize;

use crate::data::labeled_dataset::LabeledTensorSample;
use crate::dimensions::{Rank1, S};
use crate::tensor::Tensor;

/// The number of features in the Iris dataset
pub const NUM_FEATURES: usize = 4;

/// The number of classes in the Iris dataset
pub const NUM_CLASSES: usize = 3;

/// Feature names in the Iris dataset
pub const FEATURE_NAMES: [&str; NUM_FEATURES] =
    ["sepal_length", "sepal_width", "petal_length", "petal_width"];

/// Class names in the Iris dataset
pub const CLASS_NAMES: [&str; NUM_CLASSES] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];

/// Minimum standard deviation threshold to avoid division by zero
const MIN_STD_DEV: f32 = 1e-6;

/// A single record from the Iris dataset
#[derive(Debug, Deserialize)]
pub struct IrisRecord {
    sepal_length: f32,
    sepal_width: f32,
    petal_length: f32,
    petal_width: f32,
    species: String,
}

impl IrisRecord {
    /// Extracts features as a vector
    fn to_feature_vec(&self) -> Vec<f32> {
        vec![
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]
    }

    /// Converts species name to integer label (0, 1, or 2)
    fn species_to_label(&self) -> usize {
        match self.species.as_str() {
            "Iris-setosa" => 0,
            "Iris-versicolor" => 1,
            "Iris-virginica" => 2,
            _ => panic!("Unknown species: {}", self.species),
        }
    }
}

/// Loads the Iris dataset from a CSV file
///
/// This function reads the CSV and returns features and labels as ndarray structures.
///
/// # Arguments
/// * `path` - Path to the iris.csv file
///
/// # Returns
/// Tuple of (features matrix [N x 4], labels vector [N])
pub fn load_iris_dataset(
    path: impl AsRef<Path>,
) -> Result<(Array2<f32>, Array1<usize>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(File::open(path)?);

    // Read all records
    let records: Vec<IrisRecord> = rdr.deserialize().filter_map(|result| result.ok()).collect();

    let n_samples = records.len();
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for record in records {
        features.push(record.to_feature_vec());
        labels.push(record.species_to_label());
    }

    let x = Array2::from_shape_vec(
        (n_samples, NUM_FEATURES),
        features.into_iter().flatten().collect(),
    )?;
    let y = Array1::from_shape_vec(n_samples, labels)?;

    Ok((x, y))
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

/// Converts integer labels to one-hot encoded vectors
///
/// # Arguments
/// * `labels` - Vector of class labels (0, 1, or 2)
///
/// # Returns
/// Matrix of one-hot encoded labels [N x NUM_CLASSES]
pub fn labels_to_one_hot(labels: &Array1<usize>) -> Array2<f32> {
    let n_samples = labels.len();
    let mut one_hot = Array2::zeros((n_samples, NUM_CLASSES));

    for (i, &label) in labels.iter().enumerate() {
        one_hot[[i, label]] = 1.0;
    }

    one_hot
}

/// Prepares tensor data for the classification model
///
/// Converts ndarray structures into Tensor format required by the model.
///
/// # Arguments
/// * `features` - Features matrix [N x 4]
/// * `labels` - Labels vector [N] with integer class labels
///
/// # Returns
/// Vector of labeled tensor samples for the data loader
pub fn prepare_tensor_data(
    features: Array2<f32>,
    labels: Array1<usize>,
) -> Vec<LabeledTensorSample<Rank1<S<NUM_FEATURES>>, Rank1<S<NUM_CLASSES>>>> {
    let n_samples = features.shape()[0];
    let one_hot_labels = labels_to_one_hot(&labels);
    let mut labeled_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let feature_row = features.row(i).to_vec();
        let feature_array = Array::from_shape_vec(IxDyn(&[NUM_FEATURES]), feature_row).unwrap();
        let feature_tensor = Tensor::new(feature_array);

        let label_row = one_hot_labels.row(i).to_vec();
        let label_array = Array::from_shape_vec(IxDyn(&[NUM_CLASSES]), label_row).unwrap();
        let label_tensor = Tensor::new(label_array);

        labeled_data.push(LabeledTensorSample {
            input: feature_tensor,
            label: label_tensor,
        });
    }

    labeled_data
}

/// Splits data into training and test sets using stratified sampling
/// This ensures each class is proportionally represented in both train and test sets
///
/// # Arguments
/// * `features` - All features [N x 4]
/// * `labels` - All labels [N]
/// * `test_size` - Fraction of data to use for testing (e.g., 0.2 for 20%)
///
/// # Returns
/// Tuple of ((train_features, train_labels), (test_features, test_labels))
pub fn train_test_split(
    features: Array2<f32>,
    labels: Array1<usize>,
    test_size: f32,
) -> ((Array2<f32>, Array1<usize>), (Array2<f32>, Array1<usize>)) {
    let n_samples = features.shape()[0];

    // Group samples by class
    let mut class_indices: Vec<Vec<usize>> = vec![Vec::new(); NUM_CLASSES];
    for (idx, &label) in labels.iter().enumerate() {
        class_indices[label].push(idx);
    }

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    // Stratified split: take proportional samples from each class
    for class_idx in class_indices {
        let class_size = class_idx.len();
        let class_test_size = (class_size as f32 * test_size).round() as usize;
        let class_train_size = class_size - class_test_size;

        // Split this class's samples
        train_indices.extend_from_slice(&class_idx[..class_train_size]);
        test_indices.extend_from_slice(&class_idx[class_train_size..]);
    }

    // Extract features and labels for train set
    let mut train_features_vec = Vec::new();
    let mut train_labels_vec = Vec::new();
    for &idx in &train_indices {
        train_features_vec.extend(features.row(idx).iter());
        train_labels_vec.push(labels[idx]);
    }

    // Extract features and labels for test set
    let mut test_features_vec = Vec::new();
    let mut test_labels_vec = Vec::new();
    for &idx in &test_indices {
        test_features_vec.extend(features.row(idx).iter());
        test_labels_vec.push(labels[idx]);
    }

    let train_features =
        Array2::from_shape_vec((train_indices.len(), NUM_FEATURES), train_features_vec).unwrap();
    let train_labels = Array1::from_shape_vec(train_indices.len(), train_labels_vec).unwrap();

    let test_features =
        Array2::from_shape_vec((test_indices.len(), NUM_FEATURES), test_features_vec).unwrap();
    let test_labels = Array1::from_shape_vec(test_indices.len(), test_labels_vec).unwrap();

    ((train_features, train_labels), (test_features, test_labels))
}
