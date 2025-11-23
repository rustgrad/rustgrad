use std::error::Error;

use tensorboard_rs::summary_writer::SummaryWriter;

use crate::data::labeled_dataset::{LabeledTensorDataLoader, LabeledTensorSample};
use crate::data::loader::DataLoaderExt;
use crate::dimensions::{Rank1, Rank2, S};
use crate::nn::mlp::MLP;
use crate::optim::adam::AdamOptimizer;
use crate::optim::optimizer::Optimizer;
use crate::tensor::Tensor;

use super::data::{NUM_CLASSES, NUM_FEATURES};

/// Training hyperparameters
pub const BATCH_SIZE: usize = 16;
pub const HIDDEN_SIZE: usize = 16; // Reduced from 32 to reduce overfitting
pub const NUM_HIDDEN_LAYERS: usize = 1; // Reduced from 2 to reduce overfitting
pub const LEARNING_RATE: f32 = 0.001;
pub const NUM_EPOCHS: usize = 100;
pub const DROPOUT_RATE: f32 = 0.2; // Dropout probability
pub const WEIGHT_DECAY: f32 = 0.01; // L2 regularization strength

/// Calculates cross-entropy loss for multi-class classification
///
/// # Arguments
/// * `predictions` - Model predictions after softmax [batch_size x num_classes]
/// * `targets` - One-hot encoded true labels [batch_size x num_classes]
///
/// # Returns
/// Scalar loss tensor (mean cross-entropy over the batch)
pub fn cross_entropy_loss<const BATCH: usize>(
    predictions: Tensor<Rank2<S<BATCH>, S<NUM_CLASSES>>>,
    targets: Tensor<Rank2<S<BATCH>, S<NUM_CLASSES>>>,
) -> Tensor<(S<1>,)> {
    // Cross-entropy: -sum(targets * log(predictions)) / batch_size
    // Add epsilon for numerical stability
    let epsilon = 1e-7;
    let epsilon_tensor: Tensor<Rank2<S<BATCH>, S<NUM_CLASSES>>> =
        Tensor::new(ndarray::Array::from_elem(predictions.shape(), epsilon));
    let pred_clipped = predictions.clone() + epsilon_tensor;

    let log_preds = pred_clipped.log();
    let loss_per_sample = targets * log_preds;
    let mean_loss = loss_per_sample.mean();
    let loss = -mean_loss;

    loss.reshape((S {},))
}

/// Calculates accuracy for classification
///
/// # Arguments
/// * `predictions` - Model predictions [batch_size x num_classes]
/// * `targets` - One-hot encoded true labels [batch_size x num_classes]
///
/// # Returns
/// Accuracy as a float between 0 and 1
pub fn calculate_accuracy(
    predictions: &ndarray::Array2<f32>,
    targets: &ndarray::Array2<f32>,
) -> f32 {
    let batch_size = predictions.shape()[0];
    let mut correct = 0;

    for i in 0..batch_size {
        let pred_class = predictions
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let true_class = targets
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        if pred_class == true_class {
            correct += 1;
        }
    }

    correct as f32 / batch_size as f32
}

/// Trains the MLP classifier on the provided data
///
/// # Arguments
/// * `model` - The MLP model to train
/// * `optimizer` - The optimizer to use for training
/// * `dataloader` - Data loader providing batches of training data
/// * `writer` - TensorBoard summary writer for logging
/// * `num_epochs` - Number of training epochs
///
/// # Returns
/// Tuple of (per-batch losses, per-epoch average losses, per-epoch accuracies)
pub fn train_classifier(
    model: &MLP<S<NUM_FEATURES>, S<NUM_CLASSES>, S<HIDDEN_SIZE>, NUM_HIDDEN_LAYERS>,
    optimizer: &mut AdamOptimizer,
    dataloader: &mut LabeledTensorDataLoader<(S<NUM_FEATURES>,), (S<NUM_CLASSES>,), S<BATCH_SIZE>>,
    writer: &mut SummaryWriter,
    num_epochs: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let mut train_losses = Vec::with_capacity(num_epochs * 100);
    let mut train_epoch_losses = Vec::with_capacity(num_epochs);
    let mut train_epoch_accuracies = Vec::with_capacity(num_epochs);
    let mut total_steps = 0;

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let mut epoch_accuracy = 0.0;
        let mut batches: u64 = 0;

        for (_idx, batch) in dataloader.iter_shuffled(epoch as u64 + 1).enumerate() {
            let x = batch.input;
            let y_true = batch.label;

            // Forward pass
            let logits = model.forward(x.clone());

            // Apply dropout during training
            let logits_dropout = logits.dropout(DROPOUT_RATE, true);

            // Reshape for softmax: (BATCH_SIZE, NUM_FEATURES) -> (BATCH_SIZE, NUM_CLASSES)
            let logits_2d: Tensor<Rank2<S<BATCH_SIZE>, S<NUM_CLASSES>>> =
                logits_dropout.reshape((S {}, S {}));
            let y_true_2d: Tensor<Rank2<S<BATCH_SIZE>, S<NUM_CLASSES>>> =
                y_true.clone().reshape((S {}, S {}));

            // Apply softmax
            let y_pred = logits_2d.softmax();

            // Calculate cross-entropy loss
            let loss = cross_entropy_loss(y_pred.clone(), y_true_2d.clone());

            let loss_val = loss.data().into_flat()[0];
            epoch_loss += loss_val;
            train_losses.push(loss_val);
            writer.add_scalar("loss", loss_val, total_steps);

            // Calculate accuracy
            let pred_data = y_pred.data().into_dimensionality::<ndarray::Ix2>().unwrap();
            let true_data = y_true_2d
                .data()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let batch_accuracy = calculate_accuracy(&pred_data, &true_data);
            epoch_accuracy += batch_accuracy;

            batches += 1;
            total_steps += 1;

            // Backward pass and optimization
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        let avg_loss = epoch_loss / batches as f32;
        let avg_accuracy = epoch_accuracy / batches as f32;
        train_epoch_losses.push(avg_loss);
        train_epoch_accuracies.push(avg_accuracy);

        writer.add_scalar("epoch_accuracy", avg_accuracy, epoch);

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!(
                "Epoch {}/{}, Loss: {:.4}, Accuracy: {:.2}%",
                epoch + 1,
                num_epochs,
                avg_loss,
                avg_accuracy * 100.0
            );
        }
    }

    Ok((train_losses, train_epoch_losses, train_epoch_accuracies))
}

/// Evaluation metrics for classification
#[derive(Debug)]
pub struct ClassificationMetrics {
    /// Overall accuracy
    pub accuracy: f32,
    /// Per-class accuracy
    pub per_class_accuracy: Vec<f32>,
    /// Confusion matrix [true_class x predicted_class]
    pub confusion_matrix: Vec<Vec<usize>>,
}

/// Evaluates the classifier on the provided dataset
///
/// # Arguments
/// * `model` - The trained MLP model
/// * `labeled_data` - Vector of labeled samples to evaluate on
///
/// # Returns
/// Classification metrics structure
pub fn evaluate_classifier(
    model: &MLP<S<NUM_FEATURES>, S<NUM_CLASSES>, S<HIDDEN_SIZE>, NUM_HIDDEN_LAYERS>,
    labeled_data: &[LabeledTensorSample<Rank1<S<NUM_FEATURES>>, Rank1<S<NUM_CLASSES>>>],
) -> ClassificationMetrics {
    let mut confusion_matrix = vec![vec![0; NUM_CLASSES]; NUM_CLASSES];
    let mut total_correct = 0;
    let mut per_class_correct = vec![0; NUM_CLASSES];
    let mut per_class_total = vec![0; NUM_CLASSES];

    let xshape: (S<1>, S<NUM_FEATURES>) = (S {}, S {});
    let ytrue_shape: (S<1>, S<NUM_CLASSES>) = (S {}, S {});

    for sample in labeled_data {
        let x: Tensor<(S<1>, S<NUM_FEATURES>)> = sample.input.clone().reshape(xshape);
        let y_true: Tensor<(S<1>, S<NUM_CLASSES>)> = sample.label.clone().reshape(ytrue_shape);

        let logits = model.forward(x.clone());
        // No dropout during evaluation (training=false)
        let logits_eval = logits.dropout(0.0, false);
        let logits_2d: Tensor<Rank2<S<1>, S<NUM_CLASSES>>> = logits_eval.reshape((S {}, S {}));
        let y_pred = logits_2d.softmax();

        // Get predicted and true class
        let pred_data = y_pred.data();
        let true_data = y_true.data();

        let pred_class = (0..NUM_CLASSES)
            .max_by(|&a, &b| pred_data[[0, a]].partial_cmp(&pred_data[[0, b]]).unwrap())
            .unwrap();

        let true_class = (0..NUM_CLASSES)
            .max_by(|&a, &b| true_data[[0, a]].partial_cmp(&true_data[[0, b]]).unwrap())
            .unwrap();

        confusion_matrix[true_class][pred_class] += 1;
        per_class_total[true_class] += 1;

        if pred_class == true_class {
            total_correct += 1;
            per_class_correct[true_class] += 1;
        }
    }

    let accuracy = total_correct as f32 / labeled_data.len() as f32;
    let per_class_accuracy: Vec<f32> = per_class_correct
        .iter()
        .zip(per_class_total.iter())
        .map(|(&correct, &total)| {
            if total > 0 {
                correct as f32 / total as f32
            } else {
                0.0
            }
        })
        .collect();

    ClassificationMetrics {
        accuracy,
        per_class_accuracy,
        confusion_matrix,
    }
}

impl ClassificationMetrics {
    /// Prints the evaluation metrics in a formatted way
    pub fn print(&self) {
        println!("\nFinal evaluation results:");
        println!("  Overall Accuracy: {:.2}%", self.accuracy * 100.0);
        println!("\nPer-class Accuracy:");
        for (i, acc) in self.per_class_accuracy.iter().enumerate() {
            println!("    Class {}: {:.2}%", i, acc * 100.0);
        }
        println!("\nConfusion Matrix:");
        println!("         Pred 0  Pred 1  Pred 2");
        for (i, row) in self.confusion_matrix.iter().enumerate() {
            print!("True {}     ", i);
            for count in row {
                print!("{:6}  ", count);
            }
            println!();
        }
    }
}
