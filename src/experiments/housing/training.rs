use std::error::Error;

use tensorboard_rs::summary_writer::SummaryWriter;

use crate::data::labeled_dataset::{LabeledTensorDataLoader, LabeledTensorSample};
use crate::data::loader::DataLoaderExt;
use crate::dimensions::{Rank1, S};
use crate::nam::NAM;
use crate::optim::adam::AdamOptimizer;
use crate::optim::optimizer::Optimizer;
use crate::tensor::Tensor;

use super::data::NUM_FEATURES;

/// Training hyperparameters
pub const BATCH_SIZE: usize = 16;
pub const HIDDEN_SIZE: usize = 64;
pub const NUM_LAYERS: usize = 3;
pub const LEARNING_RATE: f32 = 0.0001;
pub const NUM_EPOCHS: usize = 10;

/// Trains the NAM model on the provided data
///
/// # Arguments
/// * `model` - The NAM model to train
/// * `optimizer` - The optimizer to use for training
/// * `dataloader` - Data loader providing batches of training data
/// * `writer` - TensorBoard summary writer for logging
/// * `num_epochs` - Number of training epochs
///
/// # Returns
/// Tuple of (per-batch losses, per-epoch average losses)
pub fn train_nam_model(
    model: &NAM<S<HIDDEN_SIZE>, NUM_FEATURES>,
    optimizer: &mut AdamOptimizer,
    dataloader: &mut LabeledTensorDataLoader<(S<NUM_FEATURES>,), (S<1>,), S<BATCH_SIZE>>,
    writer: &mut SummaryWriter,
    num_epochs: usize,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let mut train_losses = Vec::with_capacity(num_epochs * 100);
    let mut train_epoch_losses = Vec::with_capacity(num_epochs);
    let mut total_steps = 0;

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let mut batches: u64 = 0;

        for (_idx, batch) in dataloader.iter_shuffled(epoch as u64 + 1).enumerate() {
            let x = batch.input;
            let y_true = batch.label;

            // Forward pass
            let y_pred = model.forward(x.clone());

            // Calculate mean squared error loss
            let diff = y_pred.clone() + -y_true.clone();
            let loss = diff.clone() * diff; // squared error
            let loss = loss.mean(); // MSE

            let loss_val = loss.data().into_flat()[0];
            epoch_loss += loss_val;
            train_losses.push(loss_val);
            writer.add_scalar("loss", loss_val, total_steps);
            batches += 1;
            total_steps += 1;

            // Backward pass and optimization
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        // Learning rate decay (reduce by 50% each epoch)
        optimizer.update_lr(optimizer.lr() * 0.5);

        let avg_loss = epoch_loss / batches as f32;
        train_epoch_losses.push(avg_loss);
        println!("Epoch {}/{}, Loss: {:.4}", epoch + 1, num_epochs, avg_loss);
    }

    Ok((train_losses, train_epoch_losses))
}

/// Evaluation metrics for the model
#[derive(Debug)]
pub struct EvaluationMetrics {
    /// Mean Squared Error in normalized space
    pub normalized_mse: f32,
    /// Root Mean Squared Error in normalized space
    pub normalized_rmse: f32,
    /// Mean Squared Error in original scale
    pub denormalized_mse: f32,
    /// Root Mean Squared Error in original scale
    pub denormalized_rmse: f32,
}

/// Evaluates the NAM model on the provided dataset
///
/// Computes metrics in both normalized space (as used during training)
/// and in the original scale (denormalized predictions).
///
/// # Arguments
/// * `model` - The trained NAM model
/// * `labeled_data` - Vector of labeled samples to evaluate on
/// * `target_mean` - Mean value used for target normalization
/// * `target_std` - Standard deviation used for target normalization
///
/// # Returns
/// Evaluation metrics structure
pub fn evaluate_model(
    model: &NAM<S<HIDDEN_SIZE>, NUM_FEATURES>,
    labeled_data: &[LabeledTensorSample<Rank1<S<NUM_FEATURES>>, Rank1<S<1>>>],
    target_mean: f32,
    target_std: f32,
) -> EvaluationMetrics {
    let mut total_normalized_loss = 0.0;
    let mut total_denormalized_loss = 0.0;
    let mut samples = 0;

    let xshape: (S<1>, S<NUM_FEATURES>) = (S {}, S {});
    let ytrue_shape: (S<1>, S<1>) = (S {}, S {});

    for sample in labeled_data {
        let x: Tensor<(S<1>, S<NUM_FEATURES>)> = sample.input.clone().reshape(xshape);
        let y_true: Tensor<(S<1>, S<1>)> = sample.label.clone().reshape(ytrue_shape);

        let y_pred = model.forward(x.clone());

        // Calculate loss in normalized space
        let diff = y_pred.clone() + -y_true.clone();
        let normalized_loss = diff.clone() * diff; // squared error
        total_normalized_loss += normalized_loss.data().into_flat()[0];

        // Calculate loss in original space (denormalized)
        let y_pred_denorm = y_pred.data().into_flat()[0] * target_std + target_mean;
        let y_true_denorm = y_true.data().into_flat()[0] * target_std + target_mean;
        let denorm_diff = y_pred_denorm - y_true_denorm;
        let denormalized_loss = denorm_diff * denorm_diff;
        total_denormalized_loss += denormalized_loss;

        samples += 1;
    }

    // Compute final metrics
    let normalized_mse = total_normalized_loss / samples as f32;
    let normalized_rmse = normalized_mse.sqrt();
    let denormalized_mse = total_denormalized_loss / samples as f32;
    let denormalized_rmse = denormalized_mse.sqrt();

    EvaluationMetrics {
        normalized_mse,
        normalized_rmse,
        denormalized_mse,
        denormalized_rmse,
    }
}

impl EvaluationMetrics {
    /// Prints the evaluation metrics in a formatted way
    pub fn print(&self) {
        println!("Final evaluation results:");
        println!("  Normalized Mean Squared Error: {:.4}", self.normalized_mse);
        println!("  Normalized RMSE: {:.4}", self.normalized_rmse);
        println!(
            "  Mean Squared Error (original scale): {:.4}",
            self.denormalized_mse
        );
        println!("  RMSE (original scale): {:.4}", self.denormalized_rmse);
    }
}
