use std::error::Error;
use std::path::Path;

use tensorboard_rs::summary_writer::SummaryWriter;

use crate::data::labeled_dataset::LabeledTensorDataLoader;
use crate::dimensions::S;
use crate::nn::mlp::MLP;
use crate::optim::adam::AdamOptimizer;

use super::data::{
    load_iris_dataset, normalize_features, prepare_tensor_data, train_test_split, NUM_CLASSES,
    NUM_FEATURES,
};
use super::training::{
    evaluate_classifier, train_classifier, BATCH_SIZE, DROPOUT_RATE, HIDDEN_SIZE, LEARNING_RATE,
    NUM_EPOCHS, NUM_HIDDEN_LAYERS, WEIGHT_DECAY,
};
use super::visualization::{
    plot_confusion_matrix, plot_training_accuracy, plot_training_loss,
};

/// Runs the Iris classification experiment
///
/// This is the main orchestrator function that:
/// 1. Loads and preprocesses the Iris dataset
/// 2. Splits into train/test sets
/// 3. Creates and trains an MLP classifier
/// 4. Visualizes training progress and results
/// 5. Evaluates the model performance
///
/// # Returns
/// Ok(()) on success, or an error if any step fails
pub fn run_iris_experiment() -> Result<(), Box<dyn Error>> {
    println!("Loading Iris dataset...");
    let iris_csv_path = Path::new(file!()).parent().unwrap().join("iris.csv");
    let (mut features, labels) = load_iris_dataset(iris_csv_path)?;

    println!("Dataset loaded: {} samples", features.shape()[0]);

    println!("Normalizing features...");
    let (_feature_means, _feature_stds) = normalize_features(&mut features);

    println!("Splitting into train/test sets (80/20)...");
    let ((train_features, train_labels), (test_features, test_labels)) =
        train_test_split(features, labels, 0.2);

    println!(
        "Train samples: {}, Test samples: {}",
        train_features.shape()[0],
        test_features.shape()[0]
    );

    // Convert to tensor data
    let train_data = prepare_tensor_data(train_features, train_labels);
    let test_data = prepare_tensor_data(test_features, test_labels);

    println!("Creating MLP classifier...");
    println!(
        "  Architecture: {} -> {} (x{} layers) -> {}",
        NUM_FEATURES, HIDDEN_SIZE, NUM_HIDDEN_LAYERS, NUM_CLASSES
    );
    let model = MLP::<S<NUM_FEATURES>, S<NUM_CLASSES>, S<HIDDEN_SIZE>, NUM_HIDDEN_LAYERS>::new();
    let mut writer = SummaryWriter::new(&("./logdir_iris".to_string()));

    println!("Setting up dataloader and optimizer...");
    let mut dataloader: LabeledTensorDataLoader<(S<NUM_FEATURES>,), (S<NUM_CLASSES>,), S<BATCH_SIZE>> =
        LabeledTensorDataLoader::new(train_data.clone());

    let params = model.parameters();
    let mut opt = AdamOptimizer::new_with_weight_decay(LEARNING_RATE, WEIGHT_DECAY, params);

    // Training loop
    println!(
        "Starting training for {} epochs (batch size: {}, lr: {}, dropout: {}, weight_decay: {})...",
        NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE, WEIGHT_DECAY
    );
    let (train_losses, train_epoch_losses, train_accuracies) =
        train_classifier(&model, &mut opt, &mut dataloader, &mut writer, NUM_EPOCHS)?;
    writer.flush();

    // Create plots directory
    let plots_dir = Path::new(file!()).parent().unwrap().join("plots");
    std::fs::create_dir_all(&plots_dir)?;

    // Plot training metrics
    println!("Plotting training metrics...");
    plot_training_loss(train_losses, &plots_dir, "iris_")?;
    plot_training_loss(train_epoch_losses, &plots_dir, "iris_epoch_")?;
    plot_training_accuracy(train_accuracies, &plots_dir, "iris_")?;

    // Evaluate on training set
    println!("\nEvaluating on training set...");
    let train_metrics = evaluate_classifier(&model, &train_data);
    println!("Training Set:");
    train_metrics.print();

    // Evaluate on test set
    println!("\nEvaluating on test set...");
    let test_metrics = evaluate_classifier(&model, &test_data);
    println!("Test Set:");
    test_metrics.print();

    // Plot confusion matrix
    println!("\nPlotting confusion matrix...");
    plot_confusion_matrix(&test_metrics.confusion_matrix, &plots_dir, "iris_test_")?;
    plot_confusion_matrix(&train_metrics.confusion_matrix, &plots_dir, "iris_train_")?;

    println!(
        "\nExperiment complete! Plots saved to: {}",
        plots_dir.to_str().unwrap_or_default()
    );

    Ok(())
}

/// Binary entry point to run the experiment
pub fn main() -> Result<(), Box<dyn Error>> {
    run_iris_experiment()
}
