use std::error::Error;
use std::path::Path;

use tensorboard_rs::summary_writer::SummaryWriter;

use crate::data::labeled_dataset::LabeledTensorDataLoader;
use crate::dimensions::S;
use crate::nam::NAM;
use crate::optim::adam::AdamOptimizer;

use super::data::{
    load_housing_dataset, normalize_features, normalize_targets, prepare_tensor_data,
    FEATURE_NAMES, NUM_FEATURES,
};
use super::training::{
    evaluate_model, train_nam_model, BATCH_SIZE, HIDDEN_SIZE, LEARNING_RATE, NUM_EPOCHS,
    NUM_LAYERS,
};
use super::visualization::{plot_shape_functions, plot_training_loss};

/// Runs the NAM experiment on the California housing dataset
///
/// This is the main orchestrator function that:
/// 1. Loads and preprocesses the housing data
/// 2. Creates and trains a NAM model
/// 3. Visualizes the learned shape functions
/// 4. Evaluates the model performance
///
/// # Returns
/// Ok(()) on success, or an error if any step fails
pub fn run_housing_nam_experiment() -> Result<(), Box<dyn Error>> {
    println!("Loading California housing dataset...");
    let housing_csv_path = Path::new(file!()).parent().unwrap().join("housing.csv");
    let (mut features, mut targets) = load_housing_dataset(housing_csv_path)?;

    println!("Normalizing features...");
    let (feature_means, feature_stds) = normalize_features(&mut features);

    println!("Normalizing targets...");
    let (target_mean, target_std) = normalize_targets(&mut targets);

    // Convert to tensor data
    let labeled_data = prepare_tensor_data(features, targets);

    println!("Creating NAM model...");
    let model = NAM::<S<HIDDEN_SIZE>, NUM_FEATURES>::new(NUM_LAYERS);
    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    println!("Setting up dataloader and optimizer...");
    let mut dataloader: LabeledTensorDataLoader<(S<NUM_FEATURES>,), (S<1>,), S<BATCH_SIZE>> =
        LabeledTensorDataLoader::new(labeled_data.clone());

    let params = model.parameters();
    let mut opt = AdamOptimizer::new_with_defaults(LEARNING_RATE, params);

    // Training loop
    println!("Starting training for {} epochs...", NUM_EPOCHS);
    let (train_losses, train_epoch_losses) =
        train_nam_model(&model, &mut opt, &mut dataloader, &mut writer, NUM_EPOCHS)?;
    writer.flush();

    // Plot training loss
    let plots_dir = Path::new(file!()).parent().unwrap().join("plots");
    std::fs::create_dir_all(&plots_dir)?;

    println!("Plotting training loss...");
    plot_training_loss(train_losses, &plots_dir, "housing_")?;
    plot_training_loss(train_epoch_losses, &plots_dir, "housing_epoch_")?;

    // Plot individual shape functions
    println!("Plotting shape functions...");
    plot_shape_functions::<HIDDEN_SIZE, NUM_FEATURES>(
        &model,
        &FEATURE_NAMES,
        &feature_means,
        &feature_stds,
        &plots_dir,
    )?;

    // Evaluate model on the full dataset
    println!("Evaluating model...");
    let metrics = evaluate_model(&model, &labeled_data, target_mean, target_std);
    metrics.print();

    println!(
        "Shape function plots saved to: {}",
        plots_dir.to_str().unwrap_or_default()
    );

    Ok(())
}

/// Binary entry point to run the experiment
pub fn main() -> Result<(), Box<dyn Error>> {
    run_housing_nam_experiment()
}
