use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};

use csv::ReaderBuilder;
use ndarray::{Array, Array1, Array2, IxDyn};
use plotters::prelude::*;
use serde::Deserialize;
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::data::labeled_dataset::{LabeledTensorDataLoader, LabeledTensorSample};
use crate::data::loader::DataLoaderExt;
use crate::dimensions::{Rank1, S};
use crate::nam::NAM;
use crate::optim::adam::AdamOptimizer;
use crate::optim::optimizer::Optimizer;
use crate::tensor::Tensor;

// The number of numeric features we'll use from the dataset
// Excluding 'ocean_proximity' which is categorical
const NUM_FEATURES: usize = 8;
const BATCH_SIZE: usize = 16;
const HIDDEN_SIZE: usize = 64;
const NUM_LAYERS: usize = 3;
const LEARNING_RATE: f32 = 0.0001;
const NUM_EPOCHS: usize = 10;

#[derive(Debug, Deserialize)]
struct HousingRecord {
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

/// Loads the California housing dataset from a CSV file
pub fn load_housing_dataset(
    path: impl AsRef<Path>,
) -> Result<(Array2<f32>, Array1<f32>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(File::open(path)?);

    let mut features = Vec::new();
    let mut targets = Vec::new();

    // Calculate mean values for each column to use for filling missing values
    let mut column_sums = vec![0.0; NUM_FEATURES];
    let mut column_counts = vec![0; NUM_FEATURES];
    let mut records = Vec::new();

    // First pass: collect all valid values and calculate sums
    for result in rdr.deserialize() {
        if let Ok(record) = result {
            let record: HousingRecord = record;
            records.push(record);
        }
    }

    // Calculate means for each feature
    for record in &records {
        // For simplicity, we'll only use numeric features and ignore ocean_proximity
        let feature_values = [
            record.longitude,
            record.latitude,
            record.housing_median_age,
            record.total_rooms,
            record.total_bedrooms,
            record.population,
            record.households,
            record.median_income,
        ];

        for (i, &value) in feature_values.iter().enumerate() {
            if !value.is_nan() {
                column_sums[i] += value;
                column_counts[i] += 1;
            }
        }
    }

    // Calculate means for each column
    let column_means: Vec<f32> = column_sums
        .iter()
        .zip(column_counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
        .collect();

    // Second pass: process records, filling missing values with means
    for record in records {
        // For simplicity, we'll only use numeric features and ignore ocean_proximity
        let mut feature_row = vec![
            record.longitude,
            record.latitude,
            record.housing_median_age,
            record.total_rooms,
            record.total_bedrooms,
            record.population,
            record.households,
            record.median_income,
        ];

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

/// Normalize features to have zero mean and unit variance
fn normalize_features(features: &mut Array2<f32>) -> (Vec<f32>, Vec<f32>) {
    let mut means = Vec::with_capacity(NUM_FEATURES);
    let mut stds = Vec::with_capacity(NUM_FEATURES);

    for i in 0..NUM_FEATURES {
        let mut sum = 0.0;
        let col = features.column(i);

        // Calculate mean
        for &val in col.iter() {
            sum += val;
        }
        let mean = sum / col.len() as f32;
        means.push(mean);

        // Calculate standard deviation
        let mut sum_sq_diff = 0.0;
        for &val in col.iter() {
            sum_sq_diff += (val - mean).powi(2);
        }
        let std_dev = (sum_sq_diff / col.len() as f32).sqrt();
        stds.push(if std_dev < 1e-6 { 1.0 } else { std_dev });

        // Normalize the column
        for val in features.column_mut(i) {
            *val = (*val - mean) / stds[i];
        }
    }

    (means, stds)
}

/// Normalize targets to have zero mean and unit variance
fn normalize_targets(targets: &mut Array1<f32>) -> (f32, f32) {
    let n_samples = targets.len();

    // Calculate mean
    let sum: f32 = targets.sum();
    let mean = sum / n_samples as f32;

    // Calculate standard deviation
    let mut sum_sq_diff = 0.0;
    for &val in targets.iter() {
        sum_sq_diff += (val - mean).powi(2);
    }
    let std_dev = (sum_sq_diff / n_samples as f32).sqrt();
    let std_dev = if std_dev < 1e-6 { 1.0 } else { std_dev };

    // Normalize the targets
    for val in targets.iter_mut() {
        *val = (*val - mean) / std_dev;
    }

    (mean, std_dev)
}

/// Prepare tensor data for NAM
fn prepare_tensor_data(
    features: Array2<f32>,
    targets: Array1<f32>,
) -> Vec<LabeledTensorSample<Rank1<S<NUM_FEATURES>>, Rank1<S<1>>>> {
    let n_samples = features.shape()[0];
    let mut labeled_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let feature_row = features.row(i).to_vec();
        // Convert Vec<f32> to ndarray::Array first
        let feature_array = Array::from_shape_vec(IxDyn(&[NUM_FEATURES]), feature_row).unwrap();
        let feature_tensor = Tensor::new(feature_array);

        // Convert Vec<f32> to ndarray::Array first
        let target_array = Array::from_shape_vec(IxDyn(&[1]), vec![targets[i]]).unwrap();
        let target_tensor = Tensor::new(target_array);

        labeled_data.push(LabeledTensorSample {
            input: feature_tensor,
            label: target_tensor,
        });
    }

    labeled_data
}

/// Plot the individual feature functions learned by the NAM model
fn plot_shape_functions(
    model: &NAM<S<HIDDEN_SIZE>, NUM_FEATURES>,
    feature_names: &[&str],
    means: &[f32],
    stds: &[f32],
    output_dir: &PathBuf,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(output_dir)?;

    for (i, feature_name) in feature_names.iter().enumerate() {
        let filename = format!("{}_shape_function.png", feature_name);
        let path = output_dir.join(filename);

        let root_backend = BitMapBackend::new(&path, (800, 600));
        let root = root_backend.into_drawing_area();
        root.fill(&WHITE)?;

        let min_val = -3.0; // -3 standard deviations
        let max_val = 3.0; // +3 standard deviations
        let n_points = 100;
        let step = (max_val - min_val) / (n_points as f32 - 1.0);

        let mut values = Vec::with_capacity(n_points);
        let mut outputs = Vec::with_capacity(n_points);

        for j in 0..n_points {
            let normalized_val = min_val + j as f32 * step;

            // Get the output of just this shape function
            let shape_function = &model.shape_functions()[i];
            // Create a single feature tensor with just the normalized value
            let shape_input_data = Array::from_vec(vec![normalized_val]);
            let shape_input: Tensor<Rank1<S<1>>> = Tensor::new(shape_input_data.into_dyn());
            let shape_input: Tensor<(S<1>, S<1>)> = shape_input.reshape((S {}, S {}));
            let shape_output = shape_function.forward(shape_input);

            let output_val = shape_output.data().into_flat()[0];

            // Convert normalized value back to original scale for plotting
            let original_val = normalized_val * stds[i] + means[i];

            values.push(original_val);
            outputs.push(output_val);
        }

        let min_x = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_x = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_y = outputs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_y = outputs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let padding_x = (max_x - min_x) * 0.05;
        let padding_y = if (max_y - min_y).abs() < 1e-6 {
            1.0
        } else {
            (max_y - min_y) * 0.2
        };

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Shape Function for {}", feature_name),
                ("sans-serif", 30).into_font(),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                (min_x - padding_x)..(max_x + padding_x),
                (min_y - padding_y)..(max_y + padding_y),
            )?;

        chart
            .configure_mesh()
            .x_desc(feature_name.to_string())
            .y_desc("Contribution to prediction")
            .draw()?;

        chart.draw_series(LineSeries::new(
            values.iter().zip(outputs.iter()).map(|(&x, &y)| (x, y)),
            &BLUE,
        ))?;

        root.present()?;
    }

    Ok(())
}

/// Run the NAM experiment on the California housing dataset
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

    // Plot individual shape functions
    println!("Plotting shape functions...");
    let feature_names = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ];

    let plots_dir = Path::new(file!()).parent().unwrap().join("plots");
    std::fs::create_dir_all(&plots_dir)?;
    plot_shape_functions(
        &model,
        &feature_names,
        &feature_means,
        &feature_stds,
        &plots_dir,
    )?;

    // Training loop
    println!("Starting training for {} epochs...", NUM_EPOCHS);
    let mut train_losses = Vec::with_capacity(NUM_EPOCHS * 100);
    let mut train_epoch_losses = Vec::with_capacity(NUM_EPOCHS);
    let mut total_steps = 0;

    for epoch in 0..NUM_EPOCHS {
        let mut epoch_loss = 0.0;
        let mut batches: u64 = 0;

        for (_idx, batch) in dataloader.iter_shuffled(epoch as u64 + 1).enumerate() {
            let x = batch.input;
            let y_true = batch.label;

            let y_pred = model.forward(x.clone());

            let diff = y_pred.clone() + -y_true.clone();
            let loss = diff.clone() * diff; // squared error
            let loss = loss.mean(); // MSE

            let loss_val = loss.data().into_flat()[0];
            epoch_loss += loss_val;
            train_losses.push(loss_val);
            writer.add_scalar("loss", loss_val, total_steps);
            batches += 1;
            total_steps += 1;

            opt.zero_grad();
            loss.backward();
            opt.step();
        }
        opt.update_lr(opt.lr() * 0.5);
        let avg_loss = epoch_loss / batches as f32;
        train_epoch_losses.push(avg_loss);
        println!("Epoch {}/{}, Loss: {:.4}", epoch + 1, NUM_EPOCHS, avg_loss);
    }
    writer.flush();

    // Plot training loss
    println!("Plotting training loss...");
    plot_training_loss(train_losses, &plots_dir, "housing_")?;
    plot_training_loss(train_epoch_losses, &plots_dir, "housing_epoch_")?;
    // Plot individual shape functions
    println!("Plotting shape functions...");
    let feature_names = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ];

    plot_shape_functions(
        &model,
        &feature_names,
        &feature_means,
        &feature_stds,
        &plots_dir,
    )?;

    // Evaluate model on the full dataset
    println!("Evaluating model...");
    let mut total_normalized_loss = 0.0;
    let mut total_denormalized_loss = 0.0;
    let mut samples = 0;
    let xshape: (S<1>, S<NUM_FEATURES>) = (S {}, S {});
    let ytrue_shape: (S<1>, S<1>) = (S {}, S {});

    for sample in &labeled_data {
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

    // Metrics in normalized space
    let avg_normalized_loss = total_normalized_loss / samples as f32;
    let normalized_rmse = avg_normalized_loss.sqrt();

    // Metrics in original housing price scale
    let avg_denormalized_loss = total_denormalized_loss / samples as f32;
    let denormalized_rmse = avg_denormalized_loss.sqrt();

    println!("Final evaluation results:");
    println!(
        "  Normalized Mean Squared Error: {:.4}",
        avg_normalized_loss
    );
    println!("  Normalized RMSE: {:.4}", normalized_rmse);
    println!(
        "  Mean Squared Error (original scale): {:.4}",
        avg_denormalized_loss
    );
    println!("  RMSE (original scale): {:.4}", denormalized_rmse);
    println!(
        "Shape function plots saved to: {}",
        plots_dir.to_str().unwrap_or_default()
    );

    Ok(())
}
fn plot_training_loss(
    train_losses: Vec<f32>,
    plots_dir: &PathBuf,
    prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let path = plots_dir.join(prefix.to_string() + "training_loss.png");
    let root_backend = BitMapBackend::new(&path, (800, 600));
    let root = root_backend.into_drawing_area();
    root.fill(&WHITE)?;

    let min_y = train_losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_y = train_losses
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (0..train_losses.len()).log_scale(),
            ((min_y * 0.7)..(max_y * 1.2)).log_scale(),
        )?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss (MSE)")
        .draw()?;

    chart.draw_series(LineSeries::new(
        (0..train_losses.len())
            .zip(train_losses.iter())
            .map(|(x, &y)| (x, y)),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}
// Creates a binary to run the experiment
pub fn main() -> Result<(), Box<dyn Error>> {
    run_housing_nam_experiment()
}
