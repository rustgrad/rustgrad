use std::error::Error;
use std::path::PathBuf;

use ndarray::Array;
use plotters::prelude::*;

use crate::dimensions::{Rank1, S};
use crate::nam::NAM;
use crate::tensor::Tensor;

/// Number of standard deviations to plot for each feature's shape function
const STD_DEVS_RANGE: f32 = 3.0;

/// Number of points to plot for each shape function
const SHAPE_PLOT_POINTS: usize = 100;

/// Padding factor for x-axis in shape function plots (5% of range)
const PLOT_X_PADDING: f32 = 0.05;

/// Padding factor for y-axis in shape function plots (20% of range)
const PLOT_Y_PADDING: f32 = 0.2;

/// Minimum y-axis range to avoid degenerate plots
const MIN_Y_RANGE: f32 = 1e-6;

/// Default y-axis padding when range is too small
const DEFAULT_Y_PADDING: f32 = 1.0;

/// Loss plot y-axis margin multipliers (lower, upper)
const LOSS_PLOT_Y_MARGINS: (f32, f32) = (0.7, 1.2);

/// Plots the individual shape functions learned by the NAM model
///
/// Creates one plot per feature showing how that feature contributes to the
/// final prediction. This visualization is key to NAM's interpretability.
///
/// # Arguments
/// * `model` - The trained NAM model
/// * `feature_names` - Names of features (for plot titles and labels)
/// * `means` - Mean values used for normalization (for denormalization)
/// * `stds` - Standard deviation values used for normalization (for denormalization)
/// * `output_dir` - Directory where plot PNG files will be saved
/// * `hidden_size` - Hidden layer size of the NAM model (needed for type parameter)
/// * `num_features` - Number of features (needed for type parameter)
pub fn plot_shape_functions<const HIDDEN: usize, const FEATURES: usize>(
    model: &NAM<S<HIDDEN>, FEATURES>,
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

        // Plot range: ±3 standard deviations in normalized space
        let min_val = -STD_DEVS_RANGE;
        let max_val = STD_DEVS_RANGE;
        let step = (max_val - min_val) / (SHAPE_PLOT_POINTS as f32 - 1.0);

        let mut values = Vec::with_capacity(SHAPE_PLOT_POINTS);
        let mut outputs = Vec::with_capacity(SHAPE_PLOT_POINTS);

        // Evaluate shape function at many points to create smooth curve
        for j in 0..SHAPE_PLOT_POINTS {
            let normalized_val = min_val + j as f32 * step;

            // Get the output of just this shape function
            let shape_function = &model.shape_functions()[i];
            let shape_input_data = Array::from_vec(vec![normalized_val]);
            let shape_input: Tensor<Rank1<S<1>>> = Tensor::new(shape_input_data.into_dyn());
            let shape_input: Tensor<(S<1>, S<1>)> = shape_input.reshape((S {}, S {}));
            let shape_output = shape_function.forward(shape_input);

            let output_val = shape_output.data().into_flat()[0];

            // Convert normalized value back to original scale for x-axis
            let original_val = normalized_val * stds[i] + means[i];

            values.push(original_val);
            outputs.push(output_val);
        }

        // Calculate plot bounds with padding
        let min_x = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_x = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_y = outputs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_y = outputs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let padding_x = (max_x - min_x) * PLOT_X_PADDING;
        let padding_y = if (max_y - min_y).abs() < MIN_Y_RANGE {
            DEFAULT_Y_PADDING
        } else {
            (max_y - min_y) * PLOT_Y_PADDING
        };

        // Create chart with appropriate ranges
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

/// Plots training loss over time
///
/// Creates a log-scale plot showing how loss decreases during training.
///
/// # Arguments
/// * `train_losses` - Vector of loss values (one per epoch or batch)
/// * `plots_dir` - Directory where the plot PNG will be saved
/// * `prefix` - Prefix for the filename (e.g., "epoch_" or "batch_")
pub fn plot_training_loss(
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

    // Use log scale for both axes
    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (0..train_losses.len()).log_scale(),
            ((min_y * LOSS_PLOT_Y_MARGINS.0)..(max_y * LOSS_PLOT_Y_MARGINS.1)).log_scale(),
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
