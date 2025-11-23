use std::error::Error;
use std::path::Path;

use plotters::prelude::*;

use super::data::{CLASS_NAMES, NUM_CLASSES};

/// Plots the training loss over time
///
/// # Arguments
/// * `losses` - Vector of loss values
/// * `output_dir` - Directory to save the plot
/// * `prefix` - Filename prefix (e.g., "iris_")
///
/// # Returns
/// Ok(()) on success, or an error if plotting fails
pub fn plot_training_loss(
    losses: Vec<f32>,
    output_dir: &Path,
    prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let filename = format!("{}loss.png", prefix);
    let output_path = output_dir.join(filename);

    let root = BitMapBackend::new(&output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_loss = losses
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let min_loss = losses
        .iter()
        .fold(f32::INFINITY, |a, &b| a.min(b)) as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..losses.len(), min_loss..max_loss)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Loss")
        .draw()?;

    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(i, &loss)| (i, loss as f64)),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}

/// Plots the training accuracy over epochs
///
/// # Arguments
/// * `accuracies` - Vector of accuracy values (per epoch)
/// * `output_dir` - Directory to save the plot
/// * `prefix` - Filename prefix (e.g., "iris_")
///
/// # Returns
/// Ok(()) on success, or an error if plotting fails
pub fn plot_training_accuracy(
    accuracies: Vec<f32>,
    output_dir: &Path,
    prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let filename = format!("{}accuracy.png", prefix);
    let output_path = output_dir.join(filename);

    let root = BitMapBackend::new(&output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Accuracy", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..accuracies.len(), 0.0..1.0)?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Accuracy")
        .draw()?;

    chart.draw_series(LineSeries::new(
        accuracies
            .iter()
            .enumerate()
            .map(|(i, &acc)| (i, acc as f64)),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}

/// Plots a confusion matrix heatmap
///
/// # Arguments
/// * `confusion_matrix` - Confusion matrix [true_class x predicted_class]
/// * `output_dir` - Directory to save the plot
/// * `prefix` - Filename prefix (e.g., "iris_")
///
/// # Returns
/// Ok(()) on success, or an error if plotting fails
pub fn plot_confusion_matrix(
    confusion_matrix: &[Vec<usize>],
    output_dir: &Path,
    prefix: &str,
) -> Result<(), Box<dyn Error>> {
    let filename = format!("{}confusion_matrix.png", prefix);
    let output_path = output_dir.join(filename);

    let root = BitMapBackend::new(&output_path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find max value for color scaling
    let max_count = confusion_matrix
        .iter()
        .flat_map(|row| row.iter())
        .max()
        .unwrap_or(&1);

    let cell_size = 200;
    let margin = 100;

    // Draw title
    root.draw_text(
        "Confusion Matrix",
        &TextStyle::from(("sans-serif", 40).into_font()).color(&BLACK),
        (400, 30),
    )?;

    // Draw confusion matrix cells
    for (true_idx, row) in confusion_matrix.iter().enumerate() {
        for (pred_idx, &count) in row.iter().enumerate() {
            let x = margin + pred_idx * cell_size;
            let y = margin + 100 + true_idx * cell_size;

            // Calculate color intensity based on count
            let intensity = (count as f64 / *max_count as f64 * 200.0) as u8;
            let color = RGBColor(255 - intensity, 255 - intensity, 255);

            // Draw cell
            root.draw(&Rectangle::new(
                [(x as i32, y as i32), ((x + cell_size) as i32, (y + cell_size) as i32)],
                color.filled(),
            ))?;

            // Draw cell border
            root.draw(&Rectangle::new(
                [(x as i32, y as i32), ((x + cell_size) as i32, (y + cell_size) as i32)],
                BLACK.stroke_width(2),
            ))?;

            // Draw count text
            let text_style = TextStyle::from(("sans-serif", 30).into_font()).color(&BLACK);
            root.draw_text(
                &format!("{}", count),
                &text_style,
                ((x + cell_size / 2 - 15) as i32, (y + cell_size / 2 - 15) as i32),
            )?;
        }
    }

    // Draw axis labels
    let label_style = TextStyle::from(("sans-serif", 25).into_font()).color(&BLACK);

    // Y-axis label
    root.draw_text(
        "True Class",
        &label_style,
        (10, 400),
    )?;

    // X-axis label
    root.draw_text(
        "Predicted Class",
        &label_style,
        (350, 750),
    )?;

    // Class names on axes
    for i in 0..NUM_CLASSES {
        let class_name = CLASS_NAMES[i].split('-').last().unwrap_or(&CLASS_NAMES[i]);

        // Y-axis class names
        root.draw_text(
            class_name,
            &label_style,
            (10, (margin + 100 + i * cell_size + cell_size / 2 - 10) as i32),
        )?;

        // X-axis class names
        root.draw_text(
            class_name,
            &label_style,
            ((margin + i * cell_size + cell_size / 2 - 30) as i32, 750),
        )?;
    }

    root.present()?;
    Ok(())
}
