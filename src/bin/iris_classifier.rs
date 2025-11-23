use rustgrad::experiments::iris::iris_experiment;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting Iris Classification Experiment");
    println!("========================================\n");
    iris_experiment::run_iris_experiment()?;
    println!("\nExperiment completed successfully!");
    Ok(())
}
