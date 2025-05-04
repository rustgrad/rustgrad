use rustgrad::experiments::housing::nam_experiment;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting California Housing NAM Experiment");
    nam_experiment::run_housing_nam_experiment()?;
    println!("Experiment completed successfully");
    Ok(())
}
