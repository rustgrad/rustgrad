use std::time::Instant;

use crate::{
    dimensions::S,
    nn::MLP,
    optim::{adam::AdamOptimizer, optimizer::Optimizer},
    tensor::Tensor,
};

/// Benchmark Adam optimizer with different learning rates
pub fn run_adam_benchmark() {
    println!("Running Adam Optimizer Benchmark");
    println!("================================");

    // Create a model with fixed configuration
    let mlp: MLP<S<10>, S<1>, S<64>, 1> = MLP::new();

    // Initialize Adam optimizer with different learning rates
    let learning_rates = vec![0.1, 0.01, 0.001, 0.0001];

    // Create random input and expected output tensors
    let input: Tensor<(S<64>, S<10>)> = Tensor::new_random(0.0, 1.0);
    let expected_output: Tensor<(S<64>, S<1>)> = Tensor::new_random(0.0, 1.0);

    println!("\nBenchmarking with different learning rates:");
    for &lr in &learning_rates {
        let mut optimizer = AdamOptimizer::new_with_defaults(lr, mlp.parameters());

        // Measure time for 100 optimization steps
        let iterations = 100;
        let start_time = Instant::now();

        for _ in 0..iterations {
            // Forward pass
            let output = mlp.forward(input.clone());

            // Calculate loss
            let mut loss = -expected_output.clone() + output.clone();
            loss = loss.clone() * loss.clone(); // MSE loss

            // Backward pass and optimization
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }

        let elapsed = start_time.elapsed();
        let avg_time_per_step = elapsed.as_secs_f64() / iterations as f64;

        println!("  Learning Rate: {}", lr);
        println!("    Total Time: {:.4} seconds", elapsed.as_secs_f64());
        println!(
            "    Avg Time Per Step: {:.4} milliseconds",
            avg_time_per_step * 1000.0
        );
    }
}

/// Benchmark comparing different hidden layer sizes
pub fn benchmark_different_hidden_sizes() {
    println!("\nComparing Performance with Different Hidden Layer Sizes");
    println!("=====================================================");

    // Benchmark different hidden layer sizes
    let hidden_sizes = [32, 64, 128, 256];

    for &hidden_size in &hidden_sizes {
        println!("\nBenchmarking with hidden size: {}", hidden_size);

        // We need to match on the hidden size since Rust needs concrete types at compile time
        match hidden_size {
            32 => benchmark_specific_hidden_size::<32>(),
            64 => benchmark_specific_hidden_size::<64>(),
            128 => benchmark_specific_hidden_size::<128>(),
            256 => benchmark_specific_hidden_size::<256>(),
            _ => println!("  Unsupported hidden size"),
        }
    }
}

/// Benchmark with a specific hidden layer size
fn benchmark_specific_hidden_size<const H: usize>()
where
    S<H>: crate::dimensions::Dimension,
{
    // Create a model with the specified hidden size
    let mlp: MLP<S<10>, S<1>, S<H>, 1> = MLP::new();
    let lr = 0.001;
    let mut optimizer = AdamOptimizer::new_with_defaults(lr, mlp.parameters());

    // Create random input and expected output tensors
    let input: Tensor<(S<64>, S<10>)> = Tensor::new_random(0.0, 1.0);
    let expected_output: Tensor<(S<64>, S<1>)> = Tensor::new_random(0.0, 1.0);

    // Measure time for 50 optimization steps
    let iterations = 50;
    let start_time = Instant::now();

    for _ in 0..iterations {
        // Forward pass
        let output = mlp.forward(input.clone());

        // Calculate loss
        let mut loss = -expected_output.clone() + output.clone();
        loss = loss.clone() * loss.clone(); // MSE loss

        // Backward pass and optimization
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }

    let elapsed = start_time.elapsed();
    println!("  Total Time: {:.4} seconds", elapsed.as_secs_f64());
    println!(
        "  Avg Time Per Step: {:.4} milliseconds",
        elapsed.as_secs_f64() * 1000.0 / iterations as f64
    );
}

/// Run all benchmarks
pub fn run_all_benchmarks() {
    run_adam_benchmark();
    benchmark_different_hidden_sizes();
}
