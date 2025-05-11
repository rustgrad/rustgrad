use rustgrad::experiments::optimization::adam_benchmark::run_all_benchmarks;

fn main() {
    println!("Starting Adam Optimizer Benchmark");
    println!("=================================");

    // Run the benchmarks
    run_all_benchmarks();

    println!("\nAdam Optimizer Benchmark completed");
}
