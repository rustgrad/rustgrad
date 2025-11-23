# Adam Optimizer Benchmark

This directory contains benchmarks for the Adam optimizer implementation in the Rustgrad library.

## Benchmark Summary

The benchmark includes:

1. **Adam optimizer performance with different learning rates**: Tests the Adam optimizer with a fixed network configuration but different learning rates (0.1, 0.01, 0.001, 0.0001).
   
2. **Performance with different network sizes**: Tests how the Adam optimizer performs with different hidden layer sizes (32, 64, 128, 256).

## Running the Benchmark

### RustGrad Benchmark

To run the RustGrad benchmark:

```bash
# Debug build (slower)
cargo run --bin adam_benchmark

# Release build (for fair comparison)
cargo run --release --bin adam_benchmark
```

### PyTorch Comparison

To compare with PyTorch's Adam optimizer, run the included Python script:

```bash
# Make sure you have PyTorch installed
pip install torch

# Run the benchmark
python3 src/experiments/optimization/pytorch_benchmark.py
```

The PyTorch benchmark uses the exact same configuration as the RustGrad benchmark:
- Same MLP architecture (10 input → 64 hidden → 1 output)
- Same batch size (64)
- Same number of iterations
- Same learning rates and hidden sizes

**Note**: For a fair comparison, always compare release builds:
- RustGrad: `cargo run --release --bin adam_benchmark`
- PyTorch: Already optimized by default

## Implementation Notes

The benchmark uses a Multi-Layer Perceptron (MLP) with fixed input size (10) and output size (1), with variable hidden layer sizes.

Each benchmark measures:
- Total time taken for a fixed number of optimization steps
- Average time per optimization step

The optimization process includes:
1. Forward pass through the network
2. Loss calculation (Mean Squared Error)
3. Backward pass (gradient calculation)
4. Optimization step (parameter updates)
5. Gradient zeroing

## Expected Results

Performance characteristics:
- **Learning Rate Impact**: Minimal, as the computational cost is dominated by forward/backward passes
- **Hidden Size Impact**: Larger hidden sizes increase time proportionally to parameter count
- **RustGrad vs PyTorch**: PyTorch may be faster due to highly optimized BLAS libraries and C++/CUDA backend

## Visualizing Results

Currently, results are printed to the console. Future improvements could include:
- Generating plots comparing RustGrad vs PyTorch
- CSV output for further analysis
- Performance over training epochs
