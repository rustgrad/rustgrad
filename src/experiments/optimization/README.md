# Adam Optimizer Benchmark

This directory contains benchmarks for the Adam optimizer implementation in the Rustgrad library.

## Benchmark Summary

The benchmark includes:

1. **Adam optimizer performance with different learning rates**: Tests the Adam optimizer with a fixed network configuration but different learning rates (0.1, 0.01, 0.001, 0.0001).
   
2. **Performance with different network sizes**: Tests how the Adam optimizer performs with different hidden layer sizes (32, 64, 128, 256).

## Running the Benchmark

To run the benchmark:

```bash
cargo run --bin adam_benchmark
```

## Visualizing Results

Currently, results are printed to the console. Future improvements could include generating plots for a more visual presentation of the benchmark results.

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
