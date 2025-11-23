# RustGrad

A pure Rust deep learning framework with automatic differentiation, featuring compile-time shape checking and a flexible tensor operations API.

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

RustGrad is a from-scratch implementation of automatic differentiation and neural network training in Rust. It provides:

- **Automatic Differentiation**: Reverse-mode autodiff with dynamic computation graphs
- **Type-Safe Shape Checking**: Compile-time dimension verification using Rust's type system
- **Neural Network Layers**: Linear, Conv1D, MLP, and custom architectures
- **Optimizers**: SGD and Adam with learning rate scheduling
- **Interpretable Models**: Neural Additive Models (NAM) for explainable predictions

## Features

### Core Components

- **Tensor Operations**: Broadcasting, slicing, reshaping, and mathematical operations
- **Activation Functions**: ReLU, Softmax, and more
- **Loss Functions**: MSE, cross-entropy
- **Data Loading**: Efficient batching and shuffling utilities
- **TensorBoard Integration**: Real-time training visualization

### Neural Network Modules

- **LinearLayer**: Fully connected layers with optional activation
- **MLP**: Multi-layer perceptrons with arbitrary depth
- **NAM**: Neural Additive Models for interpretable machine learning
- **TSMixer**: Time series forecasting architecture

### Optimizers

- **SGD**: Stochastic gradient descent
- **Adam**: Adaptive moment estimation with configurable beta parameters

## Installation

Add RustGrad to your `Cargo.toml`:

```toml
[dependencies]
rustgrad = "0.1.0"
```

Or clone and build from source:

```bash
git clone https://github.com/flix59/rustgrad.git
cd rustgrad
cargo build --release
```

## Quick Start

### Basic Tensor Operations

```rust
use rustgrad::tensor::Tensor;
use rustgrad::dimensions::S;
use ndarray::array;

// Create tensors with compile-time shape checking
let a: Tensor<(S<3>,)> = Tensor::new(array![1.0, 2.0, 3.0].into_dyn());
let b: Tensor<(S<3>,)> = Tensor::new(array![4.0, 5.0, 6.0].into_dyn());

// Perform operations
let c = a.clone() * b.clone();

// Backward pass
c.backward();

// Access gradients
println!("Gradient of a: {:?}", a.grad());
```

### Building a Neural Network

```rust
use rustgrad::nn::mlp::MLP;
use rustgrad::dimensions::S;
use rustgrad::tensor::Tensor;

// Create a 3-layer MLP: 10 inputs -> 20 hidden -> 20 hidden -> 1 output
let model: MLP<S<10>, S<1>, S<20>, 2> = MLP::new();

// Forward pass
let input: Tensor<(S<1>, S<10>)> = Tensor::new(/* ... */);
let output = model.forward(input);
```

### Training a Model

```rust
use rustgrad::optim::adam::AdamOptimizer;
use rustgrad::optim::optimizer::Optimizer;

// Initialize optimizer
let mut optimizer = AdamOptimizer::new(0.001, model.parameters());

// Training loop
for epoch in 0..num_epochs {
    let predictions = model.forward(x.clone());
    let loss = calculate_loss(predictions, y_true);
    
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}
```

## Examples

### California Housing Price Prediction

Train a Neural Additive Model on the California housing dataset:

```bash
cargo run --bin housing_nam
```

This example demonstrates:
- Loading and preprocessing CSV data
- Training with NAM for interpretability
- Visualizing shape functions
- TensorBoard logging

### Iris Classification

Train a neural network classifier on the classic Iris dataset:

```bash
cargo run --bin iris_classifier
```

This example demonstrates:
- Multi-class classification with softmax
- Cross-entropy loss
- Confusion matrix visualization
- Training/test accuracy tracking
- Model evaluation metrics

### Adam Optimizer Benchmark

Compare optimizer performance across different architectures:

```bash
cargo run --bin adam_benchmark
```

## Architecture

### Type-Safe Dimensions

RustGrad uses Rust's type system to verify tensor dimensions at compile time:

```rust
// S<N> represents a static dimension of size N
type BatchedVector = Tensor<(usize, S<10>)>; // Variable batch size, 10 features

// usize represents dynamic dimensions determined at runtime
type DynamicTensor = Tensor<(usize, usize, usize)>;
```

### Automatic Differentiation

The framework builds a dynamic computation graph during the forward pass and automatically computes gradients during the backward pass:

```rust
let x = Tensor::new(/* ... */);
let y = x.clone() * Tensor::new(/* ... */);
let z = y.relu();
z.backward(); // Computes gradients for all operations
```

## Project Structure

```
src/
├── array_shape.rs       # Shape representation and utilities
├── dimensions.rs        # Type-level dimension system
├── tensor.rs           # Core tensor implementation
├── ops/                # Tensor operations (add, mul, matmul, etc.)
│   ├── add.rs
│   ├── matmul.rs
│   ├── relu.rs
│   └── ...
├── nn/                 # Neural network layers
│   ├── linear.rs
│   ├── mlp.rs
│   └── ...
├── optim/              # Optimization algorithms
│   ├── adam.rs
│   ├── sgd.rs
│   └── optimizer.rs
├── data/               # Data loading utilities
├── nam.rs              # Neural Additive Models
└── experiments/        # Example implementations
    ├── housing/        # California housing prediction
    ├── iris/           # Iris classification
    ├── optimization/   # Optimizer benchmarks
    └── ts_mixer/       # Time series forecasting
```

## Supported Operations

### Element-wise Operations
- Addition, subtraction, multiplication, division
- Negation, exponential, logarithm, square root
- ReLU, softmax

### Reduction Operations
- Sum, mean, variance
- Max (element-wise and axis-wise)

### Shape Operations
- Broadcasting
- Reshaping
- Slicing and indexing
- Permutation (transpose)
- Stacking

### Linear Algebra
- Matrix multiplication (with batch support)
- Optimized parallel execution (optional)

## Performance

RustGrad includes several optimizations:

- **Parallelization**: Optional Rayon support for parallel tensor operations
- **Optimized Release Builds**: LTO and aggressive optimization flags
- **Memory Efficiency**: Reference counting for shared tensor data
- **SIMD**: Leverages ndarray's SIMD optimizations

Build with optimizations:

```bash
cargo build --release
```

**Performance vs PyTorch**: While RustGrad is currently slower than PyTorch (which uses highly optimized BLAS libraries and a C++/CUDA backend), there are many opportunities for optimization. See [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) for detailed strategies to improve performance, including:
- BLAS integration (2-5x speedup)
- Reducing memory allocations
- Operation fusion
- GPU support

### Benchmarks

To compare RustGrad with PyTorch:
```bash
# RustGrad (use release build!)
cargo run --release --bin adam_benchmark

# PyTorch
python3 src/experiments/optimization/pytorch_benchmark.py
```

See `src/experiments/optimization/README.md` for detailed benchmarking information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Performance improvements are especially welcome! See [PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md) for optimization opportunities.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/flix59/rustgrad.git
cd rustgrad

# Run tests
cargo test

# Run with optimizations
cargo test --release

# Format code
cargo fmt

# Lint
cargo clippy
```

## Examples in Detail

### Iris Classification

The Iris dataset classification demonstrates a complete machine learning workflow:

```rust
use rustgrad::nn::mlp::MLP;
use rustgrad::dimensions::S;
use rustgrad::optim::adam::AdamOptimizer;

// Create a 3-layer MLP: 4 features -> 16 hidden -> 16 hidden -> 3 classes
let model: MLP<S<4>, S<3>, S<16>, 2> = MLP::new();

// Initialize Adam optimizer
let mut optimizer = AdamOptimizer::new(0.001, model.parameters());

// Training with cross-entropy loss
for epoch in 0..epochs {
    for batch in train_loader.iter_shuffled(epoch) {
        let logits = model.forward(batch.input);
        let probs = logits.softmax();
        let loss = cross_entropy_loss(probs, batch.label);
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
```

The experiment includes:
- One-hot encoding for multi-class targets
- Feature normalization
- Train/test split (80/20)
- Confusion matrix generation
- Accuracy tracking over epochs
- Visualization with plotters

### Neural Additive Models (NAM)

NAM provides interpretable predictions by learning individual shape functions for each feature:

```rust
use rustgrad::nam::NAM;
use rustgrad::dimensions::S;

// Create NAM with 8 features, 64 hidden units, 3 layers per shape function
let model: NAM<S<64>, 8> = NAM::new(3);

// Forward pass
let predictions = model.forward(features);

// Access individual shape functions for interpretability
let shape_functions = model.shape_functions();
```

### Custom Training Loops

```rust
use rustgrad::data::labeled_dataset::LabeledTensorDataLoader;
use tensorboard_rs::summary_writer::SummaryWriter;

let mut writer = SummaryWriter::new("./logdir");
let mut optimizer = AdamOptimizer::new(learning_rate, model.parameters());

for epoch in 0..num_epochs {
    for (idx, batch) in dataloader.iter_shuffled(epoch).enumerate() {
        let predictions = model.forward(batch.input);
        let loss = mse_loss(predictions, batch.label);
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        writer.add_scalar("loss", loss.data()[0], global_step);
    }
    
    // Learning rate scheduling
    optimizer.update_lr(optimizer.lr() * decay_factor);
}
```

## Documentation

Generate and view the documentation:

```bash
cargo doc --open
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by PyTorch
- Built on top of [ndarray](https://github.com/rust-ndarray/ndarray) for efficient array operations
- TensorBoard integration via [tensorboard-rs](https://github.com/elbaro/tensorboard-rs)

## Contact

For questions and feedback, please open an issue on GitHub.

---

**Note**: RustGrad is under active development. APIs may change between versions.