#!/usr/bin/env python3
"""
PyTorch Adam Optimizer Benchmark

This script benchmarks PyTorch's Adam optimizer with the same configuration
as the RustGrad benchmark for comparison.

Usage:
    python3 pytorch_benchmark.py
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class MLP(nn.Module):
    """
    Multi-Layer Perceptron matching RustGrad's architecture.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output units
        num_hidden_layers: Number of hidden layers (excluding first and last)
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_hidden_layers: int = 1):
        super(MLP, self).__init__()
        
        # First layer
        self.first_layer = nn.Linear(input_size, hidden_size)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) 
            for _ in range(num_hidden_layers)
        ])
        
        # Last layer (no activation)
        self.last_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize weights with He initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization (similar to RustGrad)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = self.first_layer(x)
        x = torch.relu(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
        
        x = self.last_layer(x)
        return x


def benchmark_learning_rates(
    input_size: int = 10,
    hidden_size: int = 64,
    output_size: int = 1,
    batch_size: int = 64,
    num_hidden_layers: int = 1,
    iterations: int = 100
) -> None:
    """
    Benchmark Adam optimizer with different learning rates.
    
    Matches RustGrad's benchmark: MLP<S<10>, S<1>, S<64>, 1>
    """
    print("Running PyTorch Adam Optimizer Benchmark")
    print("=" * 40)
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    # Create random input and target tensors (same for all learning rates)
    torch.manual_seed(42)
    input_data = torch.randn(batch_size, input_size)
    target_data = torch.randn(batch_size, output_size)
    
    print(f"\nBenchmarking with different learning rates:")
    print(f"Model: MLP({input_size} -> {hidden_size} -> {output_size})")
    print(f"Batch size: {batch_size}, Iterations: {iterations}\n")
    
    results = []
    
    for lr in learning_rates:
        # Create fresh model for each learning rate
        model = MLP(input_size, hidden_size, output_size, num_hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Warmup (JIT compilation, cache warming)
        for _ in range(5):
            output = model(input_data)
            loss = criterion(output, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Actual benchmark
        start_time = time.time()
        
        for _ in range(iterations):
            # Forward pass
            output = model(input_data)
            
            # Calculate MSE loss
            loss = criterion(output, target_data)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        elapsed = time.time() - start_time
        avg_time_per_step = elapsed / iterations
        
        results.append({
            'lr': lr,
            'total_time': elapsed,
            'avg_time_ms': avg_time_per_step * 1000
        })
        
        print(f"  Learning Rate: {lr}")
        print(f"    Total Time: {elapsed:.4f} seconds")
        print(f"    Avg Time Per Step: {avg_time_per_step * 1000:.4f} milliseconds")
    
    return results


def benchmark_hidden_sizes(
    input_size: int = 10,
    output_size: int = 1,
    batch_size: int = 64,
    num_hidden_layers: int = 1,
    iterations: int = 50,
    lr: float = 0.001
) -> None:
    """
    Benchmark different hidden layer sizes.
    
    Matches RustGrad's benchmark with hidden sizes: 32, 64, 128, 256
    """
    print("\nComparing Performance with Different Hidden Layer Sizes")
    print("=" * 55)
    
    hidden_sizes = [32, 64, 128, 256]
    
    # Create random input and target tensors
    torch.manual_seed(42)
    input_data = torch.randn(batch_size, input_size)
    target_data = torch.randn(batch_size, output_size)
    
    results = []
    
    for hidden_size in hidden_sizes:
        print(f"\nBenchmarking with hidden size: {hidden_size}")
        
        # Create model
        model = MLP(input_size, hidden_size, output_size, num_hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Warmup
        for _ in range(5):
            output = model(input_data)
            loss = criterion(output, target_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Actual benchmark
        start_time = time.time()
        
        for _ in range(iterations):
            # Forward pass
            output = model(input_data)
            
            # Calculate MSE loss
            loss = criterion(output, target_data)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        elapsed = time.time() - start_time
        avg_time_per_step = elapsed / iterations
        
        results.append({
            'hidden_size': hidden_size,
            'total_time': elapsed,
            'avg_time_ms': avg_time_per_step * 1000
        })
        
        print(f"  Total Time: {elapsed:.4f} seconds")
        print(f"  Avg Time Per Step: {avg_time_per_step * 1000:.4f} milliseconds")
    
    return results


def print_comparison_summary(pytorch_results: dict) -> None:
    """Print a summary comparing PyTorch results"""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nTo compare with RustGrad, run:")
    print("  cargo run --release --bin adam_benchmark")
    print("\nNote: Make sure to compare release builds for fair comparison.")
    print("      Debug builds will be significantly slower.")
    print("\nExpected trends:")
    print("  - Larger learning rates: slightly faster (fewer computations)")
    print("  - Larger hidden sizes: slower (more parameters)")
    print("  - PyTorch may be faster due to:")
    print("    * Highly optimized BLAS libraries")
    print("    * C++/CUDA backend")
    print("    * Years of performance tuning")


def main():
    """Run all benchmarks"""
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CPU: {torch.get_num_threads()} threads")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    # Run benchmarks matching RustGrad configuration
    lr_results = benchmark_learning_rates()
    hidden_size_results = benchmark_hidden_sizes()
    
    # Print comparison guide
    print_comparison_summary({
        'learning_rates': lr_results,
        'hidden_sizes': hidden_size_results
    })


if __name__ == "__main__":
    main()
