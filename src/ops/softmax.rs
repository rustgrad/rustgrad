use std::cell::RefCell;
use std::rc::Rc;

use ndarray::Axis;

use crate::dimensions::{Shape, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
struct TensorSoftmax<S: Shape> {
    inp: Tensor<S>,
    output: Tensor<S>,
}

impl<S: Shape> TensorSoftmax<S> {
    /// Applies softmax along the last dimension (dimension 1)
    /// For a batch of predictions (batch_size, num_classes), softmax is applied per sample
    fn forward(inp: Tensor<S>) -> Tensor<S> {
        let inp_data = inp.container.borrow().array.clone();
        let shape = inp_data.shape();

        // For 2D tensors, apply softmax along last axis
        if shape.len() >= 2 {
            // For numerical stability: subtract max along last axis
            let max_vals = inp_data
                .map_axis(Axis(1), |row| {
                    row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                })
                .insert_axis(Axis(1));

            let shifted = &inp_data - &max_vals;
            let exp_vals = shifted.mapv(|x| x.exp());

            // Sum along last axis and normalize
            let sum_exp = exp_vals.sum_axis(Axis(1)).insert_axis(Axis(1));

            let result = exp_vals / sum_exp;

            let output = Tensor::new(result.clone());
            let node = TensorSoftmax {
                inp: inp.clone(),
                output: output.clone(),
            };

            Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
        } else {
            // For 1D tensors, apply softmax to the whole tensor
            let max_val = inp_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let shifted = &inp_data - max_val;
            let exp_vals = shifted.mapv(|x| x.exp());
            let sum_exp = exp_vals.sum();
            let result = exp_vals / sum_exp;

            let output = Tensor::new(result.clone());
            let node = TensorSoftmax {
                inp: inp.clone(),
                output: output.clone(),
            };

            Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
        }
    }
}

impl<S: Shape> Operation<S> for TensorSoftmax<S> {
    fn backward(&self, output: &Tensor<S>) {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad_output = maybe_grad.unwrap_or(ndarray::Array::ones(output.shape()));
        let softmax_output = self.output.container.borrow().array.clone();
        let shape = softmax_output.shape();

        if shape.len() >= 2 {
            // Gradient of softmax: softmax * (grad_output - (softmax * grad_output).sum(axis=1))
            let sum_term = (&softmax_output * &grad_output)
                .sum_axis(Axis(1))
                .insert_axis(Axis(1));

            let grad_input = &softmax_output * (&grad_output - &sum_term);
            self.inp.backward_internal(grad_input);
        } else {
            // For 1D: gradient is softmax * (grad_output - dot(softmax, grad_output))
            let dot_product = (&softmax_output * &grad_output).sum();
            let grad_input = &softmax_output * (&grad_output - dot_product);
            self.inp.backward_internal(grad_input);
        }
    }

    fn zero_graph(&self) {
        self.inp.zero_graph();
        self.output.zero_graph();
    }

    fn build_graph(&self) {
        self.inp.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<UnkownShape>>> {
        Rc::new(RefCell::new(TensorSoftmax::<UnkownShape> {
            inp: self.inp.clone_into_dynamic(),
            output: self.output.clone_into_dynamic(),
        }))
    }
}

impl<S: Shape> Tensor<S> {
    pub fn softmax(self) -> Tensor<S> {
        TensorSoftmax::forward(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimensions::{Rank2, S};
    use ndarray::array;

    #[test]
    fn test_softmax_forward() {
        // Test case: 2 samples, 3 classes
        let input =
            Tensor::<Rank2<S<2>, S<3>>>::new(array![[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]].into_dyn());
        let result = input.softmax();
        let output = result.data();

        // Each row should sum to 1
        for i in 0..2 {
            let row_sum: f32 = (0..3).map(|j| output[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }

        // Values should be positive
        for val in output.iter() {
            assert!(*val > 0.0);
            assert!(*val < 1.0);
        }
    }

    #[test]
    fn test_softmax_uniform_input() {
        // When all inputs are the same, softmax should give uniform distribution
        let input = Tensor::<Rank2<S<1>, S<3>>>::new(array![[1.0, 1.0, 1.0]].into_dyn());
        let result = input.softmax();
        let output = result.data();

        let expected = 1.0 / 3.0;
        for val in output.iter() {
            assert!((*val - expected).abs() < 1e-6);
        }
    }
}
