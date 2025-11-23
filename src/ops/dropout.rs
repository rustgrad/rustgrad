use std::cell::RefCell;
use std::rc::Rc;

use crate::dimensions::{Shape, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
struct TensorDropout<S: Shape> {
    inp: Tensor<S>,
    mask: ndarray::Array<f32, ndarray::IxDyn>,
    p: f32, // dropout probability
}

impl<S: Shape> TensorDropout<S> {
    /// Applies dropout during training
    /// Each element is kept with probability (1-p) and scaled by 1/(1-p)
    fn forward(inp: Tensor<S>, p: f32, training: bool) -> Tensor<S> {
        if !training || p == 0.0 {
            // During inference or if p=0, just return input as-is
            return inp;
        }

        let inp_data = inp.container.borrow().array.clone();
        let shape = inp_data.shape();

        // Create dropout mask: 1 with probability (1-p), 0 with probability p
        let mask = ndarray::Array::from_shape_fn(shape, |_| {
            if rand::random::<f32>() > p {
                1.0 / (1.0 - p) // Scale by 1/(1-p) for inverted dropout
            } else {
                0.0
            }
        });

        let result = &inp_data * &mask;

        let node = TensorDropout {
            inp: inp.clone(),
            mask,
            p,
        };

        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S: Shape> Operation<S> for TensorDropout<S> {
    fn backward(&self, output: &Tensor<S>) {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad = maybe_grad.unwrap_or(ndarray::Array::ones(output.shape()));

        // Gradient flows only through the non-dropped neurons
        let grad_input = &grad * &self.mask;

        self.inp.backward_internal(grad_input);
    }

    fn zero_graph(&self) {
        self.inp.zero_graph();
    }

    fn build_graph(&self) {
        self.inp.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<UnkownShape>>> {
        Rc::new(RefCell::new(TensorDropout::<UnkownShape> {
            inp: self.inp.clone_into_dynamic(),
            mask: self.mask.clone(),
            p: self.p,
        }))
    }
}

impl<S: Shape> Tensor<S> {
    /// Applies dropout with probability p
    ///
    /// # Arguments
    /// * `p` - Probability of dropping a neuron (0.0 to 1.0)
    /// * `training` - Whether in training mode (true) or inference mode (false)
    ///
    /// # Example
    /// ```ignore
    /// let x = tensor.dropout(0.5, true); // Drop 50% during training
    /// let y = tensor.dropout(0.5, false); // No dropout during inference
    /// ```
    pub fn dropout(self, p: f32, training: bool) -> Tensor<S> {
        TensorDropout::forward(self, p, training)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::dimensions::{Rank1, S};
    use super::*;

    #[test]
    fn test_dropout_inference() {
        // During inference, dropout should be identity
        let input = Tensor::<Rank1<S<5>>>::new(array![1.0, 2.0, 3.0, 4.0, 5.0].into_dyn());
        let result = input.clone().dropout(0.5, false);
        assert_eq!(result.data(), input.data());
    }

    #[test]
    fn test_dropout_training() {
        // During training, some values should be zeroed
        let input = Tensor::<Rank1<S<100>>>::new(
            ndarray::Array::from_elem(ndarray::IxDyn(&[100]), 1.0)
        );
        let result = input.dropout(0.5, true);

        // Count non-zero elements
        let non_zero = result.data().iter().filter(|&&x| x != 0.0).count();

        // Should have roughly 50% non-zero (allow for randomness)
        assert!(non_zero > 30 && non_zero < 70, "Got {} non-zero elements", non_zero);
    }
}
