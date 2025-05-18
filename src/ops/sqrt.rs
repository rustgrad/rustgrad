use std::cell::RefCell;
use std::rc::Rc;

use crate::dimensions::{Shape, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct TensorSqrt<S: Shape> {
    input: Tensor<S>,
}

impl<S: Shape> TensorSqrt<S> {
    pub fn forward(input: Tensor<S>) -> Tensor<S> {
        let result = input.data().mapv(f32::sqrt);
        let node = TensorSqrt { input };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S: Shape> Operation<S> for TensorSqrt<S> {
    fn backward(&self, output: &Tensor<S>) {
        if let Some(grad) = output.grad() {
            let input_data = self.input.data();
            let sqrt_input = input_data.mapv(f32::sqrt);
            let grad_input = grad / (sqrt_input * 2.0);
            self.input.backward_internal(grad_input);
        }
    }

    fn zero_graph(&self) {
        self.input.zero_graph();
    }

    fn build_graph(&self) {
        self.input.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<UnkownShape>>> {
        Rc::new(RefCell::new(TensorSqrt::<UnkownShape> {
            input: self.input.clone_into_dynamic(),
        }))
    }
}

impl<S: Shape> Tensor<S> {
    pub fn sqrt(&self) -> Tensor<S> {
        TensorSqrt::forward(self.clone())
    }
}

#[cfg(test)]
mod tests_sqrt {
    use super::*;
    use crate::dimensions::{Rank1, S};
    use ndarray::array;

    #[test]
    fn test_sqrt_forward_and_backward() {
        let x = Tensor::<Rank1<S<3>>>::new(array![4.0, 9.0, 16.0].into_dyn());
        let y = x.clone().sqrt();

        assert_eq!(y.data(), array![2.0, 3.0, 4.0].into_dyn());

        // Testing backward pass
        let loss = y.sum();
        loss.backward();

        let grad = x.grad().unwrap();
        assert_eq!(grad, array![0.25, 0.16666667, 0.125].into_dyn());
    }
}
