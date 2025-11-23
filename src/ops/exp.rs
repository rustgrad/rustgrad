use std::cell::RefCell;
use std::rc::Rc;

use crate::dimensions::{Shape, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
struct TensorExp<S1: Shape> {
    inp: Tensor<S1>,
}

impl<S1: Shape> TensorExp<S1> {
    fn forward(inp: Tensor<S1>) -> Tensor<S1> {
        let result = inp.container.borrow().array.clone().mapv(|x| x.exp());
        let node = TensorExp { inp };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S1: Shape> Operation<S1> for TensorExp<S1> {
    fn backward(&self, output: &Tensor<S1>) {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad = maybe_grad.unwrap_or(ndarray::Array::ones(output.shape()));
        // d/dx(exp(x)) = exp(x) = output
        let output_val = output.container.borrow().array.clone();
        let grad_a = output_val * &grad;
        self.inp.backward_internal(grad_a);
    }

    fn zero_graph(&self) {
        self.inp.zero_graph();
    }

    fn build_graph(&self) {
        self.inp.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<UnkownShape>>> {
        Rc::new(RefCell::new(TensorExp::<UnkownShape> {
            inp: self.inp.clone_into_dynamic(),
        }))
    }
}

impl<S1: Shape> Tensor<S1> {
    pub fn exp(self) -> Tensor<S1> {
        TensorExp::forward(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimensions::{Rank1, S};
    use ndarray::array;

    #[test]
    fn test_exp_forward() {
        let input = Tensor::<Rank1<S<3>>>::new(array![0.0, 1.0, 2.0].into_dyn());
        let result = input.exp();
        let expected = array![1.0, 2.7182817, 7.389056].into_dyn();
        for (a, b) in result.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_exp_backward() {
        let input = Tensor::<Rank1<S<3>>>::new(array![0.0, 1.0, 2.0].into_dyn());
        let result = input.clone().exp();
        result.backward();
        // Gradient should be exp(x)
        let expected_grad = array![1.0, 2.7182817, 7.389056].into_dyn();
        let grad = input.container.borrow().grad.as_ref().unwrap();
        for (a, b) in grad.iter().zip(expected_grad.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
