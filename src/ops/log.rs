use std::cell::RefCell;
use std::rc::Rc;

use crate::dimensions::{Shape, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
struct TensorLog<S1: Shape> {
    inp: Tensor<S1>,
}

impl<S1: Shape> TensorLog<S1> {
    fn forward(inp: Tensor<S1>) -> Tensor<S1> {
        let result = inp.container.borrow().array.clone().mapv(|x| x.ln());
        let node = TensorLog { inp };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S1: Shape> Operation<S1> for TensorLog<S1> {
    fn backward(&self, output: &Tensor<S1>) {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad = maybe_grad.unwrap_or(ndarray::Array::ones(output.shape()));
        // d/dx(ln(x)) = 1/x
        let inp_val = self.inp.container.borrow().array.clone();
        let grad_a = grad / inp_val;
        self.inp.backward_internal(grad_a);
    }

    fn zero_graph(&self) {
        self.inp.zero_graph();
    }

    fn build_graph(&self) {
        self.inp.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<UnkownShape>>> {
        Rc::new(RefCell::new(TensorLog::<UnkownShape> {
            inp: self.inp.clone_into_dynamic(),
        }))
    }
}

impl<S1: Shape> Tensor<S1> {
    pub fn log(self) -> Tensor<S1> {
        TensorLog::forward(self)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::dimensions::{Rank1, S};
    use super::*;

    #[test]
    fn test_log_forward() {
        let input = Tensor::<Rank1<S<3>>>::new(array![1.0, 2.718281828, 7.389056].into_dyn());
        let result = input.log();
        let expected = array![0.0, 1.0, 2.0].into_dyn();
        for (a, b) in result.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_log_backward() {
        let input = Tensor::<Rank1<S<3>>>::new(array![1.0, 2.0, 4.0].into_dyn());
        let result = input.clone().log();
        result.backward();
        // Gradient should be 1/x
        let expected_grad = array![1.0, 0.5, 0.25].into_dyn();
        let grad = input.container.borrow().grad.as_ref().unwrap();
        for (a, b) in grad.iter().zip(expected_grad.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
