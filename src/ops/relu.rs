use std::cell::RefCell;
use std::rc::Rc;

use crate::dimensions::{DynamicShape, Shape};
use crate::tensor::{Operation, Tensor};

#[derive(Debug, Clone)]
struct TensorRelu<S1: Shape> {
    inp: Tensor<S1>,
}

impl<S1: Shape> TensorRelu<S1> {
    fn forward(inp: Tensor<S1>) -> Tensor<S1> {
        let result = inp.container.borrow().array.clone().mapv(|x| x.max(0.0));
        let node = TensorRelu { inp };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S1: Shape> Operation<S1> for TensorRelu<S1> {
    fn backward(&mut self, output: &mut Tensor<S1>) {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad = maybe_grad.unwrap_or(ndarray::Array::ones(output.shape()));
        let inp_val = self.inp.container.borrow().array.clone();
        let grad_a = inp_val.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * &grad;
        self.inp.backward_internal(grad_a);
    }

    fn zero_graph(&self) {
        self.inp.zero_graph();
    }

    fn build_graph(&self) {
        self.inp.build_graph();
    }
    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        Rc::new(RefCell::new(TensorRelu::<DynamicShape> {
            inp: self.inp.clone_into_dynamic(),
        }))
    }
}

impl<S1: Shape> Tensor<S1> {
    pub fn relu(self) -> Tensor<S1> {
        TensorRelu::forward(self)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::dimensions::{Rank1, S};

    use super::*;

    #[test]
    fn test_relu_forward_positive_values() {
        let input = Tensor::<Rank1<S<3>>>::new(array![1.0, 2.0, 3.0].into_dyn());
        let result = input.relu();
        let expected = array![1.0, 2.0, 3.0].into_dyn();
        assert_eq!(result.data(), expected);
    }
    #[test]
    fn test_relu_forward_negative_values() {
        let input = Tensor::<Rank1<S<3>>>::new(array![-1.0, -2.0, -3.0].into_dyn());
        let result = input.relu();
        let expected = array![0.0, 0.0, 0.0].into_dyn();
        assert_eq!(result.data(), expected);
    }

    #[test]
    fn test_relu_forward_mixed_values() {
        let input = Tensor::<Rank1<S<3>>>::new(array![-1.0, 0.0, 3.0].into_dyn());
        let result = input.relu();
        let expected = array![0.0, 0.0, 3.0].into_dyn();
        assert_eq!(result.data(), expected);
    }

    #[test]
    fn test_relu_backward() {
        let input = Tensor::<Rank1<S<3>>>::new(array![-1.0, 2.0, -3.0].into_dyn());
        let mut result = input.clone().relu();
        result.backward();
        let expected_grad = array![0.0, 1.0, 0.0].into_dyn();
        assert_eq!(
            input.container.borrow().grad.as_ref().unwrap(),
            &expected_grad
        );
    }
}
