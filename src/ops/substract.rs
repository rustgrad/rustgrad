use std::cell::RefCell;
use std::ops::Sub;
use std::rc::Rc;

use crate::dimensions::{DynamicShape, Shape};
use crate::ops::Operation;
use crate::tensor::{ShapeCompatible, Tensor};

#[derive(Debug, Clone)]
pub struct TensorSub<S1: Shape, S2: Shape> {
    lhs: Tensor<S1>,
    rhs: Tensor<S2>,
}

impl<S1: Shape, S2: Shape> TensorSub<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn forward(lhs: Tensor<S1>, rhs: Tensor<S2>) -> Tensor<<S1 as ShapeCompatible<S2>>::Output> {
        let result = lhs.container.borrow().array.clone() - rhs.container.borrow().array.clone();
        let node = TensorSub { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S1: Shape, S2: Shape> Operation<<S1 as ShapeCompatible<S2>>::Output> for TensorSub<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn backward(&self, output: &Tensor<<S1 as ShapeCompatible<S2>>::Output>) {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad = maybe_grad.unwrap_or(ndarray::Array::zeros(output.shape()));
        let grad_a = grad.clone();
        let grad_b = -grad; // gradient for rhs should be negated
        self.lhs.backward_internal(grad_a);
        self.rhs.backward_internal(grad_b);
    }

    fn zero_graph(&self) {
        self.lhs.zero_graph();
        self.rhs.zero_graph();
    }

    fn build_graph(&self) {
        self.lhs.build_graph();
        self.rhs.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        Rc::new(RefCell::new(TensorSub::<DynamicShape, DynamicShape> {
            lhs: self.lhs.clone_into_dynamic(),
            rhs: self.rhs.clone_into_dynamic(),
        }))
    }
}

impl<S1: Shape, S2: Shape> Sub<Tensor<S2>> for Tensor<S1>
where
    S1: ShapeCompatible<S2>,
{
    type Output = Tensor<<S1 as ShapeCompatible<S2>>::Output>;

    fn sub(self, other: Tensor<S2>) -> Self::Output {
        TensorSub::forward(self, other)
    }
}

#[cfg(test)]
mod tests_sub {
    use super::*;
    use crate::dimensions::{Rank1, S};
    use ndarray::array;

    #[test]
    fn test_subtract_tensors_forward_and_backward() {
        let x = Tensor::<Rank1<S<3>>>::new(array![6.0, 9.0, 12.0].into_dyn());
        let y = Tensor::<Rank1<S<3>>>::new(array![2.0, 4.0, 6.0].into_dyn());
        let z = x.clone() - y.clone();

        assert_eq!(z.data(), array![4.0, 5.0, 6.0].into_dyn());

        // Testing backward pass
        let loss = z.sum();
        loss.backward();

        let grad = x.grad().unwrap();
        assert_eq!(grad, array![1.0, 1.0, 1.0].into_dyn());

        let grad_y = y.grad().unwrap();
        assert_eq!(grad_y, array![-1.0, -1.0, -1.0].into_dyn());
    }

    #[test]
    fn test_subtract_tensor_and_scalar_forward_and_backward() {
        let x = Tensor::<Rank1<S<3>>>::new(array![6.0, 9.0, 12.0].into_dyn());
        let scalar = 3.0;
        let z = x.clone() - scalar;

        assert_eq!(z.data(), array![3.0, 6.0, 9.0].into_dyn());

        // Testing backward pass
        let loss = z.sum();
        loss.backward();

        let grad = x.grad().unwrap();
        assert_eq!(grad, array![1.0, 1.0, 1.0].into_dyn());
    }
}
