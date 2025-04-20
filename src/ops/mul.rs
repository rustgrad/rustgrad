use std::cell::RefCell;
use std::ops::Mul;
use std::rc::Rc;

use crate::dimensions::{DynamicShape, Shape};
use crate::ops::Operation;
use crate::tensor::{ShapeCompatible, Tensor};

#[derive(Debug, Clone)]
struct TensorMul<S1: Shape, S2: Shape> {
    lhs: Tensor<S1>,
    rhs: Tensor<S2>,
}

impl<S1: Shape, S2: Shape> TensorMul<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn forward(lhs: Tensor<S1>, rhs: Tensor<S2>) -> Tensor<<S1 as ShapeCompatible<S2>>::Output> {
        let result = lhs.container.borrow().array.clone() * rhs.container.borrow().array.clone();
        let node = TensorMul { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S1: Shape, S2: Shape> Operation<<S1 as ShapeCompatible<S2>>::Output> for TensorMul<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn backward(&self, output: &Tensor<<S1 as ShapeCompatible<S2>>::Output>) {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad = maybe_grad.unwrap_or(ndarray::Array::ones(output.shape()));

        let lhs_val = self.lhs.container.borrow().array.clone();
        let rhs_val = self.rhs.container.borrow().array.clone();

        let grad_a = &grad * &rhs_val;
        let grad_b = &grad * &lhs_val;

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
        Rc::new(RefCell::new(TensorMul::<DynamicShape, DynamicShape> {
            lhs: self.lhs.clone_into_dynamic(),
            rhs: self.rhs.clone_into_dynamic(),
        }))
    }
}

// Define Mul trait
// pub trait TensorMul<Rhs> {
//     type Output;
//     fn dot(self, rhs: Rhs) -> Self::Output;
// }

impl<S1: Shape, S2: Shape> Mul<Tensor<S2>> for Tensor<S1>
where
    S1: ShapeCompatible<S2>,
{
    type Output = Tensor<<S1 as ShapeCompatible<S2>>::Output>;

    fn mul(self, other: Tensor<S2>) -> Self::Output {
        TensorMul::forward(self, other)
    }
}

#[cfg(test)]
mod tests {
    use crate::dimensions::{Rank1, S};
    use crate::tensor::Tensor;
    use ndarray::array;

    #[test]
    fn test_dot_forward_backward() {
        let a: Tensor<Rank1<S<3>>> = Tensor::new(array![1.0, 2.0, 3.0].into_dyn());
        let b: Tensor<Rank1<S<3>>> = Tensor::new(array![4.0, 5.0, 6.0].into_dyn());
        let c = a.clone() * b.clone();

        // Forward check
        assert_eq!(c.data(), array![4.0, 10.0, 18.0].into_dyn());

        // Set gradient of output to ones
        c.backward();

        // Backward gradients should be:
        // ∂c/∂a = b = [4.0, 5.0, 6.0]
        // ∂c/∂b = a = [1.0, 2.0, 3.0]
        assert_eq!(a.grad().unwrap(), array![4.0, 5.0, 6.0].into_dyn());
        assert_eq!(b.grad().unwrap(), array![1.0, 2.0, 3.0].into_dyn());
    }
}
