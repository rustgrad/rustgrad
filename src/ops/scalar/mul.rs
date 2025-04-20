use std::cell::RefCell;

use std::rc::Rc;

use crate::dimensions::{DynamicShape, Shape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct MulScalar<S: Shape> {
    input: Tensor<S>,
    scalar: f32,
}

impl<S: Shape> MulScalar<S> {
    pub fn forward(input: Tensor<S>, scalar: f32) -> Tensor<S> {
        let array = input.data() * scalar;

        let op = MulScalar {
            input: input.clone(),
            scalar,
        };

        Tensor::new_with_prev(array, Rc::new(RefCell::new(op)))
    }
}

impl<S: Shape> Operation<S> for MulScalar<S> {
    fn backward(&self, output: &Tensor<S>) {
        if let Some(grad_out) = output.grad() {
            let grad_input = grad_out.mapv(|x| x * self.scalar);
            self.input.backward_internal(grad_input);
        }
    }

    fn zero_graph(&self) {
        self.input.zero_graph();
    }

    fn build_graph(&self) {
        self.input.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        Rc::new(RefCell::new(MulScalar::<DynamicShape> {
            input: self.input.clone_into_dynamic(),
            scalar: self.scalar,
        }))
    }
}

impl<S: Shape> Tensor<S> {
    pub fn mul_scalar(&self, scalar: f32) -> Tensor<S> {
        MulScalar::forward(self.clone(), scalar)
    }
}
use std::ops::Mul;

impl<S: Shape> Mul<f32> for Tensor<S> {
    type Output = Tensor<S>;

    fn mul(self, rhs: f32) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl<S: Shape> Mul<Tensor<S>> for f32 {
    type Output = Tensor<S>;

    fn mul(self, rhs: Tensor<S>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}
#[cfg(test)]
mod tests {
    use crate::dimensions::{Rank1, S};
    use crate::tensor::Tensor;
    use ndarray::array;

    #[test]
    fn test_mul_scalar_forward_and_backward() {
        // Forward
        let x = Tensor::<Rank1<S<3>>>::new(array![1.0, 2.0, 3.0].into_dyn());
        let y = x.clone() * 2.0;

        let expected = array![2.0, 4.0, 6.0].into_dyn();
        assert_eq!(y.data(), expected);

        // Backward
        let loss = y.sum(); // Dummy loss function
        loss.backward();

        // d(loss)/dx = d(y)/dx * d(loss)/dy = 2.0 * 1 = 2.0
        let grad = x.grad().unwrap();
        let expected_grad = array![2.0, 2.0, 2.0].into_dyn();
        assert_eq!(grad, expected_grad);
    }
}
