use std::cell::RefCell;

use std::rc::Rc;

use crate::dimensions::{Shape, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct DivScalar<S: Shape> {
    input: Tensor<S>,
    scalar: f32,
}

impl<S: Shape> DivScalar<S> {
    pub fn forward(input: Tensor<S>, scalar: f32) -> Tensor<S> {
        let array = input.data() / scalar;
        let op = DivScalar {
            input: input.clone(),
            scalar,
        };
        Tensor::new_with_prev(array, Rc::new(RefCell::new(op)))
    }
}

impl<S: Shape> Operation<S> for DivScalar<S> {
    fn backward(&self, output: &Tensor<S>) {
        if let Some(grad) = output.grad() {
            let grad_input = grad.mapv(|x| x / self.scalar);
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
        Rc::new(RefCell::new(DivScalar::<UnkownShape> {
            input: self.input.clone_into_dynamic(),
            scalar: self.scalar,
        }))
    }
}

#[derive(Debug, Clone)]
pub struct InverseDivScalar<S: Shape> {
    input: Tensor<S>,
    scalar: f32,
}
impl<S: Shape> InverseDivScalar<S> {
    pub fn forward(input: Tensor<S>, scalar: f32) -> Tensor<S> {
        let array = scalar / input.data();
        let op = InverseDivScalar {
            input: input.clone(),
            scalar,
        };
        Tensor::new_with_prev(array, Rc::new(RefCell::new(op)))
    }
}
impl<S: Shape> Operation<S> for InverseDivScalar<S> {
    fn backward(&self, output: &Tensor<S>) {
        if let Some(grad) = output.grad() {
            let grad_input = -self.input.data().mapv(|x| self.scalar / (x * x)) * grad;
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
        Rc::new(RefCell::new(InverseDivScalar::<UnkownShape> {
            input: self.input.clone_into_dynamic(),
            scalar: self.scalar,
        }))
    }
}

impl<S: Shape> Tensor<S> {
    pub fn div_scalar(&self, scalar: f32) -> Tensor<S> {
        DivScalar::forward(self.clone(), scalar)
    }
}

impl<S: Shape> Tensor<S> {
    pub fn inverse_div_scalar(&self, scalar: f32) -> Tensor<S> {
        InverseDivScalar::forward(self.clone(), scalar)
    }
}

use std::ops::Div;

impl<S: Shape> Div<f32> for Tensor<S> {
    type Output = Tensor<S>;
    fn div(self, rhs: f32) -> Self::Output {
        self.div_scalar(rhs)
    }
}

impl<S: Shape> Div<Tensor<S>> for f32 {
    type Output = Tensor<S>;
    fn div(self, rhs: Tensor<S>) -> Self::Output {
        rhs.inverse_div_scalar(self)
    }
}

#[cfg(test)]
mod tests_div {
    use super::*;
    use crate::dimensions::{Rank1, S};
    use ndarray::array;

    #[test]
    fn test_div_scalar_forward_and_backward() {
        let x = Tensor::<Rank1<S<3>>>::new(array![6.0, 9.0, 12.0].into_dyn());
        let y = x.clone() / 3.0;
        assert_eq!(y.data(), array![2.0, 3.0, 4.0].into_dyn());

        let loss = y.sum();
        loss.backward();

        let grad = x.grad().unwrap();
        assert_eq!(grad, array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0].into_dyn());
    }
    #[test]
    fn test_inverse_div_scalar_forward_and_backward() {
        let x = Tensor::<Rank1<S<3>>>::new(array![2.0, 4.0, 5.0].into_dyn());
        let y = 10.0 / x.clone(); // Uses inverse_div_scalar

        assert_eq!(y.data(), array![5.0, 2.5, 2.0].into_dyn());

        let loss = y.sum(); // L = sum(10 / x_i)
        loss.backward();

        let grad = x.grad().unwrap(); // dL/dx_i = -10 / x_i^2
        let expected_grad = array![
            -10.0 / (2.0f32 * 2.0),
            -10.0 / (4.0f32 * 4.0),
            -10.0 / (5.0f32 * 5.0)
        ]
        .into_dyn();

        for (a, b) in grad.iter().zip(expected_grad.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", b, a);
        }
    }
}
