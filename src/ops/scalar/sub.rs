use std::cell::RefCell;

use std::rc::Rc;

use crate::dimensions::{Shape, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct SubScalar<S: Shape> {
    input: Tensor<S>,
    scalar: f32,
}

impl<S: Shape> SubScalar<S> {
    pub fn forward(input: Tensor<S>, scalar: f32) -> Tensor<S> {
        let array = input.data() - scalar;
        let op = SubScalar {
            input: input.clone(),
            scalar,
        };
        Tensor::new_with_prev(array, Rc::new(RefCell::new(op)))
    }
}

impl<S: Shape> Operation<S> for SubScalar<S> {
    fn backward(&self, output: &Tensor<S>) {
        if let Some(grad) = output.grad() {
            self.input.backward_internal(grad);
        }
    }

    fn zero_graph(&self) {
        self.input.zero_graph();
    }
    fn build_graph(&self) {
        self.input.build_graph();
    }
    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<UnkownShape>>> {
        Rc::new(RefCell::new(SubScalar::<UnkownShape> {
            input: self.input.clone_into_dynamic(),
            scalar: self.scalar,
        }))
    }
}

impl<S: Shape> Tensor<S> {
    pub fn sub_scalar(&self, scalar: f32) -> Tensor<S> {
        SubScalar::forward(self.clone(), scalar)
    }
}

use std::ops::Sub;

impl<S: Shape> Sub<f32> for Tensor<S> {
    type Output = Tensor<S>;
    fn sub(self, rhs: f32) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

impl<S: Shape> Sub<Tensor<S>> for f32 {
    type Output = Tensor<S>;
    fn sub(self, rhs: Tensor<S>) -> Self::Output {
        -rhs + self
    }
}

#[cfg(test)]
mod tests_sub {
    use super::*;
    use crate::dimensions::{Rank1, S};
    use ndarray::array;

    #[test]
    fn test_sub_scalar_forward_and_backward() {
        let x = Tensor::<Rank1<S<3>>>::new(array![5.0, 6.0, 7.0].into_dyn());
        let y = x.clone() - 2.0;
        assert_eq!(y.data(), array![3.0, 4.0, 5.0].into_dyn());

        let loss = y.sum();
        loss.backward();

        let grad = x.grad().unwrap();
        assert_eq!(grad, array![1.0, 1.0, 1.0].into_dyn());
    }
}
