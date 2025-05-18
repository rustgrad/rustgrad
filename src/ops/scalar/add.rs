use std::cell::RefCell;

use std::rc::Rc;

use crate::dimensions::{Shape, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;

// ---------- Operation ----------
#[derive(Debug, Clone)]
pub struct AddScalar<S: Shape> {
    input: Tensor<S>,
    scalar: f32,
}

impl<S: Shape> AddScalar<S> {
    pub fn forward(input: Tensor<S>, scalar: f32) -> Tensor<S> {
        let array = input.data() + scalar;
        let op = AddScalar {
            input: input.clone(),
            scalar,
        };
        Tensor::new_with_prev(array, Rc::new(RefCell::new(op)))
    }
}

impl<S: Shape> Operation<S> for AddScalar<S> {
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
        Rc::new(RefCell::new(AddScalar::<UnkownShape> {
            input: self.input.clone_into_dynamic(),
            scalar: self.scalar,
        }))
    }
}

// ---------- Extension ----------
impl<S: Shape> Tensor<S> {
    pub fn add_scalar(&self, scalar: f32) -> Tensor<S> {
        AddScalar::forward(self.clone(), scalar)
    }
}

// ---------- Overload ----------
use std::ops::Add;

impl<S: Shape> Add<f32> for Tensor<S> {
    type Output = Tensor<S>;
    fn add(self, rhs: f32) -> Self::Output {
        self.add_scalar(rhs)
    }
}

impl<S: Shape> Add<Tensor<S>> for f32 {
    type Output = Tensor<S>;
    fn add(self, rhs: Tensor<S>) -> Self::Output {
        rhs.add_scalar(self)
    }
}

// ---------- Test ----------
#[cfg(test)]
mod tests_add {
    use super::*;
    use crate::dimensions::{Rank1, S};
    use ndarray::array;

    #[test]
    fn test_add_scalar_forward_and_backward() {
        let x = Tensor::<Rank1<S<3>>>::new(array![1.0, 2.0, 3.0].into_dyn());
        let y = x.clone() + 5.0;
        assert_eq!(y.data(), array![6.0, 7.0, 8.0].into_dyn());

        let loss = y.sum();
        loss.backward();

        let grad = x.grad().unwrap();
        assert_eq!(grad, array![1.0, 1.0, 1.0].into_dyn());
    }
}
