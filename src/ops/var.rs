use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::dimensions::{Shape, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;
use ndarray::Axis;

#[derive(Debug, Clone)]
pub struct Var<SIn: Shape, SOut: Shape> {
    input: Tensor<SIn>,
    axis: usize,
    keepdims: bool,
    phantom: PhantomData<SOut>,
}

impl<SIn: Shape, SOut: Shape> Var<SIn, SOut> {
    pub fn forward(input: Tensor<SIn>, axis: usize, keepdims: bool) -> Tensor<SOut> {
        let array = input.container.borrow().array.clone();
        let mean = array.mean_axis(Axis(axis)).unwrap();

        let mean_expanded = if keepdims {
            mean.insert_axis(Axis(axis))
        } else {
            mean
        };

        let diff = &array - &mean_expanded;
        let squared = &diff * &diff;

        let var = squared.mean_axis(Axis(axis)).unwrap();

        let var = if keepdims {
            var.insert_axis(Axis(axis)).into_dyn()
        } else {
            var.into_dyn()
        };

        let op = Var {
            input: input.clone(),
            axis,
            keepdims,
            phantom: PhantomData,
        };

        Tensor::new_with_prev(var, Rc::new(RefCell::new(op)))
    }
}

impl<SIn: Shape, SOut: Shape> Operation<SOut> for Var<SIn, SOut> {
    fn backward(&self, output: &Tensor<SOut>) {
        let input = self.input.clone();
        let input_array = input.data();
        let grad_out = output.grad().expect("Grad not calculated");

        let axis = Axis(self.axis);
        let mean = input_array.mean_axis(axis).unwrap();

        let mean_expanded = if self.keepdims {
            mean.insert_axis(axis)
        } else {
            mean
        };

        let diff = &input_array - &mean_expanded;

        let grad_broadcasted = grad_out.broadcast(diff.raw_dim()).unwrap();
        let input_len = input_array.len_of(axis) as f32;

        let grad_input = (&diff * 2.0 / input_len) * &grad_broadcasted;

        self.input.backward_internal(grad_input.into_dyn());
    }

    fn zero_graph(&self) {
        self.input.zero_graph();
    }

    fn build_graph(&self) {
        self.input.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<UnkownShape>>> {
        Rc::new(RefCell::new(Var::<UnkownShape, UnkownShape> {
            input: self.input.clone_into_dynamic(),
            axis: self.axis,
            keepdims: self.keepdims,
            phantom: PhantomData,
        }))
    }
}
impl<S: Shape> Tensor<S> {
    pub fn var_along<SOut: Shape>(&self, axis: usize, keepdims: bool) -> Tensor<SOut> {
        Var::forward(self.clone(), axis, keepdims)
    }
}
#[cfg(test)]
mod tests {
    use crate::dimensions::{Rank2, S};
    use crate::tensor::Tensor;
    use ndarray::array;

    #[test]
    fn test_variance() {
        let x =
            Tensor::<Rank2<S<2>, S<3>>>::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());

        let var: Tensor<Rank2<S<1>, S<3>>> = x.var_along(0, true); // Variance along axis 0, keepdims = true

        let expected = array![[4.5, 4.5, 4.5]].into_dyn(); // Variance across rows
        assert!((&var.data() - expected).iter().all(|x| x.abs() < 1e-4));
    }
}
