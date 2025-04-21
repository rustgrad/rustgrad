use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::dimensions::{DynamicShape, Shape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct TensorPermute<SIn: Shape, SOut: Shape> {
    input: Tensor<SIn>,
    output: PhantomData<SOut>,
    axes: Vec<usize>,
}

impl<SIn: Shape> Tensor<SIn> {
    pub fn permute<SOut: Shape>(&self, axes: Vec<usize>) -> Tensor<SOut> {
        let permuted = self.data().clone().permuted_axes(axes.clone());

        let op: TensorPermute<SIn, SOut> = TensorPermute {
            input: self.clone(),
            output: PhantomData,
            axes: axes,
        };

        Tensor::new_with_prev(permuted, Rc::new(RefCell::new(op)))
    }
}

impl<SIn: Shape, SOut: Shape> Operation<SOut> for TensorPermute<SIn, SOut> {
    fn backward(&self, output: &Tensor<SOut>) {
        let grad = output
            .grad()
            .expect("Called backward on Operation without gradient.");
        let grad_input = grad.permuted_axes(self.axes.clone());
        self.input.backward_internal(grad_input);
    }

    fn zero_graph(&self) {
        self.input.zero_graph();
    }

    fn build_graph(&self) {
        self.input.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        let operation = TensorPermute::<DynamicShape, DynamicShape> {
            input: self.input.clone_into_dynamic(),
            output: PhantomData,
            axes: self.axes.clone(),
        };
        return Rc::new(RefCell::new(operation));
    }
}
