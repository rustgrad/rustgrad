use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::dimensions::{DynamicShape, Rank0, Shape};
use crate::tensor::{Operation, Tensor};
use ndarray::{Array0, Axis, Ix0};

#[derive(Debug, Clone)]
pub struct Sum<SIn: Shape, SOut: Shape> {
    input: Tensor<SIn>,
    axis: Option<usize>,
    phantom: PhantomData<SOut>,
}

impl<SIn: Shape, SOut: Shape> Sum<SIn, SOut> {
    pub fn forward(input: Tensor<SIn>, axis: usize) -> Tensor<SOut> {
        let array = input.container.borrow().array.clone();
        let summed = array.sum_axis(Axis(axis)).into_dyn();

        let op = Sum {
            input: input.clone(),
            axis: Some(axis),
            phantom: PhantomData,
        };

        Tensor::new_with_prev(summed, Rc::new(RefCell::new(op)))
    }
}

impl<SIn: Shape, SOut: Shape> Operation<SOut> for Sum<SIn, SOut> {
    fn backward(&mut self, output: &mut Tensor<SOut>) {
        let grad = output
            .container
            .borrow()
            .grad
            .clone()
            .unwrap_or_else(|| ndarray::Array::zeros(output.shape()));

        let input_shape = self.input.shape();

        // Expand grad to input shape
        let broadcasted = if let Some(axis) = self.axis {
            let grad_expanded = grad.insert_axis(Axis(axis));
            grad_expanded
                .broadcast(input_shape.clone())
                .unwrap()
                .to_owned()
        } else {
            grad.broadcast(input_shape.clone()).unwrap().to_owned()
        };

        self.input.backward_internal(broadcasted.into_dyn());
    }

    fn zero_graph(&self) {
        self.input.zero_graph();
    }

    fn build_graph(&self) {
        self.input.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        Rc::new(RefCell::new(Sum::<DynamicShape, DynamicShape> {
            input: self.input.clone_into_dynamic(),
            axis: self.axis,
            phantom: PhantomData,
        }))
    }
}

impl<S: Shape> Tensor<S> {
    pub fn sum_along<SOut: Shape>(&self, axis: usize) -> Tensor<SOut> {
        Sum::<S, SOut>::forward(self.clone(), axis)
    }

    pub fn sum(&self) -> Tensor<crate::dimensions::Rank0> {
        let array = self.container.borrow().array.clone();
        let summed = Array0::from_elem(Ix0(), array.sum());
        let tensor = Tensor::<Rank0>::new_with_prev(
            summed.into_dyn(),
            Rc::new(RefCell::new(Sum::<S, crate::dimensions::Rank0> {
                input: self.clone(),
                axis: None,
                phantom: PhantomData,
            })),
        );
        tensor
    }
}
