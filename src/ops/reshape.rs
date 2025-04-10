use std::marker::PhantomData;
use std::{cell::RefCell, fmt::Debug, rc::Rc};

use crate::dimensions::{DynamicShape, Shape};
use crate::shape::ArrayShape;
use crate::tensor::{Operation, Tensor};

#[derive(Debug, Clone)]
struct TensorReshape<S: Shape, K: Shape> {
    tensor: Tensor<S>,
    input_shape: ArrayShape,
    ph: PhantomData<K>,
}

impl<S: Shape, K: Shape> TensorReshape<S, K> {
    pub fn forward(input: Tensor<S>) -> Tensor<K> {
        let output_shape = K::shape();
        let new_data = input.data().into_shape_with_order(output_shape).unwrap();
        let node = TensorReshape {
            input_shape: input.shape(),
            tensor: input,
            ph: PhantomData,
        };
        Tensor::new_with_prev(new_data, Rc::new(RefCell::new(node)))
    }
}

impl<S: Shape, K: Shape> Operation<K> for TensorReshape<S, K> {
    fn backward(&mut self, output: &mut Tensor<K>) {
        let new_grad = output
            .grad()
            .expect("Missing gradient")
            .into_shape_with_order(self.input_shape.dims.clone())
            .unwrap();
        self.tensor.backward_internal(new_grad);
    }

    fn zero_graph(&self) {
        self.tensor.zero_graph();
    }

    fn build_graph(&self) {
        self.tensor.build_graph();
    }
    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        Rc::new(RefCell::new(TensorReshape::<DynamicShape, DynamicShape> {
            tensor: self.tensor.clone_into_dynamic(),
            input_shape: self.input_shape.clone(),
            ph: PhantomData,
        }))
    }
}

impl<S: Shape> Tensor<S> {
    pub fn reshape<K: Shape>(self) -> Tensor<K> {
        TensorReshape::<S, K>::forward(self)
    }

    pub fn reshape_no_grad<K: Shape>(self) -> Tensor<K> {
        let shape = self.shape();
        let new_data = self.data().into_shape_with_order(shape.dims).unwrap();
        Tensor::new(new_data)
    }
}
