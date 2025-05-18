use std::marker::PhantomData;
use std::{cell::RefCell, fmt::Debug, rc::Rc};

use crate::array_shape::ArrayShape;
use crate::dimensions::{DynamicShape, Shape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
struct TensorReshape<S: Shape, K: Shape> {
    tensor: Tensor<S>,
    input_shape: ArrayShape,
    ph: PhantomData<K>,
}

impl<S: Shape, K: Shape> TensorReshape<S, K> {
    pub fn forward(input: &Tensor<S>, output_shape: K) -> Tensor<K> {
        let new_data = input
            .data()
            .into_shape_with_order(output_shape._shape())
            .unwrap();
        let node = TensorReshape {
            input_shape: input.shape(),
            tensor: input.clone(),
            ph: PhantomData,
        };
        Tensor::new_with_prev(new_data, Rc::new(RefCell::new(node)))
    }
}

impl<S: Shape, K: Shape> Operation<K> for TensorReshape<S, K> {
    fn backward(&self, output: &Tensor<K>) {
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
    pub fn reshape<K: Shape>(&self, output_shape: K) -> Tensor<K> {
        TensorReshape::<S, K>::forward(self, output_shape)
    }

    pub fn reshape_no_grad<K: Shape>(&self, output_shape: K) -> Tensor<K> {
        let dims = output_shape._shape().dims;
        let new_data = self.data().into_shape_with_order(dims).unwrap();
        Tensor::new(new_data)
    }
}
