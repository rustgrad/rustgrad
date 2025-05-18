use std::{cell::RefCell, ops::Neg, rc::Rc};

use crate::{
    dimensions::{Shape, UnkownShape},
    tensor::Tensor,
};

use super::Operation;

#[derive(Debug, Clone)]
pub struct TensorNeg<S: Shape> {
    tensor: Tensor<S>,
}

impl<S: Shape> TensorNeg<S> {
    fn forward(tensor: Tensor<S>) -> Tensor<S> {
        let result = -tensor.container.borrow().array.clone();
        let node = TensorNeg { tensor };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S: Shape> Operation<S> for TensorNeg<S> {
    fn backward(&self, output: &Tensor<S>) {
        let grad = output
            .grad()
            .unwrap_or(ndarray::Array::zeros(output.shape()));
        self.tensor.backward_internal(-grad);
    }

    fn zero_graph(&self) {
        self.tensor.zero_graph();
    }

    fn build_graph(&self) {
        self.tensor.build_graph();
    }
    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<UnkownShape>>> {
        Rc::new(RefCell::new(TensorNeg::<UnkownShape> {
            tensor: self.tensor.clone_into_dynamic(),
        }))
    }
}

impl<S: Shape> Neg for Tensor<S> {
    type Output = Tensor<S>;
    fn neg(self) -> Self::Output {
        TensorNeg::forward(self)
    }
}
