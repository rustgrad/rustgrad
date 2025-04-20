use std::cell::RefCell;
use std::ops::Add;
use std::rc::Rc;

use crate::dimensions::{DynamicShape, Shape};
use crate::tensor::{Operation, ShapeCompatible, Tensor};

#[derive(Debug, Clone)]
struct TensorAdd<S1: Shape, S2: Shape> {
    lhs: Tensor<S1>,
    rhs: Tensor<S2>,
}

impl<S1: Shape, S2: Shape> TensorAdd<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn forward(lhs: Tensor<S1>, rhs: Tensor<S2>) -> Tensor<<S1 as ShapeCompatible<S2>>::Output> {
        let result = lhs.container.borrow().array.clone() + rhs.container.borrow().array.clone();
        let node = TensorAdd { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S1: Shape, S2: Shape> Operation<<S1 as ShapeCompatible<S2>>::Output> for TensorAdd<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn backward(&mut self, output: &mut Tensor<<S1 as ShapeCompatible<S2>>::Output>) {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad = maybe_grad.unwrap_or(ndarray::Array::zeros(output.shape()));
        let grad_a = grad.clone();
        let grad_b = grad;
        self.lhs.backward_internal(grad_a);
        self.rhs.backward_internal(grad_b);
    }

    fn zero_graph(&self) {
        self.lhs.zero_graph();
        self.rhs.zero_graph();
    }

    fn build_graph(&self) {
        self.lhs.build_graph();
        self.rhs.build_graph();
    }
    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        Rc::new(RefCell::new(TensorAdd::<DynamicShape, DynamicShape> {
            lhs: self.lhs.clone_into_dynamic(),
            rhs: self.rhs.clone_into_dynamic(),
        }))
    }
}

impl<S1: Shape, S2: Shape> Add<Tensor<S2>> for Tensor<S1>
where
    S1: ShapeCompatible<S2>,
{
    type Output = Tensor<<S1 as ShapeCompatible<S2>>::Output>;

    fn add(self, other: Tensor<S2>) -> Self::Output {
        TensorAdd::forward(self, other)
    }
}
