use ndarray::Array;
use std::cell::RefCell;
use std::ops::Div;
use std::rc::Rc;

use crate::dimensions::{DynamicShape, Shape};
use crate::ops::Operation;
use crate::tensor::{ShapeCompatible, Tensor};

#[derive(Debug, Clone)]
struct TensorDiv<S1: Shape, S2: Shape> {
    lhs: Tensor<S1>,
    rhs: Tensor<S2>,
}

impl<S1: Shape, S2: Shape> TensorDiv<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn forward(lhs: Tensor<S1>, rhs: Tensor<S2>) -> Tensor<<S1 as ShapeCompatible<S2>>::Output> {
        let result = lhs.data() / rhs.data();
        let node = TensorDiv { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S1: Shape, S2: Shape> Operation<<S1 as ShapeCompatible<S2>>::Output> for TensorDiv<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn backward(&self, output: &Tensor<<S1 as ShapeCompatible<S2>>::Output>) {
        let grad = output.grad().unwrap_or_else(|| Array::ones(output.shape()));
        let rhs_data = self.rhs.data();
        let lhs_data = self.lhs.data();

        let lhs_grad = grad.clone() / rhs_data.clone();
        let rhs_grad = -grad * lhs_data / (rhs_data.clone() * rhs_data);
        self.lhs.backward_internal(lhs_grad);
        self.rhs.backward_internal(rhs_grad);
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
        Rc::new(RefCell::new(TensorDiv::<DynamicShape, DynamicShape> {
            lhs: self.lhs.clone_into_dynamic(),
            rhs: self.rhs.clone_into_dynamic(),
        }))
    }
}

impl<S1: Shape, S2: Shape> Div<Tensor<S2>> for Tensor<S1>
where
    S1: ShapeCompatible<S2>,
{
    type Output = Tensor<<S1 as ShapeCompatible<S2>>::Output>;
    fn div(self, rhs: Tensor<S2>) -> Self::Output {
        TensorDiv::forward(self, rhs)
    }
}
