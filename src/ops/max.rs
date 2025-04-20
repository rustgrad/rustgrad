use std::{cell::RefCell, rc::Rc};

use ndarray::Array;

use crate::{
    dimensions::{DynamicShape, Shape},
    tensor::{ShapeCompatible, Tensor},
};

use crate::ops::Operation;

#[derive(Debug, Clone)]
struct TensorMax<S1: Shape, S2: Shape> {
    lhs: Tensor<S1>,
    rhs: Tensor<S2>,
    take_from_a: Vec<bool>,
}

impl<S1: Shape, S2: Shape> TensorMax<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn forward(lhs: Tensor<S1>, rhs: Tensor<S2>) -> Tensor<<S1 as ShapeCompatible<S2>>::Output> {
        assert!(lhs.shape() == rhs.shape());
        let (take_from_a, output): (Vec<bool>, Vec<f32>) = lhs
            .data()
            .iter()
            .zip(rhs.data().iter())
            .map(|(a, b)| (a > b, if a > b { *a } else { *b }))
            .unzip();
        let output = Array::from_vec(output)
            .into_shape_with_order(lhs.shape())
            .unwrap();
        let node = TensorMax {
            lhs,
            rhs,
            take_from_a,
        };
        Tensor::new_with_prev(output, Rc::new(RefCell::new(node)))
    }
}

impl<S1: Shape, S2: Shape> Operation<<S1 as ShapeCompatible<S2>>::Output> for TensorMax<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn backward(&self, output: &Tensor<<S1 as ShapeCompatible<S2>>::Output>) {
        let grad = output.grad().expect("should have gradient");
        let (grad_a, grad_b): (Vec<f32>, Vec<f32>) = self
            .take_from_a
            .iter()
            .zip(grad.iter())
            .map(|(&take_first, &grad)| match take_first {
                true => (grad, 0.0),
                false => (0.0, grad),
            })
            .unzip();

        let grad_a = Array::from_vec(grad_a)
            .into_shape_with_order(output.shape())
            .unwrap();
        let grad_b = Array::from_vec(grad_b)
            .into_shape_with_order(output.shape())
            .unwrap();
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
        Rc::new(RefCell::new(TensorMax::<DynamicShape, DynamicShape> {
            lhs: self.lhs.clone_into_dynamic(),
            rhs: self.rhs.clone_into_dynamic(),
            take_from_a: self.take_from_a.clone(),
        }))
    }
}

pub fn max<S1: Shape, S2: Shape>(
    a: Tensor<S1>,
    b: Tensor<S2>,
) -> Tensor<<S1 as ShapeCompatible<S2>>::Output>
where
    S1: ShapeCompatible<S2>,
{
    TensorMax::forward(a, b)
}
