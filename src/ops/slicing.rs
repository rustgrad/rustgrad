use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::dimensions::{DynamicShape, Shape};
use crate::tensor::{Operation, Tensor};
use ndarray::Axis;

#[derive(Debug, Clone)]
pub struct Slice<SIn: Shape, SOut: Shape> {
    pub input: Tensor<SIn>,
    pub index: usize,
    pub axis: usize,
    pub phantom: PhantomData<SOut>,
}

impl<SIn: Shape, SOut: Shape> Operation<SOut> for Slice<SIn, SOut> {
    fn backward(&mut self, output: &mut Tensor<SOut>) {
        let grad = output.container.borrow().grad.clone().unwrap();
        let input_shape = self.input.shape();
        let mut zeros = ndarray::ArrayD::zeros(input_shape.dims.clone());

        let mut slice = zeros.index_axis_mut(Axis(self.axis), self.index);
        let grad_shape = grad.shape().to_vec();
        let grad = grad.into_shape_with_order(slice.shape()).expect(&format!(
            "Failed to reshape {:?} into {:?}",
            grad_shape,
            slice.shape()
        ));
        slice.assign(&grad);

        self.input.backward_internal(zeros);
    }

    fn zero_graph(&self) {
        self.input.zero_graph();
    }

    fn build_graph(&self) {
        self.input.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        Rc::new(RefCell::new(Slice::<DynamicShape, DynamicShape> {
            input: self.input.clone_into_dynamic(),
            index: self.index,
            axis: self.axis,
            phantom: PhantomData,
        }))
    }
}
impl<S: Shape> Tensor<S> {
    pub fn slice<Out: Shape>(&self, axis: usize, index: usize) -> Tensor<Out> {
        let view = self
            .container
            .borrow()
            .array
            .index_axis(Axis(axis), index)
            .to_owned();
        let op = Slice::<S, Out> {
            input: self.clone(),
            index,
            axis,
            phantom: PhantomData,
        };
        let shape = view.shape();
        let view = view
            .to_owned()
            .into_shape_clone(Out::shape())
            .expect(&format!(
                "Failed to reshape {:?} into {:?}",
                shape,
                Out::shape()
            ));
        Tensor::new_with_prev(view.into_dyn(), Rc::new(RefCell::new(op)))
    }
}
// use std::ops::Index;

// impl<SIn: Shape, SOut: Shape> Index<usize> for Tensor<SIn> {
//     type Output = Tensor<SOut>;

//     fn index(&self, index: usize) -> &Tensor<SOut> {
//         return &self.slice(0, index);
//     }
// }

#[cfg(test)]
mod tests {
    #[test]
    fn test_slice_and_backward() {
        use crate::dimensions::{Rank1, Rank2, S};
        use crate::tensor::Tensor;
        use ndarray::array;

        // Original tensor a: shape (2, 2)
        let a = Tensor::<Rank2<S<2>, S<2>>>::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());

        // Slice second row: shape (2,)
        let b: Tensor<Rank1<S<2>>> = a.slice(0, 1);
        println!("{:?}", b);

        // Perform dummy computation on b to trigger backward
        let mut c = b.clone() * Tensor::<(S<1>,)>::new(array![2.0].into_dyn()).broadcast_to();
        c.backward();

        // Gradient should be None for a[0], and [2.0, 2.0] for a[1]
        let grad = a.grad().unwrap();

        let expected = array![[0.0, 0.0], [2.0, 2.0]].into_dyn();
        assert_eq!(grad, expected);
    }
}
