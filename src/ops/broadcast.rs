use core::panic;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use itertools::{
    EitherOrBoth::{Both, Left, Right},
    Itertools,
};

use crate::dimensions::{Shape, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Broadcast<SIn: Shape, SOut: Shape> {
    input: Tensor<SIn>,
    phantom_data: PhantomData<SOut>,
}

impl<SIn: Shape, SOut: Shape> Broadcast<SIn, SOut> {
    pub fn forward(input: Tensor<SIn>) -> Tensor<SOut> {
        let array = input.container.borrow().array.clone();
        let target_shape = SOut::shape();

        let broadcasted = array
            .broadcast(target_shape.clone())
            .expect("Broadcast failed")
            .to_owned();

        let op = Broadcast {
            input: input.clone(),
            phantom_data: PhantomData,
        };

        Tensor::new_with_prev(broadcasted, Rc::new(RefCell::new(op)))
    }
}
impl<SIn: Shape, SOut: Shape> Operation<SOut> for Broadcast<SIn, SOut> {
    fn backward(&self, output: &Tensor<SOut>) {
        let grad = output
            .container
            .borrow()
            .grad
            .clone()
            .unwrap_or_else(|| ndarray::Array::zeros(output.shape()));

        let mut grad_owned = grad;
        let input_shape = SIn::shape();
        let output_shape = SOut::shape();
        let reversed_input_dims = input_shape.dims.iter().rev();
        let reversed_output_dims = output_shape.dims.iter().rev();
        let max_axis = input_shape.num_dims();
        for (inverted_axis, dims) in reversed_input_dims
            .zip_longest(reversed_output_dims)
            .enumerate()
        {
            let axis = max_axis - inverted_axis;
            match dims {
                Both(input_dim, output_dim) => {
                    if input_dim == output_dim {
                        continue;
                    }
                    if input_dim != &1 {
                        panic!("Input dimension is not 1");
                    }
                    let summed = grad_owned.sum_axis(ndarray::Axis(axis));
                    grad_owned = summed.insert_axis(ndarray::Axis(axis));
                }
                Left(_) => {
                    panic!("Output dimension is not in input shape");
                }
                Right(_) => {
                    let summed = grad_owned.sum_axis(ndarray::Axis(axis));
                    grad_owned = summed.insert_axis(ndarray::Axis(axis));
                }
            }
            let input_dim = *input_shape.dims.get(axis).unwrap_or(&1);
            if input_dim == 1 {
                let summed = grad_owned.sum_axis(ndarray::Axis(axis));
                grad_owned = summed.insert_axis(ndarray::Axis(axis));
            }
        }

        let reduced = grad_owned.into_shape_clone(input_shape.clone());
        let reduced = reduced.expect("Failed to reshape ");

        self.input.backward_internal(reduced);
    }

    fn zero_graph(&self) {
        self.input.zero_graph();
    }

    fn build_graph(&self) {
        self.input.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<UnkownShape>>> {
        Rc::new(RefCell::new(Broadcast::<UnkownShape, UnkownShape> {
            input: self.input.clone_into_dynamic(),
            phantom_data: PhantomData,
        }))
    }
}
impl<S: Shape> Tensor<S> {
    pub fn broadcast_to<SOut: Shape>(&self) -> Tensor<SOut> {
        Broadcast::forward(self.clone())
    }
}

#[cfg(test)]
mod tests {

    use crate::dimensions::{Rank1, Rank2, S};
    use crate::tensor::Tensor;

    use ndarray::array;

    #[test]
    fn test_broadcast_add_row_vector() {
        let a = Tensor::<Rank2<S<2>, S<2>>>::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn()); // shape: (2, 2)
        let b: Tensor<Rank1<S<2>>> = Tensor::new(array![10.0, 20.0].into_dyn()); // shape: (2,)
        let b2: Tensor<Rank2<S<2>, S<2>>> = b.broadcast_to(); // should broadcast b to (2, 1)

        let c: Tensor<Rank2<S<2>, S<2>>> = a.clone() + b2.clone(); // should broadcast b to (2, 2)

        let expected = array![[11.0, 22.0], [13.0, 24.0]].into_dyn();
        assert_eq!(c.data(), expected);
    }
}
