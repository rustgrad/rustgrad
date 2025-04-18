use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::dimensions::{DynamicShape, Shape};
use crate::tensor::{Operation, Tensor};
use ndarray::{Array0, ArrayBase, Axis, Dimension as _, Ix0, IxDyn};

#[derive(Debug, Clone)]
pub struct Mean<SIn: Shape, SOut: Shape> {
    input: Tensor<SIn>,
    axis: usize,
    phantom: PhantomData<SOut>,
}

impl<SIn: Shape, SOut: Shape> Mean<SIn, SOut> {
    pub fn forward(input: Tensor<SIn>, axis: usize) -> Tensor<SOut> {
        let array = input.container.borrow().array.clone();
        let mean = array.mean_axis(Axis(axis)).expect("Mean failed").into_dyn();

        let op = Mean {
            input: input.clone(),
            axis,
            phantom: PhantomData,
        };

        Tensor::new_with_prev(mean, Rc::new(RefCell::new(op)))
    }
}

impl<SIn: Shape, SOut: Shape> Operation<SOut> for Mean<SIn, SOut> {
    fn backward(&mut self, output: &mut Tensor<SOut>) {
        let grad = output
            .container
            .borrow()
            .grad
            .clone()
            .unwrap_or_else(|| ndarray::Array::zeros(output.shape()));

        let input_shape = self.input.shape();
        let output_shape = output.shape();
        let axis = self.axis;

        // Calculate scaling factor (1 / axis size)
        let reduction_size = input_shape.dims[axis] as f32;

        // Expand grad to input shape
        let mut grad_expanded = grad.clone();
        grad_expanded = grad_expanded.insert_axis(Axis(axis));
        let broadcasted = grad_expanded
            .broadcast(input_shape.clone())
            .unwrap()
            .to_owned();

        // Scale the gradient evenly
        let scaled = broadcasted.mapv(|x| x / reduction_size);

        self.input.backward_internal(scaled.into_dyn());
    }

    fn zero_graph(&self) {
        self.input.zero_graph();
    }

    fn build_graph(&self) {
        self.input.build_graph();
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        Rc::new(RefCell::new(Mean::<DynamicShape, DynamicShape> {
            input: self.input.clone_into_dynamic(),
            axis: self.axis,
            phantom: PhantomData,
        }))
    }
}

impl<S: Shape> Tensor<S> {
    pub fn mean_along<SOut: Shape>(&self, axis: usize) -> Tensor<SOut> {
        Mean::<S, SOut>::forward(self.clone(), axis)
    }

    pub fn mean(&self) -> Tensor<crate::dimensions::Rank0> {
        let sum = self.clone().sum(); // You may already have sum

        let numel = self.shape().num_elements() as f32;
        let scalar_array: Array0<f32> = Array0::from_elem(Ix0(), numel);
        let numel: Tensor<()> = Tensor::new(scalar_array.into_dyn());

        sum / numel
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_mean_along_axis_0() {
        use crate::dimensions::{Rank1, Rank2, S};
        use crate::tensor::Tensor;
        use ndarray::array;

        let a = Tensor::<Rank2<S<2>, S<2>>>::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn()); // shape: (2, 2)
        let mean: Tensor<Rank1<S<2>>> = a.mean_along(0); // mean along axis 0 → shape (2,)

        let expected = array![2.0, 3.0].into_dyn();
        assert_eq!(mean.data(), expected);

        // Test backward: mean -> backward() should propagate evenly
        let mut loss = mean.clone().sum(); // just sum for simplicity
        loss.backward();

        let grad = a.grad().unwrap();
        let expected_grad = array![[0.5, 0.5], [0.5, 0.5]].into_dyn();
        assert_eq!(grad, expected_grad);
    }
}
