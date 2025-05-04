use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use ndarray::{Array, Axis, IxDyn};

use crate::dimensions::{DynamicShape, Shape};
use crate::ops::Operation;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct TensorStack<SIn: Shape, SOut: Shape> {
    inputs: Vec<Tensor<SIn>>,
    axis: usize,
    phantom_data: PhantomData<SOut>,
}

impl<SIn: Shape, SOut: Shape> TensorStack<SIn, SOut> {
    pub fn forward(inputs: Vec<Tensor<SIn>>, axis: usize) -> Tensor<SOut> {
        if inputs.is_empty() {
            panic!("Cannot stack empty list of tensors");
        }

        // Create a new array with the correct shape
        let mut stacked: Array<f32, IxDyn> = Array::zeros(SOut::shape()).into_dyn();

        // Enumerate inputs
        for (i, input) in inputs.iter().enumerate() {
            stacked
                .index_axis_mut(Axis(axis), i)
                .assign(&input.container.borrow().array);
        }

        // Create the operation
        let op = TensorStack {
            inputs: inputs.clone(),
            axis,
            phantom_data: PhantomData,
        };

        Tensor::new_with_prev(stacked, Rc::new(RefCell::new(op)))
    }
}

impl<SIn: Shape, SOut: Shape> Operation<SOut> for TensorStack<SIn, SOut> {
    fn backward(&self, output: &Tensor<SOut>) {
        let grad = output
            .container
            .borrow()
            .grad
            .clone()
            .unwrap_or_else(|| ndarray::Array::zeros(output.shape()));

        // Split the gradient along the stack axis
        let num_inputs = self.inputs.len();
        let axis = self.axis;

        // Ensure the axis dimension matches the number of inputs
        if grad.shape()[axis] != num_inputs {
            panic!("Gradient shape mismatch during Stack backward");
        }

        // Split the gradient and send to each input
        for (i, input) in self.inputs.iter().enumerate() {
            // Extract the slice for this input
            let mut indices = vec![];
            for ax in 0..grad.ndim() {
                if ax == axis {
                    indices.push(ndarray::SliceInfoElem::Index(i as isize));
                } else {
                    indices.push(ndarray::SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    });
                }
            }

            let input_grad = unsafe {
                grad.slice(
                    ndarray::SliceInfo::<_, ndarray::IxDyn, ndarray::IxDyn>::new(indices)
                        .unwrap()
                        .as_ref(),
                )
            };

            assert_eq!(
                input_grad.shape(),
                input.shape().dims,
                "Gradient shape {:?} mismatch for input {} with shape {:?} during Stack backward",
                input_grad.shape(),
                i,
                input.shape()
            );
            // Pass the gradient to the input
            input.backward_internal(input_grad.to_owned());
        }
    }

    fn zero_graph(&self) {
        for input in &self.inputs {
            input.zero_graph();
        }
    }

    fn build_graph(&self) {
        for input in &self.inputs {
            input.build_graph();
        }
    }

    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        let dynamic_inputs = self.inputs.iter().map(|t| t.clone_into_dynamic()).collect();

        Rc::new(RefCell::new(TensorStack::<DynamicShape, DynamicShape> {
            inputs: dynamic_inputs,
            axis: self.axis,
            phantom_data: PhantomData,
        }))
    }
}

// Add a convenient method to stack tensors
pub fn stack<SIn: Shape, SOut: Shape>(tensors: Vec<Tensor<SIn>>, axis: usize) -> Tensor<SOut> {
    TensorStack::forward(tensors, axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimensions::{Rank1, Rank2, S};
    use ndarray::array;

    #[test]
    fn test_stack_vectors() {
        let a = Tensor::<Rank1<S<3>>>::new(array![1.0, 2.0, 3.0].into_dyn());
        let b = Tensor::<Rank1<S<3>>>::new(array![4.0, 5.0, 6.0].into_dyn());

        // Stack along axis 0 (creating a 2x3 matrix)
        let c: Tensor<Rank2<S<2>, S<3>>> = stack(vec![a.clone(), b.clone()], 0);

        let expected = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        assert_eq!(c.data(), expected);

        // Test gradient
        c.backward();

        assert_eq!(a.grad().unwrap(), array![1.0, 1.0, 1.0].into_dyn());
        assert_eq!(b.grad().unwrap(), array![1.0, 1.0, 1.0].into_dyn());
    }

    #[test]
    fn test_stack_matrices() {
        let a = Tensor::<Rank2<S<2>, S<2>>>::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
        let b = Tensor::<Rank2<S<2>, S<2>>>::new(array![[5.0, 6.0], [7.0, 8.0]].into_dyn());

        // Stack along axis 0 (creating a 4x2 matrix)
        let c: Tensor<(S<2>, S<2>, S<2>)> = stack(vec![a.clone(), b.clone()], 0);

        let expected = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]].into_dyn();
        assert_eq!(c.data(), expected);
    }

    #[test]
    fn test_stack_backward_gradients() {
        // Test stacking along axis 0 (vertical stack)
        let a =
            Tensor::<Rank2<S<2>, S<3>>>::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
        let b = Tensor::<Rank2<S<2>, S<3>>>::new(
            array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]].into_dyn(),
        );

        // Stack to create a 4x3 matrix
        let c: Tensor<(S<2>, S<2>, S<3>)> = stack(vec![a.clone(), b.clone()], 0);

        // Create a non-uniform gradient
        let upstream_grad = array![
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        ]
        .into_dyn();

        c.backward_internal(upstream_grad);

        // Verify first tensor gradient - should match first two rows of upstream_grad
        let expected_a_grad = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]].into_dyn();
        assert_eq!(a.grad().unwrap(), expected_a_grad);

        // Verify second tensor gradient - should match last two rows of upstream_grad
        let expected_b_grad = array![[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]].into_dyn();
        assert_eq!(b.grad().unwrap(), expected_b_grad);

        // Reset gradients
        a.zero_grad();
        b.zero_grad();

        // Test stacking along axis 1 (horizontal stack)
        let d: Tensor<(S<2>, S<2>, S<3>)> = stack(vec![a.clone(), b.clone()], 1);

        // Create a new non-uniform gradient for horizontal stacking
        let upstream_grad_horizontal = array![
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        ]
        .into_dyn();

        d.backward_internal(upstream_grad_horizontal);

        // Verify first tensor gradient - should match first three columns
        let expected_a_grad_h = array![[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]].into_dyn();
        assert_eq!(a.grad().unwrap(), expected_a_grad_h);

        // Verify second tensor gradient - should match last three columns
        let expected_b_grad_h = array![[0.4, 0.5, 0.6], [1.0, 1.1, 1.2]].into_dyn();
        assert_eq!(b.grad().unwrap(), expected_b_grad_h);
    }
}
