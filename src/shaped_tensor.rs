use ndarray::{Array, IxDyn};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::{rand::prelude::*, RandomExt};
use num_traits::Zero;
use std::env::consts;
use std::ops::{Div, Index};
use std::process::Output;
use std::{
    cell::RefCell,
    fmt::{Debug, Write},
    ops::{Add, Deref, Mul, Neg},
    rc::Rc,
};

use crate::{matmul::TensorMatMul, shape::Shape};

pub type TensorShape<I, J> = (I, J);

#[derive(Default, Debug, Clone)]
pub struct DataContainer {
    array: Array<f32, IxDyn>,
    pub grad: Option<Array<f32, IxDyn>>,

    pub num_consumers: usize,
}
impl DataContainer {
    fn add_value(&mut self, value: Array<f32, IxDyn>) {
        self.array = self.array.clone() + value;
    }
}

pub trait Operation<const I: usize, const J: usize>: std::fmt::Debug {
    fn backward(&mut self, output: &mut Tensor<I, J>);
    fn zero_graph(&self);
    fn build_graph(&self);
}
pub fn test_fn() {
    let tensor = Tensor::<2, 4>::ZERO(&[]);
    let tensor2 = Tensor::<4, 2>::ZERO(&[]);
    let tensor3 = Tensor::<3, 2>::ZERO(&[]);
    const TEST: usize = 3;
    let tensor4 = Tensor::<2, TEST>::ZERO(&[]);
    let result_a = tensor.dot(tensor2);
    // let result_b = tensor.dot(tensor3); This breaks, because the shapes don't fits
}

pub struct ShapeTest {}

#[derive(Default, Debug, Clone)]
pub struct Tensor<const I: usize, const J: usize> {
    pub container: Rc<RefCell<DataContainer>>,
    prev_op: Option<Rc<RefCell<dyn Operation<I, J>>>>,
}

impl<const I: usize, const J: usize> Tensor<I, J> {
    pub fn new(data: Array<f32, IxDyn>) -> Tensor<I, J> {
        let data = DataContainer {
            array: data,
            grad: None,
            num_consumers: 0,
        };
        Tensor {
            container: Rc::new(RefCell::new(data)),
            prev_op: None,
        }
    }
    pub fn ZERO<const D: usize>(batch_sizes: &[usize; D]) -> Tensor<I, J> {
        let mut dims = vec![I, J];
        dims.extend_from_slice(batch_sizes);
        let shape = Shape { dims };
        return Tensor::new(Array::<f32, IxDyn>::zeros(shape));
    }
    pub fn new_random<const D: usize>(
        batch_sizes: &[usize; D],
        mean: f32,
        std: f32,
    ) -> Tensor<I, J> {
        let mut dims = vec![I, J];
        dims.extend_from_slice(batch_sizes);
        let shape = Shape { dims };
        let mut value: Array<f32, IxDyn> = Array::<f32, IxDyn>::random(shape, StandardNormal);
        value = value * std + mean;
        return Tensor::new(value);
    }
    pub fn new_with_prev(
        data: Array<f32, IxDyn>,
        prev_op: Rc<RefCell<dyn Operation<I, J>>>,
    ) -> Tensor<I, J> {
        let data = DataContainer {
            array: data,
            grad: None,
            num_consumers: 0,
        };
        Tensor {
            container: Rc::new(RefCell::new(data)),
            prev_op: Some(prev_op),
        }
    }
    pub fn backward(&mut self) {
        // This is a leaf node, we need to build the graph
        self.build_graph();

        let start_grad = Array::ones(self.shape());
        self.backward_internal(start_grad);
    }

    pub fn backward_internal(&mut self, grad: Array<f32, IxDyn>) {
        assert!(self.container.borrow().array.dim() == grad.dim());
        let new_grad = self.grad().unwrap_or(Array::zeros(self.shape())).clone() + grad.clone();
        self.container.deref().borrow_mut().grad = Some(new_grad);

        self.container.deref().borrow_mut().num_consumers -= 1;
        if self.container.borrow().num_consumers > 0 {
            return;
        }
        if let Some(op) = self.prev_op.clone() {
            op.borrow_mut().backward(self);
        }
    }
    pub fn zero_graph(&self) {
        self.container.deref().borrow_mut().num_consumers = 0;
        if let Some(op) = self.prev_op.clone() {
            op.borrow().zero_graph();
        }
    }

    pub fn build_graph(&self) {
        self.container.deref().borrow_mut().num_consumers += 1;
        if self.container.borrow().num_consumers > 1 {
            return;
        } else if let Some(op) = self.prev_op.clone() {
            op.borrow().build_graph();
        }
    }

    pub fn grad(&self) -> Option<Array<f32, IxDyn>> {
        self.container.borrow().grad.clone()
    }
    pub fn zero_grad(&self) {
        self.container.borrow_mut().grad = None;
    }
    pub fn data(&self) -> Array<f32, IxDyn> {
        self.container.borrow().array.clone()
    }
    pub fn update_data(&self, data: Array<f32, IxDyn>) {
        self.container.borrow_mut().array = data;
    }
    pub fn add_value(&self, value: Array<f32, IxDyn>) {
        self.container.borrow_mut().add_value(value);
    }

    pub fn shape(&self) -> Shape {
        let binding = self.container.borrow();
        let shape = binding.array.shape();
        shape.to_vec().into()
    }

    pub fn dot<const K: usize>(self, rhs: Tensor<J, K>) -> Tensor<I, K> {
        unimplemented!()
    }
    // pub fn reshape(self, shape: Shape) -> Tensor {
    //     TensorReshape::forward(self, shape)
    // }
}
// impl Clone for Tensor<const I: usize, const J: usize> {
//     fn clone(&self) -> Self {
//         Tensor {
//             container: self.container.clone(),
//             prev_op: self.prev_op.clone(),
//         }
//     }
// }

#[derive(Debug, Clone)]
struct TensorAdd<const I: usize, const J: usize> {
    lhs: Tensor<I, J>,
    rhs: Tensor<I, J>,
}

impl<const I: usize, const J: usize> TensorAdd<I, J> {
    fn forward(lhs: Tensor<I, J>, rhs: Tensor<I, J>) -> Tensor<I, J> {
        let result = lhs.container.borrow().array.clone() + rhs.container.borrow().array.clone();
        let node = TensorAdd { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<const I: usize, const J: usize> Operation<I, J> for TensorAdd<I, J> {
    fn backward(&mut self, output: &mut Tensor<I, J>) {
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
}

#[derive(Debug, Clone)]
struct TensorMul<const I: usize, const J: usize> {
    lhs: Tensor<I, J>,
    rhs: Tensor<I, J>,
}

impl<const I: usize, const J: usize> TensorMul<I, J> {
    pub fn forward(lhs: Tensor<I, J>, rhs: Tensor<I, J>) -> Tensor<I, J> {
        let result = lhs.container.borrow().array.clone() * rhs.container.borrow().array.clone();
        let node = TensorMul { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<const I: usize, const J: usize> Operation<I, J> for TensorMul<I, J> {
    fn backward(&mut self, output: &mut Tensor<I, J>) {
        let grad = output.container.borrow().grad.clone();
        let grad = grad.unwrap_or(Array::ones(output.shape()));
        let grad_a = grad.clone() * self.rhs.container.borrow().array.clone();
        let grad_b = grad * self.lhs.container.borrow().array.clone();
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
}

// #[derive(Debug, Clone)]
// struct TensorDiv {
//     lhs: Tensor,
//     rhs: Tensor,
// }

// impl TensorDiv {
//     fn forward(lhs: Tensor, rhs: Tensor) -> Tensor {
//         let result = lhs.data() / rhs.data();
//         let node = TensorDiv { lhs, rhs };
//         Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
//     }
// }

// impl Operation for TensorDiv {
//     fn backward(&mut self, output: &mut Tensor) {
//         let grad = output.grad().unwrap_or_else(|| Array::ones(output.shape()));
//         let rhs_data = self.rhs.data();
//         let lhs_data = self.lhs.data();

//         let lhs_grad = grad.clone() / rhs_data.clone();
//         let rhs_grad = -grad * lhs_data / (rhs_data.clone() * rhs_data);
//         self.lhs.backward_internal(lhs_grad);
//         self.rhs.backward_internal(rhs_grad);
//     }

//     fn zero_graph(&self) {
//         self.lhs.zero_graph();
//         self.rhs.zero_graph();
//     }

//     fn build_graph(&self) {
//         self.lhs.build_graph();
//         self.rhs.build_graph();
//     }
// }

// #[derive(Debug, Clone)]
// struct TensorNeg {
//     tensor: Tensor,
// }

// impl TensorNeg {
//     fn forward(tensor: Tensor) -> Tensor {
//         let result = -tensor.container.borrow().array.clone();
//         let node = TensorNeg { tensor };
//         Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
//     }
// }

// impl Operation for TensorNeg {
//     fn backward(&mut self, output: &mut Tensor) {
//         let grad = output
//             .grad()
//             .unwrap_or(ndarray::Array::zeros(output.shape()));
//         self.tensor.backward_internal(-grad);
//     }

//     fn zero_graph(&self) {
//         self.tensor.zero_graph();
//     }

//     fn build_graph(&self) {
//         self.tensor.build_graph();
//     }
// }

// #[derive(Debug, Clone)]
// struct TensorMax {
//     lhs: Tensor,
//     rhs: Tensor,
//     take_from_a: Vec<bool>,
// }

// impl TensorMax {
//     fn forward(lhs: Tensor, rhs: Tensor) -> Tensor {
//         assert!(lhs.shape() == rhs.shape());
//         let (take_from_a, output): (Vec<bool>, Vec<f32>) = lhs
//             .data()
//             .iter()
//             .zip(rhs.data().iter())
//             .map(|(a, b)| (a > b, if a > b { *a } else { *b }))
//             .unzip();
//         let output = Array::from_vec(output)
//             .into_shape_with_order(lhs.shape())
//             .unwrap();
//         let node = TensorMax {
//             lhs,
//             rhs,
//             take_from_a,
//         };
//         Tensor::new_with_prev(output, Rc::new(RefCell::new(node)))
//     }
// }

// impl Operation for TensorMax {
//     fn backward(&mut self, output: &mut Tensor) {
//         let grad = output.grad().expect("should have gradient");
//         let (grad_a, grad_b): (Vec<f32>, Vec<f32>) = self
//             .take_from_a
//             .iter()
//             .zip(grad.iter())
//             .map(|(&take_first, &grad)| match take_first {
//                 true => (grad, 0 as f32),
//                 false => (0.0, grad),
//             })
//             .unzip();

//         let grad_a = Array::from_vec(grad_a)
//             .into_shape_with_order(output.shape())
//             .unwrap();
//         let grad_b = Array::from_vec(grad_b)
//             .into_shape_with_order(output.shape())
//             .unwrap();
//         self.lhs.backward_internal(grad_a);
//         self.rhs.backward_internal(grad_b);
//     }

//     fn zero_graph(&self) {
//         self.lhs.zero_graph();
//         self.rhs.zero_graph();
//     }

//     fn build_graph(&self) {
//         self.lhs.build_graph();
//         self.rhs.build_graph();
//     }
// }

// #[derive(Debug, Clone)]
// struct TensorReshape {
//     tensor: Tensor,
//     input_shape: Shape,
// }

// impl TensorReshape {
//     pub fn forward(input: Tensor, shape: Shape) -> Tensor {
//         let new_data = input.data().into_shape_with_order(shape.dims).unwrap();
//         let node = TensorReshape {
//             input_shape: input.shape(),
//             tensor: input,
//         };
//         Tensor::new_with_prev(new_data, Rc::new(RefCell::new(node)))
//     }
// }

// impl Operation for TensorReshape {
//     fn backward(&mut self, output: &mut Tensor) {
//         let new_grad = output
//             .grad()
//             .expect("Missing gradient")
//             .into_shape_with_order(self.input_shape.dims.clone())
//             .unwrap();
//         self.tensor.backward_internal(new_grad);
//     }

//     fn zero_graph(&self) {
//         self.tensor.zero_graph();
//     }

//     fn build_graph(&self) {
//         self.tensor.build_graph();
//     }
// }

impl<const I: usize, const J: usize> Add<Tensor<I, J>> for Tensor<I, J> {
    type Output = Tensor<I, J>;
    fn add(self, other: Tensor<I, J>) -> Tensor<I, J> {
        TensorAdd::forward(self, other)
    }
}
impl<const I: usize, const J: usize> Mul<Tensor<I, J>> for Tensor<I, J> {
    type Output = Tensor<I, J>;
    fn mul(self, other: Tensor<I, J>) -> Tensor<I, J> {
        TensorMul::forward(self, other)
    }
}
// impl Mul for Tensor {
//     type Output = Tensor;
//     fn mul(self, other: Tensor) -> Tensor {
//         TensorMul::forward(self, other)
//     }
// }
// impl Neg for Tensor {
//     type Output = Tensor;
//     fn neg(self) -> Self::Output {
//         TensorNeg::forward(self)
//     }
// }
// impl Div for Tensor {
//     type Output = Tensor;
//     fn div(self, rhs: Tensor) -> Self::Output {
//         TensorDiv::forward(self, rhs)
//     }
// }
// pub fn max(a: Tensor, b: Tensor) -> Tensor {
//     TensorMax::forward(a, b)
// }

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn it_works() {
        let test_0 = Tensor::<1, 4>::new(array![[1.0, 2.0, 3.0, 4.0]].into_dyn()); // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_1 = test_0.clone() + test_0.clone(); // grad_2 = 2 * test_1
        let mut test_2: Tensor<1, 4> = test_1.clone() * test_1.clone();
        println!("forward: {:?}", test_2);
        println!("_____________________________");
        test_2.backward();

        assert_eq!(
            test_0.grad(),
            Some(array![[8.0, 16.0, 24.0, 32.0]].into_dyn())
        );
        assert_eq!(
            test_1.grad(),
            Some(array![[4.0, 8.0, 12.0, 16.0]].into_dyn())
        );
        assert_eq!(test_2.grad(), Some(array![[1.0, 1.0, 1.0, 1.0]].into_dyn()));

        println!("{:?}", test_2);
    }
}
