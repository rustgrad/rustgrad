use ndarray::{Array, IxDyn};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::{
    rand::{self, prelude::*},
    RandomExt,
};
use std::env::consts::ARCH;
use std::{
    cell::RefCell,
    fmt::{Debug, Pointer, Write},
    ops::{Add, Deref, Mul, Neg},
    rc::Rc,
};

use crate::{matmul::TensorMatMul, shape::Shape};

#[derive(Default, Debug, Clone)]
pub struct DataContainer {
    pub array: Array<f32, IxDyn>,
    pub grad: Option<Array<f32, IxDyn>>,

    pub num_consumers: usize,
}
impl DataContainer {
    fn add_value(&mut self, value: Array<f32, IxDyn>) {
        self.array = self.array.clone() + value;
    }
}

#[derive(Default)]
pub struct Tensor {
    pub container: Rc<RefCell<DataContainer>>,
    prev_op: Option<Operation>,
}
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let repr = format!(
            "\n array: {}, grad {:?} \n num_consumer {:?} ",
            self.container.borrow().array,
            self.container.borrow().grad,
            self.container.borrow().num_consumers
        );
        f.write_str(&repr)
    }
}

impl Tensor {
    pub fn new(data: Array<f32, IxDyn>) -> Tensor {
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
    pub fn ZERO(shape: Shape) -> Tensor {
        return Tensor::new(Array::<f32, IxDyn>::zeros(shape));
    }
    pub fn new_random(shape: Shape) -> Tensor {
        let val: Array<f32, IxDyn> = Array::<f32, IxDyn>::random(shape, StandardNormal);
        return Tensor::new(val);
    }
    pub fn new_with_prev(data: Array<f32, IxDyn>, prev_op: Operation) -> Tensor {
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
        match self.prev_op.clone() {
            Some(Operation::Add(mut node)) => node.backward(self),
            Some(Operation::Mul(mut node)) => node.backward(self),
            Some(Operation::MatMul(mut node)) => node.backward(self),
            Some(Operation::Reshape(mut node)) => node.backward(self),
            Some(Operation::Neg(mut node)) => node.backward(self),
            Some(Operation::Max(mut node)) => node.backward(self),
            None => {
                // println!("No previous operation");
            }
        };
    }
    pub fn zero_graph(&mut self) {
        self.container.deref().borrow_mut().num_consumers = 0;
        match self.prev_op.clone() {
            Some(Operation::Add(mut node)) => {
                node.first.zero_graph();
                node.second.zero_graph();
            }
            Some(Operation::Mul(mut node)) => {
                node.first.zero_graph();
                node.second.zero_graph();
            }
            Some(Operation::MatMul(mut node)) => {
                node.lhs.zero_graph();
                node.rhs.zero_graph();
            }
            Some(Operation::Max(mut node)) => {
                node.a.zero_graph();
                node.b.zero_graph();
            }
            Some(Operation::Reshape(mut node)) => node.tensor.zero_grad(),
            Some(Operation::Neg(mut node)) => node.tensor.zero_grad(),
            None => {
                // println!("No previous operation");
            }
        };
    }

    pub fn build_graph(&mut self) {
        self.container.deref().borrow_mut().num_consumers += 1;
        if self.container.borrow().num_consumers > 1 {
            return;
        } else {
            match self.prev_op.clone() {
                Some(Operation::Add(mut node)) => {
                    node.first.build_graph();
                    node.second.build_graph();
                }
                Some(Operation::Mul(mut node)) => {
                    node.first.build_graph();
                    node.second.build_graph();
                }
                Some(Operation::Max(mut node)) => {
                    node.a.build_graph();
                    node.b.build_graph();
                }
                Some(Operation::MatMul(mut node)) => {
                    node.lhs.build_graph();
                    node.rhs.build_graph();
                }
                Some(Operation::Reshape(mut node)) => node.tensor.build_graph(),
                Some(Operation::Neg(mut node)) => node.tensor.build_graph(),
                None => {
                    // println!("No previous operation");
                }
            };
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

    pub fn dot(self, rhs: Tensor) -> Tensor {
        TensorMatMul::forward(self, rhs)
    }
    pub fn reshape(self, shape: Shape) -> Tensor {
        TensorReshape::forward(self, shape)
    }
}
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            container: self.container.clone(),
            prev_op: self.prev_op.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Operation {
    Add(TensorAdd),
    Mul(TensorMul),
    MatMul(TensorMatMul),
    Reshape(TensorReshape),
    Neg(TensorNeg),
    Max(TensorMax),
}
#[derive(Debug, Clone)]
struct TensorReshape {
    tensor: Box<Tensor>,
    input_shape: Shape,
}
impl TensorReshape {
    pub fn forward(input: Tensor, shape: Shape) -> Tensor {
        let input_shape = input.shape();

        let new_data = input.data().into_shape_with_order(shape.dims).unwrap();
        let node = TensorReshape {
            input_shape: input.shape(),
            tensor: Box::new(input),
        };
        Tensor::new_with_prev(new_data, Operation::Reshape(node))
    }
    pub fn backward(&mut self, output: &mut Tensor) {
        let new_grad = output
            .grad()
            .expect("Missing gradient")
            .into_shape_with_order(self.input_shape.dims.clone())
            .unwrap();
        self.tensor.backward_internal(new_grad);
    }
}

#[derive(Debug, Clone)]
struct TensorMul {
    first: Box<Tensor>,
    second: Box<Tensor>,
}

impl TensorMul {
    pub fn forward(input_a: Tensor, input_b: Tensor) -> Tensor {
        let result =
            input_a.container.borrow().array.clone() * input_b.container.borrow().array.clone();
        let node = TensorMul {
            first: Box::new(input_a),
            second: Box::new(input_b),
        };
        Tensor::new_with_prev(result, Operation::Mul(node))
    }
    pub fn backward(&mut self, output: &mut Tensor) {
        let grad = output.container.borrow().grad.clone();
        let grad = grad.unwrap_or(Array::ones(output.shape()));
        let grad_a = grad.clone() * self.second.container.borrow().array.clone();
        let grad_b = grad.clone() * self.first.container.borrow().array.clone();
        self.first.backward_internal(grad_a);
        self.second.backward_internal(grad_b);
    }
}

#[derive(Debug, Clone)]
struct TensorAdd {
    first: Box<Tensor>,
    second: Box<Tensor>,
}
impl TensorAdd {
    fn forward(input_a: Tensor, input_b: Tensor) -> Tensor {
        let result =
            input_a.container.borrow().array.clone() + input_b.container.borrow().array.clone();
        let node = TensorAdd {
            first: Box::new(input_a),
            second: Box::new(input_b),
        };
        Tensor::new_with_prev(result, Operation::Add(node))
    }
    fn backward(&mut self, output: &mut Tensor) {
        let maybe_grad = output.container.borrow().grad.clone();
        let grad = maybe_grad.unwrap_or(ndarray::Array::zeros(output.shape()));
        let grad_a = grad.clone();
        let grad_b = grad.clone();
        self.first.backward_internal(grad_a);
        self.second.backward_internal(grad_b);
    }
}

#[derive(Debug, Clone)]
struct TensorNeg {
    tensor: Box<Tensor>,
}
impl TensorNeg {
    fn forward(tensor: Tensor) -> Tensor {
        let result = -tensor.container.borrow().array.clone();
        Tensor::new_with_prev(
            result,
            Operation::Neg(TensorNeg {
                tensor: Box::new(tensor),
            }),
        )
    }
    fn backward(&mut self, output: &mut Tensor) {
        let grad = output
            .grad()
            .unwrap_or(ndarray::Array::zeros(output.shape()));
        self.tensor.backward_internal(-grad);
    }
}
#[derive(Debug, Clone)]
struct TensorMax {
    a: Box<Tensor>,
    b: Box<Tensor>,
    take_from_a: Vec<bool>,
}
impl TensorMax {
    fn forward(a: Tensor, b: Tensor) -> Tensor {
        assert!(a.shape() == b.shape());
        let first_array = a.data();
        let second_array = b.data();
        let (take_from_a, output): (Vec<bool>, Vec<f32>) = a
            .data()
            .iter()
            .zip(b.data().iter())
            .map(|(a, b)| (a > b, if a > b { *a } else { *b }))
            .unzip();
        let output = Array::from_vec(output)
            .into_shape_with_order(a.shape())
            .unwrap();
        Tensor::new_with_prev(
            output,
            Operation::Max(TensorMax {
                a: Box::new(a),
                b: Box::new(b),
                take_from_a,
            }),
        )
    }
    fn backward(&mut self, output: &mut Tensor) {
        let grad = output.grad().expect("should have gradient");
        let (grad_a, grad_b): (Vec<f32>, Vec<f32>) = self
            .take_from_a
            .iter()
            .zip(grad.iter())
            .map(|(&take_first, &grad)| match take_first {
                true => (grad, 0 as f32),
                false => (0.0, grad),
            })
            .unzip();

        let grad_a = Array::from_vec(grad_a)
            .into_shape_with_order(output.shape())
            .unwrap();
        let grad_b = Array::from_vec(grad_b)
            .into_shape_with_order(output.shape())
            .unwrap();
        self.a.backward_internal(grad_a);
        self.b.backward_internal(grad_b);
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Tensor {
        TensorAdd::forward(self, other)
    }
}
impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Tensor {
        TensorMul::forward(self, other)
    }
}
impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        TensorNeg::forward(self)
    }
}
pub fn max(a: Tensor, b: Tensor) -> Tensor {
    TensorMax::forward(a, b)
}
#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn it_works() {
        let test_0 = Tensor::new(array![[1.0, 2.0, 3.0, 4.0]].into_dyn()); // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_1 = test_0.clone() + test_0.clone(); // grad_2 = 2 * test_1
        let mut test_2: Tensor = test_1.clone() * test_1.clone();
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
