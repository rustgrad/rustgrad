use ndarray::{linalg::Dot, Array, ArrayBase, Dimension, OwnedRepr};
use num_traits::identities::Zero;
use std::{
    borrow::Borrow,
    cell::{Ref, RefCell},
    collections::HashMap,
    ops::{Add, Deref, Mul},
    process::Output,
    rc::Rc,
};
#[derive(Default, Debug, Clone)]
pub struct Tensor<A, D: Dimension> {
    pub data: Array<A, D>,
    pub grad: Option<Array<A, D>>,
    pub prev: Option<Operation<A, D>>,
}
impl<A, D: Dimension> Tensor<A, D>
where
    A: Add<Output = A> + Clone, // A must support element-wise addition and cloning
    A: Mul<Output = A> + Clone, // A must support element-wise multiplication and cloning
    A: num_traits::identities::Zero,
    A: num_traits::identities::One,
{
    pub fn backward(self) -> (Tensor<A, D>, Tensor<A, D>) {
        match self.prev.clone() {
            Some(Operation::Add(node)) => node.backward(self),
            Some(Operation::Mul(node)) => node.backward(self),
            Some(Operation::MatMul(node)) => node.backward(self),
            None => {
                panic!("No operation to backpropagate")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Operation<A, D: Dimension> {
    Add(TensorAdd<A, D>),
    Mul(TensorMul<A, D>),
    MatMul(TensorMatMul<A, D>),
}

pub struct RefTensor<A, D: Dimension> {
    pub ref_tensor: Rc<RefCell<Tensor<A, D>>>,
}

// impl<A, D: Dimension> Deref for RefTensor<A, D> {
//     type Target = Ref< Tensor<A, D>>;
//     fn deref(&self) -> Self::Target {
//         self.ref_tensor.borrow()
//     }
// }
// enum ComputationOperation<A, D> {
//     BinaryOp(Box<dyn BinaryOperation<A, D>>), // Node that performs a binary operation
//     UnaryOp(Box<dyn UnaryOperation<A, D>>),   // Node that performs a unary operation
//     TertiaryOp(Box<dyn TertiaryOperation<A, D>>),
// }

trait TertiaryOperation<A, D: Dimension> {
    fn forward(
        &self,
        input_a: Tensor<A, D>,
        input_b: Tensor<A, D>,
        input_c: Tensor<A, D>,
    ) -> Tensor<A, D>;
    fn backward(&self, input: Tensor<A, D>) -> (Tensor<A, D>, Tensor<A, D>, Tensor<A, D>);
}

trait UnaryOperation<A, D: Dimension> {
    fn forward(&self, input: Tensor<A, D>) -> Tensor<A, D>;
    fn backward(&self, input: Tensor<A, D>) -> Tensor<A, D>;
}

trait BinaryOperation<A, D: Dimension> {
    fn forward(input_a: Tensor<A, D>, input_b: Tensor<A, D>) -> Tensor<A, D>;
    fn backward(&self, input: Tensor<A, D>) -> (Tensor<A, D>, Tensor<A, D>);
}

#[derive(Debug, Clone)]
struct TensorMatMul<A, D: Dimension> {
    first: Rc<RefCell<Tensor<A, D>>>,
    second: Rc<RefCell<Tensor<A, D>>>,
}

impl<A, D> TensorMatMul<A, D>
where
    A: Mul<Output = A> + Clone + num_traits::One, // A must support element-wise multiplication and cloning
    D: Dimension,
    ArrayBase<OwnedRepr<A>, D>: Mul<Output = ArrayBase<OwnedRepr<A>, D>>, // Ensure element-wise multiplication is possible
{
    pub fn forward(input_a: Tensor<A, D>, input_b: Tensor<A, D>) -> Tensor<A, D> {
        let data = input_a.data.clone().dot(input_b.data.clone());
        let node = TensorMul {
            first: Rc::new(RefCell::new(input_a)),
            second: Rc::new(RefCell::new(input_b)),
        };
        let result = Tensor {
            data: data,
            grad: None,
            prev: Some(Operation::Mul(node)),
        };
        result
    }
    pub fn backward(self, output: Tensor<A, D>) -> (Tensor<A, D>, Tensor<A, D>) {
        let grad = output.grad.unwrap_or(Array::ones(output.data.raw_dim()));
        let grad_a = grad.clone() * (*self.second).borrow().data.clone();
        let grad_b = grad.clone() * (*self.first).borrow().data.clone();
        (
            Tensor {
                data: (*self.first).borrow().data.clone(),
                grad: Some(grad_a),
                prev: Some(Operation::Mul(self.clone())),
            },
            Tensor {
                data: (*self.second).borrow().data.clone(),
                grad: Some(grad_b),
                prev: Some(Operation::Mul(self.clone())),
            },
        )
    }
}

#[derive(Debug, Clone)]
struct TensorMul<A, D: Dimension> {
    first: Rc<RefCell<Tensor<A, D>>>,
    second: Rc<RefCell<Tensor<A, D>>>,
}

impl<A, D> TensorMul<A, D>
where
    A: Mul<Output = A> + Clone + num_traits::One, // A must support element-wise multiplication and cloning
    D: Dimension,
    ArrayBase<OwnedRepr<A>, D>: Mul<Output = ArrayBase<OwnedRepr<A>, D>>, // Ensure element-wise multiplication is possible
{
    pub fn forward(input_a: Tensor<A, D>, input_b: Tensor<A, D>) -> Tensor<A, D> {
        let data = input_a.data.clone() * input_b.data.clone();
        let node = TensorMul {
            first: Rc::new(RefCell::new(input_a)),
            second: Rc::new(RefCell::new(input_b)),
        };
        let result = Tensor {
            data: data,
            grad: None,
            prev: Some(Operation::Mul(node)),
        };
        result
    }
    pub fn backward(self, output: Tensor<A, D>) -> (Tensor<A, D>, Tensor<A, D>) {
        let grad = output.grad.unwrap_or(Array::ones(output.data.raw_dim()));
        let grad_a = grad.clone() * (*self.second).borrow().data.clone();
        let grad_b = grad.clone() * (*self.first).borrow().data.clone();
        (
            Tensor {
                data: (*self.first).borrow().data.clone(),
                grad: Some(grad_a),
                prev: Some(Operation::Mul(self.clone())),
            },
            Tensor {
                data: (*self.second).borrow().data.clone(),
                grad: Some(grad_b),
                prev: Some(Operation::Mul(self.clone())),
            },
        )
    }
}

#[derive(Debug, Clone)]
struct TensorAdd<A, D: Dimension> {
    first: Rc<RefCell<Tensor<A, D>>>,
    second: Rc<RefCell<Tensor<A, D>>>,
}
impl<A, D> TensorAdd<A, D>
where
    A: Add<Output = A> + Clone, // A must support element-wise addition and cloning
    D: Dimension,
    ArrayBase<OwnedRepr<A>, D>: Add<Output = ArrayBase<OwnedRepr<A>, D>>, // Ensure element-wise addition is possible
    A: num_traits::identities::Zero,
{
    fn forward(input_a: Tensor<A, D>, input_b: Tensor<A, D>) -> Tensor<A, D> {
        let data = input_a.data.clone() + input_b.data.clone();
        let node = TensorAdd {
            first: Rc::new(RefCell::new(input_a)),
            second: Rc::new(RefCell::new(input_b)),
        };
        let result = Tensor {
            data: data,
            grad: None,
            prev: Some(Operation::Add(node)),
        };
        result
    }
    fn backward(self, output: Tensor<A, D>) -> (Tensor<A, D>, Tensor<A, D>) {
        // unimplemented!();
        let grad = output
            .grad
            .unwrap_or(ndarray::Array::zeros(output.data.raw_dim()));
        let grad_a = grad.clone();
        let grad_b = grad.clone();
        (
            Tensor {
                data: (*self.first).borrow().data.clone(),
                grad: Some(grad_a),
                prev: Some(Operation::Add(self.clone())),
            },
            Tensor {
                data: (*self.second).borrow().data.clone(),
                grad: Some(grad_b),
                prev: Some(Operation::Add(self.clone())),
            },
        )
    }
}

impl<D: Dimension, A: Add<Output = A> + Clone> Add for Tensor<A, D>
where
    A: num_traits::identities::Zero,
{
    type Output = Tensor<A, D>;
    fn add(self, other: Tensor<A, D>) -> Tensor<A, D> {
        TensorAdd::forward(self, other)
    }
}
impl<D: Dimension, A: Mul<Output = A> + Clone> Mul for Tensor<A, D>
where
    A: num_traits::One,
{
    type Output = Tensor<A, D>;
    fn mul(self, other: Tensor<A, D>) -> Tensor<A, D> {
        TensorMul::forward(self, other)
    }
}
pub fn add<A, D>(left: Array<A, D>, right: Array<A, D>) -> Array<A, D>
where
    A: Add<Output = A> + Clone,
    D: Dimension,
{
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let test_tensor = Tensor {
            data: Array::from_vec(vec![1, 2, 3, 4]),
            grad: None,
            prev: None,
        };
        let test_tensor2 = Tensor {
            data: Array::from_vec(vec![1, 2, 3, 4]),
            grad: None,
            prev: None,
        };
        let test3 = test_tensor + test_tensor2;
        let test3 = test3.clone() * test3;
        println!("forward: {:?}", test3);
        let test3 = test3.backward();
        println!("backward: {:?}", test3);
        // println!("{:?}", test_tensor2);
        panic!();
    }
}
