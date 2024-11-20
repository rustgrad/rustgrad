use ndarray::{
    linalg::Dot, Array, ArrayBase, AsArray, Dimension, OwnedRepr, ShapeBuilder, ViewRepr,
};
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
pub struct Tensor<D: Dimension> {
    pub data: Array<f32, D>,
    pub grad: Option<Array<f32, D>>,
    pub prev: Option<Operation<D>>,
}

impl<D: Dimension> Tensor<D>
// where
//     A: Add<Output = A> + Clone, // A must support element-wise addition and cloning
//     A: Mul<Output = A> + Clone, // A must support element-wise multiplication and cloning
//     A: num_traits::identities::Zero,
// A: num_traits::identities::One,
// ArrayBase<OwnedRepr<A>, D>:
//     Dot<ArrayBase<OwnedRepr<A>, D>, Output = ArrayBase<OwnedRepr<A>, D>>,
{
    pub fn backward(&mut self, grad: Array<f32, D>) {
        self.grad = Some(
            self.grad
                .clone()
                .unwrap_or(Array::zeros(self.data.raw_dim()))
                + grad,
        );
        match self.prev.clone() {
            Some(Operation::Add(mut node)) => node.backward(self),
            Some(Operation::Mul(mut node)) => node.backward(self),
            None => {
                print!("No previous operation");
            }
        };
    }
    pub fn shape(&self) -> D {
        self.data.raw_dim()
    }
}

#[derive(Debug, Clone)]
pub enum Operation<D: Dimension> {
    Add(RefTensorAdd<D>),
    Mul(RefTensorMul<D>),
    // MatMul(TensorMatMul<A, D, E>),
}

#[derive(Debug, Clone)]
pub struct RefTensor<D: Dimension> {
    pub ref_tensor: Rc<RefCell<Tensor<D>>>,
}

impl<D: Dimension> RefTensor<D> {
    pub fn new_with_data(data: Array<f32, D>) -> Self {
        RefTensor {
            ref_tensor: Rc::new(RefCell::new(Tensor {
                data: data,
                grad: None,
                prev: None,
            })),
        }
    }
    pub fn new_with_grad(data: Array<f32, D>, grad: Array<f32, D>) -> Self {
        RefTensor {
            ref_tensor: Rc::new(RefCell::new(Tensor {
                data: data,
                grad: Some(grad),
                prev: None,
            })),
        }
    }
    pub fn new(
        data: Array<f32, D>,
        grad: Option<Array<f32, D>>,
        prev: Option<Operation<D>>,
    ) -> Self {
        RefTensor {
            ref_tensor: Rc::new(RefCell::new(Tensor {
                data: data,
                grad: grad,
                prev: prev,
            })),
        }
    }
    pub fn backward(&mut self, grad: Array<f32, D>) {
        self.ref_tensor.borrow_mut().backward(grad);
    }
    pub fn shape(&self) -> D {
        let tensor = (*self.ref_tensor).borrow();
        tensor.shape()
    }
}

// impl<A, D: Dimension> Deref for RefTensor< D> {
//     type Target = Ref< Tensor< D>>;
//     fn deref(&self) -> Self::Target {
//         self.ref_tensor.borrow()
//     }
// }
// enum ComputationOperation<A, D> {
//     BinaryOp(Box<dyn BinaryOperation<A, D>>), // Node that performs a binary operation
//     UnaryOp(Box<dyn UnaryOperation<A, D>>),   // Node that performs a unary operation
//     TertiaryOp(Box<dyn TertiaryOperation<A, D>>),
// }

trait TertiaryOperation<D: Dimension> {
    fn forward(&self, input_a: Tensor<D>, input_b: Tensor<D>, input_c: Tensor<D>) -> Tensor<D>;
    fn backward(&self, input: Tensor<D>) -> (Tensor<D>, Tensor<D>, Tensor<D>);
}

trait UnaryOperation<A, D: Dimension> {
    fn forward(&self, input: Tensor<D>) -> Tensor<D>;
    fn backward(&self, input: Tensor<D>) -> Tensor<D>;
}

trait BinaryOperation<A, D: Dimension> {
    fn forward(input_a: Tensor<D>, input_b: Tensor<D>) -> Tensor<D>;
    fn backward(&self, input: Tensor<D>) -> (Tensor<D>, Tensor<D>);
}

// #[derive(Debug, Clone)]
// struct TensorMatMul<A, D: Dimension, E: Dimension> {
//     first: Rc<RefCell<Tensor< D>>>,
//     second: Rc<RefCell<Tensor< E>>>,
// }

// impl<A, D, E> TensorMatMul<A, D, E>
// where
//     A: Clone + num_traits::One, // A must support element-wise multiplication and cloning
//     D: Dimension,
//     E: Dimension,
//     ArrayBase<OwnedRepr<A>, D>:
//         Dot<ArrayBase<OwnedRepr<A>, E>, Output = ArrayBase<OwnedRepr<A>, E>>,
//     ArrayBase<OwnedRepr<A>, E>:
//         Dot<ArrayBase<OwnedRepr<A>, E>, Output = ArrayBase<OwnedRepr<A>, D>>,
// {
//     pub fn forward(input_a: Tensor< D>, input_b: Tensor< E>) -> Tensor< E> {
//         let data_a = input_a.data.clone();
//         let data_b = input_b.data.clone();

//         let data = data_a.dot(&data_b);

//         let node = TensorMatMul {
//             first: Rc::new(RefCell::new(input_a)),
//             second: Rc::new(RefCell::new(input_b)),
//         };
//         let result: Tensor< E> = Tensor {
//             data: data,
//             grad: None,
//             prev: Some(Operation::MatMul(node)),
//         };
//         result
//     }
//     pub fn backward(self, output: Tensor< E>) -> (Tensor< D>, Tensor< E>) {
//         let grad = output.grad.unwrap_or(Array::ones(output.data.raw_dim()));

//         let input_a = (*self.first).borrow().data.clone();
//         let input_b = (*self.second).borrow().data.clone();
//         let input_a_t = input_a.reversed_axes();

//         // Ckm = Sum_n Akn * Bnm
//         // dCkm/dBij = delta(m=j)* Aki
//         // dL/dB_ij = dL/dC_km*dC_km/dB_ij
//         // dL/dB_ij = dL/dC_kj*Aki
//         // dL/dB  = A^T @ (dL/dC)
//         let grad_b = input_a_t.dot(&grad);

//         // A @ B = C
//         // Ckm = Sum_n Akn * Bnm
//         // dCjk/dAlm = d_j==l * d_i==m  Bik
//         // dCjk/dAlm = Bmk

//         // dCkm/dAij = delta(k=i)*Bjm
//         // dL/dA_ij = dL/dC_km*dC_km/dA_ij
//         // dL/dA_ij = dL/dC_km*delta(k=i)*Bjm
//         // dL/dA_ij = dL/dC_im*Bjm
//         // dL/dA = (dL/dC) @ B
//         // dL/dA = B @ (dL/dC)^T
//         let grad_t = grad.reversed_axes();
//         let grad_a = input_b.dot(&grad_t);

//         (
//             Tensor {
//                 data: (*self.first).borrow().data.clone(),
//                 grad: Some(grad_a),
//                 prev: Some(Operation::MatMul(self.clone())),
//             },
//             Tensor {
//                 data: (*self.second).borrow().data.clone(),
//                 grad: Some(grad_b),
//                 prev: Some(Operation::MatMul(self.clone())),
//             },
//         )
//     }
// }

#[derive(Debug, Clone)]
struct TensorMul<D: Dimension> {
    first: Rc<RefCell<Tensor<D>>>,
    second: Rc<RefCell<Tensor<D>>>,
}

impl<D> TensorMul<D>
where
    // A: Mul<Output = A> + Clone + num_traits::One, // A must support element-wise multiplication and cloning
    D: Dimension,
    // ArrayBase<OwnedRepr<A>, D>: Mul<Output = ArrayBase<OwnedRepr<A>, D>>, // Ensure element-wise multiplication is possible
    // A: Add<Output = A> + Clone, // A must support element-wise addition and cloning
    // A: Mul<Output = A> + Clone, // A must support element-wise multiplication and cloning
    // A: num_traits::identities::Zero,
    // A: num_traits::identities::One,
{
    pub fn forward(input_a: Tensor<D>, input_b: Tensor<D>) -> Tensor<D> {
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
    pub fn backward(&mut self, output: &mut Tensor<D>) {
        let grad = output
            .grad
            .clone()
            .unwrap_or(Array::ones(output.data.raw_dim()));
        let grad_a = grad.clone() * (*self.second).borrow().data.clone();
        let grad_b = grad.clone() * (*self.first).borrow().data.clone();
        self.first.borrow_mut().backward(grad_a);
        self.second.borrow_mut().backward(grad_b);
    }
}
#[derive(Debug, Clone)]
struct RefTensorMul<D: Dimension> {
    first: RefTensor<D>,
    second: RefTensor<D>,
}

impl<D> RefTensorMul<D>
where
    // A: Mul<Output = A> + Clone + num_traits::One, // A must support element-wise multiplication and cloning
    D: Dimension,
    // ArrayBase<OwnedRepr<A>, D>: Mul<Output = ArrayBase<OwnedRepr<A>, D>>, // Ensure element-wise multiplication is possible
    // A: Add<Output = A> + Clone, // A must support element-wise addition and cloning
    // A: Mul<Output = A> + Clone, // A must support element-wise multiplication and cloning
    // A: num_traits::identities::Zero,
    // A: num_traits::identities::One,
{
    pub fn ref_forward(input_a: RefTensor<D>, input_b: RefTensor<D>) -> RefTensor<D> {
        let data_a = (*input_a.ref_tensor).borrow().data.clone();
        let data_b = (*input_b.ref_tensor).borrow().data.clone();
        let data = data_a * data_b;
        let node = RefTensorMul {
            first: input_a,
            second: input_b,
        };
        let result = RefTensor::new(data, None, Some(Operation::Mul(node)));
        result
    }
    pub fn ref_backward(&mut self, output: &mut Tensor<D>) {
        let grad = output
            .grad
            .clone()
            .unwrap_or(Array::ones(output.data.raw_dim()));
        let grad_a = grad.clone() * (*self.second).borrow().data.clone();
        let grad_b = grad.clone() * (*self.first).borrow().data.clone();
        self.first.borrow_mut().backward(grad_a);
        self.second.borrow_mut().backward(grad_b);
    }
}
#[derive(Debug, Clone)]
struct RefTensorAdd<D: Dimension> {
    first: Rc<RefCell<Tensor<D>>>,
    second: Rc<RefCell<Tensor<D>>>,
}
#[derive(Debug, Clone)]
struct TensorAdd<D: Dimension> {
    first: Rc<RefCell<Tensor<D>>>,
    second: Rc<RefCell<Tensor<D>>>,
}
impl<D: Dimension> TensorAdd<D> {
    fn forward(input_a: Tensor<D>, input_b: Tensor<D>) -> Tensor<D> {
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
    fn backward(&mut self, output: &mut Tensor<D>) {
        let grad = output
            .grad
            .clone()
            .unwrap_or(ndarray::Array::zeros(output.data.raw_dim()));
        let grad_a = grad.clone();
        let grad_b = grad.clone();
        self.first.borrow_mut().backward(grad_a);
        self.second.borrow_mut().backward(grad_b);
    }
}

impl<D: Dimension> Add for Tensor<D> {
    type Output = Tensor<D>;
    fn add(self, other: Tensor<D>) -> Tensor<D> {
        TensorAdd::forward(self, other)
    }
}

// impl<D: Dimension, A: Clone> Tensor< D>
// where
//     A: num_traits::One,
// {
//     pub fn mat_mul<E: Dimension>(self, other: Tensor< E>) -> Tensor< E>
//     where
//         ArrayBase<OwnedRepr<A>, D>:
//             Dot<ArrayBase<OwnedRepr<A>, E>, Output = ArrayBase<OwnedRepr<A>, E>>,
//         ArrayBase<OwnedRepr<A>, E>:
//             Dot<ArrayBase<OwnedRepr<A>, E>, Output = ArrayBase<OwnedRepr<A>, E>>,
//     {
//         TensorMatMul::<A, D, E>::forward(self, other)
//     }
// }

impl<D: Dimension> Mul for Tensor<D> {
    type Output = Tensor<D>;
    fn mul(self, other: Tensor<D>) -> Tensor<D> {
        TensorMul::forward(self, other)
    }
}
impl<D: Dimension> Mul for RefTensor<D> {
    type Output = RefTensor<D>;
    fn mul(self, other: RefTensor<D>) -> Self {
        RefTensor {
            ref_tensor: Rc::new(RefCell::new(RefTensorMul::forward(
                self.ref_tensor.clone(),
                other.ref_tensor.clone(),
            ))),
        }
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
    use ndarray::array;
    use ndarray::{linalg::Dot, Array, ArrayBase, AsArray, Dimension, OwnedRepr, ViewRepr};

    use super::*;

    #[test]
    fn it_works() {
        let test_tensor = RefTensor::new(array![[1.0, 2.0], [3.0, 4.0]]);
        // let test_tensor2 = Tensor {
        //     data: array![[1.0, 2.0, 3.0, 4.0]],
        //     grad: None,
        //     prev: None,
        // };
        let mut test3 = test_tensor.clone() * test_tensor.clone();
        let start_grad = Array::ones(test3.shape());
        test3.backward(start_grad);
        println!("backward: {:?}", test3);
        println!("test_tensor: {:?}", test_tensor);
        // println!("{:?}", test_tensor2);
    }
}
