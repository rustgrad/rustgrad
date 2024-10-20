use ndarray::{Array, ArrayBase, Dimension, OwnedRepr};
use std::{
    cell::RefCell,
    ops::{Add, Deref},
    rc::Rc,
};
#[derive(Default, Debug)]
pub struct Tensor<A, D: Dimension> {
    pub data: Array<A, D>,
    pub grad: Option<Array<A, D>>,
    pub prev: Option<ComputationOperation<A, D>>,
}
impl<A, D: Dimension> Tensor<A, D> {
    pub fn new(
        data: Array<A, D>,
        grad: Option<Array<A, D>>,
        prev: Option<ComputationOperation<A, D>>,
    ) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Tensor {
            data: data,
            grad: grad,
            prev: prev,
        }))
    }
}
// #[derive(Debug)]
// pub type RefTensor<A, D> = Rc<RefCell<Tensor<A, D>>>;

#[derive(Default, Debug)]
pub struct RefTensor<A, D: Dimension>(Rc<RefCell<Tensor<A, D>>>);

impl<A, D: Dimension> Deref for RefTensor<A, D> {
    type Target = Rc<RefCell<Tensor<A, D>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<A, D: Dimension> Clone for RefTensor<A, D> {
    fn clone(&self) -> Self {
        RefTensor(self.0.clone())
    }
}
impl<A, D: Dimension> RefTensor<A, D> {
    pub fn new(
        data: Array<A, D>,
        grad: Option<Array<A, D>>,
        prev: Option<ComputationOperation<A, D>>,
    ) -> Self {
        RefTensor(Rc::new(RefCell::new(Tensor {
            data: data,
            grad: grad,
            prev: prev,
        })))
    }
}
// impl<A, D: Dimension> Borrow<Tensor<A, D>> for RefTensor<A, D> {
//     fn borrow(&self) -> &Tensor<A, D> {
//         &*self.0.borrow()
//     }
// }

// impl<A, D> Deref for RefTensor<A, D> {
//     type Target = Rc<RefCell<Tensor<A, D>>>;
//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }
// enum ComputationOperation<A, D> {
//     BinaryOp(Box<dyn BinaryOperation<A, D>>), // Node that performs a binary operation
//     UnaryOp(Box<dyn UnaryOperation<A, D>>),   // Node that performs a unary operation
//     TertiaryOp(Box<dyn TertiaryOperation<A, D>>),
// }
#[derive(Debug)]
pub enum ComputationOperation<A, D: Dimension> {
    TensorAdd(Add_<A, D>),
}

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

#[derive(Debug)]
struct Add_<A, D: Dimension> {
    first: RefTensor<A, D>,
    second: RefTensor<A, D>,
}
impl<A, D> Add_<A, D>
where
    A: Add<Output = A> + Clone, // A must support element-wise addition and cloning
    D: Dimension,
    ArrayBase<OwnedRepr<A>, D>: Add<Output = ArrayBase<OwnedRepr<A>, D>>, // Ensure element-wise addition is possible
{
    fn forward(
        input_a: RefTensor<A, D>,
        input_b: RefTensor<A, D>, // input_a: Rc<RefCell<Tensor<A, D>>>,
                                  // input_b: Rc<RefCell<Tensor<A, D>>>,
    ) -> RefTensor<A, D> {
        let data = (*input_a.0).borrow().data.clone() + (*input_b.0).borrow().data.clone();
        let node = Add_ {
            first: input_a,
            second: input_b,
        };
        let prev = Some(ComputationOperation::TensorAdd(node));
        RefTensor::new(data, None, prev)
    }
    fn backward(&mut self, output: &mut Tensor<A, D>) -> (Tensor<A, D>, Tensor<A, D>) {
        unimplemented!();
        // let grad = output.grad.unwrap();
        // let grad_a = grad.clone();
        // let grad_b = grad.clone();
        // (
        //     Tensor {
        //         data: grad_a,
        //         grad: None,
        //         prev: None,
        //     },
        //     Tensor {
        //         data: grad_b,
        //         grad: None,
        //         prev: None,
        //     },
        // )
    }
}

impl<D: Dimension, A: Add<Output = A> + Clone> Add for RefTensor<A, D> {
    type Output = RefTensor<A, D>;
    fn add(self, other: RefTensor<A, D>) -> RefTensor<A, D> {
        Add_::forward(self, other)
    }
}

// RefTensor(Add_::forward(self.clone(), other.clone()))
// pub fn add<A, D>(left: Array<A, D>, right: Array<A, D>) -> Array<A, D>
// where
//     A: Add<Output = A> + Clone,
//     D: Dimension,
// {
//     left + right
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let test_tensor = RefTensor::new(Array::from_vec(vec![1, 2, 3, 4]), None, None);
        let test_tensor2 = RefTensor::new(Array::from_vec(vec![1, 2, 3, 4]), None, None);
        let test3 = test_tensor + test_tensor2;
        println!("{:?}", test3);
        // println!("{:?}", test_tensor2);
        panic!();
    }
}
