pub mod add;
pub mod broadcast;
pub mod div;
pub mod matmul;
pub mod mean;
pub mod mul;
pub use mul::TensorMul;
pub mod relu;
pub mod reshape;
pub mod scalar;
pub mod slicing;
pub mod substract;
pub mod sum;
pub mod var;
pub use substract::TensorSub;
pub mod negate;
pub use negate::TensorNeg;
pub mod sqrt;
pub use sqrt::TensorSqrt;
pub mod permute;
pub use permute::TensorPermute;
pub mod stack;
pub use stack::TensorStack;
// pub mod update_value;
// pub use update_value::UpdateValue;

use matmul::MatMul;
pub use scalar::*;
pub mod max;
pub use max::max;
pub mod exp;
pub mod log;
pub mod softmax;

use crate::{
    dimensions::{Rank1, Rank2, Shape, UnkownShape, S},
    tensor::Tensor,
};
use std::{cell::RefCell, rc::Rc};

pub trait Operation<S: Shape>: std::fmt::Debug {
    fn backward(&self, output: &Tensor<S>);
    fn zero_graph(&self);
    fn build_graph(&self);
    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<UnkownShape>>>;
}
pub fn test_fn() {
    let tensor = Tensor::<Rank2<S<2>, S<4>>>::zero();
    let tensor2 = Tensor::<Rank2<S<4>, S<2>>>::zero();
    let _tensor3 = Tensor::<Rank2<S<3>, S<2>>>::zero();
    const TEST: usize = 3;

    let _tensor4 = Tensor::<Rank2<S<2>, S<TEST>>>::zero();
    let _result_a = tensor.matmul(tensor2);
    let tensor = Tensor::<Rank1<S<4>>>::zero();
    let tensor2 = Tensor::<Rank1<S<4>>>::zero();
    let _result_b = tensor2 + tensor;
    // let result_b = tensor.dot(tensor3); // This breaks, becaus the shapes don't fits
}
