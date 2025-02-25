use ndarray::{Array, IxDyn};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::{rand::prelude::*, RandomExt};
use num_traits::Zero;
use std::env::consts;
use std::marker::PhantomData;
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

#[derive(Debug)]
pub enum DimKind {
    Static,
    Dynamic,
}

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

pub trait Dimension: Debug + Default + Clone + 'static {
    const KIND: DimKind;
    fn value(&self) -> usize;
}

#[derive(Debug, Default, Clone)]
pub struct Static<const N: usize>;

#[derive(Debug, Default, Clone)]
pub struct Dynamic(usize);

impl<const N: usize> Dimension for Static<N> {
    const KIND: DimKind = DimKind::Static;
    fn value(&self) -> usize {
        N
    }
}

impl Dimension for Dynamic {
    const KIND: DimKind = DimKind::Dynamic;
    fn value(&self) -> usize {
        self.0
    }
}

pub trait Operation<I: Dimension, J: Dimension, B: Dimension>: std::fmt::Debug {
    fn backward(&mut self, output: &mut Tensor<I, J, B>);
    fn zero_graph(&self);
    fn build_graph(&self);
}
pub fn test_fn() {
    let tensor = Tensor::<Static<2>, Static<4>>::ZERO();
    let tensor2 = Tensor::<Static<4>, Static<2>>::ZERO();
    let tensor3 = Tensor::<Static<3>, Static<2>>::ZERO();
    const TEST: usize = 3;

    let tensor4 = Tensor::<Static<2>, Static<TEST>>::ZERO();
    // let result_a = tensor.dot(tensor2);
    let tensor = Tensor::<Static<4>>::ZERO();
    let tensor2 = Tensor::<Static<4>>::ZERO();
    let result_b = tensor2 + tensor;
    // let result_b = tensor.dot(tensor3); // This breaks, becaus the shapes don't fits
}

pub struct ShapeTest {}

#[derive(Debug, Clone)]
pub struct Tensor<I: Dimension = Static<1>, J: Dimension = Static<1>, B: Dimension = Static<1>> {
    pub container: Rc<RefCell<DataContainer>>,
    prev_op: Option<Rc<RefCell<dyn Operation<I, J, B>>>>,
}

impl<const I: usize, const J: usize, const B: usize> Tensor<Static<I>, Static<J>, Static<B>> {}

impl<
        I: Dimension + 'static + Clone,
        J: Dimension + 'static + Clone,
        B: Dimension + 'static + Clone,
    > Tensor<I, J, B>
{
    pub fn ZERO() -> Tensor<I, J, B> {
        let i = I::default();
        let j = J::default();
        let b = B::default();
        let dims = vec![b.value(), i.value(), j.value()];
        let shape = Shape { dims };
        return Tensor::new(Array::<f32, IxDyn>::zeros(shape));
    }

    pub fn new_random(mean: f32, std: f32) -> Tensor<I, J, B> {
        let i = I::default();
        let j = J::default();
        let b = B::default();
        let dims = vec![b.value(), i.value(), j.value()];
        let shape = Shape { dims };
        let mut value: Array<f32, IxDyn> = Array::<f32, IxDyn>::random(shape, StandardNormal);
        value = value * std + mean;
        return Tensor::new(value);
    }
    pub fn new(data: Array<f32, IxDyn>) -> Tensor<I, J, B> {
        let data = DataContainer {
            array: data,
            grad: None,
            num_consumers: 0,
        };
        Tensor::<I, J, B> {
            container: Rc::new(RefCell::new(data)),
            prev_op: None,
        }
    }
    pub fn new_with_prev(
        data: Array<f32, IxDyn>,
        prev_op: Rc<RefCell<dyn Operation<I, J, B>>>,
    ) -> Tensor<I, J, B> {
        let data = DataContainer {
            array: data,
            grad: None,
            num_consumers: 0,
        };
        Tensor::<I, J, B> {
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
        let new_grad = self.grad().unwrap_or(Array::zeros(self.shape()));
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

    pub fn dot<K: Dimension + Clone>(self, rhs: Tensor<J, K, B>) -> Tensor<I, K, B> {
        unimplemented!()
    }
    pub fn reshape<I2: Dimension, J2: Dimension>(self) -> Tensor<I2, J2, B> {
        TensorReshape::<I, J, B, I2, J2>::forward(self)
    }
}

pub trait DimCompatible<Rhs: Dimension> {
    type Output: Dimension;
}

impl<J: Dimension> DimCompatible<J> for J {
    type Output = J;
}

impl<const N: usize> DimCompatible<Dynamic> for Static<N> {
    type Output = Dynamic;
}

impl<const N: usize> DimCompatible<Static<N>> for Dynamic {
    type Output = Dynamic;
}

#[derive(Debug, Clone)]
struct TensorAdd<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension> {
    lhs: Tensor<I1, J1, B>,
    rhs: Tensor<I2, J2, B>,
}

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension>
    TensorAdd<I1, J1, B, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn forward(lhs: Tensor<I1, J1, B>, rhs: Tensor<I2, J2, B>) -> Tensor<I1, J1, B> {
        let result = lhs.container.borrow().array.clone() + rhs.container.borrow().array.clone();
        let node = TensorAdd { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension> Operation<I1, J1, B>
    for TensorAdd<I1, J1, B, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn backward(&mut self, output: &mut Tensor<I1, J1, B>) {
        let grad = output.grad().unwrap_or_else(|| Array::ones(output.shape()));
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

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension>
    Add<Tensor<I2, J2, B>> for Tensor<I1, J1, B>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    type Output = Tensor<I1, J1, B>;

    fn add(self, other: Tensor<I2, J2, B>) -> Self::Output {
        TensorAdd::forward(self, other)
    }
}

#[derive(Debug, Clone)]
struct TensorMul<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension> {
    lhs: Tensor<I1, J1, B>,
    rhs: Tensor<I2, J2, B>,
}

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension>
    TensorMul<I1, J1, B, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn forward(lhs: Tensor<I1, J1, B>, rhs: Tensor<I2, J2, B>) -> Tensor<I1, J1, B> {
        let result = lhs.container.borrow().array.clone() * rhs.container.borrow().array.clone();
        let node = TensorMul { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension> Operation<I1, J1, B>
    for TensorMul<I1, J1, B, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn backward(&mut self, output: &mut Tensor<I1, J1, B>) {
        let grad = output.grad().unwrap_or_else(|| Array::ones(output.shape()));
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

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension>
    Mul<Tensor<I2, J2, B>> for Tensor<I1, J1, B>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    type Output = Tensor<I1, J1, B>;

    fn mul(self, other: Tensor<I2, J2, B>) -> Self::Output {
        TensorMul::forward(self, other)
    }
}

#[derive(Debug, Clone)]
struct TensorReshape<I: Dimension, J: Dimension, B: Dimension, I2: Dimension, J2: Dimension> {
    tensor: Tensor<I, J, B>,
    input_shape: Shape,
    phantom: PhantomData<(I2, J2)>,
}

impl<I: Dimension, J: Dimension, B: Dimension, I2: Dimension, J2: Dimension>
    TensorReshape<I, J, B, I2, J2>
{
    pub fn forward(input: Tensor<I, J, B>) -> Tensor<I2, J2, B> {
        let shape = Shape {
            dims: vec![I2::default().value(), J2::default().value()],
        };
        let new_data = input.data().into_shape_with_order(shape.dims).unwrap();
        let node = TensorReshape {
            input_shape: input.shape(),
            tensor: input,
            phantom: PhantomData,
        };
        Tensor::new_with_prev(new_data, Rc::new(RefCell::new(node)))
    }
}

impl<I: Dimension, J: Dimension, B: Dimension, I2: Dimension, J2: Dimension> Operation<I2, J2, B>
    for TensorReshape<I, J, B, I2, J2>
{
    fn backward(&mut self, output: &mut Tensor<I2, J2, B>) {
        let new_grad = output
            .grad()
            .expect("Missing gradient")
            .into_shape_with_order(self.input_shape.dims.clone())
            .unwrap();
        self.tensor.backward_internal(new_grad);
    }

    fn zero_graph(&self) {
        self.tensor.zero_graph();
    }

    fn build_graph(&self) {
        self.tensor.build_graph();
    }
}

#[derive(Debug, Clone)]
struct TensorDiv<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension> {
    lhs: Tensor<I1, J1, B>,
    rhs: Tensor<I2, J2, B>,
}

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension>
    TensorDiv<I1, J1, B, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn forward(lhs: Tensor<I1, J1, B>, rhs: Tensor<I2, J2, B>) -> Tensor<I1, J1, B> {
        let result = lhs.data() / rhs.data();
        let node = TensorDiv { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension> Operation<I1, J1, B>
    for TensorDiv<I1, J1, B, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn backward(&mut self, output: &mut Tensor<I1, J1, B>) {
        let grad = output.grad().unwrap_or_else(|| Array::ones(output.shape()));
        let rhs_data = self.rhs.data();
        let lhs_data = self.lhs.data();

        let lhs_grad = grad.clone() / rhs_data.clone();
        let rhs_grad = -grad * lhs_data / (rhs_data.clone() * rhs_data);
        self.lhs.backward_internal(lhs_grad);
        self.rhs.backward_internal(rhs_grad);
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

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension>
    Div<Tensor<I2, J2, B>> for Tensor<I1, J1, B>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    type Output = Tensor<I1, J1, B>;

    fn div(self, rhs: Tensor<I2, J2, B>) -> Self::Output {
        TensorDiv::forward(self, rhs)
    }
}

#[derive(Debug, Clone)]
struct TensorNeg<I: Dimension, J: Dimension, B: Dimension> {
    tensor: Tensor<I, J, B>,
}

impl<I: Dimension, J: Dimension, B: Dimension> TensorNeg<I, J, B> {
    fn forward(tensor: Tensor<I, J, B>) -> Tensor<I, J, B> {
        let result = -tensor.container.borrow().array.clone();
        let node = TensorNeg { tensor };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<I: Dimension, J: Dimension, B: Dimension> Operation<I, J, B> for TensorNeg<I, J, B> {
    fn backward(&mut self, output: &mut Tensor<I, J, B>) {
        let grad = output
            .grad()
            .unwrap_or_else(|| Array::zeros(output.shape()));
        self.tensor.backward_internal(-grad);
    }

    fn zero_graph(&self) {
        self.tensor.zero_graph();
    }

    fn build_graph(&self) {
        self.tensor.build_graph();
    }
}

impl<I: Dimension, J: Dimension, B: Dimension> Neg for Tensor<I, J, B> {
    type Output = Tensor<I, J, B>;
    fn neg(self) -> Self::Output {
        TensorNeg::forward(self)
    }
}

#[derive(Debug, Clone)]
struct TensorMax<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension> {
    lhs: Tensor<I1, J1, B>,
    rhs: Tensor<I2, J2, B>,
    take_from_a: Vec<bool>,
}

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension>
    TensorMax<I1, J1, B, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn forward(lhs: Tensor<I1, J1, B>, rhs: Tensor<I2, J2, B>) -> Tensor<I1, J1, B> {
        assert!(lhs.shape() == rhs.shape());
        let (take_from_a, output): (Vec<bool>, Vec<f32>) = lhs
            .data()
            .iter()
            .zip(rhs.data().iter())
            .map(|(a, b)| (a > b, if a > b { *a } else { *b }))
            .unzip();
        let output = Array::from_vec(output)
            .into_shape_with_order(lhs.shape())
            .unwrap();
        let node = TensorMax {
            lhs,
            rhs,
            take_from_a,
        };
        Tensor::new_with_prev(output, Rc::new(RefCell::new(node)))
    }
}

impl<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension> Operation<I1, J1, B>
    for TensorMax<I1, J1, B, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn backward(&mut self, output: &mut Tensor<I1, J1, B>) {
        let grad = output.grad().expect("should have gradient");
        let (grad_a, grad_b): (Vec<f32>, Vec<f32>) = self
            .take_from_a
            .iter()
            .zip(grad.iter())
            .map(|(&take_first, &grad)| match take_first {
                true => (grad, 0.0),
                false => (0.0, grad),
            })
            .unzip();

        let grad_a = Array::from_vec(grad_a)
            .into_shape_with_order(output.shape())
            .unwrap();
        let grad_b = Array::from_vec(grad_b)
            .into_shape_with_order(output.shape())
            .unwrap();
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

pub fn max<I1: Dimension, J1: Dimension, B: Dimension, I2: Dimension, J2: Dimension>(
    a: Tensor<I1, J1, B>,
    b: Tensor<I2, J2, B>,
) -> Tensor<I1, J1, B>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    TensorMax::forward(a, b)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn it_works() {
        let tensor = Tensor::<Static<2>, Static<4>>::ZERO();
        let tensor2 = Tensor::<Static<4>, Static<2>>::ZERO();
        let tensor3 = Tensor::<Static<3>, Static<2>>::ZERO();
        const TEST: usize = 3;

        let tensor4 = Tensor::<Static<2>, Static<TEST>>::ZERO();
        let result_a = tensor.dot(tensor2);
        let tensor = Tensor::<Static<4>>::ZERO();
        let tensor2 = Tensor::<Static<4>>::ZERO();
        let result_b = tensor2 + tensor;
        // let result_b = tensor.dot(tensor3); // This breaks, becaus the shapes don't fits
        let test_0 = Tensor::<Dynamic, Static<4>>::new(array![[1.0, 2.0, 3.0, 4.0]].into_dyn());
        let test_1 = test_0.clone() + test_0.clone();

        let mut test_2: Tensor<Dynamic, Static<4>> = test_1.clone() + test_1.clone();
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
