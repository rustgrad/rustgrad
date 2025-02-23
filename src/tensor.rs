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

pub trait Operation<I: Dimension + 'static + Clone, J: Dimension + 'static + Clone>:
    std::fmt::Debug
{
    fn backward(&mut self, output: &mut Tensor<I, J>);
    fn zero_graph(&self);
    fn build_graph(&self);
}
pub fn test_fn() {
    let tensor = Tensor::<Static<2>, Static<4>>::ZERO(&[]);
    let tensor2 = Tensor::<Static<4>, Static<2>>::ZERO(&[]);
    let tensor3 = Tensor::<Static<3>, Static<2>>::ZERO(&[]);
    const TEST: usize = 3;

    let tensor4 = Tensor::<Static<2>, Static<TEST>>::ZERO(&[]);
    let result_a = tensor.dot(tensor2);
    let tensor = Tensor::<Static<4>>::ZERO(&[]);
    let tensor2 = Tensor::<Static<4>>::ZERO(&[]);
    let result_b = tensor2 + tensor;
    // let result_b = tensor.dot(tensor3); // This breaks, becaus the shapes don't fits
}

pub struct ShapeTest {}

#[derive(Debug, Clone)]
pub struct Tensor<I: Dimension = Static<0>, J: Dimension = Static<0>> {
    pub container: Rc<RefCell<DataContainer>>,
    prev_op: Option<Rc<RefCell<dyn Operation<I, J>>>>,
}

impl<const I: usize, const J: usize> Tensor<Static<I>, Static<J>> {
    pub fn ZERO<const D: usize>(batch_sizes: &[usize; D]) -> Tensor<Static<I>, Static<J>> {
        let mut dims = vec![I, J];
        dims.extend_from_slice(batch_sizes);
        let shape = Shape { dims };
        return Tensor::new(Array::<f32, IxDyn>::zeros(shape));
    }
    pub fn new_random<const D: usize>(
        batch_sizes: &[usize; D],
        mean: f32,
        std: f32,
    ) -> Tensor<Static<I>, Static<J>> {
        let mut dims = vec![I, J];
        dims.extend_from_slice(batch_sizes);
        let shape = Shape { dims };
        let mut value: Array<f32, IxDyn> = Array::<f32, IxDyn>::random(shape, StandardNormal);
        value = value * std + mean;
        return Tensor::new(value);
    }
}

impl<I: Dimension + 'static + Clone, J: Dimension + 'static + Clone> Tensor<I, J> {
    pub fn new(data: Array<f32, IxDyn>) -> Tensor<I, J> {
        let data = DataContainer {
            array: data,
            grad: None,
            num_consumers: 0,
        };
        Tensor::<I, J> {
            container: Rc::new(RefCell::new(data)),
            prev_op: None,
        }
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
        Tensor::<I, J> {
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

    pub fn dot<K: Dimension + Clone>(self, rhs: Tensor<J, K>) -> Tensor<I, K> {
        unimplemented!()
    }
    pub fn reshape<const I2: usize, const J2: usize>(
        self,
        shape: Shape,
    ) -> Tensor<Static<I2>, Static<J2>> {
        TensorReshape::<I, J, I2, J2>::forward(self)
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
struct TensorAdd<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> {
    lhs: Tensor<I1, J1>,
    rhs: Tensor<I2, J2>,
}

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> TensorAdd<I1, J1, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn forward(
        lhs: Tensor<I1, J1>,
        rhs: Tensor<I2, J2>,
    ) -> Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output> {
        let result = lhs.container.borrow().array.clone() + rhs.container.borrow().array.clone();
        let node = TensorAdd { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension>
    Operation<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>
    for TensorAdd<I1, J1, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn backward(
        &mut self,
        output: &mut Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>,
    ) {
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

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> Add<Tensor<I2, J2>>
    for Tensor<I1, J1>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    type Output = Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>;

    fn add(self, other: Tensor<I2, J2>) -> Self::Output {
        TensorAdd::forward(self, other)
    }
}

#[derive(Debug, Clone)]
struct TensorMul<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> {
    lhs: Tensor<I1, J1>,
    rhs: Tensor<I2, J2>,
}

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> TensorMul<I1, J1, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    pub fn forward(
        lhs: Tensor<I1, J1>,
        rhs: Tensor<I2, J2>,
    ) -> Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output> {
        let result = lhs.container.borrow().array.clone() * rhs.container.borrow().array.clone();
        let node = TensorMul { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension>
    Operation<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>
    for TensorMul<I1, J1, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn backward(
        &mut self,
        output: &mut Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>,
    ) {
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

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> Mul<Tensor<I2, J2>>
    for Tensor<I1, J1>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    type Output = Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>;
    fn mul(self, other: Tensor<I2, J2>) -> Self::Output {
        TensorMul::forward(self, other)
    }
}

#[derive(Debug, Clone)]
struct TensorReshape<I: Dimension, J: Dimension, const O1: usize, const O2: usize> {
    tensor: Tensor<I, J>,
    input_shape: Shape,
}

impl<I: Dimension, J: Dimension, const O1: usize, const O2: usize> TensorReshape<I, J, O1, O2> {
    pub fn forward(input: Tensor<I, J>) -> Tensor<Static<O1>, Static<O2>> {
        let shape = Shape { dims: vec![O1, O2] };
        let new_data = input.data().into_shape_with_order(shape.dims).unwrap();
        let node = TensorReshape {
            input_shape: input.shape(),
            tensor: input,
        };
        Tensor::new_with_prev(new_data, Rc::new(RefCell::new(node)))
    }
}

impl<I: Dimension, J: Dimension, const O1: usize, const O2: usize> Operation<Static<O1>, Static<O2>>
    for TensorReshape<I, J, O1, O2>
{
    fn backward(&mut self, output: &mut Tensor<Static<O1>, Static<O2>>) {
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
struct TensorDiv<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> {
    lhs: Tensor<I1, J1>,
    rhs: Tensor<I2, J2>,
}

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> TensorDiv<I1, J1, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn forward(
        lhs: Tensor<I1, J1>,
        rhs: Tensor<I2, J2>,
    ) -> Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output> {
        let result = lhs.data() / rhs.data();
        let node = TensorDiv { lhs, rhs };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension>
    Operation<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>
    for TensorDiv<I1, J1, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn backward(
        &mut self,
        output: &mut Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>,
    ) {
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

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> Div<Tensor<I2, J2>>
    for Tensor<I1, J1>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    type Output = Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>;
    fn div(self, rhs: Tensor<I2, J2>) -> Self::Output {
        TensorDiv::forward(self, rhs)
    }
}

#[derive(Debug, Clone)]
struct TensorNeg<I: Dimension, J: Dimension> {
    tensor: Tensor<I, J>,
}

impl<I: Dimension, J: Dimension> TensorNeg<I, J> {
    fn forward(tensor: Tensor<I, J>) -> Tensor<I, J> {
        let result = -tensor.container.borrow().array.clone();
        let node = TensorNeg { tensor };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<I: Dimension, J: Dimension> Operation<I, J> for TensorNeg<I, J> {
    fn backward(&mut self, output: &mut Tensor<I, J>) {
        let grad = output
            .grad()
            .unwrap_or(ndarray::Array::zeros(output.shape()));
        self.tensor.backward_internal(-grad);
    }

    fn zero_graph(&self) {
        self.tensor.zero_graph();
    }

    fn build_graph(&self) {
        self.tensor.build_graph();
    }
}

impl<I: Dimension, J: Dimension> Neg for Tensor<I, J> {
    type Output = Tensor<I, J>;
    fn neg(self) -> Self::Output {
        TensorNeg::forward(self)
    }
}

#[derive(Debug, Clone)]
struct TensorMax<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> {
    lhs: Tensor<I1, J1>,
    rhs: Tensor<I2, J2>,
    take_from_a: Vec<bool>,
}

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension> TensorMax<I1, J1, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn forward(
        lhs: Tensor<I1, J1>,
        rhs: Tensor<I2, J2>,
    ) -> Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output> {
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

impl<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension>
    Operation<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>
    for TensorMax<I1, J1, I2, J2>
where
    I1: DimCompatible<I2>,
    J1: DimCompatible<J2>,
{
    fn backward(
        &mut self,
        output: &mut Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>,
    ) {
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

pub fn max<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension>(
    a: Tensor<I1, J1>,
    b: Tensor<I2, J2>,
) -> Tensor<<I1 as DimCompatible<I2>>::Output, <J1 as DimCompatible<J2>>::Output>
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
        let tensor = Tensor::<Static<2>, Static<4>>::ZERO(&[]);
        let tensor2 = Tensor::<Static<4>, Static<2>>::ZERO(&[]);
        let tensor3 = Tensor::<Static<3>, Static<2>>::ZERO(&[]);
        const TEST: usize = 3;

        let tensor4 = Tensor::<Static<2>, Static<TEST>>::ZERO(&[]);
        let result_a = tensor.dot(tensor2);
        let tensor = Tensor::<Static<4>>::ZERO(&[]);
        let tensor2 = Tensor::<Static<4>>::ZERO(&[]);
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
