use crate::ops::matmul::MatMul;
use ndarray::{Array, IxDyn};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::{rand::prelude::*, RandomExt};
use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{Deref, Neg},
    rc::Rc,
};

use crate::dimensions::{Dimension, Dynamic, DynamicShape, Rank1, Rank2, Rank3, Rank4, Shape, S};
use crate::shape::ArrayShape;
// use crate::{matmul::TensorMatMul, shape::ArrayShape};

pub trait Operation<S: Shape>: std::fmt::Debug {
    fn backward(&mut self, output: &mut Tensor<S>);
    fn zero_graph(&self);
    fn build_graph(&self);
    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>>;
}
pub fn test_fn() {
    let tensor = Tensor::<Rank2<S<2>, S<4>>>::ZERO();
    let tensor2 = Tensor::<Rank2<S<4>, S<2>>>::ZERO();
    let tensor3 = Tensor::<Rank2<S<3>, S<2>>>::ZERO();
    const TEST: usize = 3;

    let tensor4 = Tensor::<Rank2<S<2>, S<TEST>>>::ZERO();
    let result_a = tensor.matmul(tensor2);
    let tensor = Tensor::<Rank1<S<4>>>::ZERO();
    let tensor2 = Tensor::<Rank1<S<4>>>::ZERO();
    let result_b = tensor2 + tensor;
    // let result_b = tensor.dot(tensor3); // This breaks, becaus the shapes don't fits
}

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

pub trait SwapLastDims {
    type Output: Shape;
}
impl SwapLastDims for DynamicShape {
    type Output = DynamicShape;
}
impl<M: Dimension, N: Dimension> SwapLastDims for Rank2<M, N> {
    type Output = Rank2<N, M>;
}
impl<O: Dimension, N: Dimension, M: Dimension> SwapLastDims for Rank3<O, N, M> {
    type Output = Rank3<O, M, N>;
}
impl<P: Dimension, O: Dimension, N: Dimension, M: Dimension> SwapLastDims for Rank4<P, O, N, M> {
    type Output = Rank4<P, O, M, N>;
}

// impl<SIN: Shape> ReshapeCompatible<SOUT> for SIN {
//     type Output = SIN;
// }

#[derive(Debug, Clone)]
pub struct Tensor<S: Shape> {
    pub container: Rc<RefCell<DataContainer>>,
    pub prev_op: Option<Rc<RefCell<dyn Operation<S>>>>,
}

impl<S: Shape> Tensor<S> {
    pub fn ZERO() -> Tensor<S> {
        let dims = S::shape().dims.clone();
        let shape = ArrayShape { dims };
        return Tensor::new(Array::<f32, IxDyn>::zeros(shape));
    }
    pub fn new_random(mean: f32, std: f32) -> Tensor<S> {
        let dims = S::shape().dims.clone();
        let shape = ArrayShape { dims };
        let mut value: Array<f32, IxDyn> = Array::<f32, IxDyn>::random(shape, StandardNormal);
        value = value * std + mean;
        return Tensor::new(value);
    }
}

impl<S: Shape> Tensor<S> {
    pub fn new(data: Array<f32, IxDyn>) -> Tensor<S> {
        let data = DataContainer {
            array: data,
            grad: None,
            num_consumers: 0,
        };
        Tensor::<S> {
            container: Rc::new(RefCell::new(data)),
            prev_op: None,
        }
    }
    pub fn clone_into_dynamic(&self) -> Tensor<DynamicShape> {
        Tensor {
            container: self.container.clone(),
            prev_op: self
                .prev_op
                .as_ref()
                .map(|op| op.borrow().clone_into_dynamic()),
        }
    }
    pub fn new_with_prev(
        data: Array<f32, IxDyn>,
        prev_op: Rc<RefCell<dyn Operation<S>>>,
    ) -> Tensor<S> {
        let data = DataContainer {
            array: data,
            grad: None,
            num_consumers: 0,
        };
        Tensor::<S> {
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

    pub fn shape(&self) -> ArrayShape {
        let binding = self.container.borrow();
        let shape = binding.array.shape();
        shape.to_vec().into()
    }
}

pub trait DimCompatible<Rhs: Dimension> {
    type Output: Dimension;
}
pub trait ShapeCompatible<Rhs: Shape> {
    type Output: Shape;
}
impl ShapeCompatible<DynamicShape> for DynamicShape {
    type Output = DynamicShape;
}
impl<I: Dimension> ShapeCompatible<Rank1<I>> for Rank1<I> {
    type Output = Rank1<I>;
}
impl<I: Dimension, J: Dimension, I2: Dimension, J2: Dimension> ShapeCompatible<Rank2<I2, J2>>
    for Rank2<I, J>
where
    I: DimCompatible<I2>,
    J: DimCompatible<J2>,
{
    type Output = Rank2<<I as DimCompatible<I2>>::Output, <J as DimCompatible<J2>>::Output>;
}
impl<A1: Dimension, A2: Dimension, A3: Dimension, B1: Dimension, B2: Dimension, B3: Dimension>
    ShapeCompatible<Rank3<B1, B2, B3>> for Rank3<A1, A2, A3>
where
    A1: DimCompatible<B1>,
    A2: DimCompatible<B2>,
    A3: DimCompatible<B3>,
{
    type Output = Rank3<
        <A1 as DimCompatible<B1>>::Output,
        <A2 as DimCompatible<B2>>::Output,
        <A3 as DimCompatible<B3>>::Output,
    >;
}
impl<
        A1: Dimension,
        A2: Dimension,
        A3: Dimension,
        A4: Dimension,
        B1: Dimension,
        B2: Dimension,
        B3: Dimension,
        B4: Dimension,
    > ShapeCompatible<Rank4<B1, B2, B3, B4>> for Rank4<A1, A2, A3, A4>
where
    A1: DimCompatible<B1>,
    A2: DimCompatible<B2>,
    A3: DimCompatible<B3>,
    A4: DimCompatible<B4>,
{
    type Output = Rank4<
        <A1 as DimCompatible<B1>>::Output,
        <A2 as DimCompatible<B2>>::Output,
        <A3 as DimCompatible<B3>>::Output,
        <A4 as DimCompatible<B4>>::Output,
    >;
}

impl<D: Dimension> DimCompatible<D> for D {
    type Output = D;
}

impl<const N: usize> DimCompatible<Dynamic> for S<N> {
    type Output = Dynamic;
}

impl<const N: usize> DimCompatible<S<N>> for Dynamic {
    type Output = Dynamic;
}

#[derive(Debug, Clone)]
struct TensorNeg<S: Shape> {
    tensor: Tensor<S>,
}

impl<S: Shape> TensorNeg<S> {
    fn forward(tensor: Tensor<S>) -> Tensor<S> {
        let result = -tensor.container.borrow().array.clone();
        let node = TensorNeg { tensor };
        Tensor::new_with_prev(result, Rc::new(RefCell::new(node)))
    }
}

impl<S: Shape> Operation<S> for TensorNeg<S> {
    fn backward(&mut self, output: &mut Tensor<S>) {
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
    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        Rc::new(RefCell::new(TensorNeg::<DynamicShape> {
            tensor: self.tensor.clone_into_dynamic(),
        }))
    }
}

impl<S: Shape> Neg for Tensor<S> {
    type Output = Tensor<S>;
    fn neg(self) -> Self::Output {
        TensorNeg::forward(self)
    }
}

#[derive(Debug, Clone)]
struct TensorMax<S1: Shape, S2: Shape> {
    lhs: Tensor<S1>,
    rhs: Tensor<S2>,
    take_from_a: Vec<bool>,
}

impl<S1: Shape, S2: Shape> TensorMax<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn forward(lhs: Tensor<S1>, rhs: Tensor<S2>) -> Tensor<<S1 as ShapeCompatible<S2>>::Output> {
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

impl<S1: Shape, S2: Shape> Operation<<S1 as ShapeCompatible<S2>>::Output> for TensorMax<S1, S2>
where
    S1: ShapeCompatible<S2>,
{
    fn backward(&mut self, output: &mut Tensor<<S1 as ShapeCompatible<S2>>::Output>) {
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
    fn clone_into_dynamic(&self) -> Rc<RefCell<dyn Operation<DynamicShape>>> {
        Rc::new(RefCell::new(TensorMax::<DynamicShape, DynamicShape> {
            lhs: self.lhs.clone_into_dynamic(),
            rhs: self.rhs.clone_into_dynamic(),
            take_from_a: self.take_from_a.clone(),
        }))
    }
}

pub fn max<S1: Shape, S2: Shape>(
    a: Tensor<S1>,
    b: Tensor<S2>,
) -> Tensor<<S1 as ShapeCompatible<S2>>::Output>
where
    S1: ShapeCompatible<S2>,
{
    TensorMax::forward(a, b)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn it_works() {
        let tensor = Tensor::<Rank2<S<2>, S<4>>>::ZERO();
        let tensor2 = Tensor::<Rank2<S<4>, S<2>>>::ZERO();
        let tensor3 = Tensor::<Rank2<S<3>, S<2>>>::ZERO();
        const TEST: usize = 3;

        let tensor4 = Tensor::<Rank2<S<2>, S<TEST>>>::ZERO();
        let result_a = tensor.matmul(tensor2);
        let tensor = Tensor::<Rank1<S<4>>>::ZERO();
        let tensor2 = Tensor::<Rank1<S<4>>>::ZERO();
        let result_b = tensor2 + tensor;
        //let result_b = tensor.dot(tensor3); // This breaks, becaus the shapes don't fits
        let test_0 = Tensor::<Rank1<Dynamic>>::new(array![[1.0, 2.0, 3.0, 4.0]].into_dyn());
        let test_1 = test_0.clone() + test_0.clone();

        let mut test_2: Tensor<Rank1<Dynamic>> = test_1.clone() + test_1.clone();
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
