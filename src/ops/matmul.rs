use ndarray::s;
use ndarray::Array;
use ndarray::IxDyn;
use std::borrow::Borrow;

use crate::dimensions::Dimension;
use crate::dimensions::DynamicShape;
use crate::dimensions::Rank2;
use crate::dimensions::Rank3;
use crate::dimensions::Rank4;
use crate::dimensions::Shape;
use crate::iter_range_par;
use crate::ops::Operation;
use crate::run_par;
use crate::shape::ArrayShape;
use crate::sharing::UnsafeSharedRef;
use crate::tensor::DimCompatible;
use crate::tensor::SwapLastDims;
use crate::tensor::Tensor;

use std::cell::RefCell;
use std::rc::Rc;
pub trait MatCompatible<Rhs: Shape> {
    type Output: Shape;
}

impl<N: Dimension, M: Dimension, K1: Dimension, K2: Dimension> MatCompatible<(K2, N)> for (M, K1)
where
    K1: DimCompatible<K2>,
{
    type Output = (M, N);
}
impl<O: Dimension, N: Dimension, M: Dimension, K1: Dimension, K2: Dimension>
    MatCompatible<(O, K2, N)> for (O, M, K1)
where
    K1: DimCompatible<K2>,
{
    type Output = (O, M, N);
}
impl<P: Dimension, O: Dimension, N: Dimension, M: Dimension, K1: Dimension, K2: Dimension>
    MatCompatible<(P, O, K2, N)> for (P, O, M, K1)
where
    K1: DimCompatible<K2>,
{
    type Output = (P, O, M, N);
}
impl MatCompatible<DynamicShape> for DynamicShape {
    type Output = DynamicShape;
}

pub trait MatMul<S1: Shape, S2: Shape>
where
    S1: MatCompatible<S2>,
{
    fn matmul(&self, rhs: Tensor<S2>) -> Tensor<<S1 as MatCompatible<S2>>::Output>;
}
impl<M: Dimension, N: Dimension, K1: Dimension, K2: Dimension> MatMul<Rank2<M, K1>, Rank2<K2, N>>
    for Tensor<Rank2<M, K1>>
where
    K1: DimCompatible<K2>,
{
    fn matmul(&self, rhs: Tensor<Rank2<K2, N>>) -> Tensor<Rank2<M, N>> {
        let node = TensorMatMul {
            lhs: self.clone(),
            rhs: rhs.clone(),
        };
        Tensor::new_with_prev(
            _matmul(self.clone(), rhs.clone()),
            Rc::new(RefCell::new(node)),
        )
    }
}

impl<O: Dimension, M: Dimension, N: Dimension, K1: Dimension, K2: Dimension>
    MatMul<Rank3<O, M, K1>, Rank3<O, K2, N>> for Tensor<Rank3<O, M, K1>>
where
    K1: DimCompatible<K2>,
{
    fn matmul(&self, rhs: Tensor<Rank3<O, K2, N>>) -> Tensor<Rank3<O, M, N>> {
        let node = TensorMatMul {
            lhs: self.clone(),
            rhs: rhs.clone(),
        };
        Tensor::new_with_prev(
            _matmul(self.clone(), rhs.clone()),
            Rc::new(RefCell::new(node)),
        )
    }
}
impl<P: Dimension, O: Dimension, M: Dimension, N: Dimension, K1: Dimension, K2: Dimension>
    MatMul<Rank4<P, O, M, K1>, Rank4<P, O, K2, N>> for Tensor<Rank4<P, O, M, K1>>
where
    K1: DimCompatible<K2>,
{
    fn matmul(&self, rhs: Tensor<Rank4<P, O, K2, N>>) -> Tensor<Rank4<P, O, M, N>> {
        let node = TensorMatMul {
            lhs: self.clone(),
            rhs: rhs.clone(),
        };
        Tensor::new_with_prev(
            _matmul(self.clone(), rhs.clone()),
            Rc::new(RefCell::new(node)),
        )
    }
}

#[derive(Debug, Clone)]
pub struct TensorMatMul<S1: Shape, S2: Shape> {
    pub lhs: Tensor<S1>,
    pub rhs: Tensor<S2>,
}

impl<S1: Shape, S2: Shape> Operation<<S1 as MatCompatible<S2>>::Output> for TensorMatMul<S1, S2>
where
    S1: MatCompatible<S2>,
    S2: SwapLastDims,
{
    fn backward(&self, output: &Tensor<<S1 as MatCompatible<S2>>::Output>) {
        let grad = output.grad().borrow().clone().unwrap();

        let input_lhs = self.lhs.data();
        let input_rhs = self.rhs.data();
        let input_lhs_t = input_lhs.reversed_axes();

        // Ckm = Sum_n Akn * Bnm
        // dCkm/dBij = delta(m=j)* Aki
        // dL/dB_ij = dL/dC_km*dC_km/dB_ij
        // dL/dB_ij = dL/dC_kj*Aki
        // dL/dB  = A^T @ (dL/dC)
        let input_lhs_t = Tensor::<S1>::new(input_lhs_t);
        let grad_tensor = Tensor::<<S1 as MatCompatible<S2>>::Output>::new(grad.clone());
        let grad_rhs = _matmul(input_lhs_t, grad_tensor);

        // A @ B = C
        // Ckm = Sum_n Akn * Bnm
        // dCjk/dAlm = d_j==l * d_i==m  Bik
        // dCjk/dAlm = Bmk

        // dCkm/dAij = delta(k=i)*Bjm
        // dL/dA_ij = dL/dC_km*dC_km/dA_ij
        // dL/dA_ij = dL/dC_km*delta(k=i)*Bjm
        // dL/dA_ij = dL/dC_im*Bjm
        // dL/dA = (dL/dC) @ B
        // dL/dA = B @ (dL/dC)^T
        let grad = Tensor::<<S1 as MatCompatible<S2>>::Output>::new(grad);
        let input_rhs_t = Tensor::<<S2 as SwapLastDims>::Output>::new(input_rhs.reversed_axes());
        let grad_lhs = _matmul(grad, input_rhs_t);

        self.lhs.backward_internal(grad_lhs);
        self.rhs.backward_internal(grad_rhs);
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
        Rc::new(RefCell::new(TensorMatMul::<DynamicShape, DynamicShape> {
            lhs: self.lhs.clone_into_dynamic(),
            rhs: self.rhs.clone_into_dynamic(),
        }))
    }
}

fn _matmul<S1: Shape, S2: Shape>(lhs: Tensor<S1>, rhs: Tensor<S2>) -> Array<f32, IxDyn> {
    let shape_lhs = lhs.shape();
    let shape_rhs = rhs.shape();
    let ndims = shape_lhs.num_dims();

    let (out_shape, strides_lhs, strides_rhs, strides_out) = output_shape(&shape_lhs, &shape_rhs);
    let m = shape_lhs.dims[ndims - 2]; // # of left rows
    let k = shape_rhs.dims[ndims - 2]; // # of left cols and right rows
    let n = shape_rhs.dims[ndims - 1]; // # of right cols
    let l_mat_size = m * k; // size of matrix component of left array
    let r_mat_size = k * n; // size of matrix component of right array
    let out_mat_size = m * n; // size of matrix component of output array

    let num_l_batches = shape_lhs.num_elements() / l_mat_size;
    let num_r_batches = shape_rhs.num_elements() / r_mat_size;
    let num_out_batches = out_shape.num_elements() / out_mat_size;

    let alpha = 1.0;
    let beta = 0.0;

    let out = run_par!(|| {
        let mut out_array = ndarray::Array3::zeros((num_out_batches, m, n));
        let unsafe_shared_out_array = UnsafeSharedRef::new(&mut out_array);

        let new_lhs_shape = ArrayShape::new([num_l_batches, m, k]);
        let new_rhs_shape = ArrayShape::new([num_r_batches, k, n]);
        let lhs_array = reshape(lhs.data(), new_lhs_shape);
        let rhs_array = reshape(rhs.data(), new_rhs_shape);

        iter_range_par!(0, num_out_batches).for_each(|out_batch| {
            // Here, we:
            //   1. Un-flatten the output batch into a component-based batch index.
            //   2. Use the strides for left and right batch indices to convert it to a flattened
            //      batch for left and right.
            let out_index = strides_out.unflatten(out_batch);
            let l_batch = strides_lhs.flatten(&out_index);
            let r_batch = strides_rhs.flatten(&out_index);

            let lhs_slice = lhs_array.slice(s!(l_batch, .., ..));
            let rhs_slice = rhs_array.slice(s!(r_batch, .., ..));

            unsafe {
                let mut out_slice = unsafe_shared_out_array
                    .get()
                    .slice_mut(s!(out_batch, .., ..));

                ndarray::linalg::general_mat_mul(
                    alpha,
                    &lhs_slice,
                    &rhs_slice,
                    beta,
                    &mut out_slice,
                )
            }
        });
        out_array
    });

    out.into_shape_with_order(out_shape.dims).unwrap()
}
/// Compute the (broadcasted) output shape of matrix multiplication, along with strides for
/// the non-matrix dimensions of all arrays.
///
/// # Arguments
/// * `lsh`: Shape of the first (left-hand) matrix multiplication argument.
/// * `rsh`: Shape of the second (right-hand) matrix multiplication argument.
///
/// # Panics
/// * If `D` is not at least 2.
/// * If the matrix multiplication dimensions (last 2) are incompatible.
/// * If any other dimension is not the same for both tensors, or equal to 1. (Any dimension where
///   one dim is equal to 1 is broadcast.)
fn output_shape(lsh: &ArrayShape, rsh: &ArrayShape) -> (ArrayShape, Strides, Strides, Strides) {
    let ndims = lsh.num_dims();
    if ndims < 2 {
        panic!("Matrix multiplication requires an array with at least 2 dimensions.");
    }

    // Fetch matrix dimensions and check compatibility.
    let l_rows = lsh.dims[ndims - 2];
    let l_cols = lsh.dims[ndims - 1];
    let r_rows = rsh.dims[ndims - 2];
    let r_cols = rsh.dims[ndims - 1];
    if l_cols != r_rows {
        panic!(
            "Dimensions {:?}, {:?} are incompatible for matrix multiplication.",
            lsh, rsh
        );
    }
    // Set matrix dimensions of the output shape.
    let mut osh = vec![0; ndims];
    osh[ndims - 2] = l_rows;
    osh[ndims - 1] = r_cols;

    // Set other array dimensions, broadcasting as necessary.
    // Compute the strides inline.
    let mut cur_l_stride: usize = 1;
    let mut cur_r_stride: usize = 1;
    let mut cur_o_stride: usize = 1;
    let mut l_strides = Vec::with_capacity(ndims - 2);
    let mut r_strides = Vec::with_capacity(ndims - 2);
    let mut o_strides = Vec::with_capacity(ndims - 2);
    for i in (0..ndims - 2).rev() {
        let l_dim = lsh.dims[i];
        let r_dim = rsh.dims[i];

        // Compatible dimensions are:
        //   1. Both dimensions are equal.
        //   2. One of the dimensions is equal to 1.
        let o_dim: usize;
        if l_dim == r_dim {
            o_dim = l_dim; // both dimensions are equal
            l_strides.push(cur_l_stride);
            r_strides.push(cur_r_stride);
        } else if l_dim == 1 {
            o_dim = r_dim; // broadcast the left
            l_strides.push(0);
            r_strides.push(cur_r_stride);
        } else if r_dim == 1 {
            o_dim = l_dim; // broadcast the right
            l_strides.push(cur_l_stride);
            r_strides.push(0);
        } else {
            panic!("Dimensions differ and cannot be broadcasted.");
        }
        osh[i] = o_dim;
        o_strides.push(cur_o_stride);
        cur_o_stride *= o_dim;

        cur_l_stride *= l_dim;
        cur_r_stride *= r_dim;
    }
    l_strides.reverse();
    r_strides.reverse();
    o_strides.reverse();

    (
        osh.into(),
        Strides::new(l_strides),
        Strides::new(r_strides),
        Strides::new(o_strides),
    )
}
fn reshape(array: Array<f32, IxDyn>, shape: ArrayShape) -> Array<f32, IxDyn> {
    array.into_shape_clone(shape.dims).unwrap()
}

#[derive(Debug, PartialEq)]
struct Strides {
    strides: Vec<usize>,
}
impl Strides {
    fn new(strides: Vec<usize>) -> Self {
        Strides { strides }
    }

    fn unflatten(&self, linear_index: usize) -> Vec<usize> {
        let mut coord = Vec::with_capacity(self.strides.len());
        let mut rem = linear_index;
        for stride in self.strides.iter() {
            coord.push(rem / stride);
            rem %= stride;
        }
        coord
    }

    fn flatten(&self, index: &Vec<usize>) -> usize {
        assert_eq!(self.strides.len(), index.len());
        self.strides
            .iter()
            .zip(index)
            .map(|(stride, index)| stride * index)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use crate::dimensions::*;
    use ndarray::{array, Array};

    use super::*;
    #[test]
    fn test_linear_layer_mm() {
        let test_0 = Tensor::<(S<1>, S<5>)>::new(Array::ones((1, 5)).into_dyn()); // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_1 = Tensor::<(S<5>, S<5>)>::new(Array::ones((5, 5)).into_dyn()); // grad_2 = 2 * test_1
        let test_2: Tensor<Rank2<S<1>, S<5>>> = test_0.clone().matmul(test_1);
        println!("forward: {:?}", test_2);
        println!("_____________________________");
        test_2.backward();
    }
    fn test<D_IN: Dimension, D_OUT: Dimension, K: Dimension>() {
        let test_0 = Tensor::<(K, D_IN)>::new(array![[1.0, 2.0, 3.0, 4.0]].into_dyn()); // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_1 = Tensor::<(D_IN, D_OUT)>::new(
            array![[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]].into_dyn(),
        ); // grad_2 = 2 * test_1
        let test_2: Tensor<(K, D_OUT)> = test_0.clone().matmul(test_1);
        println!("forward: {:?}", test_2);
        println!("_____________________________");
        test_2.backward();
    }

    #[test]
    fn test_matmul() {
        let test_0 = Tensor::<(S<1>, S<4>)>::new(array![[1.0, 2.0, 3.0, 4.0]].into_dyn()); // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_1 = Tensor::<(S<2>, S<4>)>::new(
            array![[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]].into_dyn(),
        ); // grad_2 = 2 * test_1
        let test_1 = test_1.reshape_no_grad::<(S<4>, S<2>)>();
        let test_2 = test_0.clone().matmul(test_1);
        println!("forward: {:?}", test_2);
        println!("_____________________________");
        test_2.backward();
        let test_3: Tensor<(S<2>, S<2>)> = Tensor::new(array![[2.0], [3.0]].into_dyn()); //2X1 // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_4 = Tensor::<(S<2>, S<2>)>::new(array![[1.0, 2.0], [1.0, 2.0]].into_dyn()); // 2X2 grad_2 = 2 * test_1
        let test_2 = test_4.clone().matmul(test_3);
        print!("forward passed");
        test_2.backward();

        // assert_eq!(
        //     test_0.grad(),
        //     Some(array![[8.0, 16.0, 24.0, 32.0]].into_dyn())
        // );
        // assert_eq!(
        //     test_1.grad(),
        //     Some(array![[4.0, 8.0, 12.0, 16.0]].into_dyn())
        // );
        // assert_eq!(test_2.grad(), Some(array![[1.0, 1.0, 1.0, 1.0]].into_dyn()));

        // println!("{:?}", test_2);
    }
}
