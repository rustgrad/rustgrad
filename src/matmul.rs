use ndarray::s;
// use ndarray::Dimension;
use std::borrow::Borrow;

use crate::iter_range_par;
use crate::run_par;
use crate::shape::Shape;
use crate::sharing::UnsafeSharedRef;
use crate::tensor::DimCompatible;
use crate::tensor::Dimension;
use crate::tensor::Operation;
use crate::tensor::Static;
use crate::tensor::Tensor;

use std::cell::RefCell;
use std::rc::Rc;
#[derive(Debug, Clone)]
pub struct TensorMatMul<B1, B2, I1, J1, I2, J2>
where
    B1: Dimension,
    B2: Dimension,
    I1: Dimension,
    J1: Dimension,
    I2: Dimension,
    J2: Dimension,
{
    pub lhs: Tensor<B1, I1, J1>,
    pub rhs: Tensor<B2, I2, J2>,
}

impl<B1, B2, I1, J1, I2, J2> TensorMatMul<B1, B2, I1, J1, I2, J2>
where
    B1: Dimension + DimCompatible<B2>,
    I1: Dimension,
    J1: Dimension + DimCompatible<I2>,
    I2: Dimension,
    J2: Dimension,
{
    pub fn forward(lhs: Tensor<B1, I1, J1>, rhs: Tensor<B2, I2, J2>) -> Tensor<B1, I1, J2> {
        let data = matmul(lhs.clone(), rhs.clone());
        let node = TensorMatMul { lhs, rhs };
        Tensor::new_with_prev(data.data(), Rc::new(RefCell::new(node)))
    }
}

impl<B1, B2, I1, J1, I2, J2> Operation<B1, I1, J2> for TensorMatMul<B1, B2, I1, J1, I2, J2>
where
    B1: Dimension + DimCompatible<B2>,
    I1: Dimension,
    J1: Dimension + DimCompatible<I2>,
    I2: Dimension,
    J2: Dimension,
{
    fn backward(&mut self, output: &mut Tensor<B1, I1, J2>) {
        let grad = output.grad().borrow().clone().unwrap();

        let input_lhs = self.lhs.data();
        let input_rhs = self.rhs.data();
        let input_lhs_t = input_lhs.reversed_axes();

        // Ckm = Sum_n Akn * Bnm
        // dCkm/dBij = delta(m=j)* Aki
        // dL/dB_ij = dL/dC_km*dC_km/dB_ij
        // dL/dB_ij = dL/dC_kj*Aki
        // dL/dB  = A^T @ (dL/dC)
        let input_lhs_t = Tensor::<I1, J1>::new(input_lhs_t);
        let grad_tensor = Tensor::<I2, J2>::new(grad.clone());
        let grad_rhs = matmul(input_lhs_t, grad_tensor);

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
        let grad = Tensor::<I1, J2>::new(grad);
        let input_rhs_t = Tensor::<J2, I2>::new(input_rhs.reversed_axes());
        let grad_lhs = matmul(grad, input_rhs_t);

        self.lhs.backward_internal(grad_lhs.data());
        self.rhs.backward_internal(grad_rhs.data());
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

pub(crate) fn matmul<B1, B2, I1, J1, I2, J2>(
    lhs: Tensor<B1, I1, J1>,
    rhs: Tensor<B2, I2, J2>,
) -> Tensor<B1, I1, J2>
where
    B1: Dimension + DimCompatible<B2>,
    I1: Dimension,
    J1: Dimension + DimCompatible<I2>,
    I2: Dimension,
    J2: Dimension,
{
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

        let new_lhs_shape = Shape::new([num_l_batches, m, k]);
        let new_rhs_shape = Shape::new([num_r_batches, k, n]);
        let lhs_array = reshape(lhs).data();
        let rhs_array = reshape(rhs).data();

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

        Tensor::new(out_array.into_dyn())
    });

    reshape::<I1, J1, I1, J2>(out)
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
fn output_shape(lsh: &Shape, rsh: &Shape) -> (Shape, Strides, Strides, Strides) {
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
fn reshape<I1: Dimension, J1: Dimension, I2: Dimension, J2: Dimension>(
    tensor: Tensor<I1, J1>,
) -> Tensor<I2, J2> {
    let shape = tensor.shape();
    let new_data = tensor.data().into_shape_with_order(shape.dims).unwrap();
    Tensor::new(new_data)
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
    use ndarray::{array, Array};

    use super::*;
    #[test]
    fn test_linear_layer_mm() {
        let test_0 = Tensor::new(Array::ones((1, 5)).into_dyn()); // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_1 = Tensor::new(Array::ones((5, 5)).into_dyn()); // grad_2 = 2 * test_1
        let mut test_2: Tensor = test_0.clone().dot(test_1);
        println!("forward: {:?}", test_2);
        println!("_____________________________");
        test_2.backward();
    }

    #[test]
    fn test_matmul() {
        let test_0 = Tensor::<Static<1>, Static<4>>::new(array![[1.0, 2.0, 3.0, 4.0]].into_dyn()); // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_1 = Tensor::<Static<2>, Static<4>>::new(
            array![[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]].into_dyn(),
        ); // grad_2 = 2 * test_1
        let mut test_2: Tensor = test_0.clone().dot(reshape::<
            Static<1>,
            Static<4>,
            Static<2>,
            Static<4>,
        >(test_1));
        println!("forward: {:?}", test_2);
        println!("_____________________________");
        test_2.backward();
        let test_3 = Tensor::new(array![[2.0], [3.0]].into_dyn()); //2X1 // grad = 2 * grad_1 = 4 * test_1 = 8 * test_0
        let test_4 = Tensor::new(array![[1.0, 2.0], [1.0, 2.0]].into_dyn()); // 2X2 grad_2 = 2 * test_1
        let test_3 = reshape(test_3);
        let test_4 = reshape(test_4);
        let mut test_2: Tensor = test_4.clone().dot(test_3);
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
    fn test_2() {}
}
