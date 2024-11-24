// use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArray, UnsafeSharedRef};

// use alloc::{vec, vec::Vec};
// use burn_common::{iter_range_par, run_par};
// use burn_tensor::{ops::FloatTensorOps, Shape};
// use burn_tensor::{ElementConversion, TensorMetadata};
// use ndarray::s;

// pub(crate) fn matmul<E>(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E>
// where
//     E: FloatNdArrayElement,
// {
//     let shape_lhs = lhs.shape();
//     let shape_rhs = rhs.shape();
//     let ndims = shape_lhs.num_dims();
//     let m = shape_lhs.dims[ndims - 2]; // # of left rows
//     let k = shape_rhs.dims[ndims - 2]; // # of left cols and right rows
//     let n = shape_rhs.dims[ndims - 1]; // # of right cols

//     let (out_shape, strides_lhs, strides_rhs, strides_out) = output_shape(&shape_lhs, &shape_rhs);
//     let l_mat_size = m * k; // size of matrix component of left array
//     let r_mat_size = k * n; // size of matrix component of right array
//     let out_mat_size = m * n; // size of matrix component of output array

//     let num_l_batches = shape_lhs.num_elements() / l_mat_size;
//     let num_r_batches = shape_rhs.num_elements() / r_mat_size;
//     let num_out_batches = out_shape.num_elements() / out_mat_size;

//     let alpha: E = 1.0.elem();
//     let beta: E = 0.0.elem();

//     let out: NdArrayTensor<E> = run_par!(|| {
//         let mut out_array = ndarray::Array3::<E>::zeros((num_out_batches, m, n));
//         let unsafe_shared_out_array = UnsafeSharedRef::new(&mut out_array);

//         let lhs_array = NdArray::<E>::float_reshape(lhs, Shape::new([num_l_batches, m, k])).array;
//         let rhs_array = NdArray::<E>::float_reshape(rhs, Shape::new([num_r_batches, k, n])).array;

//         iter_range_par!(0, num_out_batches).for_each(|out_batch| {
//             // Here, we:
//             //   1. Un-flatten the output batch into a component-based batch index.
//             //   2. Use the strides for left and right batch indices to convert it to a flattened
//             //      batch for left and right.
//             let out_index = strides_out.unflatten(out_batch);
//             let l_batch = strides_lhs.flatten(&out_index);
//             let r_batch = strides_rhs.flatten(&out_index);

//             let lhs_slice = lhs_array.slice(s!(l_batch, .., ..));
//             let rhs_slice = rhs_array.slice(s!(r_batch, .., ..));

//             unsafe {
//                 let mut out_slice = unsafe_shared_out_array
//                     .get()
//                     .slice_mut(s!(out_batch, .., ..));

//                 ndarray::linalg::general_mat_mul(
//                     alpha,
//                     &lhs_slice,
//                     &rhs_slice,
//                     beta,
//                     &mut out_slice,
//                 )
//             }
//         });

//         NdArrayTensor::new(out_array.into_shared().into_dyn())
//     });

//     NdArray::<E>::float_reshape(out, out_shape)
// }
