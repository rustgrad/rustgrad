use crate::dimensions::{Dimension, UnkownShape, S};
use crate::tensor::Tensor;
use ndarray_rand::rand_distr::StandardNormal;

/// 1D Convolutional Layer
pub struct Conv1DLayer<DIn: Dimension, DOut: Dimension> {
    pub weight: Tensor<(DOut, DIn, usize)>,
    pub bias: Option<Tensor<(DOut,)>>,
}

impl<DIn: Dimension, DOut: Dimension> Conv1DLayer<DIn, DOut> {
    pub fn new(use_bias: bool) -> Self {
        Conv1DLayer {
            weight: Tensor::new_random(0.0, 1.0, StandardNormal),
            bias: if use_bias { Some(Tensor::zero()) } else { None },
        }
    }

    /// Forward pass for Conv1D
    /// x shape: (batch, in_channels, d)
    /// weight shape: (out_channels, in_channels, kernel_size)
    /// output shape: (batch, out_channels, output_width)
    pub fn forward<K: Dimension>(&self, x: Tensor<(K, DIn, usize)>) -> Tensor<(K, DOut, usize)> {
        let [batch_size, in_channels, width] = x.shape().dims(); // Assuming shape() returns unpacked dims
        let [out_channels, _, kernel_size] = self.weight.shape().dims();

        let output_width = width - kernel_size + 1;
        let falttened_length = batch_size * in_channels * width;
        let mut output: Vec<f32> = Vec::from_iter((0..falttened_length).map(|_| 0.0));

        for b in 0..batch_size {
            for out_c in 0..out_channels {
                for i in 0..output_width {
                    let mut sum = Tensor::<(S<1>,)>::zero();
                    for in_c in 0..in_channels {
                        for k in 0..kernel_size {
                            let input_val: Tensor<(S<1>,)> = x.i(b).i(in_c).i(i + k);
                            let weight_val: Tensor<(S<1>,)> = self.weight.i(out_c).i(in_c).i(k);
                            sum = sum + input_val * weight_val;
                        }
                    }
                    if let Some(bias) = self.bias.as_ref() {
                        sum = sum + bias.i(out_c);
                    }
                    output.i(b).i(out_c).i(i) = output.i(b).i(out_c).i(i) + sum;
                }
            }
        }

        output
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone_into_dynamic()];
        if let Some(bias) = self.bias.as_ref() {
            params.push(bias.clone_into_dynamic());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_conv1d_forward_shape() {
        let layer = Conv1DLayer::<S<2>, S<3>>::new(true); // 2 in, 3 out
        let input = Tensor::<(usize, S<2>, usize)>::ones((4, 2, 10)); // batch=4, in_channels=2, width=10
        let output = layer.forward(input);

        let [batch, out_channels, out_width] = output.shape().dims();
        assert_eq!(batch, 4);
        assert_eq!(out_channels, 3);
        assert_eq!(out_width, 10 - layer.weight.shape().2 + 1); // kernel_size = weight.shape().2
    }
}
