use crate::dimensions::{Dimension, UnkownShape};
use crate::tensor::Tensor;
use ndarray_rand::rand_distr::StandardNormal;

/// 1D Convolutional Layer
pub struct Conv1DLayer<DIn: Dimension, DOut: Dimension> {
    pub weight: Tensor<(DOut, DIn, usize)>,
    pub bias: Tensor<(DOut,)>,
    pub use_bias: bool,
}

impl<DIn: Dimension, DOut: Dimension> Conv1DLayer<DIn, DOut> {
    pub fn new(use_bias: bool) -> Self {
        Conv1DLayer {
            weight: Tensor::randn(StandardNormal),
            bias: if use_bias {
                Tensor::zeros()
            } else {
                Tensor::empty()
            },
            use_bias,
        }
    }

    /// Forward pass for Conv1D
    /// x shape: (batch, in_channels, width)
    /// weight shape: (out_channels, in_channels, kernel_size)
    /// output shape: (batch, out_channels, output_width)
    pub fn forward<K: Dimension>(&self, x: Tensor<(K, DIn, usize)>) -> Tensor<(K, DOut, usize)> {
        let (batch_size, in_channels, width) = x.shape(); // Assuming shape() returns unpacked dims
        let (out_channels, _, kernel_size) = self.weight.shape();

        let output_width = width - kernel_size + 1;
        let mut output =
            Tensor::<(K, DOut, usize)>::zeros((batch_size, out_channels, output_width));

        for b in 0..batch_size {
            for out_c in 0..out_channels {
                for i in 0..output_width {
                    let mut sum = 0.0;
                    for in_c in 0..in_channels {
                        for k in 0..kernel_size {
                            let input_val = x[(b, in_c, i + k)];
                            let weight_val = self.weight[(out_c, in_c, k)];
                            sum += input_val * weight_val;
                        }
                    }
                    if self.use_bias {
                        sum += self.bias[out_c];
                    }
                    output[(b, out_c, i)] = sum;
                }
            }
        }

        output
    }

    pub fn parameters(&self) -> Vec<Tensor<UnkownShape>> {
        let mut params = vec![self.weight.to_unknown()];
        if self.use_bias {
            params.push(self.bias.to_unknown());
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

        let (batch, out_channels, out_width) = output.shape();
        assert_eq!(batch, 4);
        assert_eq!(out_channels, 3);
        assert_eq!(out_width, 10 - layer.weight.shape().2 + 1); // kernel_size = weight.shape().2
    }
}
