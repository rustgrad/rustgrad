use crate::dimensions::{Dimension, UnkownShape};
use crate::ops::Operation;
use crate::tensor::Tensor;
use ndarray_rand::RandomExt;

#[derive(Debug)]
pub struct Conv1DOperation<K: Dimension, DIn: Dimension, DOut: Dimension> {
    pub input_tensor: Tensor<(K, DIn, usize)>, // Input tensor shape: (batch, in_channels, width)
    pub weight: Tensor<(DOut, DIn, usize)>,
    pub bias: Option<Tensor<(DOut,)>>,
    pub slice: Vec<usize>,
}
/// 1D Convolutional Layer
#[derive(Debug)]
pub struct Conv1DLayer<DIn: Dimension, DOut: Dimension> {
    pub weight: Tensor<(DOut, DIn, usize)>,
    pub bias: Option<Tensor<(DOut,)>>,
}

impl<K: Dimension, DIn: Dimension, DOut: Dimension> Operation<(K, DOut)>
    for Conv1DOperation<K, DIn, DOut>
{
    fn backward(&self, _output: &Tensor<(K, DOut)>) {
        todo!()
    }

    fn zero_graph(&self) {
        todo!()
    }

    fn build_graph(&self) {
        todo!()
    }

    fn clone_into_dynamic(&self) -> std::rc::Rc<std::cell::RefCell<dyn Operation<UnkownShape>>> {
        todo!()
    }
}

impl<DIn: Dimension, DOut: Dimension> Conv1DLayer<DIn, DOut> {
    pub fn new(kernel_size: usize, use_bias: bool) -> Self {
        use ndarray::Array;
        use rand::distributions::Uniform;

        let in_channels = DIn::default().size();
        let out_channels = DOut::default().size();

        // Initialize weight with shape (out_channels, in_channels, kernel_size)
        let weight_data = Array::random(
            (out_channels, in_channels, kernel_size),
            Uniform::new(-0.5, 0.5),
        ) * (2.0 / (in_channels * kernel_size) as f32).sqrt();

        let bias = if use_bias {
            let bias_data = Array::zeros(out_channels);
            Some(Tensor::new(bias_data.into_dyn()))
        } else {
            None
        };

        Conv1DLayer {
            weight: Tensor::new(weight_data.into_dyn()),
            bias,
        }
    }

    /// Forward pass for Conv1D
    /// x shape: (batch, in_channels, width)
    /// weight shape: (out_channels, in_channels, kernel_size)
    /// output shape: (batch, out_channels, output_width)
    pub fn forward<K: Dimension, W: Dimension>(
        &self,
        x: Tensor<(K, DIn, W)>,
    ) -> Tensor<(K, DOut, usize)> {
        use ndarray::Array;

        let x_shape = x.shape();
        let w_shape = self.weight.shape();

        let batch_size = x_shape.dims[0];
        let in_channels = x_shape.dims[1];
        let width = x_shape.dims[2];

        let out_channels = w_shape.dims[0];
        let kernel_size = w_shape.dims[2];

        let output_width = width - kernel_size + 1;

        let x_data = x.data();
        let w_data = self.weight.data();

        // Reshape data to 3D for easier indexing
        let x_array = x_data
            .into_shape_with_order((batch_size, in_channels, width))
            .unwrap();
        let w_array = w_data
            .into_shape_with_order((out_channels, in_channels, kernel_size))
            .unwrap();

        let mut output = Array::zeros((batch_size, out_channels, output_width));

        for b in 0..batch_size {
            for out_c in 0..out_channels {
                for i in 0..output_width {
                    let mut sum = 0.0;
                    for in_c in 0..in_channels {
                        for k in 0..kernel_size {
                            sum += x_array[[b, in_c, i + k]] * w_array[[out_c, in_c, k]];
                        }
                    }
                    output[[b, out_c, i]] = sum;
                }
            }
        }

        // Add bias if present
        if let Some(bias) = self.bias.as_ref() {
            let bias_data = bias.data();
            let bias_array = bias_data.into_shape_with_order(out_channels).unwrap();
            for b in 0..batch_size {
                for out_c in 0..out_channels {
                    for i in 0..output_width {
                        output[[b, out_c, i]] += bias_array[out_c];
                    }
                }
            }
        }

        Tensor::new(output.into_dyn())
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
    use crate::dimensions::S;
    use crate::tensor::Tensor;

    #[test]
    fn test_conv1d_forward_shape() {
        let layer = Conv1DLayer::<S<2>, S<3>>::new(3, true); // 2 in, 3 out, kernel_size=3
        let input = Tensor::<(usize, S<2>, usize)>::zero(); // batch=4, in_channels=2, width=10
        let output = layer.forward(input);

        let shape = output.shape();
        let batch = shape.dims[0];
        let out_channels = shape.dims[1];
        let out_width = shape.dims[2];

        assert_eq!(batch, 4);
        assert_eq!(out_channels, 3);
        let kernel_size = layer.weight.shape().dims[2];
        assert_eq!(out_width, 10 - kernel_size + 1);
    }
}
