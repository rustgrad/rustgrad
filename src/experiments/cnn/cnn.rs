use crate::dimensions::{Dimension, UnkownShape, S};
use crate::nn::{conv1d::Conv1DLayer, LinearLayer};
use crate::tensor::Tensor;

/// A simple 1D Convolutional Neural Network model.
/// Input shape: (batch, in_channels=1, width)
/// Output shape: (batch, 1)
pub struct CNNModel<DHidden: Dimension> {
    conv1: Conv1DLayer<S<1>, DHidden>,
    conv2: Conv1DLayer<DHidden, DHidden>,
    fc1: LinearLayer<DHidden, DHidden>,
    fc2: LinearLayer<DHidden, S<1>>,
}

impl<DHidden: Dimension> CNNModel<DHidden> {
    pub fn new() -> Self {
        CNNModel {
            conv1: Conv1DLayer::new(3, true), // kernel_size=3
            conv2: Conv1DLayer::new(3, true), // kernel_size=3
            fc1: LinearLayer::new(true),
            fc2: LinearLayer::new(false),
        }
    }

    pub fn forward<K: Dimension, W: Dimension>(
        &self,
        x: Tensor<(K, S<1>, W)>,
    ) -> Tensor<(K, S<1>)> {
        // Conv1D layers: (batch, 1, width) -> (batch, DHidden, width')
        let mut x = self.conv1.forward(x).relu();
        // (batch, DHidden, width') -> (batch, DHidden, width'')
        x = self.conv2.forward(x).relu();

        // Global average pooling: take mean across width dimension
        // This reduces (batch, DHidden, width) to (batch, DHidden)
        let x = x.mean_along(2);

        // Linear layers: (batch, DHidden) -> (batch, DHidden) -> (batch, 1)
        let x = self.fc1.forward(x).relu();
        self.fc2.forward(x)
    }

    pub fn parameters(&self) -> Vec<Tensor<UnkownShape>> {
        self.conv1
            .parameters()
            .into_iter()
            .chain(self.conv2.parameters())
            .chain(self.fc1.parameters())
            .chain(self.fc2.parameters())
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_cnn_forward() {
        use ndarray::Array;

        let model = CNNModel::<S<4>>::new(); // hidden size = 4
                                             // Create input with explicit dimensions: (batch=2, channels=1, width=10)
        let input_data = Array::zeros((2, 1, 10)).into_dyn();
        let input = Tensor::<(usize, S<1>, usize)>::new(input_data);

        println!("Input shape: {:?}", input.shape());
        println!("conv1 weight shape: {:?}", model.conv1.weight.shape());
        println!("conv2 weight shape: {:?}", model.conv2.weight.shape());
        println!("fc1 weight shape: {:?}", model.fc1.parameters()[0].shape());
        println!("fc1 bias shape: {:?}", model.fc1.parameters()[1].shape());

        let x = model.conv1.forward(input);
        println!("After conv1 shape: {:?}", x.shape());

        let x = x.relu();
        let x = model.conv2.forward(x);
        println!("After conv2 shape: {:?}", x.shape());

        let x = x.relu();
        let x: Tensor<(usize, S<4>)> = x.mean_along(2);
        println!("After mean_along shape: {:?}", x.shape());

        let output = model.fc1.forward(x);
        println!("After fc1 shape: {:?}", output.shape());

        let shape = output.shape();
        assert_eq!(shape.dims[0], 2); // batch size
    }
}
