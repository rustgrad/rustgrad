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
            conv1: Conv1DLayer::new(true),
            conv2: Conv1DLayer::new(true),
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
        let model = CNNModel::<S<4>>::new(); // hidden size = 4
        let input = Tensor::<(usize, S<1>, usize)>::zero(); // batch size = 8, channels = 1, width = dynamic

        let output = model.forward(input);
        let shape = output.shape();

        assert_eq!(shape.dims[0], 8);
        assert_eq!(shape.dims[1], 1); // since final output is S<1>
    }
}
