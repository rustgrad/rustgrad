use crate::dimensions::{Dimension, UnkownShape, S};
use crate::nn::{Conv1DLayer, LinearLayer};
use crate::tensor::Tensor;

/// A simple 1D Convolutional Neural Network model.
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

    pub fn forward<K: Dimension>(&self, x: Tensor<(K, S<1>)>) -> Tensor<(K, S<1>)> {
        let mut x = self.conv1.forward(x).relu();
        x = self.conv2.forward(x).relu();
        x = self.fc1.forward(x).relu();
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
        let input = Tensor::<(usize, S<1>)>::ones((8, 1)); // batch size = 8

        let output = model.forward(input);
        let (batch_size, output_dim) = output.shape();

        assert_eq!(batch_size, 8);
        assert_eq!(output_dim, 1); // since final output is S<1>
    }
}
