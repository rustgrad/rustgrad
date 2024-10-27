use crate::tensor::Tensor;
use ndarray::array;
use ndarray::Array;
use ndarray::Ix1;
use ndarray::Ix2;
use ndarray::IxDyn;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

#[derive(Debug)]
pub struct MLP {
    layers: Vec<LinearLayer>,
}
#[derive(Debug)]
pub struct LinearLayer {
    weight: Tensor<f64, Ix2>,
    bias: Tensor<f64, Ix1>,
}
impl LinearLayer {
    pub fn forward(self, x: Tensor<f64,Ix1>) -> Tensor<f64, Ix1> {
        self.weight.mat_mul(x) + self.bias
    }
}
// impl MLP {
//     pub fn forward(self, x: Tensor) -> Tensor {

//     }
// }

pub fn build_mlp(input_size: usize, hidden_size: usize, output_size: usize, layers: usize) -> MLP {
    let mut layers = Vec::with_capacity(layers);
    for layer in 0..layers.len() {
        let (layer_input_size, layer_output_size) = match layer {
            0 => (input_size, hidden_size),
            l if l == layers.len() - 1 => (hidden_size.clone(), output_size.clone()),
            _ => (hidden_size.clone(), hidden_size.clone()),
        };
        layers.push(build_layer(layer_input_size, layer_output_size));
    }
    MLP { layers }
}
pub fn build_layer(input_size: usize, output_size: usize) -> LinearLayer {
    let data = Array::random(IxDyn(&[input_size, output_size]), StandardNormal);
    let weight = Tensor {
        data,
        grad: None,
        prev: None,
    };
    let bias = Tensor {
        data: Array::random(IxDyn(&[output_size]), StandardNormal),
        grad: None,
        prev: None,
    };
    LinearLayer { weight, bias }
}

#[cfg(test)]
mod tests {
    use std::array;

    use super::*;

    #[test]
    fn test_build_mlp() {
        let layer = build_layer(2, 3);
        println!("{:?}", layer);
        let x = Tensor {
            data: array![1., 2.].into_dyn(),
            grad: None,
            prev: None,
        };
        let forwarded = layer.forward(x);
        println!("{:?}", forwarded);
        panic!();
    }
}
