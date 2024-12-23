use crate::shape::Shape;
use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::ArrayBase;
use ndarray::IxDyn;
use ndarray_rand::rand::distributions::weighted;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

#[derive(Debug)]
pub struct MLP {
    layers: Vec<LinearLayer>,
}

#[derive(Debug)]
pub struct LinearLayer {
    weight: Tensor,
    bias: Tensor,
    input_dim: usize,
    output_dim: usize,
}
impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize) -> LinearLayer {
        // let bias: Array<f32, IxDyn> = Array::random(input_dim, StandardNormal).into_dyn();
        let bias = Array::zeros(output_dim).into_dyn();
        let bias = Tensor::new(bias);
        // let weight: Array<f32, IxDyn> = Array::random((input_dim, output_dim), StandardNormal).into_dyn();
        let weight = Array::zeros((input_dim, output_dim)).into_dyn();
        let weight = Tensor::new(weight);
        LinearLayer {
            bias,
            weight,
            input_dim,
            output_dim,
        }
    }
    pub fn forward(&self, x: Tensor) -> Tensor {
        //TODO add non linearity
        x.reshape(Shape::new([1, self.input_dim]))
            .dot(self.weight.clone())
            + self.bias.clone()
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

impl MLP {
    pub fn new(depth: usize, input_dim: usize, hidden_dim: usize, output_dim: usize) -> MLP {
        let mut layers = vec![LinearLayer::new(input_dim, hidden_dim)];
        for _ in 0..depth - 2 {
            layers.push(LinearLayer::new(hidden_dim, hidden_dim));
        }
        layers.push(LinearLayer::new(hidden_dim, output_dim));
        MLP { layers }
    }
    pub fn forward(&self, mut x: Tensor) -> Tensor {
        for layer in self.layers.iter() {
            x = layer.forward(x);
        }
        x
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];
        for layer in self.layers.iter() {
            params.extend(layer.parameters())
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_build_mlp() {
        let mlp = MLP::new(3, 10, 20, 1);
        println!("{:?}", mlp);
        let x = Tensor::new(array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].into_dyn());
        let forwarded = mlp.forward(x);
        println!("{:?}", forwarded);
        panic!();
    }
}
