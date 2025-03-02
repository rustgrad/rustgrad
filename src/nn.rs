use crate::shape::ArrayShape;
use crate::tensor::max;
use crate::tensor::Tensor;

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
    non_linearity: bool,
}
impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize, non_linearity: bool) -> LinearLayer {
        let bias = Tensor::new_random(ArrayShape::new([1, output_dim]), 0.0, 0.00);
        let std = (2.0 / (input_dim as f32)).sqrt();
        let weight = Tensor::new_random(ArrayShape::new([input_dim, output_dim]), 0.0, std);
        LinearLayer {
            bias,
            weight,
            input_dim,
            output_dim,
            non_linearity,
        }
    }
    pub fn forward(&self, x: Tensor) -> Tensor {
        let mut x = x.reshape(ArrayShape::new([1, self.input_dim]));
        x = x.dot(self.weight.clone()) + self.bias.clone();
        if self.non_linearity {
            x = max(x, Tensor::ZERO(ArrayShape::new([1, self.output_dim])));
        }
        x
    }
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

impl MLP {
    pub fn new(depth: usize, input_dim: usize, hidden_dim: usize, output_dim: usize) -> MLP {
        if depth == 1 {
            return MLP {
                layers: vec![LinearLayer::new(input_dim, output_dim, false)],
            };
        }

        let mut layers = vec![LinearLayer::new(input_dim, hidden_dim, true)];
        for _ in 0..depth - 2 {
            layers.push(LinearLayer::new(hidden_dim, hidden_dim, true));
        }
        layers.push(LinearLayer::new(hidden_dim, output_dim, false));
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
    fn test_linear_layer() {
        let layer_1 = LinearLayer::new(10, 5, true);
        let layer_2 = LinearLayer::new(5, 1, true);
        println!("{:?}", layer_1);
        let x = Tensor::new(array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].into_dyn());
        let mut x = layer_1.forward(x);
        println!("{:?}", x);
        let mut x = layer_2.forward(x);
        println!("{:?}", x);
        x.backward();
        println!("layer_1 {:?}", layer_1);
        println!("layer_2 {:?}", layer_1);
    }

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
