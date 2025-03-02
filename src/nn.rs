use crate::dimensions::Dimension;
use crate::dimensions::DynamicShape;
use crate::dimensions::Shape;
use crate::matmul::MatMul;
use crate::shape::ArrayShape;
use crate::tensor::max;
use crate::tensor::Tensor;
#[derive(Debug)]
pub struct MLP<D_IN: Dimension, D_OUT: Dimension, HIDDEN_DIM: Dimension, const HIDDEN_LAYERS: usize>
{
    first_layer: LinearLayer<D_IN, HIDDEN_DIM>,
    hidden_layers: [LinearLayer<HIDDEN_DIM, HIDDEN_DIM>; HIDDEN_LAYERS],
    last_layer: LinearLayer<HIDDEN_DIM, D_OUT>,
}

#[derive(Debug)]
pub struct LinearLayer<D_IN: Dimension, D_OUT: Dimension> {
    weight: Tensor<(D_IN, D_OUT)>,
    bias: Tensor<(D_OUT,)>,
    input_dim: usize,
    output_dim: usize,
    non_linearity: bool,
}
impl<D_IN: Dimension, D_OUT: Dimension> LinearLayer<D_IN, D_OUT> {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        non_linearity: bool,
    ) -> LinearLayer<D_IN, D_OUT> {
        let bias = Tensor::new_random(0.0, 0.00);
        let std = (2.0 / (input_dim as f32)).sqrt();
        let weight = Tensor::new_random(0.0, std);
        LinearLayer {
            bias,
            weight,
            input_dim,
            output_dim,
            non_linearity,
        }
    }
    pub fn forward<K: Dimension>(&self, x: Tensor<(K, D_IN)>) -> Tensor<(K, D_OUT)> {
        let x: Tensor<(K, D_OUT)> = x.matmul(self.weight.clone());
        let x = x + self.bias.clone();
        if self.non_linearity {
            x = max(x, Tensor::ZERO::<(K, D_OUT)>());
        }
        x
    }
    pub fn parameters(&self) -> Vec<Tensor<DynamicShape>> {
        vec![
            self.weight.clone_into_dynamic(),
            self.bias.clone_into_dynamic(),
        ]
    }
}

impl<D_IN: Dimension, D_OUT: Dimension, HIDDEN_DIM: Dimension, const HIDDEN_LAYERS: usize>
    MLP<D_IN, D_OUT, HIDDEN_DIM, HIDDEN_LAYERS>
{
    pub fn new() -> MLP<D_IN, D_OUT, HIDDEN_DIM, HIDDEN_LAYERS> {
        return MLP {
            first_layer: LinearLayer::new(
                D_IN::default().size(),
                HIDDEN_DIM::default().size(),
                true,
            ),
            hidden_layers: [LinearLayer::new(
                HIDDEN_DIM::default().size(),
                HIDDEN_DIM::default().size(),
                true,
            )],
            last_layer: LinearLayer::new(
                HIDDEN_DIM::default().size(),
                D_OUT::default().size(),
                false,
            ),
        };
    }
    pub fn forward<K: Dimension>(&self, mut x: Tensor<(K, D_IN)>) -> Tensor<(K, D_OUT)> {
        x = self.first_layer.forward(x);
        for layer in self.hidden_layers.iter() {
            x = layer.forward(x);
        }
        self.last_layer.forward(x)
    }
    pub fn parameters(&self) -> Vec<Box<Tensor<impl Shape>>> {
        let mut params = vec![];
        for layer in self.first_layer.parameters() {
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
