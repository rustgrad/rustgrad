use crate::dimensions::DynamicShape;
use crate::dimensions::Shape;
use crate::dimensions::*;
use crate::ops::matmul::MatMul;
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
    non_linearity: bool,
}
impl<D_IN: Dimension, D_OUT: Dimension> LinearLayer<D_IN, D_OUT> {
    pub fn new(non_linearity: bool) -> LinearLayer<D_IN, D_OUT> {
        let bias = Tensor::new_random(0.0, 0.00);
        let std = (2.0 / (D_IN::default().size() as f32)).sqrt();
        let weight = Tensor::new_random(0.0, std);
        LinearLayer {
            bias,
            weight,
            non_linearity,
        }
    }
    pub fn forward<K: Dimension>(&self, x: Tensor<(K, D_IN)>) -> Tensor<(K, D_OUT)> {
        let x: Tensor<(K, D_OUT)> = x.matmul(self.weight.clone());
        let mut x = x + self.bias.clone();
        if self.non_linearity {
            x = max(x, Tensor::<(K, D_OUT)>::ZERO());
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
            first_layer: LinearLayer::new(true),
            hidden_layers: core::array::from_fn(|_| LinearLayer::new(true)),
            last_layer: LinearLayer::new(false),
        };
    }
    pub fn forward<K: Dimension>(&self, mut x: Tensor<(K, D_IN)>) -> Tensor<(K, D_OUT)> {
        let mut x = self.first_layer.forward(x);
        for layer in self.hidden_layers.iter() {
            x = layer.forward(x);
        }
        self.last_layer.forward(x)
    }
    pub fn parameters(&self) -> Vec<Tensor<DynamicShape>> {
        let mut params = self.first_layer.parameters();
        for layer in self.hidden_layers.iter() {
            params.extend(layer.parameters());
        }
        params.extend(self.last_layer.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use crate::dimensions::S;

    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_layer() {
        let layer_1: LinearLayer<S<10>, S<5>> = LinearLayer::new(true);
        let layer_2: LinearLayer<S<5>, S<1>> = LinearLayer::new(true);
        println!("{:?}", layer_1);
        let x: Tensor<(S<1>, S<10>)> =
            Tensor::new(array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].into_dyn());
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
        let mlp: MLP<S<10>, S<1>, S<20>, 3> = MLP::new();
        println!("{:?}", mlp);
        let x: Tensor<(S<1>, S<10>)> =
            Tensor::new(array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].into_dyn());
        let forwarded = mlp.forward(x);
        println!("{:?}", forwarded);

        panic!();
    }
}
