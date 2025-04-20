use crate::dimensions::Dimension;
use crate::dimensions::DynamicShape;
use crate::ops::matmul::MatMul;
use crate::ops::max;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct LinearLayer<DIn: Dimension, D_OUT: Dimension> {
    weight: Tensor<(DIn, D_OUT)>,
    bias: Tensor<(D_OUT,)>,
    non_linearity: bool,
}
impl<DIn: Dimension, D_OUT: Dimension> LinearLayer<DIn, D_OUT> {
    pub fn new(non_linearity: bool) -> LinearLayer<DIn, D_OUT> {
        let bias = Tensor::new_random(0.0, 0.00);
        let std = (2.0 / (DIn::default().size() as f32)).sqrt();
        let weight = Tensor::new_random(0.0, std);
        LinearLayer {
            bias,
            weight,
            non_linearity,
        }
    }
    pub fn forward<K: Dimension>(&self, x: Tensor<(K, DIn)>) -> Tensor<(K, D_OUT)> {
        let x: Tensor<(K, D_OUT)> = x.matmul(self.weight.clone());
        let mut x = x + self.bias.broadcast_to();
        if self.non_linearity {
            x = max(x, Tensor::<(K, D_OUT)>::zero());
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
