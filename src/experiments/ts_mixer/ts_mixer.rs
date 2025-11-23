use std::marker::PhantomData;

use crate::{dimensions::Dimension, tensor::Tensor};

use super::MixingLayer;

pub struct TsMixer<L: Dimension, C: Dimension, DOut: Dimension, const HIDDEN_LAYERS: usize> {
    layers: [MixingLayer<L, C, C>; HIDDEN_LAYERS],
    last_layer: MixingLayer<L, C, DOut>,
}
impl<L: Dimension, C: Dimension, DOut: Dimension, const HIDDEN_LAYERS: usize>
    TsMixer<L, C, DOut, HIDDEN_LAYERS>
{
    pub fn new(dropout_rate: f32) -> TsMixer<L, C, DOut, HIDDEN_LAYERS> {
        return TsMixer {
            last_layer: MixingLayer::new(false, 0.0),
            layers: core::array::from_fn(|_| MixingLayer::new(true, dropout_rate)),
        };
    }
    pub fn forward<B: Dimension>(&mut self, mut x: Tensor<(B, L, C)>) -> Tensor<(B, L, DOut)> {
        for layer in self.layers.iter_mut() {
            x = layer.forward(x);
        }
        self.last_layer.forward(x)
    }
}
#[cfg(test)]
mod ts_mixer_tests {
    use crate::dimensions::S;

    use super::TsMixer;

    #[test]
    fn test_ts_mixer() {
        // let ts_mixer: TsMixer<S<5>, S<4>, S<1>, 6> = TsMixer::new(0.01);
        // println!("{:?}", ts_mixer);
    }
}
