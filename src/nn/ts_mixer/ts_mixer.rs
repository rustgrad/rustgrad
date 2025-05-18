use std::marker::PhantomData;

use crate::{dimensions::Dimension, tensor::Tensor};

use super::MixingLayer;

pub struct TsMixer<L: Dimension, C: Dimension, DOut: Dimension> {
    layers: Vec<MixingLayer<L, C, C>>,
    last_layer: MixingLayer<L, C, DOut>,
    output: PhantomData<DOut>,
}

impl<L: Dimension, C: Dimension, DOut: Dimension> TsMixer<L, C, DOut> {
    pub fn forward<B: Dimension>(&mut self, mut x: Tensor<(B, L, C)>) -> Tensor<(B, L, DOut)> {
        for layer in self.layers.iter_mut() {
            x = layer.forward(x);
        }
        self.last_layer.forward(x)
    }
}
