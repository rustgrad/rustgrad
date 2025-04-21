use std::marker::PhantomData;

use crate::{dimensions::Dimension, tensor::Tensor};

use super::MixingLayer;

pub struct TsMixer<L: Dimension, C: Dimension, DOut: Dimension, Dlc: Dimension, Dlo: Dimension> {
    layers: Vec<MixingLayer<L, C, C, Dlc, Dlo>>,
    last_layer: MixingLayer<L, C, DOut, Dlc, Dlo>,
    output: PhantomData<DOut>,
}

impl<L: Dimension, C: Dimension, DOut: Dimension, Dlc: Dimension, Dlo: Dimension>
    TsMixer<L, C, DOut, Dlc, Dlo>
{
    pub fn forward<B: Dimension>(&mut self, mut x: Tensor<(B, L, C)>) -> Tensor<(B, L, DOut)> {
        for layer in self.layers.iter_mut() {
            x = layer.forward(x);
        }
        self.last_layer.forward(x)
    }
}
