use std::usize;

use crate::dimensions::{Dimension, DynamicShape};

use crate::tensor::Tensor;
use crate::{dimensions::S, nn::MLP};

pub struct NAM<
    const D_IN: usize,
    D_OUT: Dimension,
    HIDDEN_DIM: Dimension,
    const HIDDEN_LAYERS: usize,
> {
    mlps: [MLP<S<1>, D_OUT, HIDDEN_DIM, HIDDEN_LAYERS>; D_IN],
}
impl<const D_IN: usize, D_OUT: Dimension, HIDDEN_DIM: Dimension, const HIDDEN_LAYERS: usize>
    NAM<D_IN, D_OUT, HIDDEN_DIM, HIDDEN_LAYERS>
{
    pub fn new() -> NAM<D_IN, D_OUT, HIDDEN_DIM, HIDDEN_LAYERS> {
        return NAM {
            mlps: core::array::from_fn(|_| MLP::new()),
        };
    }
    pub fn forward<K: Dimension>(&self, x: Tensor<(K, S<D_IN>)>) -> Tensor<(K, D_OUT)> {
        todo!() // most likely we need slicing
    }
    pub fn parameters(&self) -> Vec<Tensor<DynamicShape>> {
        return self
            .mlps
            .iter()
            .map(|mlp| mlp.parameters())
            .flatten()
            .collect();
    }
}
