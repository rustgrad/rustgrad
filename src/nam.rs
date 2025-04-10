use core::num;
use std::usize;

use crate::dimensions::{Dimension, Dynamic, DynamicShape, Rank2};

use crate::nn::LinearLayer;
use crate::tensor::Tensor;
use crate::{dimensions::S, nn::MLP};
use crate::ops::relu;

pub struct ShapeFunction<D_HIDDEN: Dimension> {
    pub num_layers: usize,
    hidden_layers: Vec<LinearLayer<D_HIDDEN, D_HIDDEN>>,
    first_layer: LinearLayer<S<1>, D_HIDDEN>,
    last_layer: LinearLayer<D_HIDDEN, S<1>>,
}
impl<D_HIDDEN: Dimension> ShapeFunction<D_HIDDEN> {
    pub fn new(num_layers: usize) -> ShapeFunction<D_HIDDEN> {
        return ShapeFunction {num_layers,
            hidden_layers: vec![LinearLayer::new(true); num_layers],
            first_layer: LinearLayer::new(true),
            last_layer: LinearLayer::new(false),
         };
    }
    pub fn forward(&self, x: Tensor<Rank2<Dynamic,S<1>>>) -> Tensor<Rank2<Dynamic, S<1>>> {
        let mut x = self.first_layer.forward(x);
        // TODO: Should activation be applied here?
        for layer in self.hidden_layers.iter() {
            x = layer.forward(x);
            x = x.relu();
        }
        self.last_layer.forward(x)
    }
}

pub struct NAM<D_HIDDEN: Dimension, const NUM_FEATURES: usize> {
    shape_functions: Vec<ShapeFunction<D_HIDDEN>>,
}
impl<D_HIDDEN: Dimension, const NUM_FEATURES: usize> NAM<D_HIDDEN,NUM_FEATURES>
{
    pub fn new(num_layers: usize) -> NAM<D_HIDDEN, NUM_FEATURES> {
        return NAM {
            shape_functions: (0..NUM_FEATURES).map(|_| ShapeFunction::new(num_layers)).collect(),
        };
    }
    pub fn forward<B: Dimension>(&self, x: Tensor<(B, S<NUM_FEATURES>)>) -> Tensor<(B,S<1>)> {
        let mut x = x.clone();
        let result: Tensor<(B,S<1>)> = Tensor::ZERO();
        for i in 0..NUM_FEATURES {
            let shape_function = &self.shape_functions[i];
            result = result.clone() + shape_function.forward(x[i].clone());
        }

        todo!() // most likely we need slicing
    }
    pub fn parameters(&self) -> Vec<Tensor<DynamicShape>> {
        return self
            .shape_functions
            .iter()
            .map(|mlp| mlp.parameters())
            .flatten()
            .collect();
    }
}
