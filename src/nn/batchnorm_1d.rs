use crate::dimensions::{Dimension, DynamicShape};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct BatchNorm1D<D_IN: Dimension> {
    gamma: Tensor<(D_IN,)>,        // Scale parameter
    beta: Tensor<(D_IN,)>,         // Shift parameter
    running_mean: Tensor<(D_IN,)>, // Running mean for inference
    running_var: Tensor<(D_IN,)>,  // Running variance for inference
    momentum: f32,                 // Momentum for moving average
    eps: f32,                      // Small epsilon to avoid division by zero
    is_training: bool,             // Whether to use training or inference mode
}

impl<D_IN: Dimension> BatchNorm1D<D_IN> {
    pub fn new(momentum: f32, eps: f32) -> BatchNorm1D<D_IN> {
        let gamma = Tensor::new_random(1.0, 0.0); // Init gamma = 1
        let beta = Tensor::new_random(0.0, 0.0); // Init beta = 0
        let running_mean = Tensor::new_random(0.0, 0.0);
        let running_var = Tensor::new_random(1.0, 0.0);

        Self {
            gamma,
            beta,
            running_mean,
            running_var,
            momentum,
            eps,
            is_training: true,
        }
    }

    pub fn forward<B: Dimension>(&mut self, _x: Tensor<(B, D_IN)>) -> Tensor<(B, D_IN)> {
        unimplemented!();
        // if self.is_training {
        //     let mean = x.mean_along::<(B,)>(0);
        //     let variance = x.var_along::<(B,)>(0, false);

        //     self.running_mean =
        //         self.running_mean.clone() * (1.0 - self.momentum) + mean.clone() * self.momentum;
        //     self.running_var =
        //         self.running_var.clone() * (1.0 - self.momentum) + variance.clone() * self.momentum;

        //     let normalized = (x - mean) / (variance + self.eps).sqrt();
        //     normalized * self.gamma.clone() + self.beta.clone()
        // } else {
        //     let normalized = (x - self.running_mean.clone()) / (self.running_var.clone() + self.eps).sqrt();
        //     normalized * self.gamma.clone() + self.beta.clone()
        // }
    }

    pub fn parameters(&self) -> Vec<Tensor<DynamicShape>> {
        vec![
            self.gamma.clone_into_dynamic(),
            self.beta.clone_into_dynamic(),
        ]
    }

    pub fn set_training(&mut self, is_training: bool) {
        self.is_training = is_training;
    }
}
