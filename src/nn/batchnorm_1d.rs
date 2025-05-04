use crate::dimensions::Dimension;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct BatchNorm1d<DIn: Dimension> {
    gamma: Tensor<(DIn,)>,        // Scale parameter
    beta: Tensor<(DIn,)>,         // Shift parameter
    running_mean: Tensor<(DIn,)>, // Running mean for inference
    running_var: Tensor<(DIn,)>,  // Running variance for inference
    momentum: f32,                // Momentum for moving average
    eps: f32,                     // Small epsilon to avoid division by zero
    is_training: bool,            // Whether to use training or inference mode
}

impl<DIn: Dimension> BatchNorm1d<DIn> {
    pub fn new(momentum: f32, eps: f32) -> BatchNorm1d<DIn> {
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

    pub fn forward<B: Dimension>(&mut self, x: Tensor<(B, DIn)>) -> Tensor<(B, DIn)> {
        if self.is_training {
            let mean = x.mean_along::<(DIn,)>(0);
            let variance = x.var_along::<(DIn,)>(0, false);

            self.running_mean =
                self.running_mean.clone() * (1.0 - self.momentum) + mean.clone() * self.momentum;
            self.running_var =
                self.running_var.clone() * (1.0 - self.momentum) + variance.clone() * self.momentum;

            let normalized =
                (x - mean.broadcast_to()) / (variance + self.eps).sqrt().broadcast_to();
            normalized * self.gamma.clone().broadcast_to() + self.beta.clone().broadcast_to()
        } else {
            let normalized = (x - self.running_mean.clone().broadcast_to())
                / (self.running_var.clone() + self.eps).sqrt().broadcast_to();
            normalized * self.gamma.clone().broadcast_to() + self.beta.clone().broadcast_to()
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.gamma.clone_into_dynamic(),
            self.beta.clone_into_dynamic(),
        ]
    }

    pub fn set_training(&mut self, is_training: bool) {
        self.is_training = is_training;
    }
}
