use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{optim::optimizer::Optimizer, tensor::Tensor};

pub(crate) struct AdamOptimizer {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    parameters: Vec<Tensor>,
    m: Vec<ndarray::ArrayD<f32>>, // First moment vector
    v: Vec<ndarray::ArrayD<f32>>, // Second moment vector
    t: u32,                       // Time step
}

impl AdamOptimizer {
    pub fn new(
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        parameters: Vec<Tensor>,
    ) -> AdamOptimizer {
        let m = parameters
            .iter()
            .map(|p| ndarray::ArrayD::zeros(p.shape()))
            .collect();
        let v = parameters
            .iter()
            .map(|p| ndarray::ArrayD::zeros(p.shape()))
            .collect();
        AdamOptimizer {
            lr,
            beta1,
            beta2,
            epsilon,
            parameters,
            m,
            v,
            t: 0,
        }
    }

    pub fn new_with_defaults(lr: f32, parameters: Vec<Tensor>) -> AdamOptimizer {
        let m = parameters
            .iter()
            .map(|p| ndarray::ArrayD::zeros(p.shape()))
            .collect();
        let v = parameters
            .iter()
            .map(|p| ndarray::ArrayD::zeros(p.shape()))
            .collect();
        AdamOptimizer {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            parameters,
            m,
            v,
            t: 0,
        }
    }

    pub fn update_lr(&mut self, new_lr: f32) {
        self.lr = new_lr;
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self) {
        self.t += 1;
        let grads = self
            .parameters
            .iter()
            .map(|param| param.grad().expect("Grad not calculated."))
            .collect::<Vec<_>>();

        let grad_updates = grads
            .into_par_iter()
            .enumerate()
            .map(|(i, grad)| {
                let m = &self.m[i];
                let v = &self.v[i];
                let beta1 = self.beta1;
                let beta2 = self.beta2;
                let epsilon = self.epsilon;
                let t = self.t as f32;

                // Update first moment
                let m_t = beta1 * m + (1.0 - beta1) * &grad;

                // Update second moment
                let v_t = beta2 * v + (1.0 - beta2) * grad.powi(2);

                // Compute bias-corrected first moment
                let m_hat = m_t / (1.0 - beta1.powf(t));

                // Compute bias-corrected second moment
                let v_hat = v_t / (1.0 - beta2.powf(t));

                // Compute update
                -self.lr * m_hat / (v_hat.sqrt() + epsilon)
            })
            .collect::<Vec<_>>();

        // Apply updates to parameters
        grad_updates
            .into_iter()
            .enumerate()
            .for_each(|(i, update)| {
                let param = &self.parameters[i];
                param.add_value(update);
            });
    }

    fn zero_grad(&self) {
        for param in self.parameters.iter() {
            param.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dimensions::S, nn::MLP};

    #[test]
    fn test_adam_optimizer() {
        let mlp: MLP<S<3>, S<2>, S<100>, 1> = MLP::new();
        let mut optimiser = AdamOptimizer::new(0.001, 0.9, 0.999, 1e-8, mlp.parameters());
        let input: Tensor<(S<4>, S<3>)> = Tensor::new_random(0.0, 1.0);
        let forwarded = mlp.forward(input.clone());
        let expected_output: Tensor<(S<4>, S<2>)> = Tensor::new_random(0.0, 1.0);
        let mut loss = -expected_output.clone() + forwarded.clone();
        loss = loss.clone() * loss.clone();
        loss.backward();
        optimiser.step();
        optimiser.zero_grad();
    }
}
