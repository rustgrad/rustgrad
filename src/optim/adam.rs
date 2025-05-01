use crate::{dimensions::DynamicShape, optim::optimizer::Optimizer, tensor::Tensor};

pub(crate) struct AdamOptimizer {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    parameters: Vec<Tensor<DynamicShape>>,
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
        parameters: Vec<Tensor<DynamicShape>>,
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

    pub fn new_with_defaults(lr: f32, parameters: Vec<Tensor<DynamicShape>>) -> AdamOptimizer {
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
        for (i, param) in self.parameters.iter().enumerate() {
            let grad = param.grad().expect("Grad not calculated.");
            self.m[i] = self.beta1 * &self.m[i] + (1.0 - self.beta1) * &grad;
            self.v[i] = self.beta2 * &self.v[i] + (1.0 - self.beta2) * grad.powi(2);

            let m_hat = &self.m[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = &self.v[i] / (1.0 - self.beta2.powi(self.t as i32));

            let update = -self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            param.add_value(update);
        }
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
