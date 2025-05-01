use crate::{dimensions::DynamicShape, optim::optimizer::Optimizer, tensor::Tensor};

pub(crate) struct SGDOptimizer {
    lr: f32,
    parameters: Vec<Tensor<DynamicShape>>,
}

impl SGDOptimizer {
    pub fn new(lr: f32, parameters: Vec<Tensor<DynamicShape>>) -> SGDOptimizer {
        SGDOptimizer { lr, parameters }
    }

    pub fn update_lr(&mut self, new_lr: f32) {
        self.lr = new_lr;
    }
}

impl Optimizer for SGDOptimizer {
    fn step(&mut self) {
        for param in self.parameters.iter() {
            let change = -param.grad().expect("Grad not calculated.") * self.lr;
            param.add_value(change);
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
    use crate::{dimensions::S, nn::MLP};
    use ndarray::Array;

    use super::*;

    #[test]
    fn test_optimizer() {
        let mlp: MLP<S<3>, S<2>, S<100>, 1> = MLP::new();
        println!("MLP {:?}", mlp);
        // let layer = LinearLayer::new(2, 1);
        let mut optimiser = SGDOptimizer::new(0.1 as f32, mlp.parameters());
        let epochs = 1;
        let mut accumulated_loss = 0.0;
        for i in 0..epochs {
            optimiser.zero_grad();
            let input: Tensor<(S<4>, S<3>)> = Tensor::new_random(0.0, 1.0);
            let forwarded = mlp.forward(input.clone());
            let expected_output = input.data();
            let expected_output = expected_output[0] - expected_output[1];
            let expected_output: Tensor<(S<4>, S<2>)> = Tensor::new(
                Array::from_vec(vec![expected_output])
                    .into_shape_clone(forwarded.shape())
                    .unwrap(),
            );
            let mut loss = -expected_output.clone() + forwarded.clone();
            loss = loss.clone() * loss.clone();
            loss.backward();
            // println!("network {:?}", mlp);
            optimiser.step();

            accumulated_loss += loss.data().sum();
            println!("{:?}", loss);
            if i % 1000 == 0 {
                println!("loss {:?}", accumulated_loss / 1000.0);
                // println!("input {:?}", input);
                // println!("loss {:?}", loss.data());
                println!("output {:?}", forwarded.data());
                // println!("layer {:?}", mlp);
                println!("expected_output {:?}", expected_output.data());
                println!("lr: {:?}", optimiser.lr);
                println!("===============================");
                accumulated_loss = 0.0;
            }
            optimiser.update_lr(0.01);
        }
        println!("{:?}", mlp);
    }
}
