use crate::tensor::Tensor;

struct SGDOptimizer {
    lr: f32,
    parameters: Vec<Tensor>,
}
impl SGDOptimizer {
    pub fn new(lr: f32, parameters: Vec<Tensor>) -> SGDOptimizer {
        SGDOptimizer { lr, parameters }
    }
    pub fn step(&self) {
        for param in self.parameters.iter() {
            let change = -param.grad().expect("Grad not calculated.") * self.lr;
            param.add_value(change);
        }
    }
    pub fn zero_grad(&self) {
        for param in self.parameters.iter() {
            param.zero_grad();
        }
    }

    pub fn update_lr(&mut self) {
        self.lr *= 0.9999;
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        nn::{LinearLayer, MLP},
        shape::ArrayShape,
    };
    use ndarray::{array, Array};
    use num_traits::Pow;

    use super::*;

    #[test]
    fn test_optimizer() {
        let mlp = MLP::new(3, 2, 100, 1);
        println!("MLP {:?}", mlp);
        // let layer = LinearLayer::new(2, 1);
        let mut optimiser = SGDOptimizer::new(0.1 as f32, mlp.parameters());
        let epochs = 1;
        let mut accumulated_loss = 0.0;
        for i in 0..epochs {
            optimiser.zero_grad();
            let input = Tensor::new_random(ArrayShape::new([2]), 0.0, 1.0);
            let forwarded = mlp.forward(input.clone());
            let expected_output = input.data();
            let expected_output = expected_output[0] - expected_output[1];
            let expected_output = Tensor::new(
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
            optimiser.update_lr();
        }
        println!("{:?}", mlp);
    }
}
