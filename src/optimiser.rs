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
}

#[cfg(test)]
mod tests {
    use crate::{
        nn::{LinearLayer, MLP},
        shape::Shape,
    };
    use ndarray::array;

    use super::*;

    #[test]
    fn test_optimizer() {
        let mlp = MLP::new(2, 2, 10, 1);
        // let layer = LinearLayer::new(2, 1);
        let optimiser = SGDOptimizer::new(0.01 as f32, mlp.parameters());
        let epochs = 100;
        for i in 0..epochs {
            optimiser.zero_grad();
            let input = Tensor::new_random(Shape::new([2]));
            let forwarded = mlp.forward(input.clone());
            let mut loss = forwarded.clone() * forwarded.clone();
            loss.backward();
            optimiser.step();
            if i == epochs - 1 {
                println!("last forward {:?}", forwarded);
                println!("layer {:?}", mlp);
                println!("input {:?}", input);
            }
        }
        println!("{:?}", mlp);
    }
}
