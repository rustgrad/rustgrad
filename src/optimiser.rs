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
    use ndarray::{array, Array};
    use num_traits::Pow;

    use super::*;

    #[test]
    fn test_optimizer() {
        let mlp = MLP::new(2, 2, 2, 1);
        println!("MLP {:?}", mlp);
        // let layer = LinearLayer::new(2, 1);
        let optimiser = SGDOptimizer::new(0.001 as f32, mlp.parameters());
        let epochs = 1000;
        for i in 0..epochs {
            optimiser.zero_grad();
            let input = Tensor::new_random(Shape::new([2]));
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

            println!("loss {:?}", loss.data());
            if i == epochs - 1 {
                println!("input {:?}", input);
                println!("loss {:?}", loss.data());
                println!("output {:?}", forwarded);
                // println!("layer {:?}", mlp);
                println!("expected_output {:?}", expected_output)
            }
        }
        println!("{:?}", mlp);
    }
}
