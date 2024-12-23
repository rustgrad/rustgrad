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
            let change = param.data() + param.grad().expect("Grad not calculated.") * self.lr;
            param.add_value(change);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::MLP;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_build_mlp() {
        let mlp = MLP::new(3, 10, 10, 10);
        let optimiser = SGDOptimizer::new(0.01, mlp.parameters());
        println!("{:?}", mlp);
        let x = Tensor::new(array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].into_dyn());
        let mut forwarded = mlp.forward(x);
        println!("{:?}", forwarded);
        forwarded.backward();
        optimiser.step();
        println!("{:?}", mlp);
        panic!();
    }
}
