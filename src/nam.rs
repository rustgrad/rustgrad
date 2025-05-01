use std::usize;

use crate::dimensions::{Dimension, DynamicShape};

use crate::dimensions::S;
use crate::nn::LinearLayer;
use crate::tensor::Tensor;

pub struct ShapeFunction<DHidden: Dimension> {
    pub num_layers: usize,
    hidden_layers: Vec<LinearLayer<DHidden, DHidden>>,
    first_layer: LinearLayer<S<1>, DHidden>,
    last_layer: LinearLayer<DHidden, S<1>>,
}
impl<DHidden: Dimension> ShapeFunction<DHidden> {
    pub fn new(num_layers: usize) -> ShapeFunction<DHidden> {
        return ShapeFunction {
            num_layers,
            hidden_layers: vec![LinearLayer::new(true); num_layers],
            first_layer: LinearLayer::new(true),
            last_layer: LinearLayer::new(false),
        };
    }
    pub fn forward<K: Dimension>(&self, x: Tensor<(K, S<1>)>) -> Tensor<(K, S<1>)> {
        let mut x = self.first_layer.forward(x);
        // TODO: Should activation be applied here?
        for layer in self.hidden_layers.iter() {
            x = layer.forward(x);
            x = x.relu();
        }
        self.last_layer.forward(x)
    }
    pub fn parameters(&self) -> Vec<Tensor<DynamicShape>> {
        let hidden_params = self
            .hidden_layers
            .iter()
            .map(|layer| layer.parameters())
            .flatten();

        let params = hidden_params.chain(self.first_layer.parameters());
        let params = params.chain(self.last_layer.parameters());
        return params.collect();
    }
}

pub struct NAM<DHidden: Dimension, const NUM_FEATURES: usize> {
    shape_functions: Vec<ShapeFunction<DHidden>>,
}
impl<DHidden: Dimension, const NUM_FEATURES: usize> NAM<DHidden, NUM_FEATURES> {
    pub fn new(num_layers: usize) -> NAM<DHidden, NUM_FEATURES> {
        return NAM {
            shape_functions: (0..NUM_FEATURES)
                .map(|_| ShapeFunction::new(num_layers))
                .collect(),
        };
    }
    pub fn forward<B: Dimension>(&self, x: Tensor<(B, S<NUM_FEATURES>)>) -> Tensor<(B, S<1>)> {
        let x = x.clone();
        let mut result: Tensor<(B, S<1>)> = Tensor::zero();
        for i in 0..NUM_FEATURES {
            let shape_function = &self.shape_functions[i];
            let shape_input = x.slice(1, i).clone();
            let shape_output = shape_function.forward(shape_input);
            result = result.clone() + shape_output;
        }
        return result;
    }
    pub fn parameters(&self) -> Vec<Tensor<DynamicShape>> {
        return self
            .shape_functions
            .iter()
            .map(|shape_function| shape_function.parameters())
            .flatten()
            .collect();
    }
}

#[test]
fn test_nam_learns_sum_function() {
    use crate::dimensions::S;
    use crate::nam::NAM;
    use crate::optim::adam::AdamOptimizer;
    use crate::optim::optimizer::Optimizer;
    use crate::tensor::Tensor;
    use ndarray::Array2;

    const NUM_FEATURES: usize = 3;
    const BATCH_SIZE: usize = 4;

    // Create synthetic input data (4 samples, 3 features)
    let x_data = Array2::from_shape_vec(
        (BATCH_SIZE, NUM_FEATURES),
        vec![
            1.0, 2.0, 3.0, // sum = 6.0
            0.0, 0.0, 1.0, // sum = 1.0
            1.0, 1.0, 1.0, // sum = 3.0
            2.0, 2.0, 2.0, // sum = 6.0
        ],
    )
    .unwrap();

    let y_data = Array2::from_shape_vec((BATCH_SIZE, 1), vec![6.0, 1.0, 3.0, 6.0]).unwrap();

    let x = Tensor::<(S<BATCH_SIZE>, S<NUM_FEATURES>)>::new(x_data.into_dyn());
    let y_true = Tensor::<(S<BATCH_SIZE>, S<1>)>::new(y_data.into_dyn());

    let model = NAM::<S<32>, NUM_FEATURES>::new(2); // Example with 2 hidden layers

    let params = model.parameters();
    let mut opt = AdamOptimizer::new_with_defaults(0.0001, params);

    for epoch in 0..1000 {
        let y_pred = model.forward(x.clone());

        let diff = y_pred.clone() + -y_true.clone();
        let loss = diff.clone() * diff; // squared error
        let loss = loss.mean(); // MSE

        opt.zero_grad();
        loss.backward();
        opt.step();

        if epoch % 100 == 0 {
            println!("Epoch {epoch}, Loss: {:?}", loss.data());
        }
    }

    // Final check: model output should be close to expected values
    let y_pred = model.forward(x.clone());
    let y_pred_data = y_pred.data();

    let expected = y_true.data();
    let diff = &y_pred_data - &expected;
    println!("Predicted: {:?}", y_pred_data);
    println!("Expected: {:?}", expected);
    println!("Diff: {:?}", diff);
    let max_diff = diff.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));

    assert!(
        !y_pred.data().is_any_nan() && !y_pred.data().is_any_infinite(),
        "Model output contains NaN or Inf values"
    );
    assert!(
        max_diff < 0.005,
        "Model did not learn sum function well enough"
    );
}
