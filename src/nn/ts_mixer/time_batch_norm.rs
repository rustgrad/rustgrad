use crate::dimensions::{Dimension, Dynamic};
use crate::nn::BatchNorm1d;
use crate::tensor::Tensor;

/// A batch normalization layer that normalizes over the last two dimensions (time & feature).
#[derive(Debug, Clone)]
pub struct TimeBatchNorm<DT: Dimension, DC: Dimension> {
    batch_norm: BatchNorm1d<Dynamic>,
    _phantom: std::marker::PhantomData<(DT, DC)>,
}

impl<DT: Dimension, DC: Dimension> TimeBatchNorm<DT, DC> {
    pub fn new() -> Self {
        Self {
            batch_norm: BatchNorm1d::new(0.99, 1e-5),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn forward<DB: Dimension>(&mut self, x: Tensor<(DB, DT, DC)>) -> Tensor<(DB, DT, DC)> {
        let dims: [usize; 3] = x.shape().dims();
        let dimensions: (DB, Dynamic) = (
            DB::from_size(dims[0]),
            Dynamic::from_size(dims[1] * dims[2]),
        );
        let x_reshaped = x.reshape::<(DB, Dynamic)>(dimensions); // shape: (N, T * C)
        let x_normed = self.batch_norm.forward(x_reshaped);
        x_normed.reshape(x.runtime_shape()) // shape: (N, T, C)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.batch_norm.parameters()
    }
}
