use crate::dimensions::Dimension;
use crate::nn::BatchNorm1d;
use crate::tensor::Tensor;

/// A batch normalization layer that normalizes over the last two dimensions (time & feature).
#[derive(Debug, Clone)]
pub struct TimeBatchNorm<DT: Dimension, DC: Dimension, Dtc: Dimension> {
    batch_norm: BatchNorm1d<Dtc>,
    _phantom: std::marker::PhantomData<(DT, DC)>,
}

impl<DT: Dimension, DC: Dimension, Dtc: Dimension> TimeBatchNorm<DT, DC, Dtc> {
    pub fn new() -> Self {
        Self {
            batch_norm: BatchNorm1d::new(0.99, 1e-5),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn forward<DB: Dimension>(&mut self, x: Tensor<(DB, DT, DC)>) -> Tensor<(DB, DT, DC)> {
        let x_reshaped = x.reshape::<(DB, Dtc)>(); // shape: (N, T * C)
        let x_normed = self.batch_norm.forward(x_reshaped);
        x_normed.reshape()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.batch_norm.parameters()
    }
}
