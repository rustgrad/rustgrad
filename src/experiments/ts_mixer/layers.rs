use ndarray::Array;
use ndarray_rand::rand::{thread_rng, Rng};

use crate::dimensions::{Dimension, Shape, UnkownShape};
use crate::experiments::ts_mixer::TimeBatchNorm;
use crate::nn::linear::LinearLayer;
use crate::ops::max;
use crate::tensor::{ShapeCompatible, Tensor};

#[derive(Debug, Clone)]
pub struct Dropout {
    rate: f32,
    is_training: bool,
}
impl Dropout {
    pub fn forward<S: Shape + ShapeCompatible<S>>(&self, x: Tensor<S>) -> Tensor<S>
    where
        S: ShapeCompatible<S>,
    {
        // Dropout logic would go here
        if self.is_training {
            let mut rng = thread_rng();
            let arr = Array::from_shape_fn(x.shape(), |_| {
                if rng.gen::<f32>() > self.rate {
                    1.0 as f32
                } else {
                    0.0 as f32
                }
            })
            .into_dyn();

            let mask: Tensor<S> = Tensor::new(arr);
            (mask * x).reshape(S::default())
        } else {
            x
        }
    }

    pub fn set_training(&mut self, value: bool) {
        self.is_training = value
    }
}

#[derive(Debug, Clone)]
pub struct TimeMixing<Dl: Dimension, Dc: Dimension> {
    norm: TimeBatchNorm<Dl, Dc>,
    linear: LinearLayer<Dl, Dl>,
    non_linearity: bool,
    dropout: Dropout,
}

impl<Dl: Dimension, Dc: Dimension> TimeMixing<Dl, Dc> {
    pub fn new(non_linearity: bool, dropout_rate: f32) -> Self {
        Self {
            norm: TimeBatchNorm::new(),
            linear: LinearLayer::new(non_linearity),
            non_linearity,
            dropout: Dropout {
                rate: dropout_rate,
                is_training: true,
            },
        }
    }

    pub fn forward<Db: Dimension>(&mut self, x: Tensor<(Db, Dl, Dc)>) -> Tensor<(Db, Dl, Dc)> {
        let x_temp = feature_to_time(x.clone());

        let x_temp = self.linear.forward2b(x_temp);
        let x_temp = if self.non_linearity {
            max(x_temp.clone(), Tensor::zero())
        } else {
            x_temp
        };

        let x_temp = self.dropout.forward(x_temp.clone());

        let x_res = time_to_feature(x_temp);
        self.norm.forward(x + x_res) // Residual + Norm
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];
        params.extend(self.linear.parameters());
        params.extend(self.norm.parameters());
        params
    }
}

struct FeatureMixing<Dl: Dimension, Dc: Dimension, DOut: Dimension> {
    norm: TimeBatchNorm<Dl, DOut>,
    linear: LinearLayer<Dc, DOut>,
    project_layer: LinearLayer<Dc, DOut>,
    dropout: Dropout,
}
impl<Dl: Dimension, Dc: Dimension, DOut: Dimension> FeatureMixing<Dl, Dc, DOut> {
    pub fn new(non_linearity: bool, dropout_rate: f32) -> Self {
        Self {
            norm: TimeBatchNorm::new(),
            linear: LinearLayer::new(non_linearity),
            project_layer: LinearLayer::new(false),
            dropout: Dropout {
                rate: dropout_rate,
                is_training: true,
            },
        }
    }

    pub fn forward<Db: Dimension>(&mut self, x: Tensor<(Db, Dl, Dc)>) -> Tensor<(Db, Dl, DOut)> {
        let x_project = self.project_layer.forward2b(x.clone());
        let x_temp = self.linear.forward2b(x);

        let x_temp = self.dropout.forward(x_temp);

        self.norm.forward(x_temp + x_project) // Residual + Norm
    }

    pub fn parameters(&self) -> Vec<Tensor<UnkownShape>> {
        let mut params = vec![];
        params.extend(self.linear.parameters());
        params.extend(self.norm.parameters());
        params
    }
}

fn feature_to_time<Db: Dimension, Dc: Dimension, Dl: Dimension>(
    x: Tensor<(Db, Dc, Dl)>,
) -> Tensor<(Db, Dl, Dc)> {
    return x.permute(vec![0, 2, 1]); // (N, L, C) → (N, C, L)
}
fn time_to_feature<Db: Dimension, Dc: Dimension, Dl: Dimension>(
    x: Tensor<(Db, Dl, Dc)>,
) -> Tensor<(Db, Dc, Dl)> {
    return feature_to_time(x);
}

pub struct MixingLayer<Dl: Dimension, Dc: Dimension, DOut: Dimension> {
    time_mixing: TimeMixing<Dl, Dc>,
    feature_mixing: FeatureMixing<Dl, Dc, DOut>,
}
impl<Dl: Dimension, Dc: Dimension, DOut: Dimension> MixingLayer<Dl, Dc, DOut> {
    pub fn new(non_linearity: bool, dropout_rate: f32) -> Self {
        Self {
            time_mixing: TimeMixing::new(non_linearity, dropout_rate),
            feature_mixing: FeatureMixing::new(non_linearity, dropout_rate),
        }
    }

    pub fn forward<Db: Dimension>(&mut self, x: Tensor<(Db, Dl, Dc)>) -> Tensor<(Db, Dl, DOut)> {
        let x = self.time_mixing.forward(x.clone());
        self.feature_mixing.forward(x)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];
        params.extend(self.time_mixing.parameters());
        params.extend(self.feature_mixing.parameters());
        params
    }
}
