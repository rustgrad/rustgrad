use crate::{dimensions::Shape, ops::TensorStack};

use super::{DataLoader, Dataset};
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct LabeledTensorSample<I, L>
where
    I: Shape,
    L: Shape,
{
    pub input: Tensor<I>,
    pub label: Tensor<L>,
}

impl<I, L> Dataset<LabeledTensorSample<I, L>> for Vec<LabeledTensorSample<I, L>>
where
    I: Shape,
    L: Shape,
{
    fn len(&self) -> usize {
        self.len()
    }

    fn get_sample(&self, index: usize) -> LabeledTensorSample<I, L> {
        self[index].clone()
    }
}

pub struct LabeledTensorDataLoader<I, L>
where
    I: Shape,
    L: Shape,
{
    pub dataset: Vec<LabeledTensorSample<I, L>>,
}

impl<I, L, BI, BL>
    DataLoader<
        Vec<LabeledTensorSample<I, L>>,
        LabeledTensorSample<I, L>,
        LabeledTensorSample<BI, BL>,
    > for LabeledTensorDataLoader<I, L>
where
    I: Shape,
    L: Shape,
    BI: Shape,
    BL: Shape,
{
    fn get_dataset(&self) -> &Vec<LabeledTensorSample<I, L>> {
        &self.dataset
    }

    fn get_batch_size(&self) -> usize {
        BI::shape()
            .dims
            .get(0)
            .expect("Empty output shape is invalid for batches.")
            .to_owned()
    }

    fn collate(&self, samples: Vec<LabeledTensorSample<I, L>>) -> LabeledTensorSample<BI, BL> {
        let inputs = samples.iter().map(|s| s.input.clone()).collect();
        let labels = samples.iter().map(|s| s.label.clone()).collect();

        LabeledTensorSample {
            input: TensorStack::forward(inputs, 0),
            label: TensorStack::forward(labels, 0),
        }
    }
}
