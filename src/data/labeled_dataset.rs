use crate::{
    dimensions::{Dimension, Shape, UnkownShape},
    ops::TensorStack,
};
use std::marker::PhantomData;

use super::{DataLoader, Dataset};
use crate::tensor::Tensor;

/// Trait that enforces shape compatibility between sample shapes and batch shapes
pub trait BatchCompatible<SampleShape: Shape> {
    type Output: Shape;
}

// Implementations for different shape combinations
impl<B: Dimension, I: Dimension> BatchCompatible<(I,)> for B {
    type Output = (B, I);
}

impl<B: Dimension, I: Dimension, J: Dimension> BatchCompatible<(I, J)> for B {
    type Output = (B, I, J);
}

impl<B: Dimension, I: Dimension, J: Dimension, K: Dimension> BatchCompatible<(I, J, K)> for B {
    type Output = (B, I, J, K);
}

// Allow dynamic shapes as a fallback
impl BatchCompatible<UnkownShape> for UnkownShape {
    type Output = UnkownShape;
}

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

pub struct LabeledTensorDataLoader<I, L, B>
where
    I: Shape,
    L: Shape,
    B: Dimension + BatchCompatible<I> + BatchCompatible<L>,
{
    pub dataset: Vec<LabeledTensorSample<I, L>>,
    _phantom: PhantomData<B>,
}

impl<I, L, B> LabeledTensorDataLoader<I, L, B>
where
    I: Shape,
    L: Shape,
    B: Dimension + BatchCompatible<I> + BatchCompatible<L>,
{
    pub fn new(dataset: Vec<LabeledTensorSample<I, L>>) -> Self {
        Self {
            dataset,
            _phantom: PhantomData,
        }
    }
}

impl<I, L, B>
    DataLoader<
        Vec<LabeledTensorSample<I, L>>,
        LabeledTensorSample<I, L>,
        LabeledTensorSample<<B as BatchCompatible<I>>::Output, <B as BatchCompatible<L>>::Output>,
    > for LabeledTensorDataLoader<I, L, B>
where
    I: Shape,
    L: Shape,
    B: Dimension + BatchCompatible<I> + BatchCompatible<L>,
{
    fn get_dataset(&self) -> &Vec<LabeledTensorSample<I, L>> {
        &self.dataset
    }

    fn get_batch_size(&self) -> usize {
        B::default().size()
    }

    fn collate(
        &self,
        samples: Vec<LabeledTensorSample<I, L>>,
    ) -> LabeledTensorSample<<B as BatchCompatible<I>>::Output, <B as BatchCompatible<L>>::Output>
    {
        let inputs = samples.iter().map(|s| s.input.clone()).collect();
        let labels = samples.iter().map(|s| s.label.clone()).collect();

        LabeledTensorSample {
            input: TensorStack::forward(inputs, 0),
            label: TensorStack::forward(labels, 0),
        }
    }
}
