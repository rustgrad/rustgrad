use crate::data::dataset::Dataset;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};

pub trait DataLoader<D, Sample, Batch>
where
    D: Dataset<Sample>,
{
    fn get_dataset(&self) -> &D;
    fn get_batch_size(&self) -> usize;
    fn collate(&self, samples: Vec<Sample>) -> Batch;
}

/// An iterator over batches from a DataLoader
pub struct DataLoaderIterator<'a, D, Sample, Batch>
where
    D: Dataset<Sample>,
{
    loader: &'a dyn DataLoader<D, Sample, Batch>,
    indices: Vec<usize>,
    current_index: usize,
}

impl<'a, D, Sample, Batch> Iterator for DataLoaderIterator<'a, D, Sample, Batch>
where
    D: Dataset<Sample>,
{
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've reached the end
        if self.current_index >= self.indices.len() {
            return None;
        }

        // Calculate how many samples we can retrieve
        let remaining = self.indices.len() - self.current_index;

        let batch_size = self.loader.get_batch_size();
        if remaining < batch_size || batch_size == 0 {
            return None;
        }

        // Collect samples according to our indices
        let batch: Vec<Sample> = (0..batch_size)
            .filter_map(|i| {
                let idx = self.indices[self.current_index + i];
                Some(self.loader.get_dataset().get_sample(idx))
            })
            .collect();

        self.current_index += batch_size;

        Some(self.loader.collate(batch))
    }
}

pub trait DataLoaderExt<D, Sample, Batch>
where
    D: Dataset<Sample>,
{
    fn iter(&self) -> DataLoaderIterator<'_, D, Sample, Batch>;
    fn iter_shuffled(&mut self, seed: u64) -> DataLoaderIterator<'_, D, Sample, Batch>;
}

impl<D, Sample, Batch, T: DataLoader<D, Sample, Batch>> DataLoaderExt<D, Sample, Batch> for T
where
    D: Dataset<Sample>,
{
    /// Creates an iterator over the dataset in sequential order
    fn iter(&self) -> DataLoaderIterator<'_, D, Sample, Batch> {
        DataLoaderIterator {
            loader: self,
            indices: (0..self.get_dataset().len()).collect(),
            current_index: 0,
        }
    }

    /// Creates an iterator over the dataset with shuffled indices
    fn iter_shuffled(&mut self, seed: u64) -> DataLoaderIterator<'_, D, Sample, Batch> {
        // Create a shuffled copy of the indices
        let mut shuffled_indices: Vec<usize> = (0..self.get_dataset().len()).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        shuffled_indices.shuffle(&mut rng);

        DataLoaderIterator {
            loader: self,
            indices: shuffled_indices,
            current_index: 0,
        }
    }
}
