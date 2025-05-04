pub trait Dataset<Sample> {
    /// Returns the number of samples in the dataset.
    fn len(&self) -> usize;

    /// Retrieves a sample by its index.
    fn get_sample(&self, index: usize) -> Sample;
}
