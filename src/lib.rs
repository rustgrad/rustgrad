pub mod array_shape;
pub mod data;
pub mod dimensions;
pub mod experiments;
pub mod nam;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod parallel;
pub mod sharing;
pub mod tensor;
// pub mod broadcasts;
// pub mod axes;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
