pub mod dimensions;
pub mod nn;
pub mod ops;
pub mod optimiser;
pub mod parallel;
pub mod shape;
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
