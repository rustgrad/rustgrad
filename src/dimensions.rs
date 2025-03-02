use std::{fmt::Debug, marker::PhantomData};

use crate::shape::ArrayShape;

#[derive(Debug)]
pub enum DimKind {
    Static,
    Dynamic,
}
pub trait DimCompatible<Rhs: Dimension> {
    type Output: Dimension;
}
impl<const M: usize> DimCompatible<Dynamic> for S<M> {
    type Output = Dynamic;
}
impl<const M: usize> DimCompatible<S<M>> for Dynamic {
    type Output = Dynamic;
}
impl<const M: usize> DimCompatible<S<M>> for S<M> {
    type Output = S<M>;
}

#[derive(Debug, Eq, Clone, Copy)]
pub struct Dynamic(usize);
impl Dimension for Dynamic {
    fn size(&self) -> usize {
        self.0
    }
    fn from_size(size: usize) -> Option<Self> {
        if size == usize::MAX {
            Some(Dynamic(usize::MAX))
        } else {
            None
        }
    }
}
impl PartialEq for Dynamic {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Default for Dynamic {
    fn default() -> Self {
        Dynamic(usize::MAX)
    }
}

pub trait Shape: Debug + Default + Clone + 'static {
    fn shape(&self) -> ArrayShape;
    const NUM_DIMS: usize;
    // const Dims: [Dimension; NUM_DIMS];
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DynamicShape;
impl Shape for DynamicShape {
    fn shape(&self) -> ArrayShape {
        ArrayShape {
            dims: vec![usize::MAX],
        }
    }
    const NUM_DIMS: usize = usize::MAX;
}

/// Represents a single dimension of a multi dimensional [Shape]
pub trait Dimension:
    'static + Copy + Clone + std::fmt::Debug + Send + Sync + Eq + PartialEq + Default
{
    fn size(&self) -> usize;
    fn from_size(size: usize) -> Option<Self>;
}

/// Represents a single dimension where all
/// instances are guaranteed to be the same size at compile time.
pub trait ConstDim: Default + Dimension {
    const SIZE: usize;
}

impl Dimension for usize {
    #[inline(always)]
    fn size(&self) -> usize {
        *self
    }
    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        Some(size)
    }
}

/// Represents a [Dim] with size known at compile time
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct S<const M: usize>;
impl<const M: usize> Dimension for S<M> {
    #[inline(always)]
    fn size(&self) -> usize {
        M
    }
    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        if size == M {
            Some(S)
        } else {
            None
        }
    }
}

impl<const M: usize> ConstDim for S<M> {
    const SIZE: usize = M;
}

impl<const N: usize> core::ops::Add<S<N>> for usize {
    type Output = usize;
    fn add(self, _: S<N>) -> Self::Output {
        self.size() + N
    }
}
impl<const N: usize> core::ops::Add<usize> for S<N> {
    type Output = usize;
    fn add(self, rhs: usize) -> Self::Output {
        N + rhs.size()
    }
}

/// Compile time known shape with 0 dimensions
pub type Rank0 = ();
/// Compile time known shape with 1 dimensions
pub type Rank1<M: Dimension> = (M,);
/// Compile time known shape with 2 dimensions
pub type Rank2<M: Dimension, N: Dimension> = (M, N);
/// Compile time known shape with 3 dimensions
pub type Rank3<M: Dimension, N: Dimension, O: Dimension> = (M, N, O);
/// Compile time known shape with 4 dimensions
pub type Rank4<M: Dimension, N: Dimension, O: Dimension, P: Dimension> = (M, N, O, P);
/// Compile time known shape with 5 dimensions
pub type Rank5<M: Dimension, N: Dimension, O: Dimension, P: Dimension, Q: Dimension> =
    (M, N, O, P, Q);

impl<M: Dimension> Shape for Rank1<M> {
    fn shape(&self) -> ArrayShape {
        ArrayShape {
            dims: vec![M::default().size()],
        }
    }
    const NUM_DIMS: usize = 1;
}
impl<M: Dimension, N: Dimension> Shape for Rank2<M, N> {
    fn shape(&self) -> ArrayShape {
        ArrayShape {
            dims: vec![M::default().size(), N::default().size()],
        }
    }
    const NUM_DIMS: usize = 2;
}
impl<O: Dimension, M: Dimension, N: Dimension> Shape for Rank3<O, M, N> {
    fn shape(&self) -> ArrayShape {
        ArrayShape {
            dims: vec![
                O::default().size(),
                M::default().size(),
                N::default().size(),
            ],
        }
    }
    const NUM_DIMS: usize = 3;
}

impl<P: Dimension, O: Dimension, M: Dimension, N: Dimension> Shape for Rank4<P, O, M, N> {
    fn shape(&self) -> ArrayShape {
        ArrayShape {
            dims: vec![
                P::default().size(),
                O::default().size(),
                M::default().size(),
                N::default().size(),
            ],
        }
    }
    const NUM_DIMS: usize = 4;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_dimension() {
        let dynamic = Dynamic::default();
        assert_eq!(dynamic.size(), usize::MAX);
    }
}
