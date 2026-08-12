use crate::{ToCv, TryToCv};
use anyhow::{Error, Result};
use ndarray::Array1;

impl<T> ToCv<Array1<T>> for imageproc::point::Point<T>
where
    T: Copy + Default,
{
    fn to_cv(&self) -> Array1<T> {
        Array1::from_vec(vec![self.x, self.y])
    }
}

impl<T> TryToCv<imageproc::point::Point<T>> for Array1<T>
where
    T: Copy + Default,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<imageproc::point::Point<T>, Self::Error> {
        anyhow::ensure!(self.len() == 2, "expected length 2, but got {}", self.len());
        Ok(imageproc::point::Point::new(self[0], self[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_ndarray_roundtrip() {
        let p = imageproc::point::Point::new(3i32, 4);
        let arr: Array1<i32> = p.to_cv();
        assert_eq!(arr.to_vec(), vec![3, 4]);

        let back: imageproc::point::Point<i32> = arr.try_to_cv().unwrap();
        assert_eq!(back, p);
    }
}
