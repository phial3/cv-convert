use crate::{ToCv, TryToCv};
use anyhow::{ensure, Error, Result};
use nalgebra as na;

// DVector -> Tensor
impl<T> ToCv<tch::Tensor> for na::DVector<T>
where
    T: tch::kind::Element + na::Scalar + Clone,
{
    fn to_cv(&self) -> tch::Tensor {
        let data: Vec<T> = self.as_slice().to_vec();
        tch::Tensor::from_slice(&data).view([self.len() as i64])
    }
}

// Tensor -> DVector
impl<T> TryToCv<na::DVector<T>> for tch::Tensor
where
    T: tch::kind::Element + na::Scalar,
    Vec<T>: TryFrom<tch::Tensor, Error = tch::TchError>,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<na::DVector<T>, Self::Error> {
        let from = self.shallow_clone();
        ensure!(
            from.kind() == T::KIND,
            "tensor with kind {:?} cannot convert to vector with type {:?}",
            from.kind(),
            T::KIND
        );
        ensure!(from.size().len() == 1, "expected 1D tensor");
        let data: Vec<T> = Vec::try_from(from.flatten(0, -1))?;
        Ok(na::DVector::from_vec(data))
    }
}

// DMatrix -> Tensor (row-major)
impl<T> ToCv<tch::Tensor> for na::DMatrix<T>
where
    T: tch::kind::Element + na::Scalar + Clone,
{
    fn to_cv(&self) -> tch::Tensor {
        let (rows, cols) = (self.nrows(), self.ncols());
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(self[(r, c)].clone());
            }
        }
        tch::Tensor::from_slice(&data).view([rows as i64, cols as i64])
    }
}

// Tensor -> DMatrix (interpreted as row-major)
impl<T> TryToCv<na::DMatrix<T>> for tch::Tensor
where
    T: tch::kind::Element + na::Scalar,
    Vec<T>: TryFrom<tch::Tensor, Error = tch::TchError>,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<na::DMatrix<T>, Self::Error> {
        let from = self.shallow_clone();
        ensure!(
            from.kind() == T::KIND,
            "tensor with kind {:?} cannot convert to matrix with type {:?}",
            from.kind(),
            T::KIND
        );
        let size = from.size();
        ensure!(size.len() == 2, "expected 2D tensor");
        let (rows, cols) = (size[0] as usize, size[1] as usize);
        let data: Vec<T> = Vec::try_from(from.flatten(0, -1))?;
        Ok(na::DMatrix::from_row_slice(rows, cols, &data))
    }
}

// SMatrix (fixed-size) -> Tensor (row-major)
impl<T, const R: usize, const C: usize> ToCv<tch::Tensor> for na::SMatrix<T, R, C>
where
    T: tch::kind::Element + na::Scalar + Clone,
{
    fn to_cv(&self) -> tch::Tensor {
        let (rows, cols) = self.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(self[(r, c)].clone());
            }
        }
        tch::Tensor::from_slice(&data).view([rows as i64, cols as i64])
    }
}

// Tensor -> SMatrix (fixed-size, interpreted as row-major)
impl<T, const R: usize, const C: usize> TryToCv<na::SMatrix<T, R, C>> for tch::Tensor
where
    T: tch::kind::Element + na::Scalar,
    Vec<T>: TryFrom<tch::Tensor, Error = tch::TchError>,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<na::SMatrix<T, R, C>, Self::Error> {
        let from = self.shallow_clone();
        ensure!(
            from.kind() == T::KIND,
            "tensor with kind {:?} cannot convert to matrix with type {:?}",
            from.kind(),
            T::KIND
        );
        let size = from.size();
        ensure!(size.len() == 2, "expected 2D tensor");
        ensure!(
            size[0] == R as i64 && size[1] == C as i64,
            "expected tensor shape ({}, {}), but got ({}, {})",
            R,
            C,
            size[0],
            size[1]
        );
        let data: Vec<T> = Vec::try_from(from.flatten(0, -1))?;
        Ok(na::SMatrix::<T, R, C>::from_row_slice(&data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::abs_diff_eq;

    #[test]
    fn vector_roundtrip() {
        let v = na::DVector::from_vec(vec![1.0f64, 2.0, 3.0]);
        let tensor = v.to_cv();
        let back: na::DVector<f64> = (tensor).try_to_cv().unwrap();
        assert!(back.iter().zip(v.iter()).all(|(a, b)| abs_diff_eq!(a, b)));
    }

    #[test]
    fn matrix_roundtrip() {
        let m = na::DMatrix::from_row_slice(2, 3, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tensor = m.to_cv();
        let back: na::DMatrix<f64> = (tensor).try_to_cv().unwrap();
        assert_eq!(back.shape(), (2, 3));
        assert!(back.iter().zip(m.iter()).all(|(a, b)| abs_diff_eq!(a, b)));
    }

    #[test]
    fn omatrix_roundtrip() {
        // 固定尺寸矩阵 <-> Tensor
        let m = na::SMatrix::<f64, 2, 3>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tensor: tch::Tensor = m.to_cv();
        let back: na::SMatrix<f64, 2, 3> = (tensor).try_to_cv().unwrap();
        assert_eq!(back.shape(), (2, 3));
        assert!(back.iter().zip(m.iter()).all(|(a, b)| abs_diff_eq!(a, b)));
    }

    #[test]
    fn omatrix_shape_mismatch() {
        // 形状不匹配时应报错
        let tensor = tch::Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0]).view([2, 2]);
        let back: Result<na::SMatrix<f64, 2, 3>, _> = (tensor).try_to_cv();
        assert!(back.is_err());
    }
}
