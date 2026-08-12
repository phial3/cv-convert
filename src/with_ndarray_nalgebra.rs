use crate::{ToCv, TryToCv};
use anyhow::{ensure, Error, Result};
use nalgebra as na;
use ndarray::{Array1, Array2};

// Array1 -> DVector
impl<T> ToCv<na::DVector<T>> for Array1<T>
where
    T: na::Scalar + Clone,
{
    fn to_cv(&self) -> na::DVector<T> {
        let data: Vec<T> = self.iter().cloned().collect();
        na::DVector::from_vec(data)
    }
}

// DVector -> Array1
impl<T> TryToCv<Array1<T>> for na::DVector<T>
where
    T: na::Scalar + Clone,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<Array1<T>, Self::Error> {
        Ok(Array1::from_vec(self.as_slice().to_vec()))
    }
}

// Array2 -> DMatrix
impl<T> ToCv<na::DMatrix<T>> for Array2<T>
where
    T: na::Scalar + Clone,
{
    fn to_cv(&self) -> na::DMatrix<T> {
        let (rows, cols) = self.dim();
        let data: Vec<T> = self.iter().cloned().collect();
        na::DMatrix::from_vec(rows, cols, data)
    }
}

// DMatrix -> Array2
impl<T> TryToCv<Array2<T>> for na::DMatrix<T>
where
    T: na::Scalar + Clone,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<Array2<T>, Self::Error> {
        let (rows, cols) = (self.nrows(), self.ncols());
        Ok(Array2::from_shape_vec(
            (rows, cols),
            self.as_slice().to_vec(),
        )?)
    }
}

// SVector<_, N> -> Array1
impl<T, const N: usize> ToCv<Array1<T>> for na::SVector<T, N>
where
    T: na::Scalar + Copy,
{
    fn to_cv(&self) -> Array1<T> {
        Array1::from_vec(self.as_slice().to_vec())
    }
}

// Array1 -> SVector<_, N>（尺寸不匹配时返回错误）
impl<T, const N: usize> TryToCv<na::SVector<T, N>> for Array1<T>
where
    T: na::Scalar + Copy,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<na::SVector<T, N>, Self::Error> {
        ensure!(
            self.len() == N,
            "length mismatch: got {}, expected {}",
            self.len(),
            N
        );
        Ok(na::SVector::from_row_slice(
            self.as_slice().expect("Array1 must be contiguous"),
        ))
    }
}

// SMatrix<_, R, C> -> Array2（处理 nalgebra 列主序与 ndarray 行主序的差异）
impl<T, const R: usize, const C: usize> ToCv<Array2<T>> for na::SMatrix<T, R, C>
where
    T: na::Scalar + Copy,
{
    fn to_cv(&self) -> Array2<T> {
        // SMatrix::as_slice() 为列主序；先按 (C, R) 形状读取再转置成 (R, C) 行主序
        let data = self.as_slice().to_vec();
        Array2::from_shape_vec((C, R), data)
            .expect("SMatrix has a fixed shape")
            .reversed_axes()
    }
}

// Array2 -> SMatrix<_, R, C>（形状不匹配时返回错误）
impl<T, const R: usize, const C: usize> TryToCv<na::SMatrix<T, R, C>> for Array2<T>
where
    T: na::Scalar + Copy,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<na::SMatrix<T, R, C>, Self::Error> {
        let (rows, cols) = self.dim();
        ensure!(
            rows == R && cols == C,
            "shape mismatch: got ({}, {}), expected ({}, {})",
            rows,
            cols,
            R,
            C
        );
        // 逐列提取得到列主序数据
        let mut data = Vec::with_capacity(rows * cols);
        for c in 0..cols {
            for r in 0..rows {
                data.push(self[[r, c]]);
            }
        }
        Ok(na::SMatrix::from_column_slice(&data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_roundtrip() {
        let v = na::DVector::from_vec(vec![1.0f64, 2.0, 3.0]);
        let arr: Array1<f64> = v.try_to_cv().unwrap();
        assert_eq!(arr.to_vec(), vec![1.0, 2.0, 3.0]);

        let back: na::DVector<f64> = arr.to_cv();
        assert_eq!(back, v);
    }

    #[test]
    fn matrix_roundtrip() {
        let m = na::DMatrix::from_row_slice(2, 3, &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let arr: Array2<f64> = m.try_to_cv().unwrap();
        assert_eq!(arr.dim(), (2, 3));

        let back: na::DMatrix<f64> = arr.to_cv();
        assert_eq!(back, m);
    }

    #[test]
    fn svector_roundtrip() {
        let v = na::SVector::<f64, 3>::new(1.0, 2.0, 3.0);
        let arr: Array1<f64> = v.to_cv();
        assert_eq!(arr.to_vec(), vec![1.0, 2.0, 3.0]);

        let back: na::SVector<f64, 3> = arr.try_to_cv().unwrap();
        assert_eq!(back, v);

        // 尺寸不匹配应报错
        let bad: Array1<f64> = Array1::from_vec(vec![1.0, 2.0]);
        assert!(<Array1<f64> as crate::TryToCv<na::SVector<f64, 3>>>::try_to_cv(&bad).is_err());
    }

    #[test]
    fn smatrix_roundtrip() {
        let m = na::SMatrix::<f64, 2, 3>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let arr: Array2<f64> = m.to_cv();
        assert_eq!(arr.dim(), (2, 3));

        let back: na::SMatrix<f64, 2, 3> = arr.try_to_cv().unwrap();
        assert_eq!(back, m);

        // 形状不匹配应报错
        let bad: Array2<f64> = Array2::zeros((4, 4));
        assert!(<Array2<f64> as crate::TryToCv<na::SMatrix<f64, 2, 3>>>::try_to_cv(&bad).is_err());
    }
}
