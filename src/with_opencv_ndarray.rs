use crate::with_opencv::{MatExt as _, OpenCvElement};
use crate::{ToCv, TryAsRefCv, TryToCv};
use anyhow::Result;
use opencv::core as cv_core;
use opencv::prelude::*;

impl<'a, A, D> TryAsRefCv<'a, ndarray::ArrayView<'a, A, D>> for Mat
where
    A: OpenCvElement,
    D: ndarray::Dimension + 'a,
{
    type Error = anyhow::Error;

    fn try_as_ref_cv(&'a self) -> Result<ndarray::ArrayView<'a, A, D>, Self::Error> {
        let src_shape = self.size_with_depth();
        let array = ndarray::ArrayViewD::from_shape(src_shape, self.as_slice()?)?;
        let array = array.into_dimensionality()?;
        Ok(array)
    }
}

impl<A, D> TryToCv<ndarray::Array<A, D>> for Mat
where
    A: OpenCvElement + Clone,
    D: ndarray::Dimension,
{
    type Error = anyhow::Error;

    fn try_to_cv(&self) -> Result<ndarray::Array<A, D>, Self::Error> {
        let src_shape = self.size_with_depth();
        let array = ndarray::ArrayViewD::from_shape(src_shape, self.as_slice()?)?;
        let array = array.into_dimensionality()?;
        let array = array.into_owned();
        Ok(array)
    }
}

impl<A, S, D> TryToCv<Mat> for ndarray::ArrayBase<S, D>
where
    A: DataType,
    S: ndarray::RawData<Elem = A> + ndarray::Data,
    D: ndarray::Dimension,
{
    type Error = anyhow::Error;

    fn try_to_cv(&self) -> Result<Mat, Self::Error> {
        let shape_with_channels: Vec<i32> = self.shape().iter().map(|&sz| sz as i32).collect();
        let (channels, shape) = match shape_with_channels.split_last() {
            Some(split) => split,
            None => {
                return Ok(Mat::default());
            }
        };
        let array = self.as_standard_layout();
        let slice = array.as_slice().unwrap();
        let mat = Mat::from_slice(slice)?
            .reshape_nd(*channels, shape)?
            .try_clone()?;
        Ok(mat)
    }
}

// Scalar (4 元组) <-> Array1<f64>
impl ToCv<ndarray::Array1<f64>> for cv_core::Scalar {
    fn to_cv(&self) -> ndarray::Array1<f64> {
        ndarray::Array1::from_vec(vec![self[0], self[1], self[2], self[3]])
    }
}

impl ToCv<cv_core::Scalar> for ndarray::Array1<f64> {
    fn to_cv(&self) -> cv_core::Scalar {
        let mut v = [0.0f64; 4];
        for (i, slot) in v.iter_mut().enumerate() {
            if i < self.len() {
                *slot = self[i];
            }
        }
        cv_core::Scalar::new(v[0], v[1], v[2], v[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TryToCv;
    use itertools::chain;
    use itertools::Itertools as _;
    use rand::prelude::*;

    #[test]
    fn opencv_ndarray_conversion() -> Result<()> {
        let mut rng = rand::rng();

        for _ in 0..5 {
            // Generate a random shape
            let ndim: usize = rng.random_range(2..=4);
            let shape: Vec<usize> = (0..ndim).map(|_| rng.random_range(1..=32)).collect();

            let in_mat = Mat::new_randn_nd::<f32>(&shape)?;
            let view: ndarray::ArrayViewD<f32> = in_mat.try_as_ref_cv()?;
            let array: ndarray::ArrayD<f32> = in_mat.try_to_cv()?;
            let out_mat: Mat = array.try_to_cv()?;

            shape
                .iter()
                .map(|&size| 0..size)
                .multi_cartesian_product()
                .try_for_each(|index| {
                    // OpenCV expects a &[i32] index.
                    let index_cv: Vec<_> = index.iter().map(|&size| size as i32).collect();
                    let e1: f32 = *in_mat.at_nd(&index_cv)?;

                    // It adds an extra dimension for Mat ->
                    // ndarray::ArrayView conversion.
                    let index_nd: Vec<_> = chain!(index, [0]).collect();
                    let e2 = view[index_nd.as_slice()];

                    // It adds an extra dimension for Mat -> ndarray::Array
                    // conversion.
                    let e3 = array[index_nd.as_slice()];

                    // Ensure the path Mat -> ndarray::Array -> Mat
                    // preserves the values.
                    let e4: f32 = *out_mat.at_nd(&index_cv)?;

                    anyhow::ensure!(e1 == e2);
                    anyhow::ensure!(e1 == e3);
                    anyhow::ensure!(e1 == e4);
                    anyhow::Ok(())
                })?;
        }

        Ok(())
    }
}
