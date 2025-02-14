use crate::with_opencv::{MatExt as _, OpenCvElement};
use crate::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use anyhow::Result;
use ndarray as nd;
use opencv::prelude::*;

impl<'a, A, D> TryFromCv<&'a Mat> for nd::ArrayView<'a, A, D>
where
    A: OpenCvElement,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &'a Mat) -> Result<Self, Self::Error> {
        let src_shape = from.size_with_depth();
        let array = nd::ArrayViewD::from_shape(src_shape, from.as_slice()?)?;
        let array = array.into_dimensionality()?;
        Ok(array)
    }
}

impl<A, D> TryFromCv<&Mat> for nd::Array<A, D>
where
    A: OpenCvElement + Clone,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &Mat) -> Result<Self, Self::Error> {
        let src_shape = from.size_with_depth();
        let array = nd::ArrayViewD::from_shape(src_shape, from.as_slice()?)?;
        let array = array.into_dimensionality()?;
        let array = array.into_owned();
        Ok(array)
    }
}

impl<A, D> TryFromCv<Mat> for nd::Array<A, D>
where
    A: OpenCvElement + Clone,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl<A, S, D> TryFromCv<&nd::ArrayBase<S, D>> for Mat
where
    A: DataType,
    S: nd::RawData<Elem = A> + nd::Data,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &nd::ArrayBase<S, D>) -> Result<Self> {
        let shape_with_channels: Vec<i32> = from.shape().iter().map(|&sz| sz as i32).collect();
        let (channels, shape) = match shape_with_channels.split_last() {
            Some(split) => split,
            None => {
                return Ok(Mat::default());
            }
        };
        let array = from.as_standard_layout();
        let slice = array.as_slice().unwrap();
        let mat = Mat::from_slice(slice)?
            .reshape_nd(*channels, shape)?
            .try_clone()?;
        Ok(mat)
    }
}

impl<A, S, D> TryFromCv<nd::ArrayBase<S, D>> for Mat
where
    A: DataType,
    S: nd::RawData<Elem = A> + nd::Data,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: nd::ArrayBase<S, D>) -> Result<Self> {
        (&from).try_into_cv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            let view: nd::ArrayViewD<f32> = (&in_mat).try_into_cv()?;
            let array: nd::ArrayD<f32> = (&in_mat).try_into_cv()?;
            let out_mat: Mat = (&array).try_into_cv()?;

            shape
                .iter()
                .map(|&size| 0..size)
                .multi_cartesian_product()
                .try_for_each(|index| {
                    // OpenCV expects a &[i32] index.
                    let index_cv: Vec<_> = index.iter().map(|&size| size as i32).collect();
                    let e1: f32 = *in_mat.at_nd(&index_cv)?;

                    // It adds an extra dimension for Mat ->
                    // nd::ArrayView conversion.
                    let index_nd: Vec<_> = chain!(index, [0]).collect();
                    let e2 = view[index_nd.as_slice()];

                    // It adds an extra dimension for Mat -> nd::Array
                    // conversion.
                    let e3 = array[index_nd.as_slice()];

                    // Ensure the path Mat -> nd::Array -> Mat
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
