use crate::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use anyhow::{Error, Result};
use nalgebra::geometry;
use opencv::calib3d;
use opencv::core as cv_core;
use opencv::prelude::*;

/// NOTE: for future maintainers: Since the matrixes need to accommodate any size Matrix, we are using nalgebra::OMatrix instead of SMatrix.
///
/// A pair of rvec and tvec from OpenCV, standing for rotation and translation.
#[derive(Debug, Clone)]
pub struct OpenCvPose<T> {
    pub rvec: T,
    pub tvec: T,
}

impl TryFromCv<OpenCvPose<&cv_core::Point3d>> for geometry::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(pose: OpenCvPose<&cv_core::Point3d>) -> Result<Self> {
        (&pose).try_into_cv()
    }
}

impl TryFromCv<&OpenCvPose<&cv_core::Point3d>> for geometry::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(pose: &OpenCvPose<&cv_core::Point3d>) -> Result<Self> {
        let OpenCvPose { rvec, tvec } = *pose;
        let rotation = {
            let rvec_mat = {
                let cv_core::Point3_ { x, y, z, .. } = *rvec;
                Mat::from_slice(&[x, y, z])?.try_clone()?
            };
            let mut rotation_mat = Mat::zeros(3, 3, cv_core::CV_64FC1)?.to_mat()?;
            calib3d::rodrigues(&rvec_mat, &mut rotation_mat, &mut cv_core::no_array())?;
            let rotation_matrix: nalgebra::Matrix3<f64> = TryFromCv::try_from_cv(rotation_mat)?;
            geometry::UnitQuaternion::from_matrix(&rotation_matrix)
        };

        let translation = {
            let cv_core::Point3_ { x, y, z } = *tvec;
            geometry::Translation3::new(x, y, z)
        };

        let isometry = geometry::Isometry3::from_parts(translation, rotation);
        Ok(isometry)
    }
}

impl TryFromCv<&OpenCvPose<cv_core::Point3d>> for geometry::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: &OpenCvPose<cv_core::Point3d>) -> Result<Self> {
        let OpenCvPose { rvec, tvec } = from;
        TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })
    }
}

impl TryFromCv<OpenCvPose<cv_core::Point3d>> for geometry::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: OpenCvPose<cv_core::Point3d>) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl TryFromCv<OpenCvPose<&Mat>> for geometry::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: OpenCvPose<&Mat>) -> Result<Self> {
        (&from).try_into_cv()
    }
}

impl TryFromCv<&OpenCvPose<&Mat>> for geometry::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: &OpenCvPose<&Mat>) -> Result<Self> {
        let OpenCvPose {
            rvec: rvec_mat,
            tvec: tvec_mat,
        } = *from;
        let rvec = cv_core::Point3d::try_from_cv(rvec_mat)?;
        let tvec = cv_core::Point3d::try_from_cv(tvec_mat)?;
        let isometry = TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })?;
        Ok(isometry)
    }
}

impl TryFromCv<&OpenCvPose<Mat>> for geometry::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: &OpenCvPose<Mat>) -> Result<Self> {
        let OpenCvPose { rvec, tvec } = from;
        TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })
    }
}

impl TryFromCv<OpenCvPose<Mat>> for geometry::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: OpenCvPose<Mat>) -> Result<Self> {
        let OpenCvPose { rvec, tvec } = &from;
        TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })
    }
}

impl<T> TryFromCv<&geometry::Isometry3<T>> for OpenCvPose<cv_core::Point3_<T>>
where
    T: DataType + nalgebra::RealField,
{
    type Error = Error;

    fn try_from_cv(from: &geometry::Isometry3<T>) -> Result<OpenCvPose<cv_core::Point3_<T>>> {
        let geometry::Isometry3 {
            rotation,
            translation,
            ..
        } = from;

        let rvec = {
            let rotation_mat = Mat::try_from_cv(rotation.to_rotation_matrix().into_inner())?;
            let mut rvec_mat = Mat::zeros(3, 1, cv_core::CV_64FC1)?.to_mat()?;
            calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut cv_core::no_array())?;
            cv_core::Point3_::new(
                *rvec_mat.at_2d::<T>(0, 0)?,
                *rvec_mat.at_2d::<T>(1, 0)?,
                *rvec_mat.at_2d::<T>(2, 0)?,
            )
        };
        let tvec = cv_core::Point3_::new(translation.x, translation.y, translation.z);

        Ok(OpenCvPose { rvec, tvec })
    }
}

impl<T> TryFromCv<geometry::Isometry3<T>> for OpenCvPose<cv_core::Point3_<T>>
where
    T: DataType + nalgebra::RealField,
{
    type Error = Error;

    fn try_from_cv(from: geometry::Isometry3<T>) -> Result<OpenCvPose<cv_core::Point3_<T>>> {
        TryFromCv::try_from_cv(&from)
    }
}

impl TryFromCv<&geometry::Isometry3<f64>> for OpenCvPose<Mat> {
    type Error = Error;

    fn try_from_cv(from: &geometry::Isometry3<f64>) -> Result<OpenCvPose<Mat>> {
        let geometry::Isometry3 {
            rotation,
            translation,
            ..
        } = from;

        let rvec = {
            let rotation_mat: Mat =
                TryFromCv::try_from_cv(rotation.to_rotation_matrix().into_inner())?;
            let mut rvec_mat = Mat::zeros(3, 1, cv_core::CV_64FC1)?.to_mat()?;
            calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut cv_core::no_array())?;
            rvec_mat
        };
        let tvec = Mat::from_slice(&[translation.x, translation.y, translation.z])?.try_clone()?;
        Ok(OpenCvPose { rvec, tvec })
    }
}

impl TryFromCv<geometry::Isometry3<f64>> for OpenCvPose<Mat> {
    type Error = Error;

    fn try_from_cv(from: geometry::Isometry3<f64>) -> Result<OpenCvPose<Mat>> {
        TryFromCv::try_from_cv(&from)
    }
}

impl TryFromCv<&geometry::Isometry3<f32>> for OpenCvPose<Mat> {
    type Error = Error;

    fn try_from_cv(from: &geometry::Isometry3<f32>) -> Result<OpenCvPose<Mat>> {
        let geometry::Isometry3 {
            rotation,
            translation,
            ..
        } = from;

        let rvec = {
            let rotation_mat = Mat::try_from_cv(rotation.to_rotation_matrix().into_inner())?;
            let mut rvec_mat = Mat::zeros(3, 1, cv_core::CV_32FC1)?.to_mat()?;
            calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut cv_core::no_array())?;
            rvec_mat
        };
        let tvec = Mat::from_slice(&[translation.x, translation.y, translation.z])?
            .try_clone()
            .unwrap();

        Ok(OpenCvPose { rvec, tvec })
    }
}

impl TryFromCv<geometry::Isometry3<f32>> for OpenCvPose<Mat> {
    type Error = Error;

    fn try_from_cv(from: geometry::Isometry3<f32>) -> Result<OpenCvPose<Mat>> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<N, R, C> TryFromCv<&Mat> for nalgebra::OMatrix<N, R, C>
where
    N: nalgebra::Scalar + DataType,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    nalgebra::base::default_allocator::DefaultAllocator: nalgebra::base::allocator::Allocator<R, C>,
{
    type Error = Error;

    fn try_from_cv(from: &Mat) -> Result<Self> {
        let shape = from.size()?;
        {
            let check_height = R::try_to_usize()
                .map(|size| size == shape.height as usize)
                .unwrap_or(true);
            let check_width = C::try_to_usize()
                .map(|size| size == shape.width as usize)
                .unwrap_or(true);
            let has_same_shape = check_height && check_width;
            anyhow::ensure!(has_same_shape, "input and output matrix shapes differ");
        }

        let rows: Result<Vec<&[N]>, _> = (0..shape.height)
            .map(|row_idx| from.at_row::<N>(row_idx))
            .collect();
        let rows = rows?;
        let values: Vec<N> = rows
            .into_iter()
            .flat_map(|row| row.iter().cloned())
            .collect();

        Ok(Self::from_row_slice_generic(
            R::from_usize(shape.height as usize),
            C::from_usize(shape.width as usize),
            &values,
        ))
    }
}

impl<N, R, C> TryFromCv<Mat> for nalgebra::OMatrix<N, R, C>
where
    N: nalgebra::Scalar + DataType,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    nalgebra::base::default_allocator::DefaultAllocator: nalgebra::base::allocator::Allocator<R, C>,
{
    type Error = Error;

    fn try_from_cv(from: Mat) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<N, R, C, S> TryFromCv<&nalgebra::Matrix<N, R, C, S>> for Mat
where
    N: nalgebra::Scalar + DataType,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::base::storage::Storage<N, R, C>,
    nalgebra::base::default_allocator::DefaultAllocator:
        nalgebra::base::allocator::Allocator<R, C> + nalgebra::base::allocator::Allocator<C, R>,
{
    type Error = Error;

    fn try_from_cv(from: &nalgebra::Matrix<N, R, C, S>) -> Result<Self> {
        let nrows = from.nrows();
        let mat = Mat::from_slice(from.transpose().as_slice())?
            .reshape(1, nrows as i32)?
            .try_clone()?;
        Ok(mat)
    }
}

impl<N, R, C, S> TryFromCv<nalgebra::Matrix<N, R, C, S>> for Mat
where
    N: nalgebra::Scalar + DataType,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::base::storage::Storage<N, R, C>,
    nalgebra::base::default_allocator::DefaultAllocator:
        nalgebra::base::allocator::Allocator<R, C> + nalgebra::base::allocator::Allocator<C, R>,
{
    type Error = Error;

    fn try_from_cv(from: nalgebra::Matrix<N, R, C, S>) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> FromCv<&nalgebra::Point2<T>> for cv_core::Point_<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn from_cv(from: &nalgebra::Point2<T>) -> Self {
        cv_core::Point_::new(from.x, from.y)
    }
}

impl<T> FromCv<nalgebra::Point2<T>> for cv_core::Point_<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn from_cv(from: nalgebra::Point2<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

impl<T> FromCv<&cv_core::Point_<T>> for nalgebra::Point2<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn from_cv(from: &cv_core::Point_<T>) -> Self {
        Self::new(from.x, from.y)
    }
}

impl<T> FromCv<cv_core::Point_<T>> for nalgebra::Point2<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn from_cv(from: cv_core::Point_<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

impl<T> FromCv<&nalgebra::Point3<T>> for cv_core::Point3_<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn from_cv(from: &nalgebra::Point3<T>) -> Self {
        Self::new(from.x, from.y, from.z)
    }
}

impl<T> FromCv<nalgebra::Point3<T>> for cv_core::Point3_<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn from_cv(from: nalgebra::Point3<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

impl<T> FromCv<&cv_core::Point3_<T>> for nalgebra::Point3<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn from_cv(from: &cv_core::Point3_<T>) -> Self {
        Self::new(from.x, from.y, from.z)
    }
}

impl<T> FromCv<cv_core::Point3_<T>> for nalgebra::Point3<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn from_cv(from: cv_core::Point3_<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

impl<N, const D: usize> TryFromCv<&geometry::Translation<N, D>> for Mat
where
    N: nalgebra::Scalar + DataType,
{
    type Error = Error;

    fn try_from_cv(translation: &geometry::Translation<N, D>) -> Result<Self> {
        let mat = Mat::from_exact_iter(translation.vector.into_iter().copied())?;
        Ok(mat)
    }
}

impl<N, const D: usize> TryFromCv<geometry::Translation<N, D>> for Mat
where
    N: nalgebra::Scalar + DataType,
{
    type Error = Error;

    fn try_from_cv(translation: geometry::Translation<N, D>) -> Result<Self> {
        TryFromCv::try_from_cv(&translation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IntoCv, TryIntoCv};
    use anyhow::Result;
    use approx::abs_diff_eq;
    use nalgebra::{U2, U3};
    use opencv::core as cv_core;
    use rand::prelude::*;
    use std::f64;

    #[test]
    fn convert_opencv_nalgebra() -> Result<()> {
        let mut rng = rand::rng();

        for _ in 0..5000 {
            // FromCv
            {
                let cv_point = cv_core::Point2d::new(rng.random(), rng.random());
                let na_point = nalgebra::Point2::<f64>::from_cv(&cv_point);
                anyhow::ensure!(
                    abs_diff_eq!(cv_point.x, na_point.x) && abs_diff_eq!(cv_point.y, na_point.y),
                    "point conversion failed"
                );
            }

            // IntoCv
            {
                let cv_point = cv_core::Point2d::new(rng.random(), rng.random());
                let na_point: nalgebra::Point2<f64> = cv_point.into_cv();
                anyhow::ensure!(
                    abs_diff_eq!(cv_point.x, na_point.x) && abs_diff_eq!(cv_point.y, na_point.y),
                    "point conversion failed"
                );
            }

            // TryFromCv
            {
                let na_mat = nalgebra::DMatrix::<f64>::from_vec(
                    2,
                    3,
                    vec![
                        rng.random(),
                        rng.random(),
                        rng.random(),
                        rng.random(),
                        rng.random(),
                        rng.random(),
                    ],
                );
                let cv_mat = Mat::try_from_cv(&na_mat)?;
                anyhow::ensure!(
                    abs_diff_eq!(cv_mat.at_2d(0, 0)?, na_mat.get((0, 0)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(0, 1)?, na_mat.get((0, 1)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(0, 2)?, na_mat.get((0, 2)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 0)?, na_mat.get((1, 0)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 1)?, na_mat.get((1, 1)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 2)?, na_mat.get((1, 2)).unwrap()),
                    "matrix conversion failed"
                );
            }

            // TryIntoCv
            {
                let na_mat = nalgebra::DMatrix::<f64>::from_vec(
                    2,
                    3,
                    vec![
                        rng.random(),
                        rng.random(),
                        rng.random(),
                        rng.random(),
                        rng.random(),
                        rng.random(),
                    ],
                );
                let cv_mat: Mat = (&na_mat).try_into_cv()?;
                anyhow::ensure!(
                    abs_diff_eq!(cv_mat.at_2d(0, 0)?, na_mat.get((0, 0)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(0, 1)?, na_mat.get((0, 1)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(0, 2)?, na_mat.get((0, 2)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 0)?, na_mat.get((1, 0)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 1)?, na_mat.get((1, 1)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 2)?, na_mat.get((1, 2)).unwrap()),
                    "matrix conversion failed"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn matrix_nalgebra_to_opencv_test() -> Result<()> {
        let input = nalgebra::OMatrix::<i32, U3, U2>::from_row_slice(&[1, 2, 3, 4, 5, 6]);
        let (nrows, ncols) = input.shape();
        let output = Mat::try_from_cv(input)?;
        let output_shape = output.size()?;
        anyhow::ensure!(
            output.channels() == 1
                && nrows == output_shape.height as usize
                && ncols == output_shape.width as usize,
            "the shape does not match"
        );
        Ok(())
    }

    #[test]
    fn matrix_opencv_to_nalgebra_test() -> Result<()> {
        let input = Mat::from_slice_2d(&[&[1, 2, 3], &[4, 5, 6]])?;
        let input_shape = input.size()?;
        let output = nalgebra::OMatrix::<i32, U2, U3>::try_from_cv(input)?;
        anyhow::ensure!(
            output.nrows() == input_shape.height as usize
                && output.ncols() == input_shape.width as usize,
            "the shape does not match"
        );
        Ok(())
    }

    #[test]
    fn rvec_tvec_conversion() -> Result<()> {
        let mut rng = rand::rng();

        for _ in 0..5000 {
            let orig_isometry = {
                let rotation = nalgebra::UnitQuaternion::from_euler_angles(
                    rng.random_range(0.0..(f64::consts::PI * 2.0)),
                    rng.random_range(0.0..(f64::consts::PI * 2.0)),
                    rng.random_range(0.0..(f64::consts::PI * 2.0)),
                );
                let translation =
                    nalgebra::Translation3::new(rng.random(), rng.random(), rng.random());
                nalgebra::Isometry3::from_parts(translation, rotation)
            };
            let pose = OpenCvPose::<Mat>::try_from_cv(orig_isometry)?;
            let recovered_isometry = nalgebra::Isometry3::<f64>::try_from_cv(pose)?;

            anyhow::ensure!(
                (orig_isometry.to_homogeneous() - recovered_isometry.to_homogeneous()).norm()
                    <= 1e-6,
                "the recovered isometry is not consistent the original isometry"
            );
        }
        Ok(())
    }
}
