use crate::{ToCv, TryToCv};
use anyhow::{ensure, Error, Result};
use nalgebra::geometry;
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

impl TryToCv<geometry::Isometry3<f64>> for OpenCvPose<&cv_core::Point3d> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<geometry::Isometry3<f64>, Self::Error> {
        let OpenCvPose { rvec, tvec } = *self;
        let rotation = {
            let rvec_mat = {
                let cv_core::Point3_ { x, y, z, .. } = *rvec;
                Mat::from_slice(&[x, y, z])?.clone_pointee()
            };
            let mut rotation_mat = Mat::zeros(3, 3, cv_core::CV_64FC1)?.to_mat()?;
            opencv::geometry::rodrigues(&rvec_mat, &mut rotation_mat, &mut cv_core::no_array())?;
            let rotation_matrix: nalgebra::Matrix3<f64> = rotation_mat.try_to_cv()?;
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

impl TryToCv<geometry::Isometry3<f64>> for OpenCvPose<cv_core::Point3d> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<geometry::Isometry3<f64>, Self::Error> {
        let OpenCvPose { rvec, tvec } = self;
        OpenCvPose { rvec, tvec }.try_to_cv()
    }
}

impl TryToCv<geometry::Isometry3<f64>> for OpenCvPose<&Mat> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<geometry::Isometry3<f64>, Self::Error> {
        let OpenCvPose {
            rvec: rvec_mat,
            tvec: tvec_mat,
        } = *self;
        let rvec: cv_core::Point3d = rvec_mat.try_to_cv()?;
        let tvec: cv_core::Point3d = tvec_mat.try_to_cv()?;
        let isometry = OpenCvPose {
            rvec: &rvec,
            tvec: &tvec,
        }
        .try_to_cv()?;
        Ok(isometry)
    }
}

impl TryToCv<geometry::Isometry3<f64>> for OpenCvPose<Mat> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<geometry::Isometry3<f64>, Self::Error> {
        let OpenCvPose { rvec, tvec } = self;
        OpenCvPose { rvec, tvec }.try_to_cv()
    }
}

impl<T> TryToCv<OpenCvPose<cv_core::Point3_<T>>> for geometry::Isometry3<T>
where
    T: DataType + nalgebra::RealField,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<OpenCvPose<cv_core::Point3_<T>>, Self::Error> {
        let geometry::Isometry3 {
            rotation,
            translation,
            ..
        } = self;

        let rvec = {
            let rotation_mat = rotation.to_rotation_matrix().into_inner().try_to_cv()?;
            let mut rvec_mat = Mat::zeros(3, 1, cv_core::CV_64FC1)?.to_mat()?;
            opencv::geometry::rodrigues(&rotation_mat, &mut rvec_mat, &mut cv_core::no_array())?;
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

impl TryToCv<OpenCvPose<Mat>> for geometry::Isometry3<f64> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<OpenCvPose<Mat>, Self::Error> {
        let geometry::Isometry3 {
            rotation,
            translation,
            ..
        } = self;

        let rvec = {
            let rotation_mat: Mat = rotation.to_rotation_matrix().into_inner().try_to_cv()?;
            let mut rvec_mat = Mat::zeros(3, 1, cv_core::CV_64FC1)?.to_mat()?;
            opencv::geometry::rodrigues(&rotation_mat, &mut rvec_mat, &mut cv_core::no_array())?;
            rvec_mat
        };
        let tvec = Mat::from_slice(&[translation.x, translation.y, translation.z])?.clone_pointee();
        Ok(OpenCvPose { rvec, tvec })
    }
}

impl TryToCv<OpenCvPose<Mat>> for geometry::Isometry3<f32> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<OpenCvPose<Mat>, Self::Error> {
        let geometry::Isometry3 {
            rotation,
            translation,
            ..
        } = self;

        let rvec = {
            let rotation_mat = rotation.to_rotation_matrix().into_inner().try_to_cv()?;
            let mut rvec_mat = Mat::zeros(3, 1, cv_core::CV_32FC1)?.to_mat()?;
            opencv::geometry::rodrigues(&rotation_mat, &mut rvec_mat, &mut cv_core::no_array())?;
            rvec_mat
        };
        let tvec = Mat::from_slice(&[translation.x, translation.y, translation.z])?.clone_pointee();

        Ok(OpenCvPose { rvec, tvec })
    }
}

impl<N, R, C> TryToCv<nalgebra::OMatrix<N, R, C>> for Mat
where
    N: nalgebra::Scalar + DataType,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    nalgebra::base::default_allocator::DefaultAllocator: nalgebra::base::allocator::Allocator<R, C>,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<nalgebra::OMatrix<N, R, C>, Self::Error> {
        let shape = self.size()?;
        {
            let check_height = R::try_to_usize()
                .map(|size| size == shape.height as usize)
                .unwrap_or(true);
            let check_width = C::try_to_usize()
                .map(|size| size == shape.width as usize)
                .unwrap_or(true);
            let has_same_shape = check_height && check_width;
            ensure!(has_same_shape, "input and output matrix shapes differ");
        }

        let rows: Result<Vec<&[N]>, _> = (0..shape.height)
            .map(|row_idx| self.at_row::<N>(row_idx))
            .collect();
        let rows = rows?;
        let values: Vec<N> = rows
            .into_iter()
            .flat_map(|row| row.iter().cloned())
            .collect();

        Ok(nalgebra::OMatrix::<N, R, C>::from_row_slice_generic(
            R::from_usize(shape.height as usize),
            C::from_usize(shape.width as usize),
            &values,
        ))
    }
}

impl<N, R, C, S> TryToCv<Mat> for nalgebra::Matrix<N, R, C, S>
where
    N: nalgebra::Scalar + DataType,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::base::storage::Storage<N, R, C>,
    nalgebra::base::default_allocator::DefaultAllocator:
        nalgebra::base::allocator::Allocator<R, C> + nalgebra::base::allocator::Allocator<C, R>,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<Mat, Self::Error> {
        let nrows = self.nrows();
        let mat = Mat::from_slice(self.transpose().as_slice())?
            .reshape(1, nrows as i32)?
            .clone_pointee();
        Ok(mat)
    }
}

impl<T> ToCv<cv_core::Point_<T>> for nalgebra::Point2<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn to_cv(&self) -> cv_core::Point_<T> {
        cv_core::Point_::new(self.x, self.y)
    }
}

impl<T> ToCv<nalgebra::Point2<T>> for cv_core::Point_<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn to_cv(&self) -> nalgebra::Point2<T> {
        nalgebra::Point2::new(self.x, self.y)
    }
}

impl<T> ToCv<cv_core::Point3_<T>> for nalgebra::Point3<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn to_cv(&self) -> cv_core::Point3_<T> {
        cv_core::Point3_::new(self.x, self.y, self.z)
    }
}

impl<T> ToCv<nalgebra::Point3<T>> for cv_core::Point3_<T>
where
    T: nalgebra::Scalar + Copy,
{
    fn to_cv(&self) -> nalgebra::Point3<T> {
        nalgebra::Point3::new(self.x, self.y, self.z)
    }
}

impl<N, const D: usize> TryToCv<Mat> for geometry::Translation<N, D>
where
    N: nalgebra::Scalar + DataType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<Mat, Self::Error> {
        let mat = Mat::from_exact_iter(self.vector.into_iter().copied())?;
        Ok(mat)
    }
}

// Scalar (4 元组) <-> Vector4<f64>
impl ToCv<nalgebra::Vector4<f64>> for cv_core::Scalar {
    fn to_cv(&self) -> nalgebra::Vector4<f64> {
        nalgebra::Vector4::new(self[0], self[1], self[2], self[3])
    }
}

impl ToCv<cv_core::Scalar> for nalgebra::Vector4<f64> {
    fn to_cv(&self) -> cv_core::Scalar {
        cv_core::Scalar::new(self[0], self[1], self[2], self[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToCv, TryToCv};
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
            // ToCv
            {
                let cv_point = cv_core::Point2d::new(rng.random(), rng.random());
                let na_point: nalgebra::Point2<f64> = cv_point.to_cv();
                anyhow::ensure!(
                    abs_diff_eq!(cv_point.x, na_point.x) && abs_diff_eq!(cv_point.y, na_point.y),
                    "point conversion failed"
                );
            }

            // ToCv
            {
                let cv_point = cv_core::Point2d::new(rng.random(), rng.random());
                let na_point: nalgebra::Point2<f64> = cv_point.to_cv();
                anyhow::ensure!(
                    abs_diff_eq!(cv_point.x, na_point.x) && abs_diff_eq!(cv_point.y, na_point.y),
                    "point conversion failed"
                );
            }

            // ToCv
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
                let cv_mat: Mat = na_mat.try_to_cv()?;
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

            // ToCv
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
                let cv_mat: Mat = na_mat.try_to_cv()?;
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
        let output: Mat = input.try_to_cv()?;
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
        let output: nalgebra::OMatrix<i32, U2, U3> = input.try_to_cv()?;
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
            let pose: OpenCvPose<Mat> = orig_isometry.try_to_cv()?;
            let recovered_isometry: nalgebra::Isometry3<f64> = pose.try_to_cv()?;

            anyhow::ensure!(
                (orig_isometry.to_homogeneous() - recovered_isometry.to_homogeneous()).norm()
                    <= 1e-6,
                "the recovered isometry is not consistent the original isometry"
            );
        }
        Ok(())
    }

    #[test]
    fn scalar_vector4_roundtrip() -> Result<()> {
        let scalar = cv_core::Scalar::new(1.0, 2.0, 3.0, 4.0);
        let v: nalgebra::Vector4<f64> = scalar.to_cv();
        anyhow::ensure!(
            abs_diff_eq!(v.x, 1.0)
                && abs_diff_eq!(v.y, 2.0)
                && abs_diff_eq!(v.z, 3.0)
                && abs_diff_eq!(v.w, 4.0),
            "scalar -> vector4 failed"
        );

        let back: cv_core::Scalar = v.to_cv();
        anyhow::ensure!(
            abs_diff_eq!(back[0], 1.0)
                && abs_diff_eq!(back[1], 2.0)
                && abs_diff_eq!(back[2], 3.0)
                && abs_diff_eq!(back[3], 4.0),
            "vector4 -> scalar failed"
        );
        Ok(())
    }
}
