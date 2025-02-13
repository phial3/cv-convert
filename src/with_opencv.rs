use crate::{TryFromCv};

use half::f16;
use opencv::{core as cv_core, prelude::*};
use anyhow::{Error, Result};

pub use element_type::*;
mod element_type {
    use super::*;

    pub trait OpenCvElement {
        const DEPTH: i32;
    }

    impl OpenCvElement for u8 {
        const DEPTH: i32 = cv_core::CV_8U;
    }

    impl OpenCvElement for i8 {
        const DEPTH: i32 = cv_core::CV_8S;
    }

    impl OpenCvElement for u16 {
        const DEPTH: i32 = cv_core::CV_16U;
    }

    impl OpenCvElement for i16 {
        const DEPTH: i32 = cv_core::CV_16S;
    }

    impl OpenCvElement for i32 {
        const DEPTH: i32 = cv_core::CV_32S;
    }

    impl OpenCvElement for f16 {
        const DEPTH: i32 = cv_core::CV_16F;
    }

    impl OpenCvElement for f32 {
        const DEPTH: i32 = cv_core::CV_32F;
    }

    impl OpenCvElement for f64 {
        const DEPTH: i32 = cv_core::CV_64F;
    }
}

pub(crate) use mat_ext::*;
mod mat_ext {
    use super::*;

    pub trait MatExt {
        fn size_with_depth(&self) -> Vec<usize>;

        fn numel(&self) -> usize {
            self.size_with_depth().iter().product()
        }

        fn as_slice<T>(&self) -> Result<&[T]>
        where
            T: OpenCvElement;

        fn type_name(&self) -> String;

        #[cfg(test)]
        fn new_randn_2d(rows: i32, cols: i32, typ: i32) -> Result<Self>
        where
            Self: Sized;

        #[cfg(test)]
        fn new_randn_nd<T>(shape: &[usize]) -> Result<Self>
        where
            Self: Sized,
            T: OpenCvElement;
    }

    impl MatExt for cv_core::Mat {
        fn size_with_depth(&self) -> Vec<usize> {
            let size = self.mat_size();
            let size = size.iter().map(|&dim| dim as usize);
            let channels = self.channels() as usize;
            size.chain([channels]).collect()
        }

        fn as_slice<T>(&self) -> Result<&[T]>
        where
            T: OpenCvElement,
        {
            anyhow::ensure!(self.depth() == T::DEPTH, "element type mismatch");
            anyhow::ensure!(self.is_continuous(), "Mat data must be continuous");

            let numel = self.numel();
            let ptr = self.ptr(0)? as *const T;

            let slice = unsafe { std::slice::from_raw_parts(ptr, numel) };
            Ok(slice)
        }

        fn type_name(&self) -> String {
            cv_core::type_to_string(self.typ()).unwrap()
        }

        #[cfg(test)]
        fn new_randn_2d(rows: i32, cols: i32, typ: i32) -> Result<Self>
        where
            Self: Sized,
        {
            let mut mat = Self::zeros(rows, cols, typ)?.to_mat()?;
            cv_core::randn(&mut mat, &0.0, &1.0)?;
            Ok(mat)
        }

        #[cfg(test)]
        fn new_randn_nd<T>(shape: &[usize]) -> Result<Self>
        where
            T: OpenCvElement,
        {
            let shape: Vec<_> = shape.iter().map(|&val| val as i32).collect();
            let mut mat = Self::zeros_nd(&shape, T::DEPTH)?.to_mat()?;
            cv_core::randn(&mut mat, &0.0, &1.0)?;
            Ok(mat)
        }
    }
}

impl<T> TryFromCv<&cv_core::Mat> for cv_core::Point_<T>
where
    T: cv_core::DataType,
{
    type Error = Error;

    fn try_from_cv(from: &cv_core::Mat) -> Result<Self> {
        let slice = from.data_typed::<T>()?;
        anyhow::ensure!(slice.len() == 2, "invalid length");
        let point = Self {
            x: slice[0],
            y: slice[1],
        };
        Ok(point)
    }
}

impl<T> TryFromCv<cv_core::Mat> for cv_core::Point_<T>
where
    T: cv_core::DataType,
{
    type Error = Error;

    fn try_from_cv(from: cv_core::Mat) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&cv_core::Mat> for cv_core::Point3_<T>
where
    T: cv_core::DataType,
{
    type Error = Error;

    fn try_from_cv(from: &cv_core::Mat) -> Result<Self> {
        let slice = from.data_typed::<T>()?;
        anyhow::ensure!(slice.len() == 3, "invalid length");
        let point = Self {
            x: slice[0],
            y: slice[1],
            z: slice[2],
        };
        Ok(point)
    }
}

impl<T> TryFromCv<cv_core::Mat> for cv_core::Point3_<T>
where
    T: cv_core::DataType,
{
    type Error = Error;

    fn try_from_cv(from: cv_core::Mat) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&cv_core::Point_<T>> for cv_core::Mat
where
    T: cv_core::DataType,
{
    type Error = Error;

    fn try_from_cv(from: &cv_core::Point_<T>) -> Result<Self> {
        let cv_core::Point_ { x, y, .. } = *from;
        let binding = [x, y];
        let mat = cv_core::Mat::from_slice(&binding)?;
        Ok(mat.try_clone()?)
    }
}

impl<T> TryFromCv<cv_core::Point_<T>> for cv_core::Mat
where
    T: cv_core::DataType,
{
    type Error = Error;

    fn try_from_cv(from: cv_core::Point_<T>) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&cv_core::Point3_<T>> for cv_core::Mat
where
    T: cv_core::DataType,
{
    type Error = Error;

    fn try_from_cv(from: &cv_core::Point3_<T>) -> Result<Self> {
        let cv_core::Point3_ { x, y, z, .. } = *from;
        let binding = [x, y, z];
        let mat = cv_core::Mat::from_slice(&binding)?;
        Ok(mat.try_clone()?)
    }
}

impl<T> TryFromCv<cv_core::Point3_<T>> for cv_core::Mat
where
    T: cv_core::DataType,
{
    type Error = Error;

    fn try_from_cv(from: cv_core::Point3_<T>) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use cv_core::{Point2f, Point2i, Point3f, Point3i};

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_opencv_element_depth() {
        assert_eq!(u8::DEPTH, cv_core::CV_8U);
        assert_eq!(i8::DEPTH, cv_core::CV_8S);
        assert_eq!(u16::DEPTH, cv_core::CV_16U);
        assert_eq!(i16::DEPTH, cv_core::CV_16S);
        assert_eq!(i32::DEPTH, cv_core::CV_32S);
        assert_eq!(f16::DEPTH, cv_core::CV_16F);
        assert_eq!(f32::DEPTH, cv_core::CV_32F);
        assert_eq!(f64::DEPTH, cv_core::CV_64F);
    }

    #[test]
    fn test_mat_size_with_depth() -> Result<()> {
        // 测试 2D Mat
        let mat = cv_core::Mat::new_randn_2d(3, 4, cv_core::CV_32FC2)?;
        assert_eq!(mat.size_with_depth(), vec![3, 4, 2]);

        // 测试 3D Mat
        let mat = cv_core::Mat::new_randn_nd::<f32>(&[2, 3, 4])?;
        assert_eq!(mat.size_with_depth(), vec![2, 3, 4, 1]);

        Ok(())
    }

    #[test]
    fn test_mat_numel() -> Result<()> {
        // 测试 2D Mat 的元素数量
        let mat = cv_core::Mat::new_randn_2d(3, 4, cv_core::CV_32FC2)?;
        assert_eq!(mat.numel(), 24); // 3 * 4 * 2

        // 测试 3D Mat 的元素数量
        let mat = cv_core::Mat::new_randn_nd::<f32>(&[2, 3, 4])?;
        assert_eq!(mat.numel(), 24); // 2 * 3 * 4 * 1

        Ok(())
    }

    #[test]
    fn test_mat_as_slice() -> Result<()> {
        // 测试 f32 类型
        let mat = cv_core::Mat::new_randn_2d(2, 3, cv_core::CV_32F)?;
        let _slice = mat.as_slice::<f32>()?;

        // 测试类型不匹配的情况
        let mat = cv_core::Mat::new_randn_2d(2, 3, cv_core::CV_8U)?;
        assert!(mat.as_slice::<f32>().is_err());

        Ok(())
    }

    #[test]
    fn test_point2_conversion() -> Result<()> {
        // 测试 Point2f
        let point = Point2f::new(1.5, 2.5);
        let mat: cv_core::Mat = (&point).try_into_cv()?;
        let converted_point: Point2f = (&mat).try_into_cv()?;

        assert!((point.x - converted_point.x).abs() < EPSILON as f32);
        assert!((point.y - converted_point.y).abs() < EPSILON as f32);

        // 测试 Point2i
        let point = Point2i::new(1, 2);
        let mat: cv_core::Mat = (&point).try_into_cv()?;
        let converted_point: Point2i = (&mat).try_into_cv()?;

        assert_eq!(point.x, converted_point.x);
        assert_eq!(point.y, converted_point.y);

        Ok(())
    }

    #[test]
    fn test_point3_conversion() -> Result<()> {
        // 测试 Point3f
        let point = Point3f::new(1.5, 2.5, 3.5);
        let mat: cv_core::Mat = (&point).try_into_cv()?;
        let converted_point: Point3f = (&mat).try_into_cv()?;

        assert!((point.x - converted_point.x).abs() < EPSILON as f32);
        assert!((point.y - converted_point.y).abs() < EPSILON as f32);
        assert!((point.z - converted_point.z).abs() < EPSILON as f32);

        // 测试 Point3i
        let point = Point3i::new(1, 2, 3);
        let mat: cv_core::Mat = (&point).try_into_cv()?;
        let converted_point: Point3i = (&mat).try_into_cv()?;

        assert_eq!(point.x, converted_point.x);
        assert_eq!(point.y, converted_point.y);
        assert_eq!(point.z, converted_point.z);

        Ok(())
    }

    #[test]
    fn test_invalid_point_conversion() -> Result<()> {
        // 测试无效的数据长度
        let mat = cv_core::Mat::new_randn_2d(1, 1, cv_core::CV_32F)?;
        assert!(Point2f::try_from_cv(&mat).is_err());
        assert!(Point3f::try_from_cv(&mat).is_err());

        Ok(())
    }

    #[test]
    fn test_type_name() -> Result<()> {
        let mat = cv_core::Mat::new_randn_2d(2, 3, cv_core::CV_32F)?;
        assert_eq!(mat.type_name(), "CV_32FC1");  // 修正：添加通道数 C1

        let mat = cv_core::Mat::new_randn_2d(2, 3, cv_core::CV_8UC3)?;
        assert_eq!(mat.type_name(), "CV_8UC3");   // 这个是正确的，因为已经包含通道数 C3

        // 添加更多测试用例以验证不同类型和通道数
        let mat = cv_core::Mat::new_randn_2d(2, 3, cv_core::CV_64F)?;
        assert_eq!(mat.type_name(), "CV_64FC1");

        let mat = cv_core::Mat::new_randn_2d(2, 3, cv_core::CV_8U)?;
        assert_eq!(mat.type_name(), "CV_8UC1");

        Ok(())
    }

    #[test]
    fn test_mat_new_randn() -> Result<()> {
        // 测试 2D random Mat
        let mat = cv_core::Mat::new_randn_2d(3, 4, cv_core::CV_32F)?;
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 4);
        assert_eq!(mat.depth(), cv_core::CV_32F);

        // 测试 ND random Mat
        let mat = cv_core::Mat::new_randn_nd::<f32>(&[2, 3, 4])?;
        let size = mat.size_with_depth();
        assert_eq!(size[..3], vec![2, 3, 4]);
        assert_eq!(mat.depth(), cv_core::CV_32F);

        Ok(())
    }
}
