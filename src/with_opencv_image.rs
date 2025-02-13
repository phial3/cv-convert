use crate::with_opencv::{MatExt, OpenCvElement};
use crate::{TryFromCv, TryIntoCv};
use anyhow::{Context, Error, Result};
use opencv::{core::Mat, prelude::*};
use std::ops::Deref;

// ImageBuffer -> Mat
impl<P, Container> TryFromCv<image::ImageBuffer<P, Container>> for Mat
where
    P: image::Pixel,
    P::Subpixel: OpenCvElement,
    Container: Deref<Target=[P::Subpixel]> + Clone,
{
    type Error = Error;
    fn try_from_cv(from: image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &ImageBuffer -> Mat
impl<P, Container> TryFromCv<&image::ImageBuffer<P, Container>> for Mat
where
    P: image::Pixel,
    P::Subpixel: OpenCvElement,
    Container: Deref<Target=[P::Subpixel]> + Clone,
{
    type Error = Error;
    fn try_from_cv(from: &image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
        let (width, height) = from.dimensions();
        let cv_type = opencv::core::CV_MAKETYPE(P::Subpixel::DEPTH, P::CHANNEL_COUNT as i32);
        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                height as i32, // 行数
                width as i32,  // 列数
                cv_type,       //
                from.as_ptr() as *mut _,     // 获取 ImageBuffer 底层数据切片
                opencv::core::Mat_AUTO_STEP, // 步长
            )?
                .try_clone()?
        };
        Ok(mat)
    }
}

// &DynamicImage -> Mat
impl TryFromCv<&image::DynamicImage> for Mat {
    type Error = Error;

    fn try_from_cv(from: &image::DynamicImage) -> Result<Self, Self::Error> {
        use image::DynamicImage as D;

        let mat = match from {
            D::ImageLuma8(image) => image.try_into_cv()?,
            D::ImageLumaA8(image) => image.try_into_cv()?,
            D::ImageRgb8(image) => image.try_into_cv()?,
            D::ImageRgba8(image) => image.try_into_cv()?,
            D::ImageLuma16(image) => image.try_into_cv()?,
            D::ImageLumaA16(image) => image.try_into_cv()?,
            D::ImageRgb16(image) => image.try_into_cv()?,
            D::ImageRgba16(image) => image.try_into_cv()?,
            D::ImageRgb32F(image) => image.try_into_cv()?,
            D::ImageRgba32F(image) => image.try_into_cv()?,
            image => anyhow::bail!("the color type {:?} is not supported", image.color()),
        };
        Ok(mat)
    }
}

// DynamicImage -> Mat
impl TryFromCv<image::DynamicImage> for Mat {
    type Error = Error;
    fn try_from_cv(from: image::DynamicImage) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &Mat -> DynamicImage
impl TryFromCv<&Mat> for image::DynamicImage {
    type Error = Error;

    fn try_from_cv(from: &Mat) -> Result<Self, Self::Error> {
        let rows = from.rows();
        let cols = from.cols();
        anyhow::ensure!(
            rows != -1 && cols != -1,
            "Mat with more than 2 dimensions is not supported."
        );

        let depth = from.depth();
        let n_channels = from.channels();
        let width = cols as u32;
        let height = rows as u32;

        let image: image::DynamicImage = match (depth, n_channels) {
            (opencv::core::CV_8U, 1) => mat_to_image_buffer_gray::<u8>(from, width, height).into(),
            (opencv::core::CV_16U, 1) => mat_to_image_buffer_gray::<u16>(from, width, height).into(),
            (opencv::core::CV_8U, 3) => mat_to_image_buffer_rgb::<u8>(from, width, height).into(),
            (opencv::core::CV_16U, 3) => mat_to_image_buffer_rgb::<u16>(from, width, height).into(),
            (opencv::core::CV_32F, 3) => mat_to_image_buffer_rgb::<f32>(from, width, height).into(),
            _ => anyhow::bail!("Mat of type {} is not supported", from.type_name()),
        };

        Ok(image)
    }
}

// Mat -> DynamicImage
impl TryFromCv<Mat> for image::DynamicImage {
    type Error = Error;

    fn try_from_cv(from: Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &Mat -> gray ImageBuffer
impl<T> TryFromCv<&Mat> for image::ImageBuffer<image::Luma<T>, Vec<T>>
where
    image::Luma<T>: image::Pixel,
    T: OpenCvElement + image::Primitive + DataType,
{
    type Error = Error;

    fn try_from_cv(from: &Mat) -> Result<Self, Self::Error> {
        let rows = from.rows();
        let cols = from.cols();
        anyhow::ensure!(
            rows != -1 && cols != -1,
            "Mat with more than 2 dimensions is not supported."
        );

        let depth = from.depth();
        let n_channels = from.channels();
        let width = cols as u32;
        let height = rows as u32;

        anyhow::ensure!(
            n_channels == 1,
            "Unable to convert a multi-channel Mat to a gray image"
        );
        anyhow::ensure!(depth == T::DEPTH, "Subpixel type is not supported");

        let image = mat_to_image_buffer_gray::<T>(from, width, height);
        Ok(image)
    }
}

// Mat -> gray ImageBuffer
impl<T> TryFromCv<Mat> for image::ImageBuffer<image::Luma<T>, Vec<T>>
where
    image::Luma<T>: image::Pixel,
    T: OpenCvElement + image::Primitive + DataType,
{
    type Error = Error;

    fn try_from_cv(from: Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &Mat -> rgb ImageBuffer
impl<T> TryFromCv<&Mat> for image::ImageBuffer<image::Rgb<T>, Vec<T>>
where
    image::Rgb<T>: image::Pixel<Subpixel=T>,
    T: OpenCvElement + image::Primitive + DataType,
{
    type Error = Error;

    fn try_from_cv(from: &Mat) -> Result<Self, Self::Error> {
        let rows = from.rows();
        let cols = from.cols();
        anyhow::ensure!(
            rows != -1 && cols != -1,
            "Mat with more than 2 dimensions is not supported."
        );

        let depth = from.depth();
        let n_channels = from.channels();
        let width = cols as u32;
        let height = rows as u32;

        anyhow::ensure!(
            n_channels == 3,
            "Expect 3 channels, but get {n_channels} channels"
        );
        anyhow::ensure!(depth == T::DEPTH, "Subpixel type is not supported");

        let image = mat_to_image_buffer_rgb::<T>(from, width, height);
        Ok(image)
    }
}

// Mat -> rgb ImageBuffer
impl<T> TryFromCv<Mat> for image::ImageBuffer<image::Rgb<T>, Vec<T>>
where
    image::Rgb<T>: image::Pixel<Subpixel=T>,
    T: OpenCvElement + image::Primitive + DataType,
{
    type Error = Error;

    fn try_from_cv(from: Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// Utility functions
fn mat_to_image_buffer_gray<T>(
    mat: &Mat,
    width: u32,
    height: u32,
) -> image::ImageBuffer<image::Luma<T>, Vec<T>>
where
    T: image::Primitive + OpenCvElement + DataType,
{
    type Image<T> = image::ImageBuffer<image::Luma<T>, Vec<T>>;

    match mat.as_slice::<T>() {
        Ok(slice) => Image::<T>::from_vec(width, height, slice.to_vec()).unwrap(),
        Err(_) => Image::<T>::from_fn(width, height, |col, row| {
            let pixel: T = *mat.at_2d(row as i32, col as i32).unwrap();
            image::Luma([pixel])
        }),
    }
}

fn mat_to_image_buffer_rgb<T>(
    mat: &Mat,
    width: u32,
    height: u32,
) -> image::ImageBuffer<image::Rgb<T>, Vec<T>>
where
    T: image::Primitive + OpenCvElement + DataType,
    image::Rgb<T>: image::Pixel<Subpixel=T>,
{
    type Image<T> = image::ImageBuffer<image::Rgb<T>, Vec<T>>;

    match mat.as_slice::<T>() {
        Ok(slice) => Image::<T>::from_vec(width, height, slice.to_vec()).unwrap(),
        Err(_) => Image::<T>::from_fn(width, height, |col, row| {
            let opencv::core::Point3_::<T> { x, y, z } = *mat.at_2d(row as i32, col as i32).unwrap();
            image::Rgb([x, y, z])
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::with_opencv::MatExt;
    use crate::TryIntoCv;
    use anyhow::Result;
    use itertools::iproduct;
    use opencv::{core::Mat, prelude::*};

    #[test]
    fn convert_opencv_image() -> Result<()> {
        const WIDTH: usize = 250;
        const HEIGHT: usize = 100;

        // gray
        {
            let mat = Mat::new_randn_2d(HEIGHT as i32, WIDTH as i32, opencv::core::CV_8UC1)?;
            let image: image::GrayImage = (&mat).try_into_cv()?;
            let mat2: Mat = (&image).try_into_cv()?;

            iproduct!(0..HEIGHT, 0..WIDTH).try_for_each(|(row, col)| {
                let p1: u8 = *mat.at_2d(row as i32, col as i32)?;
                let p2 = image[(col as u32, row as u32)].0[0];
                let p3: u8 = *mat2.at_2d(row as i32, col as i32)?;
                anyhow::ensure!(p1 == p2 && p1 == p3);
                anyhow::Ok(())
            })?;
        }

        // rgb
        {
            let mat = Mat::new_randn_2d(HEIGHT as i32, WIDTH as i32, opencv::core::CV_8UC3)?;
            let image: image::RgbImage = (&mat).try_into_cv()?;
            let mat2: Mat = (&image).try_into_cv()?;

            iproduct!(0..HEIGHT, 0..WIDTH).try_for_each(|(row, col)| {
                let p1: opencv::core::Point3_<u8> = *mat.at_2d(row as i32, col as i32)?;
                let p2: image::Rgb<u8> = image[(col as u32, row as u32)];
                let p3: opencv::core::Point3_<u8> = *mat2.at_2d(row as i32, col as i32)?;
                anyhow::ensure!(p1 == p3);
                anyhow::ensure!({
                    let a1 = {
                        let opencv::core::Point3_ { x, y, z } = p1;
                        [x, y, z]
                    };
                    let a2 = p2.0;
                    a1 == a2
                });
                anyhow::Ok(())
            })?;
        }

        Ok(())
    }
}
