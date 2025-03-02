use crate::with_opencv::{MatExt, OpenCvElement};
use crate::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use anyhow::{Context, Error, Result};
use opencv::prelude::*;
use std::ops::Deref;

// ImageBuffer -> Mat
impl<P, Container> TryFromCv<image::ImageBuffer<P, Container>> for Mat
where
    P: image::Pixel,
    P::Subpixel: OpenCvElement,
    Container: Deref<Target = [P::Subpixel]> + Clone,
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
    Container: Deref<Target = [P::Subpixel]> + Clone,
{
    type Error = Error;
    fn try_from_cv(from: &image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
        let (width, height) = from.dimensions();
        let cv_type = opencv::core::CV_MAKETYPE(P::Subpixel::DEPTH, P::CHANNEL_COUNT as i32);
        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe_def(
                height as i32,           // 行数
                width as i32,            // 列数
                cv_type,                 // 类型
                from.as_ptr() as *mut _, // image 数据
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
        use image::DynamicImage;

        let mat = match from {
            DynamicImage::ImageLuma8(image) => image.try_into_cv()?,
            DynamicImage::ImageLumaA8(image) => image.try_into_cv()?,
            DynamicImage::ImageRgb8(image) => image.try_into_cv()?,
            DynamicImage::ImageRgba8(image) => image.try_into_cv()?,
            DynamicImage::ImageLuma16(image) => image.try_into_cv()?,
            DynamicImage::ImageLumaA16(image) => image.try_into_cv()?,
            DynamicImage::ImageRgb16(image) => image.try_into_cv()?,
            DynamicImage::ImageRgba16(image) => image.try_into_cv()?,
            DynamicImage::ImageRgb32F(image) => image.try_into_cv()?,
            DynamicImage::ImageRgba32F(image) => image.try_into_cv()?,
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
            (opencv::core::CV_16U, 1) => {
                mat_to_image_buffer_gray::<u16>(from, width, height).into()
            }
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
    image::Rgb<T>: image::Pixel<Subpixel = T>,
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
    image::Rgb<T>: image::Pixel<Subpixel = T>,
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
    image::Rgb<T>: image::Pixel<Subpixel = T>,
{
    type Image<T> = image::ImageBuffer<image::Rgb<T>, Vec<T>>;

    match mat.as_slice::<T>() {
        Ok(slice) => Image::<T>::from_vec(width, height, slice.to_vec()).unwrap(),
        Err(_) => Image::<T>::from_fn(width, height, |col, row| {
            let opencv::core::Point3_::<T> { x, y, z } =
                *mat.at_2d(row as i32, col as i32).unwrap();
            image::Rgb([x, y, z])
        }),
    }
}

/// RgbImage 转换为 OpenCV Mat
#[allow(dead_code)]
fn image_to_mat(img: &image::RgbImage) -> Result<Mat> {
    let width = img.width() as i32;
    let height = img.height() as i32;

    // 创建 RGB Mat
    let rgb_mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe_def(
            height,
            width,
            opencv::core::CV_8UC3,
            img.as_raw().as_ptr() as *mut _,
        )
        .context("Failed to create RGB Mat")?
    };

    // 转换为 openCV 默认 BGR 格式
    let mut bgr_mat = Mat::default();
    opencv::imgproc::cvt_color_def(&rgb_mat, &mut bgr_mat, opencv::imgproc::COLOR_RGB2BGR)
        .context("Failed to convert RGB to BGR")?;

    Ok(bgr_mat)
}

/// OpenCV Mat 转换为 RgbImage
#[allow(dead_code)]
fn mat_to_image(mat: &Mat) -> Result<image::RgbImage> {
    if mat.empty() {
        return Err(anyhow::anyhow!("Input Mat is empty"));
    }

    // 转换为 RGB
    let mut rgb_mat = Mat::default();
    opencv::imgproc::cvt_color_def(mat, &mut rgb_mat, opencv::imgproc::COLOR_BGR2RGB)
        .context("Failed to convert BGR to RGB")?;

    let width = rgb_mat.cols() as u32;
    let height = rgb_mat.rows() as u32;

    // 获取连续数据
    let buffer = rgb_mat
        .data_bytes()
        .context("Failed to get mat data")?
        .to_vec();

    image::RgbImage::from_raw(width, height, buffer)
        .context("Failed to create RgbImage from Mat data")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::with_opencv::MatExt;
    use crate::TryIntoCv;
    use anyhow::{Context, Result};
    use opencv::prelude::*;

    #[test]
    fn convert_opencv_image() -> Result<()> {
        const WIDTH: usize = 250;
        const HEIGHT: usize = 100;

        // gray
        {
            let mat = Mat::new_randn_2d(HEIGHT as i32, WIDTH as i32, opencv::core::CV_8UC1)?;
            let image: image::GrayImage = (&mat).try_into_cv()?;
            let mat2: Mat = (&image).try_into_cv()?;

            itertools::iproduct!(0..HEIGHT, 0..WIDTH).try_for_each(|(row, col)| {
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

            itertools::iproduct!(0..HEIGHT, 0..WIDTH).try_for_each(|(row, col)| {
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

    fn create_rgb_image() -> image::RgbImage {
        let width = 320;
        let height = 240;
        let mut img = image::RgbImage::new(width, height);

        // 创建一个简单的渐变图案
        for y in 0..height {
            for x in 0..width {
                let r = (x as f32 / width as f32 * 255.0) as u8;
                let g = (y as f32 / height as f32 * 255.0) as u8;
                let b = ((x + y) as f32 / (width + height) as f32 * 255.0) as u8;
                img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
        img
    }

    fn create_test_mat() -> Result<Mat> {
        let width = 320;
        let height = 240;

        // 创建一个 3 通道的空白图像
        let mut mat = unsafe {
            Mat::new_rows_cols(height, width, opencv::core::CV_8UC3)
                .context("Failed to create Mat")?
        };

        // 创建渐变效果
        for y in 0..height {
            for x in 0..width {
                let b = (y * 255 / height) as u8;
                let g = (x * 255 / width) as u8;
                let r = ((x + y) * 255 / (width + height)) as u8;

                // 使用 Vec3b 设置像素值
                let color = opencv::core::Vec3b::from([b, g, r]);
                mat.at_2d_mut::<opencv::core::Vec3b>(y, x)
                    .context("Failed to set pixel value")?
                    .copy_from_slice(&color.0);
            }
        }

        // 添加一些测试图形
        // 1. 画一个矩形
        opencv::imgproc::rectangle_points(
            &mut mat,
            opencv::core::Point::new(50, 50),
            opencv::core::Point::new(100, 100),
            opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0), // 红色
            2,                                               // 线宽
            opencv::imgproc::LINE_8,
            0,
        )
        .context("Failed to draw rectangle")?;

        // 2. 画一个圆
        opencv::imgproc::circle(
            &mut mat,
            opencv::core::Point::new(width / 2, height / 2),
            40,
            opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0), // 绿色
            2,                                               // 线宽
            opencv::imgproc::LINE_8,
            0,
        )
        .context("Failed to draw circle")?;

        // 3. 画一条线
        opencv::imgproc::line(
            &mut mat,
            opencv::core::Point::new(0, 0),
            opencv::core::Point::new(width - 1, height - 1),
            opencv::core::Scalar::new(255.0, 0.0, 0.0, 0.0), // 蓝色
            2,                                               // 线宽
            opencv::imgproc::LINE_8,
            0,
        )
        .context("Failed to draw line")?;

        Ok(mat)
    }

    #[test]
    fn test_mat_to_image() -> Result<()> {
        let mat = create_test_mat()?;

        let img = mat_to_image(&mat)?;

        assert_eq!(img.width(), mat.cols() as u32);
        assert_eq!(img.height(), mat.rows() as u32);

        img.save("/tmp/test_mat_to_image.png")
            .expect("mat_to_image error");

        Ok(())
    }

    #[test]
    fn test_image_to_mat() -> Result<()> {
        let img = create_rgb_image();

        let mat = image_to_mat(&img)?;

        assert_eq!(mat.cols(), img.width() as i32);
        assert_eq!(mat.rows(), img.height() as i32);
        assert_eq!(mat.channels(), 3);

        println!("test_image_to_mat depth:{}", mat.depth());

        Ok(())
    }
}
