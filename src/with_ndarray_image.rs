use crate::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use anyhow::{Error, Result};
use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use ndarray::Array3;
use num_traits::{NumCast, Zero};

// Array3<T> -> RgbImage
impl<T> TryFromCv<Array3<T>> for RgbImage
where
    T: Copy + Clone + NumCast + Zero,
{
    type Error = Error;

    fn try_from_cv(from: Array3<T>) -> Result<Self, Self::Error> {
        let (height, width, channels) = from.dim();
        if channels != 3 {
            return Err(Error::msg(format!(
                "Expected {} channels, but got {}",
                3, channels
            )));
        }

        let mut img = RgbImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let pixel = Rgb([
                    from[[y, x, 0]].to_u8().unwrap(),
                    from[[y, x, 1]].to_u8().unwrap(),
                    from[[y, x, 2]].to_u8().unwrap(),
                ]);
                img.put_pixel(x as u32, y as u32, pixel);
            }
        }

        Ok(img)
    }
}

// RgbImage -> Array3<T>
impl<T> TryFromCv<RgbImage> for Array3<T>
where
    T: Copy + Clone + NumCast + Zero,
{
    type Error = Error;

    fn try_from_cv(from: RgbImage) -> Result<Self, Self::Error> {
        let (width, height) = from.dimensions();
        let mut array = Array3::zeros((height as usize, width as usize, 3));

        // 将 image 的 RGB 数据拷贝到 frame 中
        // let data_arr = Array3::from_shape_vec((height as usize, width as usize, 3), from.into_raw())
        //     .expect("Failed to create ndarray from raw image data");

        for y in 0..height {
            for x in 0..width {
                let pixel = from.get_pixel(x, y);
                array[[y as usize, x as usize, 0]] = T::from(pixel[0]).unwrap();
                array[[y as usize, x as usize, 1]] = T::from(pixel[1]).unwrap();
                array[[y as usize, x as usize, 2]] = T::from(pixel[2]).unwrap();
            }
        }

        Ok(array)
    }
}

// Array3<T> -> RgbaImage
impl<T> TryFromCv<Array3<T>> for RgbaImage
where
    T: Copy + Clone + NumCast + Zero,
{
    type Error = Error;

    fn try_from_cv(from: Array3<T>) -> Result<Self, Self::Error> {
        let (height, width, channels) = from.dim();
        if channels != 4 {
            return Err(Error::msg(format!(
                "Expected {} channels, but got {}",
                4, channels
            )));
        }

        let mut img = RgbaImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let pixel = Rgba([
                    from[[y, x, 0]].to_u8().unwrap(),
                    from[[y, x, 1]].to_u8().unwrap(),
                    from[[y, x, 2]].to_u8().unwrap(),
                    from[[y, x, 3]].to_u8().unwrap(),
                ]);
                img.put_pixel(x as u32, y as u32, pixel);
            }
        }

        Ok(img)
    }
}

// RgbaImage -> Array3<T>
impl<T> TryFromCv<RgbaImage> for Array3<T>
where
    T: Copy + Clone + NumCast + Zero,
{
    type Error = Error;

    fn try_from_cv(from: RgbaImage) -> Result<Self, Self::Error> {
        let (width, height) = from.dimensions();
        let mut array = Array3::zeros((height as usize, width as usize, 4));

        for y in 0..height {
            for x in 0..width {
                let pixel = from.get_pixel(x, y);
                array[[y as usize, x as usize, 0]] = T::from(pixel[0]).unwrap();
                array[[y as usize, x as usize, 1]] = T::from(pixel[1]).unwrap();
                array[[y as usize, x as usize, 2]] = T::from(pixel[2]).unwrap();
                array[[y as usize, x as usize, 3]] = T::from(pixel[3]).unwrap();
            }
        }

        Ok(array)
    }
}

// Array3<T> -> GrayImage
impl<T> TryFromCv<Array3<T>> for GrayImage
where
    T: Copy + Clone + NumCast + Zero,
{
    type Error = Error;

    fn try_from_cv(from: Array3<T>) -> Result<Self, Self::Error> {
        let (height, width, channels) = from.dim();
        if channels != 1 {
            return Err(Error::msg(format!(
                "Expected {} channel, but got {}",
                1, channels
            )));
        }

        let mut img = GrayImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let pixel = Luma([from[[y, x, 0]].to_u8().unwrap()]);
                img.put_pixel(x as u32, y as u32, pixel);
            }
        }

        Ok(img)
    }
}

// GrayImage -> Array3<T>
impl<T> TryFromCv<GrayImage> for Array3<T>
where
    T: Copy + Clone + NumCast + Zero,
{
    type Error = Error;

    fn try_from_cv(from: GrayImage) -> Result<Self, Self::Error> {
        let (width, height) = from.dimensions();
        let mut array = Array3::zeros((height as usize, width as usize, 1));

        for y in 0..height {
            for x in 0..width {
                array[[y as usize, x as usize, 0]] = T::from(from.get_pixel(x, y)[0]).unwrap();
            }
        }

        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_rgb_to_array3() {
        // 创建一个具有特定模式的 RGB 测试数组
        let mut rgb_array = Array3::<u8>::zeros((100, 100, 3));
        for i in 0..100 {
            for j in 0..100 {
                rgb_array[[i, j, 0]] = (i % 256) as u8; // R channel
                rgb_array[[i, j, 1]] = (j % 256) as u8; // G channel
                rgb_array[[i, j, 2]] = ((i + j) % 256) as u8; // B channel
            }
        }

        // 转换到图像并返回
        let rgb_image = RgbImage::try_from_cv(rgb_array.clone()).unwrap();
        let back_rgb_array = Array3::<u8>::try_from_cv(rgb_image).unwrap();

        // 比较转换前后的数组
        assert_eq!(rgb_array.shape(), back_rgb_array.shape());
        assert_eq!(rgb_array, back_rgb_array);
    }

    #[test]
    fn test_rgba_to_array3() {
        // 创建一个具有特定模式的 RGBA 测试数组
        let mut rgba_array = Array3::<u8>::zeros((100, 100, 4));
        for i in 0..100 {
            for j in 0..100 {
                rgba_array[[i, j, 0]] = (i % 256) as u8; // R channel
                rgba_array[[i, j, 1]] = (j % 256) as u8; // G channel
                rgba_array[[i, j, 2]] = ((i + j) % 256) as u8; // B channel
                rgba_array[[i, j, 3]] = 255; // A channel
            }
        }

        // 转换到图像并返回
        let rgba_image = RgbaImage::try_from_cv(rgba_array.clone()).unwrap();
        let back_rgba_array = Array3::<u8>::try_from_cv(rgba_image).unwrap();

        // 比较转换前后的数组
        assert_eq!(rgba_array.shape(), back_rgba_array.shape());
        assert_eq!(rgba_array, back_rgba_array);
    }

    #[test]
    fn test_gray_to_array3() {
        // 创建一个具有渐变的灰度测试数组
        let mut gray_array = Array3::<u8>::zeros((100, 100, 1));
        for i in 0..100 {
            for j in 0..100 {
                gray_array[[i, j, 0]] = ((i + j) / 2) as u8;
            }
        }

        // 转换到图像并返回
        let gray_image = GrayImage::try_from_cv(gray_array.clone()).unwrap();
        let back_gray_array = Array3::<u8>::try_from_cv(gray_image).unwrap();

        // 比较转换前后的数组
        assert_eq!(gray_array.shape(), back_gray_array.shape());
        assert_eq!(gray_array, back_gray_array);
    }

    #[test]
    fn test_invalid_channel_conversions() {
        // 测试通道数不匹配的情况
        let invalid_rgb = Array3::<u8>::zeros((100, 100, 2));
        assert!(RgbImage::try_from_cv(invalid_rgb).is_err());

        let invalid_rgba = Array3::<u8>::zeros((100, 100, 3));
        assert!(RgbaImage::try_from_cv(invalid_rgba).is_err());

        let invalid_gray = Array3::<u8>::zeros((100, 100, 2));
        assert!(GrayImage::try_from_cv(invalid_gray).is_err());
    }

    #[test]
    fn test_different_types() {
        // 测试不同数值类型的转换
        let mut f32_array = Array3::<f32>::zeros((100, 100, 3));
        for i in 0..100 {
            for j in 0..100 {
                // 映射值到合适的范围，确保能准确量化到 u8
                // 将值范围设置为 0 到 1，每个步进至少 1/255
                f32_array[[i, j, 0]] = (i as f32 * 255.0 / 100.0) / 255.0;
                f32_array[[i, j, 1]] = (j as f32 * 255.0 / 100.0) / 255.0;
                f32_array[[i, j, 2]] = ((i + j) as f32 * 255.0 / 200.0) / 255.0;
            }
        }

        let rgb_image = RgbImage::try_from_cv(f32_array.clone()).unwrap();
        let back_f32_array = Array3::<f32>::try_from_cv(rgb_image).unwrap();

        for i in 0..100 {
            for j in 0..100 {
                for k in 0..3 {
                    let original = f32_array[[i, j, k]];
                    let converted = back_f32_array[[i, j, k]];

                    // 计算期望的量化值
                    let u8_val = (original * 255.0).round() as u8;
                    let expected = u8_val as f32 / 255.0;

                    assert!(expected - converted < 1.0,
                        "Values different at [{}, {}, {}]: original {:.3} -> expected {:.3} but got {:.3}",
                        i, j, k,
                        original,
                        expected,
                        converted
                    );
                }
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        // 测试边界值
        let mut edge_array = Array3::<u8>::zeros((2, 2, 3));
        edge_array[[0, 0, 0]] = 0;
        edge_array[[0, 0, 1]] = 255;
        edge_array[[0, 0, 2]] = 128;
        edge_array[[1, 1, 0]] = 255;
        edge_array[[1, 1, 1]] = 0;
        edge_array[[1, 1, 2]] = 128;

        let rgb_image = RgbImage::try_from_cv(edge_array.clone()).unwrap();
        let back_edge_array = Array3::<u8>::try_from_cv(rgb_image).unwrap();

        assert_eq!(edge_array, back_edge_array);
    }

    #[test]
    fn test_empty_image() {
        // 测试空图像（1x1）的情况
        let empty_array = Array3::<u8>::zeros((1, 1, 3));
        let rgb_image = RgbImage::try_from_cv(empty_array.clone()).unwrap();
        let back_empty_array = Array3::<u8>::try_from_cv(rgb_image).unwrap();

        assert_eq!(empty_array, back_empty_array);
    }
}
