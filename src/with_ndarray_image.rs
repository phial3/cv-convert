use crate::pixel::PixelType;
use crate::with_ndarray::ArrayWithFormat;
use crate::TryToCv;
use anyhow::{Error, Result};
use image::{GrayAlphaImage, GrayImage, ImageBuffer, Luma, LumaA, Rgb, RgbImage, Rgba, RgbaImage};
use ndarray::Array3;

// Array3<T> -> RgbImage
impl<T> TryToCv<RgbImage> for Array3<T>
where
    T: PixelType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<RgbImage, Self::Error> {
        let (height, width, channels) = self.dim();
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
                    self[[y, x, 0]].to_u8().unwrap(),
                    self[[y, x, 1]].to_u8().unwrap(),
                    self[[y, x, 2]].to_u8().unwrap(),
                ]);
                img.put_pixel(x as u32, y as u32, pixel);
            }
        }

        Ok(img)
    }
}

// RgbImage -> Array3<T>
impl<T> TryToCv<Array3<T>> for RgbImage
where
    T: PixelType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<Array3<T>, Self::Error> {
        let (width, height) = self.dimensions();
        let mut array = Array3::zeros((height as usize, width as usize, 3));

        // 将 image 的 RGB 数据拷贝到 frame 中
        // let data_arr = Array3::from_shape_vec((height as usize, width as usize, 3), from.into_raw())
        //     .expect("Failed to create ndarray from raw image data");

        for y in 0..height {
            for x in 0..width {
                let pixel = self.get_pixel(x, y);
                array[[y as usize, x as usize, 0]] = T::from(pixel[0]).unwrap();
                array[[y as usize, x as usize, 1]] = T::from(pixel[1]).unwrap();
                array[[y as usize, x as usize, 2]] = T::from(pixel[2]).unwrap();
            }
        }

        Ok(array)
    }
}

// Array3<T> -> RgbaImage
impl<T> TryToCv<RgbaImage> for Array3<T>
where
    T: PixelType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<RgbaImage, Self::Error> {
        let (height, width, channels) = self.dim();
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
                    self[[y, x, 0]].to_u8().unwrap(),
                    self[[y, x, 1]].to_u8().unwrap(),
                    self[[y, x, 2]].to_u8().unwrap(),
                    self[[y, x, 3]].to_u8().unwrap(),
                ]);
                img.put_pixel(x as u32, y as u32, pixel);
            }
        }

        Ok(img)
    }
}

// RgbaImage -> Array3<T>
impl<T> TryToCv<Array3<T>> for RgbaImage
where
    T: PixelType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<Array3<T>, Self::Error> {
        let (width, height) = self.dimensions();
        let mut array = Array3::zeros((height as usize, width as usize, 4));

        for y in 0..height {
            for x in 0..width {
                let pixel = self.get_pixel(x, y);
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
impl<T> TryToCv<GrayImage> for Array3<T>
where
    T: PixelType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<GrayImage, Self::Error> {
        let (height, width, channels) = self.dim();
        if channels != 1 {
            return Err(Error::msg(format!(
                "Expected {} channel, but got {}",
                1, channels
            )));
        }

        let mut img = GrayImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let pixel = Luma([self[[y, x, 0]].to_u8().unwrap()]);
                img.put_pixel(x as u32, y as u32, pixel);
            }
        }

        Ok(img)
    }
}

// GrayImage -> Array3<T>
impl<T> TryToCv<Array3<T>> for GrayImage
where
    T: PixelType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<Array3<T>, Self::Error> {
        let (width, height) = self.dimensions();
        let mut array = Array3::zeros((height as usize, width as usize, 1));

        for y in 0..height {
            for x in 0..width {
                array[[y as usize, x as usize, 0]] = T::from(self.get_pixel(x, y)[0]).unwrap();
            }
        }

        Ok(array)
    }
}

// Array3<T> -> GrayAlphaImage
impl<T> TryToCv<GrayAlphaImage> for Array3<T>
where
    T: PixelType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<GrayAlphaImage, Self::Error> {
        let (height, width, channels) = self.dim();
        if channels != 2 {
            return Err(Error::msg(format!(
                "Expected {} channels, but got {}",
                2, channels
            )));
        }

        let mut img = GrayAlphaImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let pixel = LumaA([
                    self[[y, x, 0]].to_u8().unwrap(),
                    self[[y, x, 1]].to_u8().unwrap(),
                ]);
                img.put_pixel(x as u32, y as u32, pixel);
            }
        }

        Ok(img)
    }
}

// GrayAlphaImage -> Array3<T>
impl<T> TryToCv<Array3<T>> for GrayAlphaImage
where
    T: PixelType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<Array3<T>, Self::Error> {
        let (width, height) = self.dimensions();
        let mut array = Array3::zeros((height as usize, width as usize, 2));

        for y in 0..height {
            for x in 0..width {
                let pixel = self.get_pixel(x, y);
                array[[y as usize, x as usize, 0]] = T::from(pixel[0]).unwrap();
                array[[y as usize, x as usize, 1]] = T::from(pixel[1]).unwrap();
            }
        }

        Ok(array)
    }
}

// Array3<T> -> DynamicImage
impl<T> TryToCv<image::DynamicImage> for Array3<T>
where
    T: PixelType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<image::DynamicImage, Self::Error> {
        let (_, _, channels) = self.dim();
        match channels {
            1 => Ok(<Array3<T> as TryToCv<GrayImage>>::try_to_cv(self)?.into()),
            2 => Ok(<Array3<T> as TryToCv<GrayAlphaImage>>::try_to_cv(self)?.into()),
            3 => Ok(<Array3<T> as TryToCv<RgbImage>>::try_to_cv(self)?.into()),
            4 => Ok(<Array3<T> as TryToCv<RgbaImage>>::try_to_cv(self)?.into()),
            n => Err(Error::msg(format!(
                "Unsupported channel count for DynamicImage: {}",
                n
            ))),
        }
    }
}

// DynamicImage -> Array3<T>
impl<T> TryToCv<Array3<T>> for image::DynamicImage
where
    T: PixelType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<Array3<T>, Self::Error> {
        use image::DynamicImage;

        // 统一转换为 RGBA，保证通道数一致且可逆
        let rgba = self.to_rgba8();
        rgba.try_to_cv()
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
        let rgb_image: RgbImage = rgb_array.try_to_cv().unwrap();
        let back_rgb_array: Array3<u8> = rgb_image.try_to_cv().unwrap();

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
        let rgba_image: RgbaImage = rgba_array.try_to_cv().unwrap();
        let back_rgba_array: Array3<u8> = rgba_image.try_to_cv().unwrap();

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
        let gray_image: GrayImage = gray_array.try_to_cv().unwrap();
        let back_gray_array: Array3<u8> = gray_image.try_to_cv().unwrap();

        // 比较转换前后的数组
        assert_eq!(gray_array.shape(), back_gray_array.shape());
        assert_eq!(gray_array, back_gray_array);
    }

    #[test]
    fn test_gray_alpha_to_array3() {
        // 创建一个具有渐变的灰度+alpha 测试数组
        let mut gray_alpha_array = Array3::<u8>::zeros((100, 100, 2));
        for i in 0..100 {
            for j in 0..100 {
                gray_alpha_array[[i, j, 0]] = ((i + j) / 2) as u8; // Gray channel
                gray_alpha_array[[i, j, 1]] = 255; // Alpha channel
            }
        }

        // 转换到图像并返回
        let gray_alpha_image: GrayAlphaImage = gray_alpha_array.try_to_cv().unwrap();
        let back_gray_alpha_array: Array3<u8> = gray_alpha_image.try_to_cv().unwrap();

        // 比较转换前后的数组
        assert_eq!(gray_alpha_array.shape(), back_gray_alpha_array.shape());
        assert_eq!(gray_alpha_array, back_gray_alpha_array);
    }

    #[test]
    fn test_dynamic_image_to_array3() {
        // DynamicImage -> Array3 (RGBA)
        let rgb = RgbImage::from_pixel(2, 2, Rgb([1u8, 2, 3]));
        let dyn_image: image::DynamicImage = rgb.clone().into();
        let array: Array3<u8> = dyn_image.try_to_cv().unwrap();
        assert_eq!(array.shape(), &[2, 2, 4]); // to_rgba8 产生 4 通道

        // Array3 -> DynamicImage (4 通道)
        let back_dyn: image::DynamicImage = array.try_to_cv().unwrap();
        assert!(matches!(back_dyn, image::DynamicImage::ImageRgba8(_)));
    }

    #[test]
    fn test_invalid_channel_conversions() {
        // 测试通道数不匹配的情况
        let invalid_rgb = Array3::<u8>::zeros((100, 100, 2));
        let result: Result<RgbImage, _> = invalid_rgb.try_to_cv();
        assert!(result.is_err());

        let invalid_rgba = Array3::<u8>::zeros((100, 100, 3));
        let result: Result<RgbaImage, _> = invalid_rgba.try_to_cv();
        assert!(result.is_err());

        let invalid_gray = Array3::<u8>::zeros((100, 100, 2));
        let result: Result<GrayImage, _> = invalid_gray.try_to_cv();
        assert!(result.is_err());
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

        let rgb_image: RgbImage = f32_array.try_to_cv().unwrap();
        let back_f32_array: Array3<f32> = rgb_image.try_to_cv().unwrap();

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

        let rgb_image: RgbImage = edge_array.try_to_cv().unwrap();
        let back_edge_array: Array3<u8> = rgb_image.try_to_cv().unwrap();

        assert_eq!(edge_array, back_edge_array);
    }

    #[test]
    fn test_empty_image() {
        // 测试空图像（1x1）的情况
        let empty_array = Array3::<u8>::zeros((1, 1, 3));
        let rgb_image: RgbImage = empty_array.try_to_cv().unwrap();
        let back_empty_array: Array3<u8> = rgb_image.try_to_cv().unwrap();

        assert_eq!(empty_array, back_empty_array);
    }
}
