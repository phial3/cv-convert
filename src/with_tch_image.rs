use crate::with_tch::{TchTensorAsImage, TchTensorImageShape};
use crate::{ToCv, TryToCv};
use anyhow::{ensure, Error, Result};
use std::ops::Deref;

impl<P, Container> ToCv<TchTensorAsImage> for image::ImageBuffer<P, Container>
where
    P: image::Pixel + 'static,
    P::Subpixel: 'static + tch::kind::Element,
    Container: Deref<Target = [P::Subpixel]>,
{
    fn to_cv(&self) -> TchTensorAsImage {
        let (width, height) = self.dimensions();
        let channels = P::CHANNEL_COUNT;
        let tensor =
            tch::Tensor::from_slice(self).view([width as i64, height as i64, channels as i64]);
        TchTensorAsImage {
            tensor,
            kind: TchTensorImageShape::Whc,
        }
    }
}

impl TryToCv<TchTensorAsImage> for image::DynamicImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<TchTensorAsImage, Self::Error> {
        use image::DynamicImage;

        let tensor = match self {
            DynamicImage::ImageLuma8(image) => image.to_cv(),
            DynamicImage::ImageLumaA8(image) => image.to_cv(),
            DynamicImage::ImageRgb8(image) => image.to_cv(),
            DynamicImage::ImageRgba8(image) => image.to_cv(),
            DynamicImage::ImageRgb32F(image) => image.to_cv(),
            DynamicImage::ImageRgba32F(image) => image.to_cv(),
            _ => anyhow::bail!("the color type {:?} is not supported", self.color()),
        };
        Ok(tensor)
    }
}

/// 将 [TchTensorAsImage] 的 tensor 归一化为 [H, W, C] 顺序，返回 (height, width, channels, data)。
fn tensor_to_image_data(from: &TchTensorAsImage) -> Result<(u32, u32, usize, Vec<u8>)> {
    let tensor = from.tensor.shallow_clone();
    ensure!(
        tensor.kind() == tch::kind::Kind::Uint8,
        "expected an uint8 image tensor, but got {:?}",
        tensor.kind()
    );
    ensure!(
        tensor.dim() == 3,
        "the image tensor must have 3 dimensions, but got {}",
        tensor.dim()
    );

    // 归一化到 [H, W, C]
    let hwc = match from.kind {
        TchTensorImageShape::Whc => tensor.permute(&[1, 0, 2]),
        TchTensorImageShape::Hwc => tensor,
        TchTensorImageShape::Chw => tensor.permute(&[1, 2, 0]),
        TchTensorImageShape::Cwh => tensor.permute(&[2, 0, 1]),
    };

    let size = hwc.size();
    let (height, width, channels) = (size[0] as u32, size[1] as u32, size[2] as usize);
    let data: Vec<u8> = Vec::try_from(hwc.flatten(0, -1))?;
    Ok((height, width, channels, data))
}

// TchTensorAsImage -> DynamicImage
impl TryToCv<image::DynamicImage> for TchTensorAsImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<image::DynamicImage, Self::Error> {
        let (height, width, channels, data) = tensor_to_image_data(self)?;
        let image: Option<image::DynamicImage> = match channels {
            1 => {
                image::GrayImage::from_raw(width, height, data).map(image::DynamicImage::ImageLuma8)
            }
            2 => image::GrayAlphaImage::from_raw(width, height, data)
                .map(image::DynamicImage::ImageLumaA8),
            3 => image::RgbImage::from_raw(width, height, data).map(image::DynamicImage::ImageRgb8),
            4 => {
                image::RgbaImage::from_raw(width, height, data).map(image::DynamicImage::ImageRgba8)
            }
            _ => None,
        };
        image.ok_or_else(|| {
            Error::msg(format!(
                "failed to create image from tensor data or unsupported channel count ({})",
                channels
            ))
        })
    }
}

// TchTensorAsImage -> ImageBuffer<Rgb<u8>>
impl TryToCv<image::RgbImage> for TchTensorAsImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<image::RgbImage, Self::Error> {
        let (height, width, channels, data) = tensor_to_image_data(self)?;
        ensure!(channels == 3, "expected 3 channels, but got {}", channels);
        image::RgbImage::from_raw(width, height, data)
            .ok_or_else(|| Error::msg("failed to create RgbImage from tensor data"))
    }
}

// TchTensorAsImage -> ImageBuffer<Rgba<u8>>
impl TryToCv<image::RgbaImage> for TchTensorAsImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<image::RgbaImage, Self::Error> {
        let (height, width, channels, data) = tensor_to_image_data(self)?;
        ensure!(channels == 4, "expected 4 channels, but got {}", channels);
        image::RgbaImage::from_raw(width, height, data)
            .ok_or_else(|| Error::msg("failed to create RgbaImage from tensor data"))
    }
}

// TchTensorAsImage -> ImageBuffer<Luma<u8>>
impl TryToCv<image::GrayImage> for TchTensorAsImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<image::GrayImage, Self::Error> {
        let (height, width, channels, data) = tensor_to_image_data(self)?;
        ensure!(channels == 1, "expected 1 channel, but got {}", channels);
        image::GrayImage::from_raw(width, height, data)
            .ok_or_else(|| Error::msg("failed to create GrayImage from tensor data"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToCv;
    use image::{GenericImageView, GrayImage, Rgb, RgbImage, Rgba, RgbaImage};

    #[test]
    fn tch_tensor_as_image_roundtrip_rgb() {
        let img = RgbImage::from_pixel(2, 3, Rgb([10u8, 20, 30]));
        let t: TchTensorAsImage = (&img).to_cv();
        let back: RgbImage = (&t).try_to_cv().unwrap();
        assert_eq!(back.dimensions(), img.dimensions());
        assert_eq!(back.get_pixel(1, 2), &Rgb([10u8, 20, 30]));
    }

    #[test]
    fn tch_tensor_as_image_roundtrip_rgba() {
        let img = RgbaImage::from_pixel(2, 3, Rgba([10u8, 20, 30, 255]));
        let t: TchTensorAsImage = (&img).to_cv();
        let back: RgbaImage = (&t).try_to_cv().unwrap();
        assert_eq!(back.dimensions(), img.dimensions());
        assert_eq!(back.get_pixel(1, 2), &Rgba([10u8, 20, 30, 255]));
    }

    #[test]
    fn tch_tensor_as_image_roundtrip_gray() {
        let img = GrayImage::from_pixel(2, 3, image::Luma([128u8]));
        let t: TchTensorAsImage = (&img).to_cv();
        let back: GrayImage = (&t).try_to_cv().unwrap();
        assert_eq!(back.dimensions(), img.dimensions());
        assert_eq!(back.get_pixel(1, 2), &image::Luma([128u8]));
    }

    #[test]
    fn tch_tensor_as_image_to_dynamic() {
        let img = RgbaImage::from_pixel(2, 3, Rgba([10u8, 20, 30, 255]));
        let t: TchTensorAsImage = (&img).to_cv();
        let dyn_image: image::DynamicImage = (&t).try_to_cv().unwrap();
        assert!(matches!(dyn_image, image::DynamicImage::ImageRgba8(_)));
        assert_eq!(dyn_image.dimensions(), (2, 3));
    }
}
