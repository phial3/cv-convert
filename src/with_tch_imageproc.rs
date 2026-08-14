use crate::{ToCv, TryToCv};
use anyhow::{ensure, Error, Result};

// RgbImage -> Tensor
impl ToCv<tch::Tensor> for imageproc::image::RgbImage {
    fn to_cv(&self) -> tch::Tensor {
        let (width, height) = self.dimensions();
        tch::Tensor::from_slice(self.as_raw()).view([height as i64, width as i64, 3i64])
    }
}

// Tensor -> RgbImage
impl TryToCv<imageproc::image::RgbImage> for tch::Tensor {
    type Error = Error;

    fn try_to_cv(&self) -> Result<imageproc::image::RgbImage, Self::Error> {
        let from = self.shallow_clone();
        ensure!(
            from.kind() == tch::kind::Kind::Uint8,
            "expected an uint8 tensor, but got {:?}",
            from.kind()
        );
        let size = from.size();
        ensure!(size.len() == 3, "expected a 3D tensor");
        ensure!(size[2] == 3, "expected 3 channels, but got {}", size[2]);
        let (height, width) = (size[0] as u32, size[1] as u32);

        let data: Vec<u8> = Vec::try_from(from.flatten(0, -1))?;
        imageproc::image::RgbImage::from_raw(width, height, data)
            .ok_or_else(|| Error::msg("failed to create image from tensor data"))
    }
}

// GrayImage -> Tensor
impl ToCv<tch::Tensor> for imageproc::image::GrayImage {
    fn to_cv(&self) -> tch::Tensor {
        let (width, height) = self.dimensions();
        tch::Tensor::from_slice(self.as_raw()).view([height as i64, width as i64, 1i64])
    }
}

// Tensor -> GrayImage
impl TryToCv<imageproc::image::GrayImage> for tch::Tensor {
    type Error = Error;

    fn try_to_cv(&self) -> Result<imageproc::image::GrayImage, Self::Error> {
        let from = self.shallow_clone();
        ensure!(
            from.kind() == tch::kind::Kind::Uint8,
            "expected an uint8 tensor, but got {:?}",
            from.kind()
        );
        let size = from.size();
        ensure!(size.len() == 3, "expected a 3D tensor");
        ensure!(size[2] == 1, "expected 1 channel, but got {}", size[2]);
        let (height, width) = (size[0] as u32, size[1] as u32);

        let data: Vec<u8> = Vec::try_from(from.flatten(0, -1))?;
        imageproc::image::GrayImage::from_raw(width, height, data)
            .ok_or_else(|| Error::msg("failed to create image from tensor data"))
    }
}

// RgbaImage -> Tensor
impl ToCv<tch::Tensor> for imageproc::image::RgbaImage {
    fn to_cv(&self) -> tch::Tensor {
        let (width, height) = self.dimensions();
        tch::Tensor::from_slice(self.as_raw()).view([height as i64, width as i64, 4i64])
    }
}

// Tensor -> RgbaImage
impl TryToCv<imageproc::image::RgbaImage> for tch::Tensor {
    type Error = Error;

    fn try_to_cv(&self) -> Result<imageproc::image::RgbaImage, Self::Error> {
        let from = self.shallow_clone();
        ensure!(
            from.kind() == tch::kind::Kind::Uint8,
            "expected an uint8 tensor, but got {:?}",
            from.kind()
        );
        let size = from.size();
        ensure!(size.len() == 3, "expected a 3D tensor");
        ensure!(size[2] == 4, "expected 4 channels, but got {}", size[2]);
        let (height, width) = (size[0] as u32, size[1] as u32);

        let data: Vec<u8> = Vec::try_from(from.flatten(0, -1))?;
        imageproc::image::RgbaImage::from_raw(width, height, data)
            .ok_or_else(|| Error::msg("failed to create image from tensor data"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb, RgbImage, Rgba, RgbaImage};

    #[test]
    fn tensor_image_roundtrip() {
        let img = RgbImage::from_pixel(2, 3, Rgb([10u8, 20, 30]));
        let tensor: tch::Tensor = img.to_cv();
        let back: RgbImage = (tensor).try_to_cv().unwrap();
        assert_eq!(back.dimensions(), img.dimensions());
        assert_eq!(back.get_pixel(1, 2), &Rgb([10u8, 20, 30]));
    }

    #[test]
    fn tensor_gray_image_roundtrip() {
        let img = image::GrayImage::from_pixel(2, 3, Luma([128u8]));
        let tensor: tch::Tensor = img.to_cv();
        let back: image::GrayImage = (tensor).try_to_cv().unwrap();
        assert_eq!(back.dimensions(), img.dimensions());
        assert_eq!(back.get_pixel(1, 2), &Luma([128u8]));
    }

    #[test]
    fn tensor_rgba_image_roundtrip() {
        let img = RgbaImage::from_pixel(2, 3, Rgba([10u8, 20, 30, 255]));
        let tensor: tch::Tensor = img.to_cv();
        let back: RgbaImage = (tensor).try_to_cv().unwrap();
        assert_eq!(back.dimensions(), img.dimensions());
        assert_eq!(back.get_pixel(1, 2), &Rgba([10u8, 20, 30, 255]));
    }
}
