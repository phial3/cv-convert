use crate::{ToCv, TryToCv};
use anyhow::{Error, Result};
use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};

// ImageBuffer -> DynamicImage
impl ToCv<DynamicImage> for RgbImage {
    fn to_cv(&self) -> DynamicImage {
        DynamicImage::ImageRgb8(self.clone())
    }
}

impl ToCv<DynamicImage> for RgbaImage {
    fn to_cv(&self) -> DynamicImage {
        DynamicImage::ImageRgba8(self.clone())
    }
}

impl ToCv<DynamicImage> for GrayImage {
    fn to_cv(&self) -> DynamicImage {
        DynamicImage::ImageLuma8(self.clone())
    }
}

impl ToCv<DynamicImage> for image::GrayAlphaImage {
    fn to_cv(&self) -> DynamicImage {
        DynamicImage::ImageLumaA8(self.clone())
    }
}

// DynamicImage -> ImageBuffer
impl TryToCv<RgbImage> for DynamicImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<RgbImage, Self::Error> {
        Ok(self.to_rgb8())
    }
}

impl TryToCv<RgbaImage> for DynamicImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<RgbaImage, Self::Error> {
        Ok(self.to_rgba8())
    }
}

impl TryToCv<GrayImage> for DynamicImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<GrayImage, Self::Error> {
        Ok(self.to_luma8())
    }
}

impl TryToCv<image::GrayAlphaImage> for DynamicImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<image::GrayAlphaImage, Self::Error> {
        Ok(self.to_luma_alpha8())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb, Rgba};

    #[test]
    fn image_buffer_to_dynamic_image() {
        let rgb = RgbImage::from_pixel(2, 2, Rgb([1u8, 2, 3]));
        let dyn_image: DynamicImage = (&rgb).to_cv();
        assert!(matches!(dyn_image, DynamicImage::ImageRgb8(_)));

        let rgba = RgbaImage::from_pixel(2, 2, Rgba([1u8, 2, 3, 4]));
        let dyn_image: DynamicImage = (&rgba).to_cv();
        assert!(matches!(dyn_image, DynamicImage::ImageRgba8(_)));

        let gray = GrayImage::from_pixel(2, 2, Luma([128u8]));
        let dyn_image: DynamicImage = (&gray).to_cv();
        assert!(matches!(dyn_image, DynamicImage::ImageLuma8(_)));
    }

    #[test]
    fn dynamic_image_to_image_buffer() {
        let rgb = RgbImage::from_pixel(2, 2, Rgb([1u8, 2, 3]));
        let dyn_image: DynamicImage = (&rgb).to_cv();

        let back: RgbImage = (&dyn_image).try_to_cv().unwrap();
        assert_eq!(back.dimensions(), (2, 2));
        assert_eq!(back.get_pixel(0, 0), &Rgb([1u8, 2, 3]));
    }
}
