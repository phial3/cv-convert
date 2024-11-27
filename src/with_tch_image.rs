use crate::{FromCv, IntoCv, TchTensorAsImage, TchTensorImageShape, TryFromCv};
use anyhow::{Error, Result};
use image;
use std::ops::Deref;
use tch;

impl<P, Container> FromCv<&image::ImageBuffer<P, Container>> for TchTensorAsImage
where
    P: image::Pixel + 'static,
    P::Subpixel: 'static + tch::kind::Element,
    Container: Deref<Target=[P::Subpixel]>,
{
    fn from_cv(from: &image::ImageBuffer<P, Container>) -> Self {
        let (width, height) = from.dimensions();
        let channels = P::CHANNEL_COUNT;
        let tensor =
            tch::Tensor::from_slice(&*from).view([width as i64, height as i64, channels as i64]);
        TchTensorAsImage {
            tensor,
            kind: TchTensorImageShape::Whc,
        }
    }
}

impl<P, Container> FromCv<image::ImageBuffer<P, Container>> for TchTensorAsImage
where
    P: image::Pixel + 'static,
    P::Subpixel: 'static + tch::kind::Element,
    Container: Deref<Target=[P::Subpixel]>,
{
    fn from_cv(from: image::ImageBuffer<P, Container>) -> Self {
        Self::from_cv(&from)
    }
}

impl TryFromCv<&image::DynamicImage> for TchTensorAsImage {
    type Error = Error;

    fn try_from_cv(from: &image::DynamicImage) -> Result<Self, Self::Error> {
        use image::DynamicImage as D;

        let tensor = match from {
            D::ImageLuma8(image) => image.into_cv(),
            D::ImageLumaA8(image) => image.into_cv(),
            D::ImageRgb8(image) => image.into_cv(),
            D::ImageRgba8(image) => image.into_cv(),
            D::ImageRgb32F(image) => image.into_cv(),
            D::ImageRgba32F(image) => image.into_cv(),
            _ => anyhow::bail!("the color type {:?} is not supported", from.color()),
        };
        Ok(tensor)
    }
}

impl TryFromCv<image::DynamicImage> for TchTensorAsImage {
    type Error = Error;

    fn try_from_cv(from: image::DynamicImage) -> Result<Self, Self::Error> {
        Self::try_from_cv(&from)
    }
}
