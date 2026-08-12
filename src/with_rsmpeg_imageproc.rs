use crate::with_rsmpeg;
use crate::TryToCv;
use anyhow::{ensure, Error, Result};
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;

// NOTE: `imageproc::Image<Rgb<u8>>` is a type alias of `image::RgbImage`.
// When the `image` feature is enabled, `with_rsmpeg_image` provides the
// equivalent conversions, so these impls are gated to avoid conflicts.

// AVFrame -> imageproc::Image<Rgb<u8>>
#[cfg(not(feature = "image"))]
impl TryToCv<imageproc::image::RgbImage> for AVFrame {
    type Error = Error;

    fn try_to_cv(&self) -> Result<imageproc::image::RgbImage, Self::Error> {
        let from = if self.format == ffi::AV_PIX_FMT_YUV420P {
            with_rsmpeg::convert_avframe(self, self.width, self.height, ffi::AV_PIX_FMT_RGB24)?
        } else {
            self.clone()
        };

        ensure!(
            from.format == ffi::AV_PIX_FMT_RGB24,
            "unsupported pixel format: {}",
            from.format
        );

        let width = from.width as usize;
        let height = from.height as usize;
        let channels = 3usize;
        let linesize = from.linesize[0] as usize;

        let mut data = vec![0u8; width * height * channels];
        for y in 0..height {
            let src = unsafe {
                std::slice::from_raw_parts(from.data[0].add(y * linesize), width * channels)
            };
            data[y * width * channels..(y + 1) * width * channels].copy_from_slice(src);
        }

        imageproc::image::RgbImage::from_raw(width as u32, height as u32, data)
            .ok_or_else(|| Error::msg("failed to build image from frame"))
    }
}

// imageproc::Image<Rgb<u8>> -> AVFrame
#[cfg(not(feature = "image"))]
impl TryToCv<AVFrame> for imageproc::image::RgbImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<AVFrame, Self::Error> {
        let (width, height) = self.dimensions();

        let mut frame = AVFrame::new();
        frame.set_width(width as i32);
        frame.set_height(height as i32);
        frame.set_format(ffi::AV_PIX_FMT_RGB24);
        frame.alloc_buffer()?;

        let rgb_data = self.as_raw();
        let linesize = frame.linesize[0] as usize;
        let channels = 3usize;
        for y in 0..height as usize {
            let row = &rgb_data[y * width as usize * channels..(y + 1) * width as usize * channels];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    frame.data[0].add(y * linesize),
                    row.len(),
                );
            }
        }

        Ok(frame)
    }
}

// AVFrame -> imageproc::Image<Luma<u8>>
#[cfg(not(feature = "image"))]
impl TryToCv<imageproc::image::GrayImage> for AVFrame {
    type Error = Error;

    fn try_to_cv(&self) -> Result<imageproc::image::GrayImage, Self::Error> {
        let from = if self.format == ffi::AV_PIX_FMT_YUV420P {
            with_rsmpeg::convert_avframe(self, self.width, self.height, ffi::AV_PIX_FMT_GRAY8)?
        } else {
            self.clone()
        };

        ensure!(
            from.format == ffi::AV_PIX_FMT_GRAY8,
            "unsupported pixel format: {}",
            from.format
        );

        let width = from.width as usize;
        let height = from.height as usize;
        let linesize = from.linesize[0] as usize;

        let mut data = vec![0u8; width * height];
        for y in 0..height {
            let src = unsafe { std::slice::from_raw_parts(from.data[0].add(y * linesize), width) };
            data[y * width..(y + 1) * width].copy_from_slice(src);
        }

        imageproc::image::GrayImage::from_raw(width as u32, height as u32, data)
            .ok_or_else(|| Error::msg("failed to build image from frame"))
    }
}

// imageproc::Image<Luma<u8>> -> AVFrame
#[cfg(not(feature = "image"))]
impl TryToCv<AVFrame> for imageproc::image::GrayImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<AVFrame, Self::Error> {
        let (width, height) = self.dimensions();

        let mut frame = AVFrame::new();
        frame.set_width(width as i32);
        frame.set_height(height as i32);
        frame.set_format(ffi::AV_PIX_FMT_GRAY8);
        frame.alloc_buffer()?;

        let gray_data = self.as_raw();
        let linesize = frame.linesize[0] as usize;
        for y in 0..height as usize {
            let row = &gray_data[y * width as usize..(y + 1) * width as usize];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    frame.data[0].add(y * linesize),
                    row.len(),
                );
            }
        }

        Ok(frame)
    }
}

#[cfg(all(test, not(feature = "image")))]
mod tests {
    use super::*;
    use imageproc::image::{GrayImage, Luma, Rgb, RgbImage};

    #[test]
    fn rgb_roundtrip() {
        let img = RgbImage::from_pixel(3, 2, Rgb([10u8, 20, 30]));
        let frame: AVFrame = (&img).try_to_cv().unwrap();
        assert_eq!(frame.format, ffi::AV_PIX_FMT_RGB24);
        assert_eq!(frame.width, 3);
        assert_eq!(frame.height, 2);

        let back: imageproc::image::RgbImage = (&frame).try_to_cv().unwrap();
        assert_eq!(back.dimensions(), img.dimensions());
        assert_eq!(back.get_pixel(2, 1), &Rgb([10u8, 20, 30]));
    }

    #[test]
    fn gray_roundtrip() {
        let img = GrayImage::from_pixel(3, 2, Luma([128u8]));
        let frame: AVFrame = (&img).try_to_cv().unwrap();
        assert_eq!(frame.format, ffi::AV_PIX_FMT_GRAY8);
        assert_eq!(frame.width, 3);
        assert_eq!(frame.height, 2);

        let back: imageproc::image::GrayImage = (&frame).try_to_cv().unwrap();
        assert_eq!(back.dimensions(), img.dimensions());
        assert_eq!(back.get_pixel(2, 1), &Luma([128u8]));
    }
}
