use crate::with_rsmpeg;
use crate::TryToCv;
use anyhow::{ensure, Error, Result};
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;
use std::slice;

// AVFrame -> Tensor (RGB24, shape [H, W, 3])
impl TryToCv<tch::Tensor> for AVFrame {
    type Error = Error;

    fn try_to_cv(&self) -> Result<tch::Tensor, Self::Error> {
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
            let src =
                unsafe { slice::from_raw_parts(from.data[0].add(y * linesize), width * channels) };
            data[y * width * channels..(y + 1) * width * channels].copy_from_slice(src);
        }

        Ok(tch::Tensor::from_slice(&data).view([height as i64, width as i64, channels as i64]))
    }
}

// Tensor -> AVFrame (RGB24)
impl TryToCv<AVFrame> for tch::Tensor {
    type Error = Error;

    fn try_to_cv(&self) -> Result<AVFrame, Self::Error> {
        let from = self.shallow_clone();
        ensure!(
            from.kind() == tch::kind::Kind::Uint8,
            "expected an uint8 tensor, but got {:?}",
            from.kind()
        );

        let size = from.size();
        ensure!(size.len() == 3, "expected a 3D tensor");
        ensure!(size[2] == 3, "expected 3 channels, but got {}", size[2]);
        let (height, width) = (size[0] as usize, size[1] as usize);
        let channels = 3usize;

        let data: Vec<u8> = Vec::try_from(from.flatten(0, -1))?;

        let mut frame = AVFrame::new();
        frame.set_width(width as i32);
        frame.set_height(height as i32);
        frame.set_format(ffi::AV_PIX_FMT_RGB24);
        frame.alloc_buffer()?;

        let linesize = frame.linesize[0] as usize;
        for y in 0..height {
            let row = &data[y * width * channels..(y + 1) * width * channels];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TryToCv;

    #[test]
    fn avframe_tensor_roundtrip() {
        let width = 16;
        let height = 16;

        let mut frame = AVFrame::new();
        frame.set_width(width);
        frame.set_height(height);
        frame.set_format(ffi::AV_PIX_FMT_RGB24);
        frame.alloc_buffer().unwrap();

        // Fill with a pattern.
        let linesize = frame.linesize[0] as usize;
        for y in 0..height as usize {
            for x in 0..width as usize {
                unsafe {
                    let ptr = frame.data[0].add(y * linesize + x * 3);
                    *ptr = (x % 256) as u8;
                    *ptr.add(1) = (y % 256) as u8;
                    *ptr.add(2) = 128;
                }
            }
        }

        let tensor: tch::Tensor = (&frame).try_to_cv().unwrap();
        assert_eq!(tensor.size(), &[height as i64, width as i64, 3]);

        let back: AVFrame = (&tensor).try_to_cv().unwrap();
        assert_eq!(back.width, width);
        assert_eq!(back.height, height);
        assert_eq!(back.format, ffi::AV_PIX_FMT_RGB24);
    }
}
