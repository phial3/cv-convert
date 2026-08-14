use crate::with_rsmpeg;
use crate::TryToCv;
use anyhow::{ensure, Error, Result};
use nalgebra as na;
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;
use std::slice;

// AVFrame -> DMatrix<u8> (RGB24 data laid out as a [height, width * 3] row-major matrix)
impl TryToCv<na::DMatrix<u8>> for AVFrame {
    type Error = Error;

    fn try_to_cv(&self) -> Result<na::DMatrix<u8>, Self::Error> {
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

        Ok(na::DMatrix::from_row_slice(height, width * channels, &data))
    }
}

// DMatrix<u8> -> AVFrame (RGB24)
impl TryToCv<AVFrame> for na::DMatrix<u8> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<AVFrame, Self::Error> {
        let height = self.nrows();
        let cols = self.ncols();
        ensure!(
            cols.is_multiple_of(3),
            "expected a multiple of 3 columns, but got {}",
            cols
        );
        let width = cols / 3;

        let mut frame = AVFrame::new();
        frame.set_width(width as i32);
        frame.set_height(height as i32);
        frame.set_format(ffi::AV_PIX_FMT_RGB24);
        frame.alloc_buffer()?;

        let linesize = frame.linesize[0] as usize;
        for y in 0..height {
            let row = &self.row(y);
            let src: Vec<u8> = row.iter().cloned().collect();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    frame.data[0].add(y * linesize),
                    src.len(),
                );
            }
        }

        Ok(frame)
    }
}

// ---------- 扩展到 u16 / f32 像素类型（RGB24 的位宽扩展/压缩） ----------

// AVFrame -> DMatrix<u16>（RGB24 逐通道扩展到 u16）
impl TryToCv<na::DMatrix<u16>> for AVFrame {
    type Error = Error;

    fn try_to_cv(&self) -> Result<na::DMatrix<u16>, Self::Error> {
        let rgb: na::DMatrix<u8> = self.try_to_cv()?;
        let data: Vec<u16> = rgb.iter().map(|&v| v as u16).collect();
        Ok(na::DMatrix::from_vec(rgb.nrows(), rgb.ncols(), data))
    }
}

// DMatrix<u16> -> AVFrame（压缩回 RGB24，超过 255 的值被截断）
impl TryToCv<AVFrame> for na::DMatrix<u16> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<AVFrame, Self::Error> {
        let data: Vec<u8> = self.iter().map(|&v| v.min(255) as u8).collect();
        let rgb = na::DMatrix::from_vec(self.nrows(), self.ncols(), data);
        rgb.try_to_cv()
    }
}

// AVFrame -> DMatrix<f32>（RGB24 归一化到 [0, 255] 的 f32）
impl TryToCv<na::DMatrix<f32>> for AVFrame {
    type Error = Error;

    fn try_to_cv(&self) -> Result<na::DMatrix<f32>, Self::Error> {
        let rgb: na::DMatrix<u8> = self.try_to_cv()?;
        let data: Vec<f32> = rgb.iter().map(|&v| v as f32).collect();
        Ok(na::DMatrix::from_vec(rgb.nrows(), rgb.ncols(), data))
    }
}

// DMatrix<f32> -> AVFrame（压缩回 RGB24，非法值截断到 [0, 255]）
impl TryToCv<AVFrame> for na::DMatrix<f32> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<AVFrame, Self::Error> {
        let data: Vec<u8> = self.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
        let rgb = na::DMatrix::from_vec(self.nrows(), self.ncols(), data);
        rgb.try_to_cv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TryToCv;

    #[test]
    fn avframe_matrix_roundtrip() {
        let width = 16;
        let height = 16;

        let mut frame = AVFrame::new();
        frame.set_width(width);
        frame.set_height(height);
        frame.set_format(ffi::AV_PIX_FMT_RGB24);
        frame.alloc_buffer().unwrap();

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

        let mat: na::DMatrix<u8> = frame.try_to_cv().unwrap();
        assert_eq!(mat.shape(), (height as usize, (width * 3) as usize));

        let back: AVFrame = mat.try_to_cv().unwrap();
        assert_eq!(back.width, width);
        assert_eq!(back.height, height);
        assert_eq!(back.format, ffi::AV_PIX_FMT_RGB24);
    }

    #[test]
    fn avframe_u16_matrix_roundtrip() {
        let width = 4;
        let height = 4;

        let mut frame = AVFrame::new();
        frame.set_width(width);
        frame.set_height(height);
        frame.set_format(ffi::AV_PIX_FMT_RGB24);
        frame.alloc_buffer().unwrap();

        let linesize = frame.linesize[0] as usize;
        for y in 0..height as usize {
            for x in 0..width as usize {
                unsafe {
                    let ptr = frame.data[0].add(y * linesize + x * 3);
                    *ptr = (x * 10 % 256) as u8;
                    *ptr.add(1) = (y * 10 % 256) as u8;
                    *ptr.add(2) = 128;
                }
            }
        }

        let mat: na::DMatrix<u16> = frame.try_to_cv().unwrap();
        assert_eq!(mat.shape(), (height as usize, (width * 3) as usize));

        let back: AVFrame = mat.try_to_cv().unwrap();
        assert_eq!(back.width, width);
        assert_eq!(back.height, height);
        assert_eq!(back.format, ffi::AV_PIX_FMT_RGB24);
    }
}
