use crate::with_rsmpeg;
use crate::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use anyhow::{Error, Result};
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;

impl TryFromCv<&AVFrame> for image::DynamicImage {
    type Error = Error;

    fn try_from_cv(from: &AVFrame) -> Result<Self, Self::Error> {
        let format = from.format;
        let img: image::DynamicImage = match format {
            ffi::AV_PIX_FMT_GRAY8 => avframe_to_image_buffer_gray(from),
            ffi::AV_PIX_FMT_RGBA | ffi::AV_PIX_FMT_BGRA => avframe_to_image_buffer_rgba(from),
            ffi::AV_PIX_FMT_RGB24
            | ffi::AV_PIX_FMT_BGR24
            | ffi::AV_PIX_FMT_YUYV422
            | ffi::AV_PIX_FMT_YVYU422
            | ffi::AV_PIX_FMT_UYVY422
            | ffi::AV_PIX_FMT_UYYVYY411
            | ffi::AV_PIX_FMT_YUVA422P
            | ffi::AV_PIX_FMT_YUVA444P
            | ffi::AV_PIX_FMT_YUV410P
            | ffi::AV_PIX_FMT_YUV411P
            | ffi::AV_PIX_FMT_YUV420P
            | ffi::AV_PIX_FMT_YUV422P
            | ffi::AV_PIX_FMT_YUV440P
            | ffi::AV_PIX_FMT_YUV444P => avframe_to_image_buffer_rgb(from),
            _ => return Err(Error::msg(format!("Unsupported pixel format: {}", format))),
        };

        Ok(img)
    }
}

impl TryFromCv<AVFrame> for image::DynamicImage {
    type Error = Error;
    fn try_from_cv(from: AVFrame) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl TryFromCv<&image::DynamicImage> for AVFrame {
    type Error = Error;

    fn try_from_cv(from: &image::DynamicImage) -> Result<Self, Self::Error> {
        use image::DynamicImage;
        let frame = match from {
            DynamicImage::ImageLuma8(img) => img.try_into_cv()?,
            // DynamicImage::ImageLumaA8(img) => img.try_into_cv()?,
            DynamicImage::ImageRgb8(img) => img.try_into_cv()?,
            DynamicImage::ImageRgba8(img) => img.try_into_cv()?,
            // DynamicImage::ImageLuma16(img) => img.try_into_cv()?,
            // DynamicImage::ImageLumaA16(img) => img.try_into_cv()?,
            // DynamicImage::ImageRgb16(img) => img.try_into_cv()?,
            // DynamicImage::ImageRgba16(img) => img.try_into_cv()?,
            // DynamicImage::ImageRgb32F(img) => img.try_into_cv()?,
            // DynamicImage::ImageRgba32F(img) => img.try_into_cv()?,
            _ => return Err(Error::msg("Unsupported image format")),
        };
        Ok(frame)
    }
}

impl TryFromCv<image::DynamicImage> for AVFrame {
    type Error = Error;
    fn try_from_cv(from: image::DynamicImage) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

/// YUV to RGB conversion formulas (BT.601):
/// R = Y + 1.402 * (V - 128)
/// G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
/// B = Y + 1.772 * (U - 128)
///
/// RGB to YUV conversion formulas (BT.601):
/// Y = 0.299 * R + 0.587 * G + 0.114 * B
/// U = -0.169 * R - 0.331 * G + 0.500 * B + 128
/// V = 0.500 * R - 0.419 * G - 0.081 * B + 128
///
/// 优先考虑基于 ffmpeg 转换
///
// &AVFrame -> RgbImage
impl TryFromCv<&AVFrame> for image::RgbImage {
    type Error = Error;

    fn try_from_cv(from: &AVFrame) -> Result<Self, Self::Error> {
        let width = from.width as u32;
        let height = from.height as u32;
        let yuv_vec = [
            ffi::AV_PIX_FMT_YUYV422,
            ffi::AV_PIX_FMT_YVYU422,
            ffi::AV_PIX_FMT_UYVY422,
            ffi::AV_PIX_FMT_UYYVYY411,
            ffi::AV_PIX_FMT_YUVA422P,
            ffi::AV_PIX_FMT_YUVA444P,
            ffi::AV_PIX_FMT_YUV410P,
            ffi::AV_PIX_FMT_YUV411P,
            ffi::AV_PIX_FMT_YUV420P,
            ffi::AV_PIX_FMT_YUV422P,
            ffi::AV_PIX_FMT_YUV440P,
            ffi::AV_PIX_FMT_YUV444P,
        ];

        // 基于 ffmpeg 转换  YUV420P => RGB
        let from = if yuv_vec.contains(&from.format) {
            with_rsmpeg::convert_avframe(from, from.width, from.height, ffi::AV_PIX_FMT_RGB24)?
        } else {
            from.clone()
        };

        match from.format {
            // RGB24 或 BGR24
            ffi::AV_PIX_FMT_RGB24 | ffi::AV_PIX_FMT_BGR24 => {
                let stride = from.linesize[0] as usize;
                let data = unsafe {
                    std::slice::from_raw_parts(from.data[0], stride * from.height as usize)
                };

                let mut buffer = image::RgbImage::new(width, height);
                let is_bgr = from.format == ffi::AV_PIX_FMT_BGR24;

                for y in 0..height {
                    for x in 0..width {
                        let pos = (y as usize * stride + x as usize * 3) as usize;
                        let (r, g, b) = if is_bgr {
                            (data[pos + 2], data[pos + 1], data[pos])
                        } else {
                            (data[pos], data[pos + 1], data[pos + 2])
                        };
                        buffer.put_pixel(x, y, image::Rgb([r, g, b]));
                    }
                }
                Ok(buffer)
            }

            // YUV420P => RGB (可以基于 ffmpeg 转换)
            format if format == ffi::AV_PIX_FMT_YUV420P => {
                let y_stride = from.linesize[0] as usize;
                let u_stride = from.linesize[1] as usize;
                let v_stride = from.linesize[2] as usize;

                let y_data = unsafe {
                    std::slice::from_raw_parts(from.data[0], y_stride * from.height as usize)
                };
                let u_data = unsafe {
                    std::slice::from_raw_parts(from.data[1], u_stride * (from.height as usize / 2))
                };
                let v_data = unsafe {
                    std::slice::from_raw_parts(from.data[2], v_stride * (from.height as usize / 2))
                };

                let mut buffer = image::RgbImage::new(width, height);
                for y in 0..height {
                    for x in 0..width {
                        let y_val = y_data[y as usize * y_stride + x as usize] as f32;
                        let u_val =
                            u_data[(y as usize / 2) * u_stride + (x as usize / 2)] as f32 - 128.0;
                        let v_val =
                            v_data[(y as usize / 2) * v_stride + (x as usize / 2)] as f32 - 128.0;

                        // YUV to RGB conversion formulas (BT.601):
                        // R = Y + 1.402 * (V - 128)
                        // G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
                        // B = Y + 1.772 * (U - 128)
                        let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                        let g =
                            (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
                        let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

                        buffer.put_pixel(x, y, image::Rgb([r, g, b]));
                    }
                }
                Ok(buffer)
            }

            format => Err(Error::msg(format!("Unsupported pixel format: {}", format))),
        }
    }
}

// AVFrame -> DynamicImage
impl TryFromCv<AVFrame> for image::RgbImage {
    type Error = Error;

    fn try_from_cv(from: AVFrame) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &AVFrame -> RgbaImage
impl TryFromCv<&AVFrame> for image::RgbaImage {
    type Error = Error;

    fn try_from_cv(from: &AVFrame) -> Result<Self, Self::Error> {
        let width = from.width as u32;
        let height = from.height as u32;

        match from.format {
            // RGBA 和 BGRA
            format if format == ffi::AV_PIX_FMT_RGBA || format == ffi::AV_PIX_FMT_BGRA => {
                let stride = from.linesize[0] as usize;
                let data = unsafe {
                    std::slice::from_raw_parts(from.data[0], stride * from.height as usize)
                };

                let mut buffer = image::RgbaImage::new(width, height);
                let is_bgra = format == ffi::AV_PIX_FMT_BGRA;

                for y in 0..height {
                    for x in 0..width {
                        let pos = y as usize * stride + x as usize * 4;
                        let (r, g, b, a) = if is_bgra {
                            (data[pos + 2], data[pos + 1], data[pos], data[pos + 3])
                        } else {
                            (data[pos], data[pos + 1], data[pos + 2], data[pos + 3])
                        };
                        buffer.put_pixel(x, y, image::Rgba([r, g, b, a]));
                    }
                }
                Ok(buffer)
            }

            // RGB24/BGR24 format (add alpha channel)
            format if format == ffi::AV_PIX_FMT_RGB24 || format == ffi::AV_PIX_FMT_BGR24 => {
                let stride = from.linesize[0] as usize;
                let data = unsafe {
                    std::slice::from_raw_parts(from.data[0], stride * from.height as usize)
                };

                let mut buffer = image::RgbaImage::new(width, height);
                let is_bgr = format == ffi::AV_PIX_FMT_BGR24;

                for y in 0..height {
                    for x in 0..width {
                        let pos = y as usize * stride + x as usize * 3;
                        let (r, g, b) = if is_bgr {
                            (data[pos + 2], data[pos + 1], data[pos])
                        } else {
                            (data[pos], data[pos + 1], data[pos + 2])
                        };
                        buffer.put_pixel(x, y, image::Rgba([r, g, b, 255]));
                    }
                }
                Ok(buffer)
            }

            // YUV420P => RGB (可以基于 ffmpeg 转换)
            format if format == ffi::AV_PIX_FMT_YUV420P => {
                let y_stride = from.linesize[0] as usize;
                let u_stride = from.linesize[1] as usize;
                let v_stride = from.linesize[2] as usize;

                let y_data = unsafe {
                    std::slice::from_raw_parts(from.data[0], y_stride * from.height as usize)
                };
                let u_data = unsafe {
                    std::slice::from_raw_parts(from.data[1], u_stride * (from.height as usize / 2))
                };
                let v_data = unsafe {
                    std::slice::from_raw_parts(from.data[2], v_stride * (from.height as usize / 2))
                };

                let mut buffer = image::RgbaImage::new(width, height);

                for y in 0..height {
                    for x in 0..width {
                        // Get YUV values
                        let y_val = y_data[y as usize * y_stride + x as usize] as f32;
                        let u_val =
                            u_data[(y as usize / 2) * u_stride + (x as usize / 2)] as f32 - 128.0;
                        let v_val =
                            v_data[(y as usize / 2) * v_stride + (x as usize / 2)] as f32 - 128.0;

                        // YUV to RGB conversion formulas (BT.601):
                        // R = Y + 1.402 * (V - 128)
                        // G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
                        // B = Y + 1.772 * (U - 128)
                        let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                        let g =
                            (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
                        let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

                        buffer.put_pixel(x, y, image::Rgba([r, g, b, 255]));
                    }
                }
                Ok(buffer)
            }

            format => Err(Error::msg(format!("Unsupported pixel format: {}", format))),
        }
    }
}

// AVFrame -> RgbaImage
impl TryFromCv<AVFrame> for image::RgbaImage {
    type Error = Error;

    fn try_from_cv(from: AVFrame) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &AVFrame -> GrayImage
impl TryFromCv<&AVFrame> for image::GrayImage {
    type Error = Error;
    fn try_from_cv(from: &AVFrame) -> Result<Self, Self::Error> {
        let width = from.width as u32;
        let height = from.height as u32;

        match from.format {
            // Grayscale
            format if format == ffi::AV_PIX_FMT_GRAY8 => {
                let stride = from.linesize[0] as usize;
                let data = unsafe {
                    std::slice::from_raw_parts(from.data[0], stride * from.height as usize)
                };

                let mut buffer = image::GrayImage::new(width, height);
                for y in 0..height {
                    for x in 0..width {
                        let pos = y as usize * stride + x as usize;
                        buffer.put_pixel(x, y, image::Luma([data[pos]]));
                    }
                }
                Ok(buffer)
            }

            // RGB/BGR to Grayscale Conversion
            // 原理：人眼对不同颜色的敏感程度不同，对绿色最敏感，其次是红色，最后是蓝色
            // BT.601 标准转换公式：
            // Gray = 0.299 * R + 0.587 * G + 0.114 * B
            // BT.709 标准转换公式：
            // Gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
            // 代码中使用的公式（接近 BT.601）：
            // Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
            // RGB24/BGR24
            format if format == ffi::AV_PIX_FMT_RGB24 || format == ffi::AV_PIX_FMT_BGR24 => {
                let stride = from.linesize[0] as usize;
                let data = unsafe {
                    std::slice::from_raw_parts(from.data[0], stride * from.height as usize)
                };

                let mut buffer = image::GrayImage::new(width, height);
                let is_bgr = format == ffi::AV_PIX_FMT_BGR24;

                for y in 0..height {
                    for x in 0..width {
                        let pos = y as usize * stride + x as usize * 3;
                        let (r, g, b) = if is_bgr {
                            (data[pos + 2], data[pos + 1], data[pos])
                        } else {
                            (data[pos], data[pos + 1], data[pos + 2])
                        };

                        // 计算灰度值
                        // 1. 将 RGB 值转换为 f32 进行浮点运算
                        // 0.2989 (R): 红色对亮度的贡献
                        // 0.5870 (G): 绿色对亮度的贡献（最大，因为人眼对绿色最敏感）
                        // 0.1140 (B): 蓝色对亮度的贡献（最小）
                        // 注意：这些系数的总和等于1，确保不会溢出
                        let gray =
                            ((0.2989 * r as f32) + (0.5870 * g as f32) + (0.1140 * b as f32)) as u8;
                        buffer.put_pixel(x, y, image::Luma([gray]));
                    }
                }
                Ok(buffer)
            }

            // YUV420P pixel to grayscale (use Y channel only)
            format if format == ffi::AV_PIX_FMT_YUV420P => {
                let stride = from.linesize[0] as usize;
                let data = unsafe {
                    std::slice::from_raw_parts(from.data[0], stride * from.height as usize)
                };

                let mut buffer = image::GrayImage::new(width, height);
                for y in 0..height {
                    for x in 0..width {
                        let pos = y as usize * stride + x as usize;
                        buffer.put_pixel(x, y, image::Luma([data[pos]]));
                    }
                }
                Ok(buffer)
            }

            format => Err(Error::msg(format!("Unsupported pixel format: {}", format))),
        }
    }
}

// AVFrame -> GrayImage
impl TryFromCv<AVFrame> for image::GrayImage {
    type Error = Error;

    fn try_from_cv(from: AVFrame) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

//////////////////////////////////////////////////
//////////////////////////////////////////////////

// &RgbImage -> AVFrame
impl TryFromCv<&image::RgbImage> for AVFrame {
    type Error = Error;

    fn try_from_cv(from: &image::RgbImage) -> Result<Self, Self::Error> {
        let (width, height) = from.dimensions();

        // 创建源 AVFrame，并分配缓冲区
        let mut frame = AVFrame::new();
        frame.set_width(width as i32);
        frame.set_height(height as i32);
        frame.set_format(ffi::AV_PIX_FMT_RGB24);
        frame.set_pts(0);
        frame.alloc_buffer().unwrap();

        // 方式一：直接系统级别内存复制，性能更好，适合处理连续的内存布局
        unsafe {
            let rgb_data = from.as_raw();
            let buffer_slice = std::slice::from_raw_parts_mut(frame.data[0], rgb_data.len());
            buffer_slice.copy_from_slice(rgb_data);
        }

        // 方式二：逐像素复制，可以处理不同的内存布局，便于进行像素级的转换或处理，性能相对较低
        // unsafe {
        //     let linesize = (*frame.as_mut_ptr()).linesize[0] as usize;
        //     let data_ptr = (*frame.as_mut_ptr()).data[0];
        //
        //     // 复制 RGB 数据
        //     for y in 0..height {
        //         for x in 0..width {
        //             let pixel = from.get_pixel(x, y);
        //             let offset = y as usize * linesize + (x as usize * 3);
        //             *data_ptr.add(offset) = pixel[0];           // R
        //             *data_ptr.add(offset + 1) = pixel[1]; // G
        //             *data_ptr.add(offset + 2) = pixel[2]; // B
        //         }
        //     }
        // }

        Ok(frame)
    }
}

// RgbImage -> AVFrame
impl TryFromCv<image::RgbImage> for AVFrame {
    type Error = Error;

    fn try_from_cv(from: image::RgbImage) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &RgbaImage -> AVFrame
impl TryFromCv<&image::RgbaImage> for AVFrame {
    type Error = Error;
    fn try_from_cv(from: &image::RgbaImage) -> Result<Self, Self::Error> {
        let (width, height) = from.dimensions();

        // 创建源 AVFrame，并分配缓冲区
        let mut frame = AVFrame::new();
        frame.set_width(width as i32);
        frame.set_height(height as i32);
        frame.set_format(ffi::AV_PIX_FMT_RGBA);
        frame.set_pts(0);
        frame.alloc_buffer().unwrap();

        unsafe {
            let rgba_data = from.as_raw();
            let buffer_slice = std::slice::from_raw_parts_mut(frame.data[0], rgba_data.len());
            buffer_slice.copy_from_slice(rgba_data);
        }

        Ok(frame)
    }
}

// RgbaImage -> AVFrame
impl TryFromCv<image::RgbaImage> for AVFrame {
    type Error = Error;
    fn try_from_cv(from: image::RgbaImage) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &GrayImage -> AVFrame
impl TryFromCv<&image::GrayImage> for AVFrame {
    type Error = Error;
    fn try_from_cv(from: &image::GrayImage) -> Result<Self, Self::Error> {
        let (width, height) = from.dimensions();

        // 创建源 AVFrame，并分配缓冲区
        let mut frame = AVFrame::new();
        frame.set_width(width as i32);
        frame.set_height(height as i32);
        frame.set_format(ffi::AV_PIX_FMT_GRAY8);
        frame.set_pts(0);
        frame.alloc_buffer().unwrap();

        unsafe {
            let gray_data = from.as_raw();
            let buffer_slice = std::slice::from_raw_parts_mut(frame.data[0], gray_data.len());
            buffer_slice.copy_from_slice(gray_data);
        }

        Ok(frame)
    }
}

// GrayImage -> AVFrame
impl TryFromCv<image::GrayImage> for AVFrame {
    type Error = Error;
    fn try_from_cv(from: image::GrayImage) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

fn avframe_to_image_buffer_gray(frame: &AVFrame) -> image::DynamicImage {
    let width = frame.width as u32;
    let height = frame.height as u32;
    let mut img = image::GrayImage::new(width, height);

    unsafe {
        let linesize = frame.linesize[0] as usize;
        let data = std::slice::from_raw_parts(frame.data[0], height as usize * linesize);

        for y in 0..height {
            for x in 0..width {
                let val = data[y as usize * linesize + x as usize];
                img.put_pixel(x, y, image::Luma([val]));
            }
        }
    }

    image::DynamicImage::ImageLuma8(img)
}

fn avframe_to_image_buffer_rgb(frame: &AVFrame) -> image::DynamicImage {
    let frame = if frame.format != ffi::AV_PIX_FMT_RGB24 {
        with_rsmpeg::convert_avframe(frame, frame.width, frame.height, ffi::AV_PIX_FMT_RGB24)
            .unwrap()
    } else {
        frame.clone()
    };

    let width = frame.width as u32;
    let height = frame.height as u32;
    let mut img = image::RgbImage::new(width, height);

    unsafe {
        let linesize = frame.linesize[0] as usize;
        let data = std::slice::from_raw_parts(frame.data[0], height as usize * linesize);

        for y in 0..height {
            for x in 0..width {
                let offset = y as usize * linesize + x as usize * 3;
                img.put_pixel(
                    x,
                    y,
                    image::Rgb([data[offset], data[offset + 1], data[offset + 2]]),
                );
            }
        }
    }

    image::DynamicImage::ImageRgb8(img)
}

fn avframe_to_image_buffer_rgba(frame: &AVFrame) -> image::DynamicImage {
    let width = frame.width as u32;
    let height = frame.height as u32;
    let is_bgra = frame.format == ffi::AV_PIX_FMT_BGRA;
    let mut img = image::RgbaImage::new(width, height);

    unsafe {
        let linesize = frame.linesize[0] as usize;

        // 每一行实际的数据长度 * 4 bytes per pixel (RGBA/BGRA)
        let row_size = width as usize * 4;

        for y in 0..height {
            // 获取当前行的起始指针
            let row_ptr = frame.data[0].add(y as usize * linesize);
            // 将当前行转换为切片
            let row_data = std::slice::from_raw_parts(row_ptr, row_size);

            for x in 0..width {
                let pixel_offset = x as usize * 4;

                // 设置正确的颜色通道顺序
                let (r, g, b, a) = if is_bgra {
                    (
                        row_data[pixel_offset + 2], // B -> R
                        row_data[pixel_offset + 1], // G -> G
                        row_data[pixel_offset],     // R -> B
                        row_data[pixel_offset + 3], // A -> A
                    )
                } else {
                    (
                        row_data[pixel_offset],     // R
                        row_data[pixel_offset + 1], // G
                        row_data[pixel_offset + 2], // B
                        row_data[pixel_offset + 3], // A
                    )
                };

                img.put_pixel(x, y, image::Rgba([r, g, b, a]));
            }
        }
    }

    image::DynamicImage::ImageRgba8(img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GenericImageView, GrayImage, RgbImage, RgbaImage};
    use std::ptr;
    use std::time::Instant;

    // 辅助函数：创建测试用 AVFrame
    fn create_test_frame(
        format: i32,
        width: i32,
        height: i32,
        data: Vec<u8>,
        data2: Option<Vec<u8>>,
        data3: Option<Vec<u8>>,
    ) -> AVFrame {
        let mut frame = AVFrame::new();

        frame.set_format(format);
        frame.set_width(width);
        frame.set_height(height);

        let mut_frame = frame.as_mut_ptr();
        unsafe {
            let data_ptr = Box::new(data);
            (*mut_frame).data[0] = Box::leak(data_ptr).as_mut_ptr();
            (*mut_frame).linesize[0] = width
                * match format {
                    f if f == ffi::AV_PIX_FMT_GRAY8 => 1,
                    f if f == ffi::AV_PIX_FMT_RGBA => 4,
                    _ => 3,
                };

            if let Some(u_data) = data2 {
                let u_ptr = Box::new(u_data);
                (*mut_frame).data[1] = Box::leak(u_ptr).as_mut_ptr();
                (*mut_frame).linesize[1] = width / 2;
            }

            if let Some(v_data) = data3 {
                let v_ptr = Box::new(v_data);
                (*mut_frame).data[2] = Box::leak(v_ptr).as_mut_ptr();
                (*mut_frame).linesize[2] = width / 2;
            }
        }

        frame
    }

    #[test]
    fn test_bidirectional_rgb_conversion() {
        let start = Instant::now();

        // 创建测试数据
        let test_data = vec![
            255, 0, 0, // Red pixel
            0, 255, 0, // Green pixel
            0, 0, 255, // Blue pixel
            255, 255, 0, // Yellow pixel
        ];

        // 创建 AVFrame
        let frame = create_test_frame(ffi::AV_PIX_FMT_RGB24, 2, 2, test_data.clone(), None, None);

        // AVFrame -> RgbImage
        let rgb_image =
            image::RgbImage::try_from_cv(&frame).expect("Failed to convert AVFrame to RgbImage");

        // 验证转换结果
        assert_eq!(rgb_image.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
        assert_eq!(rgb_image.get_pixel(1, 0), &image::Rgb([0, 255, 0]));
        assert_eq!(rgb_image.get_pixel(0, 1), &image::Rgb([0, 0, 255]));
        assert_eq!(rgb_image.get_pixel(1, 1), &image::Rgb([255, 255, 0]));

        // RgbImage -> AVFrame
        let new_frame =
            AVFrame::try_from_cv(&rgb_image).expect("Failed to convert RgbImage back to AVFrame");

        // 验证格式和尺寸
        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_RGB24);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_bidirectional_rgba_conversion() {
        let start = Instant::now();

        // 创建测试数据
        let test_data = vec![
            255, 0, 0, 255, // Red pixel
            0, 255, 0, 255, // Green pixel
            0, 0, 255, 255, // Blue pixel
            255, 255, 0, 128, // Semi-transparent yellow pixel
        ];

        // 创建 AVFrame
        let frame = create_test_frame(ffi::AV_PIX_FMT_RGBA, 2, 2, test_data.clone(), None, None);

        // AVFrame -> RgbaImage
        let rgba_image =
            image::RgbaImage::try_from_cv(&frame).expect("Failed to convert AVFrame to RgbaImage");

        // 验证转换结果
        assert_eq!(rgba_image.get_pixel(0, 0), &image::Rgba([255, 0, 0, 255]));
        assert_eq!(rgba_image.get_pixel(1, 0), &image::Rgba([0, 255, 0, 255]));
        assert_eq!(rgba_image.get_pixel(0, 1), &image::Rgba([0, 0, 255, 255]));
        assert_eq!(rgba_image.get_pixel(1, 1), &image::Rgba([255, 255, 0, 128]));

        // RgbaImage -> AVFrame
        let new_frame =
            AVFrame::try_from_cv(&rgba_image).expect("Failed to convert RgbaImage back to AVFrame");

        // 验证格式和尺寸
        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_RGBA);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_bidirectional_yuv420p_conversion() {
        let start = Instant::now();

        // 创建 YUV420P 测试数据
        let y_data = vec![235, 128, 16, 235]; // Y plane
        let u_data = vec![128]; // U plane (2x2 -> 1x1)
        let v_data = vec![128]; // V plane (2x2 -> 1x1)

        // 创建 AVFrame
        let frame = create_test_frame(
            ffi::AV_PIX_FMT_YUV420P,
            2,
            2,
            y_data.clone(),
            Some(u_data.clone()),
            Some(v_data.clone()),
        );

        // YUV420P -> RgbImage
        let rgb_image = image::RgbImage::try_from_cv(&frame)
            .expect("Failed to convert YUV420P AVFrame to RgbImage");

        // 验证转换结果（注意：YUV->RGB 转换可能有轻微的舍入误差）
        let first_pixel = rgb_image.get_pixel(0, 0);
        println!(
            "First pixel: ({}, {}, {})",
            first_pixel[0], first_pixel[1], first_pixel[2]
        );
        // FIXME: ffmpeg 转换之后是 255， 但是自定义转换之后不变
        // assert!((first_pixel[0] as i32 - 235).abs() <= 1);

        // RgbImage -> YUV420P AVFrame
        let new_frame =
            AVFrame::try_from_cv(&rgb_image).expect("Failed to convert RgbImage back to AVFrame");

        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_RGB24);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_bidirectional_gray_conversion() {
        let start = Instant::now();

        // 创建灰度测试数据
        let test_data = vec![0, 85, 170, 255]; // 不同灰度值

        // 创建 AVFrame
        let frame = create_test_frame(ffi::AV_PIX_FMT_GRAY8, 2, 2, test_data.clone(), None, None);

        // AVFrame -> GrayImage
        let gray_image =
            image::GrayImage::try_from_cv(&frame).expect("Failed to convert AVFrame to GrayImage");

        // 验证转换结果
        assert_eq!(gray_image.get_pixel(0, 0), &image::Luma([0]));
        assert_eq!(gray_image.get_pixel(1, 0), &image::Luma([85]));
        assert_eq!(gray_image.get_pixel(0, 1), &image::Luma([170]));
        assert_eq!(gray_image.get_pixel(1, 1), &image::Luma([255]));

        // GrayImage -> AVFrame
        let new_frame =
            AVFrame::try_from_cv(&gray_image).expect("Failed to convert GrayImage back to AVFrame");

        // 验证格式和尺寸
        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_GRAY8);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_format_preservation() {
        // RGB24 格式保持测试
        let rgb_frame = create_test_frame(
            ffi::AV_PIX_FMT_RGB24,
            2,
            2,
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0],
            None,
            None,
        );
        let rgb_image = image::RgbImage::try_from_cv(&rgb_frame).unwrap();
        let new_rgb_frame = AVFrame::try_from_cv(&rgb_image).unwrap();
        assert_eq!(new_rgb_frame.format, ffi::AV_PIX_FMT_RGB24);

        // RGBA 格式保持测试
        let rgba_frame = create_test_frame(
            ffi::AV_PIX_FMT_RGBA,
            2,
            2,
            vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 128,
            ],
            None,
            None,
        );
        let rgba_image = image::RgbaImage::try_from_cv(&rgba_frame).unwrap();
        let new_rgba_frame = AVFrame::try_from_cv(&rgba_image).unwrap();
        assert_eq!(new_rgba_frame.format, ffi::AV_PIX_FMT_RGBA);
    }

    #[test]
    fn test_rgb_to_avframe() {
        // 创建测试用 RGB 图像
        let mut rgb_image = image::RgbImage::new(2, 2);
        rgb_image.put_pixel(0, 0, image::Rgb([255, 0, 0])); // Red
        rgb_image.put_pixel(1, 0, image::Rgb([0, 255, 0])); // Green
        rgb_image.put_pixel(0, 1, image::Rgb([0, 0, 255])); // Blue
        rgb_image.put_pixel(1, 1, image::Rgb([255, 255, 0])); // Yellow

        let frame =
            AVFrame::try_from_cv(&rgb_image).expect("Failed to convert RgbImage to AVFrame");

        assert_eq!(frame.format, ffi::AV_PIX_FMT_RGB24);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);

        // 验证像素数据
        unsafe {
            let data_ptr = (*frame.as_ptr()).data[0];
            assert_eq!(*data_ptr, 255); // Red component of first pixel
        }
    }

    #[test]
    fn test_rgba_to_avframe() {
        // 创建测试用 RGBA 图像
        let mut rgba_image = image::RgbaImage::new(2, 2);
        rgba_image.put_pixel(0, 0, image::Rgba([255, 0, 0, 255])); // Red
        rgba_image.put_pixel(1, 0, image::Rgba([0, 255, 0, 255])); // Green
        rgba_image.put_pixel(0, 1, image::Rgba([0, 0, 255, 255])); // Blue
        rgba_image.put_pixel(1, 1, image::Rgba([255, 255, 0, 128])); // Semi-transparent Yellow

        let frame =
            AVFrame::try_from_cv(&rgba_image).expect("Failed to convert RgbaImage to AVFrame");

        assert_eq!(frame.format, ffi::AV_PIX_FMT_RGBA);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
    }

    #[test]
    fn test_gray_to_avframe() {
        // 创建测试用灰度图像
        let mut gray_image = image::GrayImage::new(2, 2);
        gray_image.put_pixel(0, 0, image::Luma([0])); // Black
        gray_image.put_pixel(1, 0, image::Luma([85])); // Dark gray
        gray_image.put_pixel(0, 1, image::Luma([170])); // Light gray
        gray_image.put_pixel(1, 1, image::Luma([255])); // White

        let frame =
            AVFrame::try_from_cv(&gray_image).expect("Failed to convert GrayImage to AVFrame");

        assert_eq!(frame.format, ffi::AV_PIX_FMT_GRAY8);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
    }

    #[test]
    fn test_rgb_image_conversion() {
        let start = Instant::now();

        // 创建测试数据
        let test_data = vec![
            255, 0, 0, // Red pixel
            0, 255, 0, // Green pixel
            0, 0, 255, // Blue pixel
            255, 255, 0, // Yellow pixel
        ];

        // 创建 AVFrame
        let frame = create_test_frame(ffi::AV_PIX_FMT_RGB24, 2, 2, test_data.clone(), None, None);

        // AVFrame -> DynamicImage
        let dynamic_image =
            DynamicImage::try_from_cv(&frame).expect("Failed to convert AVFrame to DynamicImage");

        // DynamicImage -> AVFrame
        let new_frame = AVFrame::try_from_cv(dynamic_image)
            .expect("Failed to convert DynamicImage back to AVFrame");

        // 验证格式和尺寸
        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_RGB24);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_rgba_image_conversion() {
        let start = Instant::now();

        // 创建测试数据
        let test_data = vec![
            255, 0, 0, 255, // Red pixel
            0, 255, 0, 255, // Green pixel
            0, 0, 255, 255, // Blue pixel
            255, 255, 0, 128, // Semi-transparent yellow pixel
        ];

        // 创建 AVFrame
        let frame = create_test_frame(ffi::AV_PIX_FMT_RGBA, 2, 2, test_data.clone(), None, None);

        // AVFrame -> DynamicImage
        let dynamic_image =
            DynamicImage::try_from_cv(&frame).expect("Failed to convert AVFrame to DynamicImage");

        // DynamicImage -> AVFrame
        let new_frame = AVFrame::try_from_cv(&dynamic_image)
            .expect("Failed to convert DynamicImage back to AVFrame");

        // 验证格式和尺寸
        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_RGBA);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_bgr_image_conversion() {
        let start = Instant::now();

        // 创建测试数据
        let test_data = vec![
            0, 0, 255, // Blue pixel
            0, 255, 0, // Green pixel
            255, 0, 0, // Red pixel
            255, 255, 0, // Yellow pixel
        ];

        // 创建 AVFrame
        let frame = create_test_frame(ffi::AV_PIX_FMT_BGR24, 2, 2, test_data.clone(), None, None);

        // AVFrame -> DynamicImage
        let dynamic_image =
            DynamicImage::try_from_cv(&frame).expect("Failed to convert AVFrame to DynamicImage");

        // DynamicImage -> AVFrame
        let new_frame = AVFrame::try_from_cv(&dynamic_image)
            .expect("Failed to convert DynamicImage back to AVFrame");

        // 验证格式和尺寸
        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_RGB24);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_bgra_image_conversion() {
        let start = Instant::now();

        // 创建测试数据
        let test_data = vec![
            0, 0, 255, 255, // Blue pixel
            0, 255, 0, 255, // Green pixel
            255, 0, 0, 255, // Red pixel
            255, 255, 0, 128, // Semi-transparent yellow pixel
        ];

        // 创建 AVFrame
        let frame = create_test_frame(ffi::AV_PIX_FMT_BGRA, 2, 2, test_data.clone(), None, None);

        // AVFrame -> DynamicImage
        let dynamic_image =
            DynamicImage::try_from_cv(&frame).expect("Failed to convert AVFrame to DynamicImage");

        // DynamicImage -> AVFrame
        let new_frame = AVFrame::try_from_cv(&dynamic_image)
            .expect("Failed to convert DynamicImage back to AVFrame");

        // 验证格式和尺寸
        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_RGBA);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_gray_image_conversion() {
        let start = Instant::now();

        // 创建灰度测试数据
        let test_data = vec![0, 85, 170, 255]; // 不同灰度值

        // 创建 AVFrame
        let frame = create_test_frame(ffi::AV_PIX_FMT_GRAY8, 2, 2, test_data.clone(), None, None);

        // AVFrame -> DynamicImage
        let dynamic_image =
            DynamicImage::try_from_cv(&frame).expect("Failed to convert AVFrame to DynamicImage");

        // DynamicImage -> AVFrame
        let new_frame = AVFrame::try_from_cv(&dynamic_image)
            .expect("Failed to convert DynamicImage back to AVFrame");

        // 验证格式和尺寸
        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_GRAY8);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn testyuv420p_image_conversion() {
        let start = Instant::now();

        // 创建 YUV420P 测试数据
        let y_data = vec![235, 128, 16, 235]; // Y plane
        let u_data = vec![128]; // U plane (2x2 -> 1x1)
        let v_data = vec![128]; // V plane (2x2 -> 1x1)

        // 创建 AVFrame
        let frame = create_test_frame(
            ffi::AV_PIX_FMT_YUV420P,
            2,
            2,
            y_data.clone(),
            Some(u_data.clone()),
            Some(v_data.clone()),
        );

        // AVFrame YUV420P -> DynamicImage RGB24
        let dynamic_image =
            DynamicImage::try_from_cv(&frame).expect("Failed to convert AVFrame to DynamicImage");

        // DynamicImage RGB24 -> AVFrame RGB24
        let new_frame = AVFrame::try_from_cv(&dynamic_image)
            .expect("Failed to convert DynamicImage back to AVFrame");

        // 验证格式和尺寸
        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_RGB24);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_image_format_preservation() {
        let start = Instant::now();

        // RGB24 格式保持测试
        let rgb_frame = create_test_frame(
            ffi::AV_PIX_FMT_RGB24,
            2,
            2,
            vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0],
            None,
            None,
        );
        let rgb_image = DynamicImage::try_from_cv(&rgb_frame).unwrap();
        let new_rgb_frame = AVFrame::try_from_cv(&rgb_image).unwrap();
        assert_eq!(new_rgb_frame.format, ffi::AV_PIX_FMT_RGB24);

        // RGBA 格式保持测试
        let rgba_frame = create_test_frame(
            ffi::AV_PIX_FMT_RGBA,
            2,
            2,
            vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 128,
            ],
            None,
            None,
        );
        let rgba_image = DynamicImage::try_from_cv(&rgba_frame).unwrap();
        let new_rgba_frame = AVFrame::try_from_cv(&rgba_image).unwrap();
        assert_eq!(new_rgba_frame.format, ffi::AV_PIX_FMT_RGBA);

        // BGR24 格式保持测试
        let bgr_frame = create_test_frame(
            ffi::AV_PIX_FMT_BGR24,
            2,
            2,
            vec![0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 255],
            None,
            None,
        );
        let bgr_image = DynamicImage::try_from_cv(&bgr_frame).unwrap();
        let new_bgr_frame = AVFrame::try_from_cv(&bgr_image).unwrap();
        assert_eq!(new_bgr_frame.format, ffi::AV_PIX_FMT_RGB24);

        // BGRA 格式保持测试
        let bgra_frame = create_test_frame(
            ffi::AV_PIX_FMT_BGRA,
            2,
            2,
            vec![
                0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255, 255, 255, 0, 128,
            ],
            None,
            None,
        );
        let bgra_image = DynamicImage::try_from_cv(&bgra_frame).unwrap();
        let new_bgra_frame = AVFrame::try_from_cv(&bgra_image).unwrap();
        assert_eq!(new_bgra_frame.format, ffi::AV_PIX_FMT_RGBA);

        // GRAY8 格式保持测试
        let gray_frame = create_test_frame(
            ffi::AV_PIX_FMT_GRAY8,
            2,
            2,
            vec![0, 85, 170, 255],
            None,
            None,
        );
        let gray_image = DynamicImage::try_from_cv(&gray_frame).unwrap();
        let new_gray_frame = AVFrame::try_from_cv(&gray_image).unwrap();
        assert_eq!(new_gray_frame.format, ffi::AV_PIX_FMT_GRAY8);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_rgb_to_avframe_and_back() {
        let start = Instant::now();

        // 创建测试用 RGB 图像
        let mut rgb_image = image::RgbImage::new(2, 2);
        rgb_image.put_pixel(0, 0, image::Rgb([255, 0, 0])); // Red
        rgb_image.put_pixel(1, 0, image::Rgb([0, 255, 0])); // Green
        rgb_image.put_pixel(0, 1, image::Rgb([0, 0, 255])); // Blue
        rgb_image.put_pixel(1, 1, image::Rgb([255, 255, 0])); // Yellow

        // RgbImage -> DynamicImage
        let dynamic_image = DynamicImage::ImageRgb8(rgb_image.clone());

        // DynamicImage -> AVFrame
        let frame = AVFrame::try_from_cv(&dynamic_image)
            .expect("Failed to convert DynamicImage to AVFrame");

        assert_eq!(frame.format, ffi::AV_PIX_FMT_RGB24);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);

        // 验证像素数据
        unsafe {
            let data_ptr = (*frame.as_ptr()).data[0];
            assert_eq!(*data_ptr, 255); // Red component of first pixel
            assert_eq!(*data_ptr.offset(1), 0); // Green component of first pixel
            assert_eq!(*data_ptr.offset(2), 0); // Blue component of first pixel
        }

        // AVFrame -> DynamicImage
        let _new_dynamic_image =
            DynamicImage::try_from_cv(&frame).expect("Failed to convert AVFrame to DynamicImage");

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_rgba_to_avframe_and_back() {
        let start = Instant::now();

        // 创建测试用 RGBA 图像
        let mut rgba_image = image::RgbaImage::new(2, 2);
        rgba_image.put_pixel(0, 0, image::Rgba([255, 0, 0, 255])); // Red
        rgba_image.put_pixel(1, 0, image::Rgba([0, 255, 0, 255])); // Green
        rgba_image.put_pixel(0, 1, image::Rgba([0, 0, 255, 255])); // Blue
        rgba_image.put_pixel(1, 1, image::Rgba([255, 255, 0, 128])); // Semi-transparent Yellow

        // RgbaImage -> DynamicImage
        let dynamic_image = DynamicImage::ImageRgba8(rgba_image.clone());

        // DynamicImage -> AVFrame
        let frame = AVFrame::try_from_cv(&dynamic_image)
            .expect("Failed to convert DynamicImage to AVFrame");

        // 验证基本属性
        assert_eq!(frame.format, ffi::AV_PIX_FMT_RGBA);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);

        // 验证像素数据
        unsafe {
            let data_ptr = frame.data[0];
            assert_eq!(*data_ptr, 255); // R
            assert_eq!(*data_ptr.offset(1), 0); // G
            assert_eq!(*data_ptr.offset(2), 0); // B
            assert_eq!(*data_ptr.offset(3), 255); // A
        }

        // AVFrame -> DynamicImage
        let new_dynamic_image =
            DynamicImage::try_from_cv(&frame).expect("Failed to convert AVFrame to DynamicImage");

        println!("Test completed in: {}ms", start.elapsed().as_millis());

        // 验证边界条件
        assert_eq!(rgba_image.dimensions(), (2, 2));
        assert_eq!(new_dynamic_image.dimensions(), (2, 2));
    }

    #[test]
    fn test_gray_to_avframe_and_back() {
        let start = Instant::now();

        // 创建测试用灰度图像
        let mut gray_image = image::GrayImage::new(2, 2);
        gray_image.put_pixel(0, 0, image::Luma([0])); // Black
        gray_image.put_pixel(1, 0, image::Luma([85])); // Dark gray
        gray_image.put_pixel(0, 1, image::Luma([170])); // Light gray
        gray_image.put_pixel(1, 1, image::Luma([255])); // White

        // GrayImage -> DynamicImage
        let dynamic_image = DynamicImage::ImageLuma8(gray_image.clone());

        // DynamicImage -> AVFrame
        let frame = AVFrame::try_from_cv(&dynamic_image)
            .expect("Failed to convert DynamicImage to AVFrame");

        assert_eq!(frame.format, ffi::AV_PIX_FMT_GRAY8);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);

        // 验证像素数据
        unsafe {
            let data_ptr = (*frame.as_ptr()).data[0];
            assert_eq!(*data_ptr, 0); // First pixel
            assert_eq!(*data_ptr.offset(1), 85); // Second pixel
            assert_eq!(*data_ptr.offset(2), 170); // Third pixel
            assert_eq!(*data_ptr.offset(3), 255); // Fourth pixel
        }

        // AVFrame -> DynamicImage
        let _new_dynamic_image =
            DynamicImage::try_from_cv(&frame).expect("Failed to convert AVFrame to DynamicImage");

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_yuv420p_to_avframe_and_back() {
        let start = Instant::now();

        // 创建 YUV420P 测试数据
        let y_data = vec![235, 128, 16, 235]; // Y plane
        let u_data = vec![128]; // U plane (2x2 -> 1x1)
        let v_data = vec![128]; // V plane (2x2 -> 1x1)

        // 创建 AVFrame
        let frame = create_test_frame(
            ffi::AV_PIX_FMT_YUV420P,
            2,
            2,
            y_data.clone(),
            Some(u_data.clone()),
            Some(v_data.clone()),
        );

        // AVFrame -> DynamicImage
        let dynamic_image =
            DynamicImage::try_from_cv(&frame).expect("Failed to convert AVFrame to DynamicImage");

        // DynamicImage -> AVFrame
        let new_frame = AVFrame::try_from_cv(&dynamic_image)
            .expect("Failed to convert DynamicImage back to AVFrame");

        // 验证格式和尺寸
        assert_eq!(new_frame.format, ffi::AV_PIX_FMT_RGB24);
        assert_eq!(new_frame.width, 2);
        assert_eq!(new_frame.height, 2);

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }
}
