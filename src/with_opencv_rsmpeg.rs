use crate::with_rsmpeg;
use crate::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use anyhow::{Error, Result};
use opencv::core::Mat;
use opencv::prelude::*;
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;
use std::ops::Mul;

/// Convert OpenCV Mat to FFmpeg AVFrame
/// 支持的格式转换：
/// - CV_8UC3 (BGR) -> AV_PIX_FMT_BGR24
/// - CV_8UC3 (RGB) -> AV_PIX_FMT_RGB24
/// - CV_8UC1 (Gray) -> AV_PIX_FMT_GRAY8
/// - CV_8UC4 (BGRA) -> AV_PIX_FMT_BGRA
/// - CV_8UC4 (RGBA) -> AV_PIX_FMT_RGBA
impl TryFromCv<&Mat> for AVFrame {
    type Error = Error;

    fn try_from_cv(from: &Mat) -> Result<Self, Self::Error> {
        // 创建一个标准化的 Mat（如果需要转换）
        let normalized_mat = match (from.channels(), from.depth()) {
            // 1 通道格式
            (1, opencv::core::CV_8U) => Ok((from.clone(), ffi::AV_PIX_FMT_GRAY8, None)),
            (1, opencv::core::CV_16U) => {
                let mut converted = Mat::default();
                // FFmpeg 的常用像素格式主要基于 8 位/通道
                // 对于 16 位数据，我们需要缩放到 8 位范围
                // mat 不同格式的数据范围：
                // CV_8U:   0 - 255        (8位无符号整数)
                // CV_16U:  0 - 65535      (16位无符号整数)
                // CV_32F:  0.0 - 1.0      (32位浮点数，通常归一化)
                opencv::core::normalize(
                    from,
                    &mut converted,
                    0.0,
                    255.0,
                    opencv::core::NORM_MINMAX,
                    opencv::core::CV_8U,
                    &opencv::core::no_array(),
                )?;
                Ok((converted, ffi::AV_PIX_FMT_GRAY8, None))
            }
            (1, opencv::core::CV_32F) => {
                let mut converted = Mat::default();
                // 对于浮点数据，我们假设范围在 0-1 之间，需要缩放到 0-255
                opencv::core::normalize(
                    from,
                    &mut converted,
                    0.0,
                    255.0,
                    opencv::core::NORM_MINMAX,
                    opencv::core::CV_8U,
                    &opencv::core::no_array(),
                )?;
                Ok((converted, ffi::AV_PIX_FMT_GRAY8, None))
            }

            // 3 通道格式
            (3, opencv::core::CV_8U) => Ok((
                from.clone(),
                ffi::AV_PIX_FMT_RGB24,
                Some(opencv::imgproc::COLOR_BGR2RGB),
            )),
            (3, opencv::core::CV_16U) => {
                let mut converted = Mat::default();
                opencv::core::normalize(
                    from,
                    &mut converted,
                    0.0,
                    255.0,
                    opencv::core::NORM_MINMAX,
                    opencv::core::CV_8U,
                    &opencv::core::no_array(),
                )?;
                Ok((
                    converted,
                    ffi::AV_PIX_FMT_RGB24,
                    Some(opencv::imgproc::COLOR_BGR2RGB),
                ))
            }
            (3, opencv::core::CV_32F) => {
                let mut converted = Mat::default();
                opencv::core::normalize(
                    from,
                    &mut converted,
                    0.0,
                    255.0,
                    opencv::core::NORM_MINMAX,
                    opencv::core::CV_8U,
                    &opencv::core::no_array(),
                )?;
                Ok((
                    converted,
                    ffi::AV_PIX_FMT_RGB24,
                    Some(opencv::imgproc::COLOR_BGR2RGB),
                ))
            }

            // 4 通道格式
            (4, opencv::core::CV_8U) => Ok((
                from.clone(),
                ffi::AV_PIX_FMT_RGBA,
                Some(opencv::imgproc::COLOR_BGRA2RGBA),
            )),
            (4, opencv::core::CV_16U) => {
                let mut converted = Mat::default();
                opencv::core::normalize(
                    from,
                    &mut converted,
                    0.0,
                    255.0,
                    opencv::core::NORM_MINMAX,
                    opencv::core::CV_8U,
                    &opencv::core::no_array(),
                )?;
                Ok((
                    converted,
                    ffi::AV_PIX_FMT_RGBA,
                    Some(opencv::imgproc::COLOR_BGRA2RGBA),
                ))
            }
            (4, opencv::core::CV_32F) => {
                let mut converted = Mat::default();
                opencv::core::normalize(
                    from,
                    &mut converted,
                    0.0,
                    255.0,
                    opencv::core::NORM_MINMAX,
                    opencv::core::CV_8U,
                    &opencv::core::no_array(),
                )?;
                Ok((
                    converted,
                    ffi::AV_PIX_FMT_RGBA,
                    Some(opencv::imgproc::COLOR_BGRA2RGBA),
                ))
            }

            // 不支持的格式
            (channels, depth) => Err(Error::msg(format!(
                "Unsupported mat channels: {} depth: {}",
                channels, depth
            ))),
        }?;

        let mut frame = AVFrame::new();
        frame.set_width(from.cols());
        frame.set_height(from.rows());
        frame.set_pts(0);
        let (src_mat, format, color_conversion) = normalized_mat;
        frame.set_format(format);
        frame.alloc_buffer()?;

        // 如果需要颜色空间转换
        let mut converted = if let Some(conv) = color_conversion {
            let mut cvt = Mat::default();
            opencv::imgproc::cvt_color_def(&src_mat, &mut cvt, conv)?;
            cvt
        } else {
            src_mat
        };

        // 获取 mat 基本信息
        let src_step = converted.step1_def()? as usize;
        let height = converted.rows() as usize;
        // frame 的步长
        let dst_step = frame.linesize[0] as usize;

        // 逐行复制数据
        unsafe {
            for y in 0..height {
                let src_ptr = converted.ptr_mut(y as i32)?;
                let src_row = std::slice::from_raw_parts(src_ptr, src_step);

                let dst_ptr = frame.data[0].add(y * dst_step);

                // 只复制实际的图像数据宽度
                std::ptr::copy_nonoverlapping(
                    src_row.as_ptr(),
                    dst_ptr,
                    (converted.cols() * converted.channels()) as usize,
                );
            }
        }

        Ok(frame)
    }
}

/// Convert FFmpeg AVFrame to OpenCV Mat
/// 支持的格式转换：
/// - AV_PIX_FMT_BGR24 -> CV_8UC3 (BGR)
/// - AV_PIX_FMT_RGB24 -> CV_8UC3 (RGB)
/// - AV_PIX_FMT_GRAY8 -> CV_8UC1 (Gray)
/// - AV_PIX_FMT_BGRA -> CV_8UC4 (BGRA)
/// - AV_PIX_FMT_RGBA -> CV_8UC4 (RGBA)
impl TryFromCv<&AVFrame> for Mat {
    type Error = Error;

    fn try_from_cv(from: &AVFrame) -> Result<Self, Self::Error> {
        // 基于 ffmpeg 转换
        // YUV420P => RGB24
        let from = if from.format == ffi::AV_PIX_FMT_YUV420P {
            with_rsmpeg::convert_avframe(from, from.width, from.height, ffi::AV_PIX_FMT_RGB24)?
        } else {
            from.clone()
        };

        // 根据格式确定通道数和颜色转换
        let (channels, color_conversion) = match from.format {
            f if f == ffi::AV_PIX_FMT_GRAY8 => (1, None),
            f if f == ffi::AV_PIX_FMT_RGB24 => (3, Some(opencv::imgproc::COLOR_RGB2BGR)),
            f if f == ffi::AV_PIX_FMT_BGR24 => (3, None),
            f if f == ffi::AV_PIX_FMT_RGBA => (4, Some(opencv::imgproc::COLOR_RGBA2BGRA)),
            f if f == ffi::AV_PIX_FMT_BGRA => (4, None),
            _ => {
                return Err(Error::msg(format!(
                    "Unsupported pixel format: {}",
                    from.format
                )))
            }
        };

        let dst_mat = unsafe {
            // 创建目标 Mat
            let mut mat = Mat::new_rows_cols(
                from.height,
                from.width,
                opencv::core::CV_8U + ((channels - 1) << 3), // 计算正确的 Mat 类型
            )?;

            let src_step = from.linesize[0] as usize;
            let width = from.width;
            let height = from.height as usize;
            let row_size = (width * channels) as usize;

            // 逐行复制数据
            for y in 0..height {
                let src_ptr = from.data[0].add(y * src_step);
                let src_row = std::slice::from_raw_parts(src_ptr, row_size);

                let dst_ptr = mat.ptr_mut(y as i32)?;

                // 只复制实际的图像数据宽度
                std::ptr::copy_nonoverlapping(src_row.as_ptr(), dst_ptr, row_size);
            }

            mat
        };

        // 如果需要颜色空间转换
        match color_conversion {
            Some(conversion) => {
                let mut converted_mat = Mat::default();
                opencv::imgproc::cvt_color_def(&dst_mat, &mut converted_mat, conversion)?;
                Ok(converted_mat)
            }
            None => Ok(dst_mat),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_mat_to_avframe_conversion() {
        let start = Instant::now();
        // OpenCV 默认 BGR, 创建Mat: 2x2 BGR
        let mut mat = unsafe { Mat::new_rows_cols(2, 2, opencv::core::CV_8UC3).unwrap() };
        let data = mat.data_bytes_mut().unwrap();
        // BGR 格式数据
        data[0..3].copy_from_slice(&[0, 0, 255]); // Red
        data[3..6].copy_from_slice(&[0, 255, 0]); // Green
        data[6..9].copy_from_slice(&[255, 0, 0]); // Blue
        data[9..12].copy_from_slice(&[0, 255, 255]); // Yellow

        // BGR Mat -> RGB24 AVFrame
        let frame = AVFrame::try_from_cv(&mat).expect("Failed to convert Mat to AVFrame");
        // 验证 AVFrame 属性
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.format, ffi::AV_PIX_FMT_RGB24);
        // 验证 AVFrame 数据
        unsafe {
            let data = std::slice::from_raw_parts(
                frame.data[0],
                (frame.linesize[0] * frame.height) as usize,
            );
            // 验证第一个像素 (Red in RGB)
            assert_eq!(data[0], 255); // R
            assert_eq!(data[1], 0); // G
            assert_eq!(data[2], 0); // B
        }

        // RGB24 AVFrame -> BGR Mat
        let new_mat = Mat::try_from_cv(&frame).expect("Failed to convert AVFrame back to Mat");
        // 验证 Mat 属性
        assert_eq!(new_mat.typ(), opencv::core::CV_8UC3);
        assert_eq!(new_mat.depth(), opencv::core::CV_8U);
        assert_eq!(new_mat.channels(), 3);
        assert_eq!(new_mat.rows(), 2);
        assert_eq!(new_mat.cols(), 2);
        assert_eq!(new_mat.channels(), 3);
        // 验证 Mat 数据
        let data = new_mat.data_bytes().unwrap();
        // 验证第一个像素 (Red in BGR)
        assert_eq!(data[0], 0); // B
        assert_eq!(data[1], 0); // G
        assert_eq!(data[2], 255); // R

        // 验证两个 Mat 是否完全相同
        let mut diff = Mat::default();
        opencv::core::absdiff(&mat, &new_mat, &mut diff).unwrap();
        let sum = opencv::core::sum_elems(&diff).unwrap();
        assert_eq!(sum[0], 0.0); // 确保没有任何差异

        // save
        save_mat_as_image(&mat, "rgb_mat.png").unwrap();
        save_mat_as_image(&new_mat, "rgb_new_mat.png").unwrap();

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_grayscale_conversion() {
        let start = Instant::now();

        // 创建灰度 Mat
        let mut mat = unsafe { Mat::new_rows_cols(2, 2, opencv::core::CV_8UC1).unwrap() };
        let data = mat.data_bytes_mut().unwrap();
        data[0] = 0; // Black
        data[1] = 85; // Dark gray
        data[2] = 170; // Light gray
        data[3] = 255; // White

        // 验证原始 Mat 属性
        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.channels(), 1);
        assert_eq!(mat.typ(), opencv::core::CV_8UC1);

        // Mat -> AVFrame
        let frame = AVFrame::try_from_cv(&mat).expect("Failed to convert Mat to AVFrame");
        // 验证 AVFrame 属性
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.format, ffi::AV_PIX_FMT_GRAY8);
        // 验证 AVFrame 数据
        unsafe {
            let linesize = frame.linesize[0] as usize;
            println!("AVFrame linesize: {}", linesize);

            // 正确访问每个像素，考虑行步长
            let row0 = std::slice::from_raw_parts(frame.data[0], linesize);
            let row1 = std::slice::from_raw_parts(frame.data[0].add(linesize), linesize);

            // 验证第一行的像素
            assert_eq!(row0[0], 0); // Black
            assert_eq!(row0[1], 85); // Dark gray

            // 验证第二行的像素
            assert_eq!(row1[0], 170); // Light gray
            assert_eq!(row1[1], 255); // White
        }

        // AVFrame -> Mat
        let new_mat = Mat::try_from_cv(&frame).expect("Failed to convert AVFrame back to Mat");
        // 验证转换后的 Mat 属性
        assert_eq!(new_mat.rows(), 2);
        assert_eq!(new_mat.cols(), 2);
        assert_eq!(new_mat.channels(), 1);
        assert_eq!(new_mat.typ(), opencv::core::CV_8UC1);
        // 验证转换后的 Mat 数据
        let data = new_mat.data_bytes().unwrap();
        assert_eq!(data[0], 0); // Black
        assert_eq!(data[1], 85); // Dark gray
        assert_eq!(data[2], 170); // Light gray
        assert_eq!(data[3], 255); // White

        // 验证两个 Mat 是否完全相同
        let mut diff = Mat::default();
        opencv::core::absdiff(&mat, &new_mat, &mut diff).unwrap();
        let sum = opencv::core::sum_elems(&diff).unwrap();
        assert_eq!(sum[0], 0.0); // 确保没有任何差异

        // save
        save_mat_as_image(&mat, "gray_mat.png").unwrap();
        save_mat_as_image(&new_mat, "gray_new_mat.png").unwrap();

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    /// 辅助函数：创建和填充 Mat
    fn create_test_mat(width: i32, height: i32, mat_type: i32) -> Mat {
        let mut mat = unsafe { Mat::new_rows_cols(height, width, mat_type).unwrap() };

        match mat_type {
            opencv::core::CV_8UC1 => {
                // 灰度图像
                let data = mat.data_bytes_mut().unwrap();
                for i in 0..data.len() {
                    data[i] = (i % 256) as u8;
                }
            }
            opencv::core::CV_8UC3 => {
                // BGR 图像
                for y in 0..height {
                    for x in 0..width {
                        *mat.at_2d_mut::<opencv::core::Vec3b>(y, x).unwrap() =
                            opencv::core::Vec3b::from([
                                (x * 30) as u8,       // B
                                (y * 30) as u8,       // G
                                ((x + y) * 20) as u8, // R
                            ]);
                    }
                }
            }
            opencv::core::CV_8UC4 => {
                // BGRA 图像
                for y in 0..height {
                    for x in 0..width {
                        *mat.at_2d_mut::<opencv::core::Vec4b>(y, x).unwrap() =
                            opencv::core::Vec4b::from([
                                (x * 30) as u8,       // B
                                (y * 30) as u8,       // G
                                ((x + y) * 20) as u8, // R
                                200,                  // A (固定透明度)
                            ]);
                    }
                }
            }
            opencv::core::CV_16UC1 => {
                // 16位灰度图像
                for y in 0..height {
                    for x in 0..width {
                        *mat.at_2d_mut::<u16>(y, x).unwrap() = ((x + y) * 1000) as u16;
                    }
                }
            }
            opencv::core::CV_32FC3 => {
                // 浮点数 BGR 图像
                for y in 0..height {
                    for x in 0..width {
                        *mat.at_2d_mut::<opencv::core::Vec3f>(y, x).unwrap() =
                            opencv::core::Vec3f::from([
                                x as f32 / width as f32,  // B
                                y as f32 / height as f32, // G
                                0.5,                      // R
                            ]);
                    }
                }
            }
            _ => panic!("Unsupported mat type"),
        }
        mat
    }

    /// 验证两个 Mat 是否相似
    fn verify_mats_similar(mat1: &Mat, mat2: &Mat, tolerance: f64) -> bool {
        let mut diff = Mat::default();
        opencv::core::absdiff(mat1, mat2, &mut diff).unwrap();
        let mean = opencv::core::Scalar::default();
        opencv::core::mean(&diff, &opencv::core::no_array()).unwrap();
        mean[0] < tolerance
    }

    /// 常用格式
    /// - PNG (.png)    // 无损压缩,适合截图、图标等
    /// - JPG (.jpg,)   // 有损压缩,文件小，适合照片
    /// - JPEG (.jpeg)  // 有损压缩,文件小，适合照片
    /// - BMP (.bmp)    // 无压缩
    /// - TIFF (.tiff)
    /// - WebP (.webp)  // 无压缩，保存最快
    fn save_mat_as_image(mat: &Mat, path: &str) -> Result<()> {
        // 保存图像
        opencv::imgcodecs::imwrite(path, mat, &opencv::core::Vector::new()).unwrap();

        // 读取图像
        // let loaded_mat =
        //     opencv::imgcodecs::imread("img_path.png", opencv::imgcodecs::IMREAD_UNCHANGED)
        //         .expect("Failed to load saved image");

        Ok(())
    }

    #[test]
    fn test_gray8_conversion() {
        let start = Instant::now();

        // 创建 8位灰度图
        let original_mat = create_test_mat(64, 48, opencv::core::CV_8UC1);

        // Mat -> AVFrame
        let frame = AVFrame::try_from_cv(&original_mat).expect("Failed to convert Mat to AVFrame");

        // 验证 AVFrame 属性
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 48);
        assert_eq!(frame.format, ffi::AV_PIX_FMT_GRAY8);

        // AVFrame -> Mat
        let converted_mat =
            Mat::try_from_cv(&frame).expect("Failed to convert AVFrame back to Mat");

        // 验证转换结果
        assert!(verify_mats_similar(&original_mat, &converted_mat, 1.0));

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_bgr24_conversion() {
        let start = Instant::now();

        // 创建 BGR 图像
        let original_mat = create_test_mat(64, 48, opencv::core::CV_8UC3);

        // Mat -> AVFrame
        let frame = AVFrame::try_from_cv(&original_mat).expect("Failed to convert Mat to AVFrame");

        // 验证 AVFrame 属性
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 48);
        assert_eq!(frame.format, ffi::AV_PIX_FMT_RGB24);

        // AVFrame -> Mat
        let converted_mat =
            Mat::try_from_cv(&frame).expect("Failed to convert AVFrame back to Mat");

        // 验证转换结果
        assert!(verify_mats_similar(&original_mat, &converted_mat, 1.0));

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_bgra_conversion() {
        let start = Instant::now();

        // 创建 BGRA 图像
        let original_mat = create_test_mat(64, 48, opencv::core::CV_8UC4);

        // Mat -> AVFrame
        let frame = AVFrame::try_from_cv(&original_mat).expect("Failed to convert Mat to AVFrame");

        // 验证 AVFrame 属性
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 48);
        assert_eq!(frame.format, ffi::AV_PIX_FMT_RGBA);

        // AVFrame -> Mat
        let converted_mat =
            Mat::try_from_cv(&frame).expect("Failed to convert AVFrame back to Mat");

        // 验证转换结果
        assert!(verify_mats_similar(&original_mat, &converted_mat, 1.0));

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_16bit_gray_conversion() {
        let start = Instant::now();

        // 创建 16位灰度图
        let original_mat = create_test_mat(64, 48, opencv::core::CV_16UC1);

        // Mat -> AVFrame
        let frame = AVFrame::try_from_cv(&original_mat).expect("Failed to convert Mat to AVFrame");

        // 验证 AVFrame 属性
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 48);
        assert_eq!(frame.format, ffi::AV_PIX_FMT_GRAY8);

        // AVFrame -> Mat
        // 注意：这里会有精度损失，因为从16位转换到8位
        let converted_mat =
            Mat::try_from_cv(&frame).expect("Failed to convert AVFrame back to Mat");

        // 转换原始 mat 为 8位以进行比较
        let mut original_8bit = Mat::default();
        opencv::core::normalize(
            &original_mat,
            &mut original_8bit,
            0.0,
            255.0,
            opencv::core::NORM_MINMAX,
            opencv::core::CV_8UC1,
            &opencv::core::no_array(),
        )
        .unwrap();

        // 验证转换结果
        assert!(verify_mats_similar(&original_8bit, &converted_mat, 1.0));

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn test_float_bgr_conversion() {
        let start = Instant::now();

        // 创建浮点数 BGR 图像
        let original_mat = create_test_mat(64, 48, opencv::core::CV_32FC3);

        // Mat -> AVFrame
        let frame = AVFrame::try_from_cv(&original_mat).expect("Failed to convert Mat to AVFrame");

        // 验证 AVFrame 属性
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 48);
        assert_eq!(frame.format, ffi::AV_PIX_FMT_RGB24);

        // AVFrame -> Mat
        let converted_mat =
            Mat::try_from_cv(&frame).expect("Failed to convert AVFrame back to Mat");

        // 转换原始 mat 为 8位以进行比较
        let mut original_8bit = Mat::default();
        opencv::core::normalize(
            &original_mat,
            &mut original_8bit,
            0.0,
            255.0,
            opencv::core::NORM_MINMAX,
            opencv::core::CV_8UC3,
            &opencv::core::no_array(),
        )
        .unwrap();

        // 验证转换结果
        assert!(verify_mats_similar(&original_8bit, &converted_mat, 1.0));

        println!("Test completed in: {}ms", start.elapsed().as_millis());
    }
}
