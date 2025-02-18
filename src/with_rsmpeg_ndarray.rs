use crate::with_ndarray::{AVFramePixel, ArrayWithFormat, PixelFormat, PixelType};
use crate::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use anyhow::{Error, Result};
use ndarray::Array3;
use num_traits::{NumCast, Zero};
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;

// &AVFrame -> Array3
impl<T: PixelType> TryFromCv<&AVFrame> for Array3<T> {
    type Error = Error;

    fn try_from_cv(frame: &AVFrame) -> Result<Self, Self::Error> {
        if frame.data[0].is_null() {
            return Err(Error::msg("Cannot get frame data error"));
        }

        let width = frame.width as usize;
        let height = frame.height as usize;
        let pix_fmt = get_pixel_format(frame.format).unwrap();
        match pix_fmt {
            PixelFormat::RGB4
            | PixelFormat::RGB8
            | PixelFormat::RGB24
            | PixelFormat::BGR4
            | PixelFormat::BGR8
            | PixelFormat::BGR24 => handle_rgb(frame, height, width, pix_fmt.bits_per_pixel()),
            PixelFormat::RGBA | PixelFormat::BGRA => {
                handle_rgba(frame, height, width, pix_fmt.bits_per_pixel())
            }
            PixelFormat::GRAY8 => handle_gray(frame, height, width, pix_fmt.bits_per_pixel()),
            PixelFormat::YUV410P
            | PixelFormat::YUV411P
            | PixelFormat::YUV420P
            | PixelFormat::YUV422P
            | PixelFormat::YUV440P
            | PixelFormat::YUV444P => {
                let (params, _dimensions) = pix_fmt.yuv_params().unwrap();
                handle_yuv(frame, height, width, params.subsample_x, params.subsample_y)
            }
            _ => Err(Error::msg(format!(
                "Unsupported pixel format: {:?}",
                pix_fmt
            ))),
        }
    }
}

// AVFrame -> Array3
impl<T: PixelType> TryFromCv<AVFrame> for Array3<T> {
    type Error = Error;

    fn try_from_cv(frame: AVFrame) -> Result<Self, Self::Error> {
        (&frame).try_into_cv()
    }
}

// Array3 -> AVFrame
impl<T: PixelType, F: AVFramePixel> TryFromCv<ArrayWithFormat<T, F>> for AVFrame {
    type Error = Error;

    fn try_from_cv(arr_with_fmt: ArrayWithFormat<T, F>) -> Result<Self, Self::Error> {
        let array = arr_with_fmt.array;
        let format = arr_with_fmt.format;
        let (height, width, _channels) = array.dim();

        // 创建并设置 AVFrame
        let mut frame = AVFrame::new();
        frame.set_format(format.pix_fmt());
        frame.set_width(width as i32);
        frame.set_height(height as i32);
        frame.alloc_buffer()?;

        // 获取像素格式的YUV参数
        let yuv_params = format.yuv_params();

        unsafe {
            let frame_ptr = frame.as_mut_ptr();

            match format.pix_fmt() {
                // RGB 格式处理
                f if f == PixelFormat::RGB24.pix_fmt() || f == PixelFormat::BGR24.pix_fmt() => {
                    let line_size = (*frame_ptr).linesize[0] as usize;
                    let data_ptr = (*frame_ptr).data[0];

                    for y in 0..height {
                        let row_ptr = data_ptr.add(y * line_size);
                        for x in 0..width {
                            for c in 0..3 {
                                *row_ptr.add(x * 3 + c) = array[[y, x, c]].to_u8().unwrap();
                            }
                        }
                    }
                }

                // RGBA/BGRA 格式处理
                f if f == PixelFormat::RGBA.pix_fmt() || f == PixelFormat::BGRA.pix_fmt() => {
                    let line_size = (*frame_ptr).linesize[0] as usize;
                    let data_ptr = (*frame_ptr).data[0];

                    for y in 0..height {
                        let row_ptr = data_ptr.add(y * line_size);
                        for x in 0..width {
                            for c in 0..4 {
                                *row_ptr.add(x * 4 + c) = array[[y, x, c]].to_u8().unwrap();
                            }
                        }
                    }
                }

                // YUV 平面格式处理
                f if f == PixelFormat::YUV420P.pix_fmt()
                    || f == PixelFormat::YUV422P.pix_fmt()
                    || f == PixelFormat::YUV444P.pix_fmt() =>
                {
                    let (yuv_params, uv_dims) = yuv_params.unwrap();

                    // Y平面处理
                    let y_line_size = (*frame_ptr).linesize[0] as usize;
                    let y_ptr = (*frame_ptr).data[0];
                    for y in 0..height {
                        let row_ptr = y_ptr.add(y * y_line_size);
                        for x in 0..width {
                            *row_ptr.add(x) = array[[y, x, 0]].to_u8().unwrap();
                        }
                    }

                    // UV平面处理
                    let uv_width = width >> yuv_params.subsample_x;
                    let uv_height = height >> yuv_params.subsample_y;
                    let u_line_size = (*frame_ptr).linesize[1] as usize;
                    let v_line_size = (*frame_ptr).linesize[2] as usize;
                    let u_ptr = (*frame_ptr).data[1];
                    let v_ptr = (*frame_ptr).data[2];

                    for y in 0..uv_height {
                        let u_row_ptr = u_ptr.add(y * u_line_size);
                        let v_row_ptr = v_ptr.add(y * v_line_size);
                        for x in 0..uv_width {
                            let src_y = y * uv_dims.height;
                            let src_x = x * uv_dims.width;

                            *u_row_ptr.add(x) = array[[src_y, src_x, 1]].to_u8().unwrap();
                            *v_row_ptr.add(x) = array[[src_y, src_x, 2]].to_u8().unwrap();
                        }
                    }
                }

                // 灰度图格式处理
                f if f == PixelFormat::GRAY8.pix_fmt() => {
                    let line_size = (*frame_ptr).linesize[0] as usize;
                    let data_ptr = (*frame_ptr).data[0];

                    for y in 0..height {
                        let row_ptr = data_ptr.add(y * line_size);
                        for x in 0..width {
                            *row_ptr.add(x) = array[[y, x, 0]].to_u8().unwrap();
                        }
                    }
                }

                _ => return Err(Error::msg("Unsupported pixel format")),
            }
        }

        Ok(frame)
    }
}

fn get_pixel_format(format: ffi::AVPixelFormat) -> Result<PixelFormat> {
    match format {
        // RGB
        ffi::AV_PIX_FMT_RGB4 => Ok(PixelFormat::RGB4),
        ffi::AV_PIX_FMT_RGB8 => Ok(PixelFormat::RGB8),
        ffi::AV_PIX_FMT_RGB24 => Ok(PixelFormat::RGB24),
        // ffi::AV_PIX_FMT_RGB32 => Ok(PixelFormat::RGB32), // RGB32 == BGRA : 28
        // BGR
        ffi::AV_PIX_FMT_BGR4 => Ok(PixelFormat::BGR4),
        ffi::AV_PIX_FMT_BGR8 => Ok(PixelFormat::BGR8),
        ffi::AV_PIX_FMT_BGR24 => Ok(PixelFormat::BGR24),
        // ffi::AV_PIX_FMT_BGR32 => Ok(PixelFormat::BGR32), // BGR32 == RGBA : 26
        // RGBA/BGRA
        ffi::AV_PIX_FMT_RGBA => Ok(PixelFormat::RGBA),
        ffi::AV_PIX_FMT_BGRA => Ok(PixelFormat::BGRA),
        // Gray
        ffi::AV_PIX_FMT_GRAY8 => Ok(PixelFormat::GRAY8),
        // YUV
        ffi::AV_PIX_FMT_YUV410P => Ok(PixelFormat::YUV410P),
        ffi::AV_PIX_FMT_YUV411P => Ok(PixelFormat::YUV411P),
        ffi::AV_PIX_FMT_YUV420P => Ok(PixelFormat::YUV420P),
        ffi::AV_PIX_FMT_YUV422P => Ok(PixelFormat::YUV422P),
        ffi::AV_PIX_FMT_YUV440P => Ok(PixelFormat::YUV440P),
        ffi::AV_PIX_FMT_YUV444P => Ok(PixelFormat::YUV444P),
        ffi::AV_PIX_FMT_YUYV422 => Ok(PixelFormat::YUYV422),
        // 不支持的格式返回错误
        _ => Err(Error::msg(format!("Unsupported pixel format: {}", format))),
    }
}

// 处理 RGB 格式
fn handle_rgb<T>(
    frame: &AVFrame,
    height: usize,
    width: usize,
    bits_per_pixel: u32,
) -> Result<Array3<T>, Error>
where
    T: Copy + Clone + NumCast + Zero + 'static,
{
    let channels = bits_per_pixel.div_ceil(8);
    let mut array = Array3::zeros((height, width, channels as usize));
    let stride = frame.linesize[0] as usize;

    for y in 0..height {
        for x in 0..width {
            let offset = y * stride + x * (channels as usize);
            for c in 0..channels {
                let value = unsafe { *frame.data[0].add(offset + c as usize) };
                array[[y, x, c as usize]] = T::from(value).unwrap();
            }
        }
    }

    Ok(array)
}

fn handle_rgba<T>(
    frame: &AVFrame,
    height: usize,
    width: usize,
    bits_per_pixel: u32,
) -> Result<Array3<T>>
where
    T: Copy + Clone + NumCast + Zero + 'static,
{
    // 参数验证
    if bits_per_pixel != 32 {
        return Err(Error::msg("Only 32-bit RGBA format is supported"));
    }

    // 确保输入frame数据有效
    let data = frame.data[0];

    // 计算每个颜色通道的字节数
    let _bytes_per_channel = (bits_per_pixel / 32) * 8;
    let linesize = frame.linesize[0] as usize;
    let mut array = Array3::<T>::zeros((height, width, 4));

    // 安全地将frame数据转换为切片
    let frame_data = unsafe { std::slice::from_raw_parts(data as *const u8, linesize * height) };

    // 遍历每个像素并复制数据
    for y in 0..height {
        for x in 0..width {
            let pixel_offset = y * linesize + x * 4;

            // 直接按照RGBA顺序读取数据
            for c in 0..4 {
                let idx = pixel_offset + c;
                let normalized: T = NumCast::from(frame_data[idx]).unwrap();
                array[[y, x, c]] = normalized;
            }
        }
    }

    Ok(array)
}

fn handle_gray<T>(
    frame: &AVFrame,
    height: usize,
    width: usize,
    _bits: u32,
) -> Result<Array3<T>, Error>
where
    T: Copy + Clone + NumCast + Zero + 'static,
{
    let mut array = Array3::zeros((height, width, 1));
    let stride = frame.linesize[0] as usize;

    for y in 0..height {
        for x in 0..width {
            let offset = y * stride + x;
            let value = unsafe { *frame.data[0].add(offset) };
            array[[y, x, 0]] = T::from(value).unwrap();
        }
    }

    Ok(array)
}

fn handle_yuv<T>(
    frame: &AVFrame,
    height: usize,
    width: usize,
    subsample_x: u32,
    subsample_y: u32,
) -> Result<Array3<T>, Error>
where
    T: Copy + Clone + NumCast + Zero + 'static,
{
    let mut array = Array3::zeros((height, width, 3));

    // 检查所有平面的数据指针
    if frame.data[0].is_null() || frame.data[1].is_null() || frame.data[2].is_null() {
        return Err(Error::msg("YUV plane buffer is null"));
    }

    let y_stride = frame.linesize[0] as usize;
    let u_stride = frame.linesize[1] as usize;
    let v_stride = frame.linesize[2] as usize;

    // Y平面
    for y in 0..height {
        for x in 0..width {
            let value = unsafe { *frame.data[0].add(y * y_stride + x) };
            array[[y, x, 0]] = T::from(value).unwrap();
        }
    }

    // U和V平面
    for y in 0..height {
        let uv_y = y >> subsample_y;
        for x in 0..width {
            let uv_x = x >> subsample_x;

            let u_value = unsafe { *frame.data[1].add(uv_y * u_stride + uv_x) };
            let v_value = unsafe { *frame.data[2].add(uv_y * v_stride + uv_x) };

            array[[y, x, 1]] = T::from(u_value).unwrap();
            array[[y, x, 2]] = T::from(v_value).unwrap();
        }
    }

    Ok(array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::with_ndarray::{self, ArrayExt};

    fn create_test_frame(
        width: i32,
        height: i32,
        format: PixelFormat,
        pattern: Box<dyn Fn(i32, i32, i32) -> u8>,
    ) -> Result<AVFrame> {
        // 创建并初始化帧
        let mut frame = AVFrame::new();
        frame.set_format(format.pix_fmt());
        frame.set_width(width);
        frame.set_height(height);
        frame.alloc_buffer()?;

        match format {
            // 打包格式处理 (RGB, BGR, RGBA, BGRA)
            PixelFormat::RGB24 | PixelFormat::BGR24 | PixelFormat::RGBA | PixelFormat::BGRA => {
                let channels = format.channels();
                let stride = frame.linesize[0] as usize;
                let data = unsafe {
                    std::slice::from_raw_parts_mut(frame.data[0], (height as usize) * stride)
                };

                // 处理颜色通道顺序
                let channel_map: Vec<usize> = match format {
                    PixelFormat::BGR24 => vec![2, 1, 0],
                    PixelFormat::BGRA => vec![2, 1, 0, 3],
                    PixelFormat::RGB24 => vec![0, 1, 2],
                    PixelFormat::RGBA => vec![0, 1, 2, 3],
                    _ => unreachable!(),
                };

                for y in 0..height {
                    for x in 0..width {
                        for c in 0..channels {
                            let mapped_c = channel_map[c];
                            data[(y as usize) * stride + (x as usize) * channels + c] =
                                pattern(x, y, mapped_c as i32);
                        }
                    }
                }
            }

            // 单通道格式处理
            PixelFormat::GRAY8 => {
                let stride = frame.linesize[0] as usize;
                let data = unsafe {
                    std::slice::from_raw_parts_mut(frame.data[0], (height as usize) * stride)
                };

                for y in 0..height {
                    for x in 0..width {
                        data[(y as usize) * stride + x as usize] = pattern(x, y, 0);
                    }
                }
            }

            // YUV 平面格式处理
            PixelFormat::YUV420P
            | PixelFormat::YUV422P
            | PixelFormat::YUV444P
            | PixelFormat::YUV410P
            | PixelFormat::YUV411P
            | PixelFormat::YUV440P => {
                // 获取子采样参数
                let (params, _dimensions) = format.yuv_params().unwrap();
                let (subsample_x, subsample_y) = (params.subsample_x, params.subsample_y);

                // Y平面
                let y_stride = frame.linesize[0] as usize;
                let y_data = unsafe {
                    std::slice::from_raw_parts_mut(frame.data[0], (height as usize) * y_stride)
                };

                // UV平面尺寸
                let uv_width = width >> subsample_x;
                let uv_height = height >> subsample_y;

                // U平面
                let u_stride = frame.linesize[1] as usize;
                let u_data = unsafe {
                    std::slice::from_raw_parts_mut(frame.data[1], (uv_height as usize) * u_stride)
                };

                // V平面
                let v_stride = frame.linesize[2] as usize;
                let v_data = unsafe {
                    std::slice::from_raw_parts_mut(frame.data[2], (uv_height as usize) * v_stride)
                };

                // 填充Y平面
                for y in 0..height {
                    for x in 0..width {
                        y_data[(y as usize) * y_stride + x as usize] = pattern(x, y, 0);
                    }
                }

                // 填充UV平面
                for y in 0..uv_height {
                    for x in 0..uv_width {
                        // 计算原始坐标
                        let orig_x = x << subsample_x;
                        let orig_y = y << subsample_y;

                        u_data[(y as usize) * u_stride + x as usize] = pattern(orig_x, orig_y, 1);
                        v_data[(y as usize) * v_stride + x as usize] = pattern(orig_x, orig_y, 2);
                    }
                }
            }

            // 打包YUV格式
            PixelFormat::YUYV422 => {
                let stride = frame.linesize[0] as usize;
                let data = unsafe {
                    std::slice::from_raw_parts_mut(frame.data[0], (height as usize) * stride)
                };

                for y in 0..height {
                    for x in 0..(width / 2) {
                        // YUYV = 2 pixels
                        let base = (y as usize) * stride + (x as usize) * 4;
                        // Y1 U Y2 V
                        data[base + 0] = pattern(x * 2, y, 0); // Y1
                        data[base + 1] = pattern(x * 2, y, 1); // U
                        data[base + 2] = pattern(x * 2 + 1, y, 0); // Y2
                        data[base + 3] = pattern(x * 2, y, 2); // V
                    }
                }
            }

            _ => {
                return Err(Error::msg(format!(
                    "Unsupported pixel format: {:?}",
                    format
                )))
            }
        }

        Ok(frame)
    }

    #[test]
    fn test_frame_rgb24_array_conversion() -> Result<()> {
        let width = 64;
        let height = 48;

        // 创建测试帧
        let frame = create_test_frame(
            width,
            height,
            PixelFormat::RGB24,
            Box::new(|x, y, c| ((x + y + c) % 256) as u8),
        )?;

        // 转换为 Array3
        let array = Array3::<u8>::try_from_cv(&frame)?;

        // 验证维度
        assert_eq!(array.dim(), (height as usize, width as usize, 3));

        // 验证数据
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    assert_eq!(
                        array[[y as usize, x as usize, c as usize]],
                        ((x + y + c) % 256) as u8
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_frame_rgba_array_conversion() -> Result<()> {
        let width = 64;
        let height = 48;

        let frame = create_test_frame(
            width,
            height,
            PixelFormat::RGBA,
            Box::new(|x, y, c| ((x + y + c) % 256) as u8),
        )?;

        let array = Array3::<u8>::try_from_cv(&frame)?;

        assert_eq!(array.dim(), (height as usize, width as usize, 4));

        for y in 0..height {
            for x in 0..width {
                for c in 0..4 {
                    assert_eq!(
                        array[[y as usize, x as usize, c as usize]],
                        ((x + y + c) % 256) as u8
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_frame_gray8_array_conversion() -> Result<()> {
        let width = 64;
        let height = 48;

        let frame = create_test_frame(
            width,
            height,
            PixelFormat::GRAY8,
            Box::new(|x, y, _| ((x + y) % 256) as u8),
        )?;

        let array = Array3::<u8>::try_from_cv(&frame)?;

        assert_eq!(array.dim(), (height as usize, width as usize, 1));

        for y in 0..height {
            for x in 0..width {
                assert_eq!(array[[y as usize, x as usize, 0]], ((x + y) % 256) as u8);
            }
        }

        Ok(())
    }

    #[test]
    fn test_frame_yuv420p_arrray_conversion() -> Result<()> {
        let width = 64;
        let height = 48;

        let frame = create_test_frame(
            width,
            height,
            PixelFormat::YUV420P,
            Box::new(|x, y, c| ((x + y + c * 50) % 256) as u8),
        )?;

        let array = Array3::<u8>::try_from_cv(&frame)?;

        assert_eq!(array.dim(), (height as usize, width as usize, 3));

        // 只验证 Y 平面的完整分辨率
        for y in 0..height {
            for x in 0..width {
                assert_eq!(
                    array[[y as usize, x as usize, 0]],
                    ((x + y) % 256) as u8,
                    "Y plane mismatch at ({}, {})",
                    x,
                    y
                );
            }
        }

        // UV 平面是子采样的，验证采样点
        for y in (0..height).step_by(2) {
            for x in (0..width).step_by(2) {
                assert_eq!(
                    array[[y as usize, x as usize, 1]],
                    ((x + y + 50) % 256) as u8,
                    "U plane mismatch at ({}, {})",
                    x,
                    y
                );
                assert_eq!(
                    array[[y as usize, x as usize, 2]],
                    ((x + y + 100) % 256) as u8,
                    "V plane mismatch at ({}, {})",
                    x,
                    y
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_zero_dimensions() {
        let mut frame = AVFrame::new();
        frame.set_format(ffi::AV_PIX_FMT_RGB24);
        frame.set_width(0);
        frame.set_height(0);

        let result = Array3::<u8>::try_from_cv(&frame);
        assert!(result.is_err());
    }

    #[test]
    fn test_different_numeric_types() -> Result<()> {
        let width = 64;
        let height = 48;

        let frame = create_test_frame(
            width,
            height,
            PixelFormat::RGB24,
            Box::new(|x, y, c| ((x + y + c) % 256) as u8),
        )?;

        // 测试转换为 f32
        let array_f32 = Array3::<f32>::try_from_cv(&frame)?;
        assert_eq!(array_f32.dim(), (height as usize, width as usize, 3));

        // 测试转换为 u16
        let array_u16 = Array3::<u16>::try_from_cv(&frame)?;
        assert_eq!(array_u16.dim(), (height as usize, width as usize, 3));

        Ok(())
    }

    #[test]
    fn test_array_to_frame() -> Result<()> {
        let width = 64;
        let height = 48;

        let frame = create_test_frame(
            width,
            height,
            PixelFormat::RGB24,
            Box::new(|x, y, c| ((x + y + c) % 256) as u8),
        )?;

        // 测试转换为 f32
        let rgb_arr = Array3::<u8>::try_from_cv(&frame)?;
        assert_eq!(rgb_arr.dim(), (height as usize, width as usize, 3));

        let frame = AVFrame::try_from_cv(rgb_arr.with_format(PixelFormat::RGB24))?;
        assert_eq!(frame.width, width);
        assert_eq!(frame.height, height);
        assert_eq!(frame.format, ffi::AV_PIX_FMT_RGB24);

        Ok(())
    }

    /// 生成测试用的 RGB 数据
    fn create_test_rgb_data(height: usize, width: usize, channels: usize) -> Array3<u8> {
        let mut array = Array3::zeros((height, width, channels));
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    array[[y, x, c]] = ((y + x + c) % 255) as u8;
                }
            }
        }
        array
    }

    /// 验证 AVFrame 中的数据是否正确
    fn verify_frame_data(
        frame: &AVFrame,
        expected: &Array3<u8>,
        format: PixelFormat,
    ) -> Result<()> {
        unsafe {
            let frame_ptr = frame.as_ptr();
            match format {
                PixelFormat::RGB24 | PixelFormat::BGR24 => {
                    let channels = 3;
                    let line_size = (*frame_ptr).linesize[0] as usize;
                    let data_ptr = (*frame_ptr).data[0];

                    for y in 0..frame.height as usize {
                        for x in 0..frame.width as usize {
                            for c in 0..channels {
                                assert_eq!(
                                    *data_ptr.add(y * line_size + x * channels + c),
                                    expected[[y, x, c]],
                                    "Mismatch at position y={}, x={}, c={}",
                                    y,
                                    x,
                                    c
                                );
                            }
                        }
                    }
                }
                PixelFormat::YUV420P => {
                    let y_line_size = (*frame_ptr).linesize[0] as usize;
                    let u_line_size = (*frame_ptr).linesize[1] as usize;
                    let v_line_size = (*frame_ptr).linesize[2] as usize;
                    let height = frame.height as usize;
                    let width = frame.width as usize;

                    // 验证 Y 平面
                    for y in 0..height {
                        for x in 0..width {
                            assert_eq!(
                                *(*frame_ptr).data[0].add(y * y_line_size + x),
                                expected[[y, x, 0]],
                                "Y plane mismatch at y={}, x={}",
                                y,
                                x
                            );
                        }
                    }

                    // 验证 U/V 平面
                    let uv_height = height / 2;
                    let uv_width = width / 2;
                    for y in 0..uv_height {
                        for x in 0..uv_width {
                            assert_eq!(
                                *(*frame_ptr).data[1].add(y * u_line_size + x),
                                expected[[y * 2, x * 2, 1]],
                                "U plane mismatch at y={}, x={}",
                                y,
                                x
                            );
                            assert_eq!(
                                *(*frame_ptr).data[2].add(y * v_line_size + x),
                                expected[[y * 2, x * 2, 2]],
                                "V plane mismatch at y={}, x={}",
                                y,
                                x
                            );
                        }
                    }
                }
                _ => panic!("Unsupported format for verification: {:?}", format),
            }
        }
        Ok(())
    }

    #[test]
    fn test_rgb24_conversion() -> Result<()> {
        let height = 64;
        let width = 64;
        let channels = 3;

        // 创建测试数据
        let array = create_test_rgb_data(height, width, channels);

        // Array3 -> AVFrame
        let frame = AVFrame::try_from_cv(array.clone().with_format(PixelFormat::RGB24))?;

        // 验证转换结果
        verify_frame_data(&frame, &array, PixelFormat::RGB24)?;

        // AVFrame -> Array3
        let array_back: Array3<u8> = Array3::try_from_cv(&frame)?;
        assert_eq!(array, array_back);

        Ok(())
    }

    #[test]
    fn test_yuv420p_conversion() -> Result<()> {
        let height = 64;
        let width = 64;
        let channels = 3;

        // 创建测试数据
        let mut array = Array3::<u8>::zeros((height, width, channels));
        // 首先将 UV 分量初始化为 128
        for y in 0..height {
            for x in 0..width {
                array[[y, x, 1]] = 128; // U
                array[[y, x, 2]] = 128; // V
            }
        }

        // 然后设置渐变的 RGB 测试数据
        for y in 0..height {
            for x in 0..width {
                // 为了简化测试，我们使用易于验证的值
                let r = if x < width / 2 { 255 } else { 0 };
                let g = if y < height / 2 { 255 } else { 0 };
                let b = 128; // 固定中间值

                // 使用 rgb_to_yuv 计算对应的 YUV 值
                let (y_val, u_val, v_val) = with_ndarray::rgb_to_yuv(r as f64, g as f64, b as f64);

                // 设置对应的值
                array[[y, x, 0]] = y_val as u8; // Y
                array[[y, x, 1]] = u_val as u8; // U
                array[[y, x, 2]] = v_val as u8; // V
            }
        }

        // Array3 -> AVFrame
        let frame = AVFrame::try_from_cv(array.clone().with_format(PixelFormat::YUV420P))?;

        // 验证 YUV 值
        unsafe {
            let frame_ptr = frame.as_ptr();
            let y_line_size = (*frame_ptr).linesize[0] as usize;
            let u_line_size = (*frame_ptr).linesize[1] as usize;
            let v_line_size = (*frame_ptr).linesize[2] as usize;

            for by in 0..(height / 2) {
                for bx in 0..(width / 2) {
                    // 取 2x2 块的 YUV 值平均值
                    let mut sum_y = 0u32;
                    let mut sum_u = 0u32;
                    let mut sum_v = 0u32;

                    for dy in 0..2 {
                        for dx in 0..2 {
                            let y = by * 2 + dy;
                            let x = bx * 2 + dx;
                            sum_y += array[[y, x, 0]] as u32;
                            sum_u += array[[y, x, 1]] as u32;
                            sum_v += array[[y, x, 2]] as u32;
                        }
                    }

                    let avg_y = (sum_y / 4) as u8;
                    let avg_u = (sum_u / 4) as u8;
                    let avg_v = (sum_v / 4) as u8;

                    // 检查实际值
                    let actual_y = *(*frame_ptr).data[0].add(by * 2 * y_line_size + bx * 2);
                    let actual_u = *(*frame_ptr).data[1].add(by * u_line_size + bx);
                    let actual_v = *(*frame_ptr).data[2].add(by * v_line_size + bx);

                    // 允许小的误差范围
                    let y_diff = (avg_y as i16 - actual_y as i16).abs();
                    let u_diff = (avg_u as i16 - actual_u as i16).abs();
                    let v_diff = (avg_v as i16 - actual_v as i16).abs();

                    assert!(
                        y_diff <= 2,
                        "Y value difference too large at block ({}, {}): expected={}, actual={}",
                        bx,
                        by,
                        avg_y,
                        actual_y
                    );
                    assert!(
                        u_diff <= 2,
                        "U value difference too large at block ({}, {}): expected={}, actual={}",
                        bx,
                        by,
                        avg_u,
                        actual_u
                    );
                    assert!(
                        v_diff <= 2,
                        "V value difference too large at block ({}, {}): expected={}, actual={}",
                        bx,
                        by,
                        avg_v,
                        actual_v
                    );
                }
            }
        }

        // AVFrame -> Array3
        let array_back: Array3<u8> = Array3::try_from_cv(&frame)?;

        // 验证转换回 RGB 的结果
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let diff = (array[[y, x, c]] as i16 - array_back[[y, x, c]] as i16).abs();
                    assert!(
                        diff <= 1,
                        "Color difference too large at y={}, x={}, c={}: original={}, converted={}",
                        y,
                        x,
                        c,
                        array[[y, x, c]],
                        array_back[[y, x, c]]
                    );
                }
            }
        }

        Ok(())
    }

    #[allow(unused)]
    fn create_yuv_test_data(height: usize, width: usize, channels: usize) -> Array3<u8> {
        let mut array = Array3::<u8>::zeros((height, width, channels));
        for y in 0..height {
            for x in 0..width {
                // Y: 亮度渐变
                array[[y, x, 0]] = ((y * x * 255) / (height * width)) as u8;
                // U: 中等值
                array[[y, x, 1]] = 128;
                // V: 中等值
                array[[y, x, 2]] = 128;
            }
        }
        array
    }

    #[test]
    fn test_gray8_conversion() -> Result<()> {
        let height = 64;
        let width = 64;

        // 创建灰度图测试数据
        let mut array = Array3::zeros((height, width, 1));
        for y in 0..height {
            for x in 0..width {
                array[[y, x, 0]] = ((y + x) % 255) as u8;
            }
        }

        // Array3 -> AVFrame
        let frame = AVFrame::try_from_cv(array.clone().with_format(PixelFormat::GRAY8))?;

        // 验证数据
        unsafe {
            let frame_ptr = frame.as_ptr();
            let line_size = (*frame_ptr).linesize[0] as usize;
            let data_ptr = (*frame_ptr).data[0];

            for y in 0..height {
                for x in 0..width {
                    assert_eq!(
                        *data_ptr.add(y * line_size + x),
                        array[[y, x, 0]],
                        "Mismatch at position y={}, x={}",
                        y,
                        x
                    );
                }
            }
        }

        // AVFrame -> Array3
        let array_back: Array3<u8> = Array3::try_from_cv(&frame)?;
        assert_eq!(array, array_back);

        Ok(())
    }

    #[test]
    fn test_rgba_conversion() -> Result<()> {
        let height = 64;
        let width = 64;
        let channels = 4;

        // 创建测试数据
        let mut array = Array3::zeros((height, width, channels));
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    array[[y, x, c]] = ((y + x + c) % 255) as u8;
                }
            }
        }

        // Array3 -> AVFrame
        let frame = AVFrame::try_from_cv(array.clone().with_format(PixelFormat::RGBA))?;

        // 验证转换结果
        unsafe {
            let frame_ptr = frame.as_ptr();
            let line_size = (*frame_ptr).linesize[0] as usize;
            let data_ptr = (*frame_ptr).data[0];

            for y in 0..height {
                for x in 0..width {
                    for c in 0..channels {
                        assert_eq!(
                            *data_ptr.add(y * line_size + x * channels + c),
                            array[[y, x, c]],
                            "Mismatch at position y={}, x={}, c={}",
                            y,
                            x,
                            c
                        );
                    }
                }
            }
        }

        // AVFrame -> Array3
        let array_back: Array3<u8> = Array3::try_from_cv(&frame)?;
        assert_eq!(array, array_back);

        Ok(())
    }
}
