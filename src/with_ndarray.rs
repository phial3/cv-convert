use anyhow::{Error, Result};
use ndarray::{Array3, ArrayView3};
use num_traits::{NumCast, Zero};

/// RGB
pub const PIXEL_FORMAT_RGB4: &str = "RGB4";
pub const PIXEL_FORMAT_RGB8: &str = "RGB8";
pub const PIXEL_FORMAT_RGB24: &str = "RGB24";
pub const PIXEL_FORMAT_RGB32: &str = "RGB32";
/// BGR
pub const PIXEL_FORMAT_BGR4: &str = "BGR4";
pub const PIXEL_FORMAT_BGR8: &str = "BGR8";
pub const PIXEL_FORMAT_BGR24: &str = "BGR24";
pub const PIXEL_FORMAT_BGR32: &str = "BGR32";
/// RGBA
pub const PIXEL_FORMAT_RGBA: &str = "RGBA";
pub const PIXEL_FORMAT_RGBA64: &str = "RGBA64";
/// BGRA
pub const PIXEL_FORMAT_BGRA: &str = "BGRA";
pub const PIXEL_FORMAT_BGRA64: &str = "BGRA64";
/// YUV
/// YUV410P: U/V 平面是 Y 平面的 1/4 宽度和 1/4 高度
/// YUV411P: U/V 平面是 Y 平面的 1/4 宽度，相同高度
/// YUV420P: U/V 平面是 Y 平面的 1/2 宽度和 1/2 高度
/// YUV422P: U/V 平面是 Y 平面的 1/2 宽度，相同高度
/// YUV440P: U/V 平面是 Y 平面的相同宽度，1/2 高度
/// YUV444P: U/V 平面与 Y 平面相同大小
pub const PIXEL_FORMAT_YUV410P: &str = "YUV410P";
pub const PIXEL_FORMAT_YUV411P: &str = "YUV411P";
pub const PIXEL_FORMAT_YUV420P: &str = "YUV420P";
pub const PIXEL_FORMAT_YUV422P: &str = "YUV422P";
pub const PIXEL_FORMAT_YUV440P: &str = "YUV440P";
pub const PIXEL_FORMAT_YUV444P: &str = "YUV444P";
/// GRAY
pub const PIXEL_FORMAT_GRAY8: &str = "GRAY8";
pub const PIXEL_FORMAT_GRAY9: &str = "GRAY9";
pub const PIXEL_FORMAT_GRAY10: &str = "GRAY10";
pub const PIXEL_FORMAT_GRAY12: &str = "GRAY12";
pub const PIXEL_FORMAT_GRAY16: &str = "GRAY16";

/// 有效的格式列表
pub const VALID_FORMATS: [&str; 23] = [
    PIXEL_FORMAT_RGB4,
    PIXEL_FORMAT_RGB8,
    PIXEL_FORMAT_RGB24,
    PIXEL_FORMAT_RGB32,
    PIXEL_FORMAT_BGR4,
    PIXEL_FORMAT_BGR8,
    PIXEL_FORMAT_BGR24,
    PIXEL_FORMAT_BGR32,
    PIXEL_FORMAT_RGBA,
    PIXEL_FORMAT_RGBA64,
    PIXEL_FORMAT_BGRA,
    PIXEL_FORMAT_BGRA64,
    PIXEL_FORMAT_GRAY8,
    PIXEL_FORMAT_GRAY9,
    PIXEL_FORMAT_GRAY10,
    PIXEL_FORMAT_GRAY12,
    PIXEL_FORMAT_GRAY16,
    PIXEL_FORMAT_YUV410P,
    PIXEL_FORMAT_YUV411P,
    PIXEL_FORMAT_YUV420P,
    PIXEL_FORMAT_YUV422P,
    PIXEL_FORMAT_YUV440P,
    PIXEL_FORMAT_YUV444P,
];

/// ITU-R BT.601-7: <https://www.itu.int/rec/R-REC-BT.601>
/// For conventional RGB with range [0, 255] to YUV conversion:
/// Y = 0.299R + 0.587G + 0.114B          // Luma
/// U = -0.169R - 0.331G + 0.500B + 128   // Cb (Blue difference)
/// V = 0.500R - 0.419G - 0.081B + 128    // Cr (Red difference)
///
/// For YUV to RGB conversion:
/// R = Y + 1.4021U                       // Red
/// G = Y - 0.3441U - 0.7142V             // Green
/// B = Y + 1.7718U                       // Blue
///
/// YUV 到 RGB 的转换系数
pub const YUV2RGB_MAT: [[f64; 3]; 3] = [
    [1.0, 0.0, 1.4021],      // R
    [1.0, -0.3441, -0.7142], // G
    [1.0, 1.7718, 0.0],      // B
];

/// RGB 到 YUV 的转换系数
pub const RGB2YUV_MAT: [[f64; 3]; 3] = [
    [0.299, 0.587, 0.114], // Y
    [-0.169, -0.331, 0.5], // U
    [0.5, -0.419, -0.081], // V
];

/// Convert between different pixel formats
pub fn convert_pixel_format<T, U>(
    src: &Array3<T>,
    src_format: &str,
    dst_format: &str,
) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 转换为小写并移除空白字符
    let src_fmt = src_format.trim().to_uppercase();
    let dst_fmt = dst_format.trim().to_uppercase();

    if !VALID_FORMATS.contains(&src_fmt.as_str()) {
        return Err(Error::msg(format!(
            "Unsupported source format: {}",
            src_fmt
        )));
    }
    if !VALID_FORMATS.contains(&dst_fmt.as_str()) {
        return Err(Error::msg(format!(
            "Unsupported destination format: {}",
            dst_fmt
        )));
    }

    // 检查通道数是否匹配
    let expected_channels = match src_fmt.as_str() {
        // GRAY
        PIXEL_FORMAT_GRAY8 | PIXEL_FORMAT_GRAY9 | PIXEL_FORMAT_GRAY10 | PIXEL_FORMAT_GRAY12
        | PIXEL_FORMAT_GRAY16 => 1,
        // RGB/BGR
        PIXEL_FORMAT_RGB4 | PIXEL_FORMAT_RGB8 | PIXEL_FORMAT_RGB24 | PIXEL_FORMAT_RGB32
        | PIXEL_FORMAT_BGR4 | PIXEL_FORMAT_BGR8 | PIXEL_FORMAT_BGR24 | PIXEL_FORMAT_BGR32 => 3,
        // YUV
        PIXEL_FORMAT_YUV410P | PIXEL_FORMAT_YUV411P | PIXEL_FORMAT_YUV420P
        | PIXEL_FORMAT_YUV422P | PIXEL_FORMAT_YUV440P | PIXEL_FORMAT_YUV444P => 3,
        // RGBA/BGRA
        PIXEL_FORMAT_RGBA | PIXEL_FORMAT_RGBA64 | PIXEL_FORMAT_BGRA | PIXEL_FORMAT_BGRA64 => 4,
        _ => {
            return Err(Error::msg(format!(
                "Unsupported source format: {}",
                src_fmt
            )))
        }
    };

    let (_, _, channels) = src.dim();
    if channels != expected_channels {
        return Err(Error::msg(format!(
            "Source format {} expects {} channels, but got {}",
            src_fmt, expected_channels, channels
        )));
    }

    // 如果源格式和目标格式相同，直接复制数据
    if src_fmt == dst_fmt {
        return Ok(Array3::from_shape_fn(src.raw_dim(), |idx| {
            NumCast::from(src[idx].clone().to_f64().unwrap()).unwrap()
        }));
    }

    // 执行转换
    match (src_fmt.as_str(), dst_fmt.as_str()) {
        // RGB to BGR conversions
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_BGR8) => rgb8_to_bgr8(src),
        (PIXEL_FORMAT_BGR8, PIXEL_FORMAT_RGB8) => bgr8_to_rgb8(src),
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_BGR24) => rgb24_to_bgr24(src),
        (PIXEL_FORMAT_BGR24, PIXEL_FORMAT_RGB24) => bgr24_to_rgb24(src),
        (PIXEL_FORMAT_RGB32, PIXEL_FORMAT_BGR32) => rgb32_to_bgr32(src),
        (PIXEL_FORMAT_BGR32, PIXEL_FORMAT_RGB32) => bgr32_to_rgb32(src),

        // RGB/BGR to RGBA/BGRA conversions
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_RGBA) => rgb8_to_rgba(src),
        (PIXEL_FORMAT_BGR8, PIXEL_FORMAT_BGRA) => bgr8_to_bgra(src),
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_RGBA) => rgb24_to_rgba(src),
        (PIXEL_FORMAT_BGR24, PIXEL_FORMAT_BGRA) => bgr24_to_bgra(src),

        // RGBA/BGRA to RGB/BGR conversions
        (PIXEL_FORMAT_RGBA, PIXEL_FORMAT_RGB8) => rgba_to_rgb8(src),
        (PIXEL_FORMAT_BGRA, PIXEL_FORMAT_BGR8) => bgra_to_bgr8(src),
        (PIXEL_FORMAT_RGBA, PIXEL_FORMAT_RGB24) => rgba_to_rgb24(src),
        (PIXEL_FORMAT_BGRA, PIXEL_FORMAT_BGR24) => bgra_to_bgr24(src),

        // Gray to RGB/BGR conversions
        (PIXEL_FORMAT_GRAY8, PIXEL_FORMAT_RGB8) => gray8_to_rgb8(src),
        (PIXEL_FORMAT_GRAY8, PIXEL_FORMAT_BGR8) => gray8_to_bgr8(src),
        (PIXEL_FORMAT_GRAY16, PIXEL_FORMAT_RGB8) => gray16_to_rgb8(src),
        (PIXEL_FORMAT_GRAY16, PIXEL_FORMAT_BGR8) => gray16_to_bgr8(src),

        // GRAY to GRAY
        (PIXEL_FORMAT_GRAY8, PIXEL_FORMAT_GRAY16) => gray8_to_gray16(src),
        (PIXEL_FORMAT_GRAY16, PIXEL_FORMAT_GRAY8) => gray16_to_gray8(src),

        // YUV to RGB8 conversions
        (PIXEL_FORMAT_YUV410P, PIXEL_FORMAT_RGB8) => yuv_to_rgb(src, PIXEL_FORMAT_YUV410P),
        (PIXEL_FORMAT_YUV411P, PIXEL_FORMAT_RGB8) => yuv_to_rgb(src, PIXEL_FORMAT_YUV411P),
        (PIXEL_FORMAT_YUV420P, PIXEL_FORMAT_RGB8) => yuv_to_rgb(src, PIXEL_FORMAT_YUV420P),
        (PIXEL_FORMAT_YUV422P, PIXEL_FORMAT_RGB8) => yuv_to_rgb(src, PIXEL_FORMAT_YUV422P),
        (PIXEL_FORMAT_YUV440P, PIXEL_FORMAT_RGB8) => yuv_to_rgb(src, PIXEL_FORMAT_YUV440P),
        (PIXEL_FORMAT_YUV444P, PIXEL_FORMAT_RGB8) => yuv_to_rgb(src, PIXEL_FORMAT_YUV444P),

        // YUV to RGB24 conversions
        (PIXEL_FORMAT_YUV410P, PIXEL_FORMAT_RGB24) => yuv_to_rgb(src, PIXEL_FORMAT_YUV410P),
        (PIXEL_FORMAT_YUV411P, PIXEL_FORMAT_RGB24) => yuv_to_rgb(src, PIXEL_FORMAT_YUV411P),
        (PIXEL_FORMAT_YUV420P, PIXEL_FORMAT_RGB24) => yuv_to_rgb(src, PIXEL_FORMAT_YUV420P),
        (PIXEL_FORMAT_YUV422P, PIXEL_FORMAT_RGB24) => yuv_to_rgb(src, PIXEL_FORMAT_YUV422P),
        (PIXEL_FORMAT_YUV440P, PIXEL_FORMAT_RGB24) => yuv_to_rgb(src, PIXEL_FORMAT_YUV440P),
        (PIXEL_FORMAT_YUV444P, PIXEL_FORMAT_RGB24) => yuv_to_rgb(src, PIXEL_FORMAT_YUV444P),

        // RGB8 to YUV conversions
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV410P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV410P),
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV411P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV411P),
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV420P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV420P),
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV422P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV422P),
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV440P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV440P),
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV444P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV444P),

        // RGB24 to YUV conversions
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV410P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV410P),
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV411P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV411P),
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV420P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV420P),
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV422P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV422P),
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV440P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV440P),
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV444P) => rgb_to_yuv(src, PIXEL_FORMAT_YUV444P),

        _ => Err(Error::msg(format!(
            "Unsupported conversion path: {} to {}",
            src_format, dst_format
        ))),
    }
}

/// @author: phial3
fn rgb8_to_bgr8<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 3));

    for h in 0..height {
        for w in 0..width {
            // RGB -> BGR: swap R and B channels
            dst[[h, w, 0]] = NumCast::from(src[[h, w, 2]].to_f64().unwrap()).unwrap(); // B <- R
            dst[[h, w, 1]] = NumCast::from(src[[h, w, 1]].to_f64().unwrap()).unwrap(); // G <- G
            dst[[h, w, 2]] = NumCast::from(src[[h, w, 0]].to_f64().unwrap()).unwrap();
            // R <- B
        }
    }
    Ok(dst)
}

/// BGR8 to RGB8 conversion
fn bgr8_to_rgb8<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 3));

    for h in 0..height {
        for w in 0..width {
            // BGR -> RGB: swap B and R channels
            dst[[h, w, 0]] = NumCast::from(src[[h, w, 2]].to_f64().unwrap()).unwrap(); // R <- B
            dst[[h, w, 1]] = NumCast::from(src[[h, w, 1]].to_f64().unwrap()).unwrap(); // G <- G
            dst[[h, w, 2]] = NumCast::from(src[[h, w, 0]].to_f64().unwrap()).unwrap();
            // B <- R
        }
    }
    Ok(dst)
}

/// RGB24 to BGR24 conversion
fn rgb24_to_bgr24<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 与 RGB8 to BGR8 相同的实现，因为都是 3 通道，只是数据类型可能不同
    rgb8_to_bgr8(src)
}

/// BGR24 to RGB24 conversion
fn bgr24_to_rgb24<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 与 BGR8 to RGB8 相同的实现
    bgr8_to_rgb8(src)
}

/// RGB32 to BGR32 conversion
fn rgb32_to_bgr32<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 与 RGB8/24 to BGR8/24 相同的实现
    rgb8_to_bgr8(src)
}

/// BGR32 to RGB32 conversion
fn bgr32_to_rgb32<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 与 BGR8/24 to RGB8/24 相同的实现
    bgr8_to_rgb8(src)
}

/// RGB8 to RGBA conversion
fn rgb8_to_rgba<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 4));
    let alpha: U = NumCast::from(255).unwrap();

    for h in 0..height {
        for w in 0..width {
            dst[[h, w, 0]] = NumCast::from(src[[h, w, 0]].to_f64().unwrap()).unwrap(); // R
            dst[[h, w, 1]] = NumCast::from(src[[h, w, 1]].to_f64().unwrap()).unwrap(); // G
            dst[[h, w, 2]] = NumCast::from(src[[h, w, 2]].to_f64().unwrap()).unwrap(); // B
            dst[[h, w, 3]] = alpha.clone(); // A (255)
        }
    }
    Ok(dst)
}

/// BGR8 to BGRA conversion
fn bgr8_to_bgra<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 4));
    let alpha: U = NumCast::from(255).unwrap();

    for h in 0..height {
        for w in 0..width {
            dst[[h, w, 0]] = NumCast::from(src[[h, w, 0]].to_f64().unwrap()).unwrap(); // B
            dst[[h, w, 1]] = NumCast::from(src[[h, w, 1]].to_f64().unwrap()).unwrap(); // G
            dst[[h, w, 2]] = NumCast::from(src[[h, w, 2]].to_f64().unwrap()).unwrap(); // R
            dst[[h, w, 3]] = alpha.clone(); // A (255)
        }
    }
    Ok(dst)
}

/// RGB24 to RGBA conversion
fn rgb24_to_rgba<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 与 RGB8 to RGBA 相同的实现
    rgb8_to_rgba(src)
}

/// BGR24 to BGRA conversion
fn bgr24_to_bgra<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 与 BGR8 to BGRA 相同的实现
    bgr8_to_bgra(src)
}

/// RGBA to RGB8 conversion
fn rgba_to_rgb8<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 3));

    for h in 0..height {
        for w in 0..width {
            dst[[h, w, 0]] = NumCast::from(src[[h, w, 0]].to_f64().unwrap()).unwrap(); // R
            dst[[h, w, 1]] = NumCast::from(src[[h, w, 1]].to_f64().unwrap()).unwrap(); // G
            dst[[h, w, 2]] = NumCast::from(src[[h, w, 2]].to_f64().unwrap()).unwrap();
            // B
        }
    }
    Ok(dst)
}

/// BGRA to BGR8 conversion
fn bgra_to_bgr8<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 3));

    for h in 0..height {
        for w in 0..width {
            dst[[h, w, 0]] = NumCast::from(src[[h, w, 0]].to_f64().unwrap()).unwrap(); // B
            dst[[h, w, 1]] = NumCast::from(src[[h, w, 1]].to_f64().unwrap()).unwrap(); // G
            dst[[h, w, 2]] = NumCast::from(src[[h, w, 2]].to_f64().unwrap()).unwrap();
            // R
        }
    }
    Ok(dst)
}

/// RGBA to RGB24 conversion
fn rgba_to_rgb24<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 与 RGBA to RGB8 相同的实现
    rgba_to_rgb8(src)
}

/// BGRA to BGR24 conversion
fn bgra_to_bgr24<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 与 BGRA to BGR8 相同的实现
    bgra_to_bgr8(src)
}

/// GRAY8 to RGB8 conversion
fn gray8_to_rgb8<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 3));

    for h in 0..height {
        for w in 0..width {
            let val = src[[h, w, 0]].to_f64().unwrap();
            let rgb: U = NumCast::from(val).unwrap();
            dst[[h, w, 0]] = rgb.clone(); // R
            dst[[h, w, 1]] = rgb.clone(); // G
            dst[[h, w, 2]] = rgb; // B
        }
    }
    Ok(dst)
}

/// GRAY8 to BGR8 conversion
fn gray8_to_bgr8<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 灰度图转 BGR 与转 RGB 实现相同，因为所有通道值相等
    gray8_to_rgb8(src)
}

/// GRAY16 to RGB8 conversion
fn gray16_to_rgb8<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 3));

    for h in 0..height {
        for w in 0..width {
            // 将 16 位值转换为 8 位值
            let val = src[[h, w, 0]].to_f64().unwrap();
            let scaled = (val / 256.0).round(); // 16位到8位的转换
            let rgb: U = NumCast::from(scaled).unwrap();
            dst[[h, w, 0]] = rgb.clone(); // R
            dst[[h, w, 1]] = rgb.clone(); // G
            dst[[h, w, 2]] = rgb; // B
        }
    }
    Ok(dst)
}

/// GRAY16 to BGR8 conversion
fn gray16_to_bgr8<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    // 与 GRAY16 to RGB8 相同的实现
    gray16_to_rgb8(src)
}

/// GRAY8 to GRAY16 conversion
fn gray8_to_gray16<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 1));

    for h in 0..height {
        for w in 0..width {
            let val = src[[h, w, 0]].to_f64().unwrap();
            let scaled = val * 256.0; // 8位到16位的转换
            dst[[h, w, 0]] = NumCast::from(scaled).unwrap();
        }
    }
    Ok(dst)
}

/// GRAY16 to GRAY8 conversion
fn gray16_to_gray8<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 1));

    for h in 0..height {
        for w in 0..width {
            let val = src[[h, w, 0]].to_f64().unwrap();
            let scaled = (val / 256.0).round(); // 16位到8位的转换
            dst[[h, w, 0]] = NumCast::from(scaled).unwrap();
        }
    }
    Ok(dst)
}

/// Convert YUV planar formats to RGB (supports both RGB8 and RGB24)
fn yuv_to_rgb<T, U>(src: &Array3<T>, src_format: &str) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 3));

    // Get UV sampling ratios for different YUV formats
    let (uv_width_ratio, uv_height_ratio) = match src_format {
        PIXEL_FORMAT_YUV410P => (4, 4),
        PIXEL_FORMAT_YUV411P => (4, 1),
        PIXEL_FORMAT_YUV420P => (2, 2),
        PIXEL_FORMAT_YUV422P => (2, 1),
        PIXEL_FORMAT_YUV440P => (1, 2),
        PIXEL_FORMAT_YUV444P => (1, 1),
        _ => {
            return Err(Error::msg(format!(
                "Unsupported YUV format: {}",
                src_format
            )))
        }
    };

    for y in 0..height {
        for x in 0..width {
            // Get YUV values with proper subsampling
            let y_val = src[[y, x, 0]].to_f64().unwrap() - 16.0;
            let u_val = src[[y / uv_height_ratio, x / uv_width_ratio, 1]]
                .to_f64()
                .unwrap()
                - 128.0;
            let v_val = src[[y / uv_height_ratio, x / uv_width_ratio, 2]]
                .to_f64()
                .unwrap()
                - 128.0;

            // YUV to RGB conversion
            let r =
                (YUV2RGB_MAT[0][0] * y_val + YUV2RGB_MAT[0][1] * u_val + YUV2RGB_MAT[0][2] * v_val)
                    .round()
                    .clamp(0.0, 255.0);
            let g =
                (YUV2RGB_MAT[1][0] * y_val + YUV2RGB_MAT[1][1] * u_val + YUV2RGB_MAT[1][2] * v_val)
                    .round()
                    .clamp(0.0, 255.0);
            let b =
                (YUV2RGB_MAT[2][0] * y_val + YUV2RGB_MAT[2][1] * u_val + YUV2RGB_MAT[2][2] * v_val)
                    .round()
                    .clamp(0.0, 255.0);

            // Store RGB values
            dst[[y, x, 0]] = NumCast::from(r).unwrap();
            dst[[y, x, 1]] = NumCast::from(g).unwrap();
            dst[[y, x, 2]] = NumCast::from(b).unwrap();
        }
    }

    Ok(dst)
}

/// Convert RGB (RGB8 or RGB24) to YUV planar format
fn rgb_to_yuv<T, U>(src: &Array3<T>, dst_format: &str) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _) = src.dim();

    // Get UV plane dimensions based on format
    let (uv_width_ratio, uv_height_ratio) = match dst_format {
        PIXEL_FORMAT_YUV410P => (4, 4),
        PIXEL_FORMAT_YUV411P => (4, 1),
        PIXEL_FORMAT_YUV420P => (2, 2),
        PIXEL_FORMAT_YUV422P => (2, 1),
        PIXEL_FORMAT_YUV440P => (1, 2),
        PIXEL_FORMAT_YUV444P => (1, 1),
        _ => {
            return Err(Error::msg(format!(
                "Unsupported YUV format: {}",
                dst_format
            )))
        }
    };

    let uv_height = height.div_ceil(uv_height_ratio);
    let _uv_width = width.div_ceil(uv_width_ratio);
    let mut dst = Array3::<U>::zeros((height, uv_height, 3));

    // RGB to YUV conversion with subsampling
    for y in 0..height {
        for x in 0..width {
            let r = src[[y, x, 0]].to_f64().unwrap();
            let g = src[[y, x, 1]].to_f64().unwrap();
            let b = src[[y, x, 2]].to_f64().unwrap();

            // Calculate Y (luminance)
            let y_val = (RGB2YUV_MAT[0][0] * r + RGB2YUV_MAT[0][1] * g + RGB2YUV_MAT[0][2] * b)
                .round()
                .clamp(0.0, 255.0)
                + 16.0;

            dst[[y, x, 0]] = NumCast::from(y_val).unwrap();

            // Calculate and subsample U and V (chrominance)
            if y % uv_height_ratio == 0 && x % uv_width_ratio == 0 {
                let mut u_sum = 0.0;
                let mut v_sum = 0.0;
                let mut count = 0;

                // Average RGB values over the sampling block
                for sy in 0..uv_height_ratio {
                    for sx in 0..uv_width_ratio {
                        if y + sy < height && x + sx < width {
                            let sr = src[[y + sy, x + sx, 0]].to_f64().unwrap();
                            let sg = src[[y + sy, x + sx, 1]].to_f64().unwrap();
                            let sb = src[[y + sy, x + sx, 2]].to_f64().unwrap();

                            // Calculate U and V
                            u_sum += RGB2YUV_MAT[1][0] * sr
                                + RGB2YUV_MAT[1][1] * sg
                                + RGB2YUV_MAT[1][2] * sb;
                            v_sum += RGB2YUV_MAT[2][0] * sr
                                + RGB2YUV_MAT[2][1] * sg
                                + RGB2YUV_MAT[2][2] * sb;
                            count += 1;
                        }
                    }
                }

                // Store average U and V values
                let u_val = (u_sum / count as f64).round().clamp(-128.0, 127.0) + 128.0;
                let v_val = (v_sum / count as f64).round().clamp(-128.0, 127.0) + 128.0;

                let uv_y = y / uv_height_ratio;
                let uv_x = x / uv_width_ratio;

                dst[[uv_y, uv_x, 1]] = NumCast::from(u_val).unwrap();
                dst[[uv_y, uv_x, 2]] = NumCast::from(v_val).unwrap();
            }
        }
    }

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use std::time::Instant;

    /// Helper function to create a test image with specified format
    fn create_test_image<T>(
        height: usize,
        width: usize,
        channels: usize,
        pattern: &str,
    ) -> Array3<T>
    where
        T: Clone + NumCast + Zero,
    {
        let mut img = Array3::<T>::zeros((height, width, channels));

        match pattern {
            "gradient" => {
                for h in 0..height {
                    for w in 0..width {
                        for c in 0..channels {
                            let val = ((h + w) % 256) as f64;
                            img[[h, w, c]] = NumCast::from(val).unwrap();
                        }
                    }
                }
            }
            "checkerboard" => {
                for h in 0..height {
                    for w in 0..width {
                        for c in 0..channels {
                            let val = if (h / 8 + w / 8) % 2 == 0 { 255.0 } else { 0.0 };
                            img[[h, w, c]] = NumCast::from(val).unwrap();
                        }
                    }
                }
            }
            _ => {}
        }

        img
    }

    /// Test RGB8/RGB24/BGR8/BGR24 format conversions
    #[test]
    fn test_rgb_bgr_conversions() {
        let start = Instant::now();

        let test_size = (32, 32);
        let rgb8_img: Array3<u8> = create_test_image(test_size.0, test_size.1, 3, "gradient");

        // Test RGB8 <-> BGR8
        let result =
            convert_pixel_format::<u8, u8>(&rgb8_img, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_BGR8)
                .unwrap();

        assert_eq!(result.dim(), (test_size.0, test_size.1, 3));

        // Convert back
        let back_to_rgb =
            convert_pixel_format::<u8, u8>(&result, PIXEL_FORMAT_BGR8, PIXEL_FORMAT_RGB8).unwrap();

        assert_eq!(rgb8_img, back_to_rgb);

        println!(
            "RGB/BGR conversion test completed in: {}ms",
            start.elapsed().as_millis()
        );
    }

    /// Test RGBA/BGRA format conversions
    #[test]
    fn test_rgba_bgra_conversions() {
        let start = Instant::now();

        let test_size = (32, 32);
        let rgba_img: Array3<u8> = create_test_image(test_size.0, test_size.1, 4, "gradient");

        // Test RGBA -> RGB8 -> RGBA conversions
        let rgb8 = convert_pixel_format::<u8, u8>(&rgba_img, PIXEL_FORMAT_RGBA, PIXEL_FORMAT_RGB8)
            .unwrap();

        // 验证 RGB8 转换结果的维度和RGB通道
        assert_eq!(rgb8.dim(), (test_size.0, test_size.1, 3));
        for h in 0..test_size.0 {
            for w in 0..test_size.1 {
                for c in 0..3 {
                    assert_eq!(rgba_img[[h, w, c]], rgb8[[h, w, c]]);
                }
            }
        }

        let back_to_rgba =
            convert_pixel_format::<u8, u8>(&rgb8, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_RGBA).unwrap();

        // 验证 RGBA 转换结果的维度
        assert_eq!(back_to_rgba.dim(), (test_size.0, test_size.1, 4));

        // 仅比较 RGB 通道，不比较 alpha 通道
        // 因为 RGB8 -> RGBA 转换会将 alpha 设为 255
        for h in 0..test_size.0 {
            for w in 0..test_size.1 {
                // 验证 RGB 通道
                for c in 0..3 {
                    assert_eq!(
                        rgba_img[[h, w, c]],
                        back_to_rgba[[h, w, c]],
                        "Mismatch at position [{}, {}, {}]",
                        h,
                        w,
                        c
                    );
                }
                // 验证 alpha 通道总是 255
                assert_eq!(
                    back_to_rgba[[h, w, 3]],
                    255,
                    "Alpha channel should be 255 at position [{}, {}]",
                    h,
                    w
                );
            }
        }

        println!(
            "RGBA/BGRA conversion test completed in: {}ms",
            start.elapsed().as_millis()
        );
    }

    /// Test Gray format conversions
    #[test]
    fn test_gray_conversions() {
        let start = Instant::now();

        let test_size = (32, 32);
        let gray8_img: Array3<u8> = create_test_image(test_size.0, test_size.1, 1, "gradient");

        // Test GRAY8 -> GRAY16
        let result =
            convert_pixel_format::<u8, u16>(&gray8_img, PIXEL_FORMAT_GRAY8, PIXEL_FORMAT_GRAY16)
                .unwrap();

        assert_eq!(result.dim(), (test_size.0, test_size.1, 1));

        // Convert back
        let back_to_gray8 =
            convert_pixel_format::<u16, u8>(&result, PIXEL_FORMAT_GRAY16, PIXEL_FORMAT_GRAY8)
                .unwrap();

        // Allow small differences due to conversion
        for h in 0..test_size.0 {
            for w in 0..test_size.1 {
                let diff = (gray8_img[[h, w, 0]] as i16 - back_to_gray8[[h, w, 0]] as i16).abs();
                assert!(diff <= 1);
            }
        }

        println!(
            "Gray conversion test completed in: {}ms",
            start.elapsed().as_millis()
        );
    }

    /// Test YUV format conversions
    #[test]
    fn test_yuv_conversions() {
        let start = Instant::now();

        let test_size = (32, 32);
        let rgb8_img: Array3<u8> = create_test_image(test_size.0, test_size.1, 3, "gradient");

        // Test one YUV format at a time
        let yuv_format = PIXEL_FORMAT_YUV444P; // Start with simplest format

        // Convert RGB8 -> YUV444P
        let yuv_result =
            convert_pixel_format::<u8, u8>(&rgb8_img, PIXEL_FORMAT_RGB8, yuv_format).unwrap();

        // Convert back YUV444P -> RGB8
        let back_to_rgb =
            convert_pixel_format::<u8, u8>(&yuv_result, yuv_format, PIXEL_FORMAT_RGB8).unwrap();

        // Check dimensions
        assert_eq!(back_to_rgb.dim(), rgb8_img.dim());

        // Allow small differences due to YUV conversion
        for h in 0..test_size.0 {
            for w in 0..test_size.1 {
                for c in 0..3 {
                    let diff = (rgb8_img[[h, w, c]] as i16 - back_to_rgb[[h, w, c]] as i16).abs();
                    assert!(diff <= 5);
                }
            }
        }

        println!(
            "YUV conversion test completed in: {}ms",
            start.elapsed().as_millis()
        );
    }

    /// Test invalid format handling
    #[test]
    fn test_invalid_formats() {
        let start = Instant::now();

        let test_size = (32, 32);
        let img: Array3<u8> = create_test_image(test_size.0, test_size.1, 3, "gradient");

        // Test invalid source format
        let result = convert_pixel_format::<u8, u8>(&img, "INVALID_FORMAT", PIXEL_FORMAT_RGB8);
        assert!(result.is_err());

        // Test invalid destination format
        let result = convert_pixel_format::<u8, u8>(&img, PIXEL_FORMAT_RGB8, "INVALID_FORMAT");
        assert!(result.is_err());

        // Test invalid channel count
        let invalid_img: Array3<u8> = Array3::zeros((32, 32, 5)); // 5 channels
        let result =
            convert_pixel_format::<u8, u8>(&invalid_img, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_BGR8);
        assert!(result.is_err());

        println!(
            "Invalid format test completed in: {}ms",
            start.elapsed().as_millis()
        );
    }

    /// Test type conversion
    #[test]
    fn test_type_conversions() {
        let start = Instant::now();

        let test_size = (32, 32);
        let img_u8: Array3<u8> = create_test_image(test_size.0, test_size.1, 3, "gradient");

        // Test RGB8 -> BGR8 -> RGB8 with type conversion
        let result_u16 =
            convert_pixel_format::<u8, u16>(&img_u8, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_BGR8).unwrap();

        let back_to_u8 =
            convert_pixel_format::<u16, u8>(&result_u16, PIXEL_FORMAT_BGR8, PIXEL_FORMAT_RGB8)
                .unwrap();

        assert_eq!(img_u8, back_to_u8);

        println!(
            "Type conversion test completed in: {}ms",
            start.elapsed().as_millis()
        );
    }

    /// Test performance with larger images
    #[test]
    fn test_performance() {
        let start = Instant::now();

        let test_size = (1920, 1080); // Full HD size
        let img: Array3<u8> = create_test_image(test_size.0, test_size.1, 3, "checkerboard");

        // Test RGB -> BGR -> RGB conversion instead
        let bgr_result =
            convert_pixel_format::<u8, u8>(&img, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_BGR8).unwrap();

        let rgb_result =
            convert_pixel_format::<u8, u8>(&bgr_result, PIXEL_FORMAT_BGR8, PIXEL_FORMAT_RGB8)
                .unwrap();

        assert_eq!(rgb_result.dim(), img.dim());

        println!(
            "Performance test (1920x1080) completed in: {}ms",
            start.elapsed().as_millis()
        );
    }
}
