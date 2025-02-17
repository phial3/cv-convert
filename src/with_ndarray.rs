use anyhow::{Error, Result};
use ndarray::Array3;
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
/// GRAY
pub const PIXEL_FORMAT_GRAY8: &str = "GRAY8";
pub const PIXEL_FORMAT_GRAY9: &str = "GRAY9";
pub const PIXEL_FORMAT_GRAY10: &str = "GRAY10";
pub const PIXEL_FORMAT_GRAY12: &str = "GRAY12";
pub const PIXEL_FORMAT_GRAY16: &str = "GRAY16";
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

/// BT.709-6: <https://www.itu.int/rec/R-REC-BT.709>
/// BT.601-7: <https://www.itu.int/rec/R-REC-BT.601>
/// YUV to RGB conversion (BT.601):
///
/// `R = Y + 1.4021*V`
/// `G = Y - 0.3441*U - 0.7142*V`
/// `B = Y + 1.7718*U`
///
/// Input: YUV values in range [0, 255]
/// Output: RGB values in range [0, 255]
#[inline]
#[allow(unused)]
pub fn yuv_to_rgb(y: f64, u: f64, v: f64) -> (f64, f64, f64) {
    let y = y - 16.0;
    let u = u - 128.0;
    let v = v - 128.0;

    let r = y + 1.4021 * v;
    let g = y - 0.3441 * u - 0.7142 * v;
    let b = y + 1.7718 * u;
    (
        r.round().clamp(0.0, 255.0),
        g.round().clamp(0.0, 255.0),
        b.round().clamp(0.0, 255.0),
    )
}

/// BT.709-6: <https://www.itu.int/rec/R-REC-BT.709>
/// BT.601-7: <https://www.itu.int/rec/R-REC-BT.601>
/// RGB to YUV conversion (BT.601):
///
/// `Y = 0.299*R  + 0.587*G + 0.114*B`
/// `U = -0.169*R - 0.331*G + 0.500*B + 128`
/// `V = 0.500*R  - 0.419*G - 0.081*B + 128`
///
/// Input: RGB values in range [0, 255]
/// Output: YUV values in range [0, 255]
#[inline]
pub fn rgb_to_yuv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let u = -0.169 * r - 0.331 * g + 0.500 * b;
    let v = 0.500 * r - 0.419 * g - 0.081 * b;
    (
        y.round().clamp(0.0, 255.0) + 16.0,
        u.round().clamp(0.0, 255.0) + 128.0,
        v.round().clamp(0.0, 255.0) + 128.0,
    )
}

/// RGB to Grayscale conversion (BT.601):
/// Gray = 0.299R + 0.587G + 0.114B
///
/// The coefficients are based on human perception:
/// - Green light contributes the most to intensity perception
/// - Red contributes the second most
/// - Blue contributes the least
///
/// Note: Input RGB range [0, 255]
///       Output Grayscale range [0, 255]
#[inline]
#[allow(unused)]
pub fn rgb_to_gray(r: f64, g: f64, b: f64) -> f64 {
    ((0.299 * r) + (0.587 * g) + (0.114 * b))
        .round()
        .clamp(0.0, 255.0)
}

/// RGB to HSV conversion
/// H = Hue [0, 360), S = Saturation [0, 1], V = Value [0, 1]
///
/// Formula:
/// V = max(R, G, B)
/// S = (V - min(R, G, B)) / V
/// H = {
///   undefined,                  if V = 0
///   60° × (G-B)/(V-min(R,G,B)), if V = R
///   60° × (2 + (B-R)/(V-min)),  if V = G
///   60° × (4 + (R-G)/(V-min)),  if V = B
/// }
#[inline]
#[allow(unused)]
pub fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let r = r / 255.0;
    let g = g / 255.0;
    let b = b / 255.0;

    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let delta = max - min;

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };
    let s = if max == 0.0 { 0.0 } else { delta / max };
    let v = max;

    (h, s, v)
}

/// HSV to RGB conversion
/// Input: H [0, 360), S [0, 1], V [0, 1]
/// Output: RGB [0, 255]
///
/// Formula:
/// C = V × S
/// X = C × (1 - |(H/60°) mod 2 - 1|)
/// m = V - C
///
/// (R,G,B) = {
///   (C,X,0) + m,   if 0° ≤ H < 60°
///   (X,C,0) + m,   if 60° ≤ H < 120°
///   (0,C,X) + m,   if 120° ≤ H < 180°
///   (0,X,C) + m,   if 180° ≤ H < 240°
///   (X,0,C) + m,   if 240° ≤ H < 300°
///   (C,0,X) + m,   if 300° ≤ H < 360°
/// }
#[inline]
#[allow(unused)]
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let c = v * s;
    let h = h / 60.0;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (
        ((r + m) * 255.0).clamp(0.0, 255.0),
        ((g + m) * 255.0).clamp(0.0, 255.0),
        ((b + m) * 255.0).clamp(0.0, 255.0),
    )
}

/// RGB to XYZ conversion (CIE 1931)
/// Using D65 white point
///
/// Formula:
/// |X|   |0.4124 0.3576 0.1805| |R|
/// |Y| = |0.2126 0.7152 0.0722| |G|
/// |Z|   |0.0193 0.1192 0.9505| |B|
#[inline]
#[allow(unused)]
pub fn rgb_to_xyz(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let r = r / 255.0;
    let g = g / 255.0;
    let b = b / 255.0;

    // Linear RGB to XYZ (D65)
    let x = 0.4124 * r + 0.3576 * g + 0.1805 * b;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = 0.0193 * r + 0.1192 * g + 0.9505 * b;

    (x, y, z)
}

/// XYZ to RGB conversion (CIE 1931)
/// Using D65 white point
///
/// Formula:
/// |R|   | 3.2406 -1.5372 -0.4986| |X|
/// |G| = |-0.9689  1.8758  0.0415| |Y|
/// |B|   | 0.0557 -0.2040  1.0570| |Z|
#[inline]
#[allow(unused)]
pub fn xyz_to_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    (
        (r * 255.0).clamp(0.0, 255.0),
        (g * 255.0).clamp(0.0, 255.0),
        (b * 255.0).clamp(0.0, 255.0),
    )
}

/// Convert between different pixel formats
pub fn convert_pixel_format<T, U>(
    src: &Array3<T>,
    src_format: &str,
    dst_format: &str,
) -> Result<Array3<U>>
where
    T: Copy + Clone + NumCast + Zero,
    U: Copy + Clone + NumCast + Zero,
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

    let (_height, _width, channels) = src.dim();
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
        // RGB to BGR
        (PIXEL_FORMAT_RGB4, PIXEL_FORMAT_BGR4) |
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_BGR8) |
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_BGR24) |
        (PIXEL_FORMAT_RGB32, PIXEL_FORMAT_BGR32) |
        // BGR to RGB
        (PIXEL_FORMAT_BGR4, PIXEL_FORMAT_RGB4) |
        (PIXEL_FORMAT_BGR8, PIXEL_FORMAT_RGB8) |
        (PIXEL_FORMAT_BGR24, PIXEL_FORMAT_RGB24) |
        (PIXEL_FORMAT_BGR32, PIXEL_FORMAT_RGB32) => {
            swap_rgb_bgr(src)
        }

        // RGB to RGBA
        (PIXEL_FORMAT_RGB4, PIXEL_FORMAT_RGBA) |
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_RGBA) |
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_RGBA) |
        (PIXEL_FORMAT_RGB32, PIXEL_FORMAT_RGBA) |
        // BGR to BGRA
        (PIXEL_FORMAT_BGR4, PIXEL_FORMAT_BGRA) |
        (PIXEL_FORMAT_BGR8, PIXEL_FORMAT_BGRA) |
        (PIXEL_FORMAT_BGR24, PIXEL_FORMAT_BGRA) |
        (PIXEL_FORMAT_BGR32, PIXEL_FORMAT_BGRA) => {
            let alpha_value = U::from(255).unwrap();
            add_alpha_channel(src, alpha_value)
        }

        // RGBA to RGB
        (PIXEL_FORMAT_RGBA, PIXEL_FORMAT_RGB4) |
        (PIXEL_FORMAT_RGBA, PIXEL_FORMAT_RGB8) |
        (PIXEL_FORMAT_RGBA, PIXEL_FORMAT_RGB24) |
        (PIXEL_FORMAT_RGBA, PIXEL_FORMAT_RGB32) |
        // BGRA to BGR
        (PIXEL_FORMAT_BGRA, PIXEL_FORMAT_BGR4) |
        (PIXEL_FORMAT_BGRA, PIXEL_FORMAT_BGR8) |
        (PIXEL_FORMAT_BGRA, PIXEL_FORMAT_BGR24) |
        (PIXEL_FORMAT_BGRA, PIXEL_FORMAT_BGR32) => {
            remove_alpha_channel(src)
        }

        // RGB to GRAY
        (PIXEL_FORMAT_RGB4, PIXEL_FORMAT_GRAY8) => rgb_to_gray8(src, PIXEL_FORMAT_RGB4),
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_GRAY8) => rgb_to_gray8(src, PIXEL_FORMAT_RGB8),
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_GRAY8) => rgb_to_gray8(src, PIXEL_FORMAT_RGB24),
        (PIXEL_FORMAT_RGB32, PIXEL_FORMAT_GRAY8) => rgb_to_gray8(src, PIXEL_FORMAT_RGB32),

        // GRAY to GRAY
        (PIXEL_FORMAT_GRAY8, PIXEL_FORMAT_GRAY16) => gray8_to_gray16(src),
        (PIXEL_FORMAT_GRAY16, PIXEL_FORMAT_GRAY8) => gray16_to_gray8(src),

        // YUV to RGB8/RGB24
        (PIXEL_FORMAT_YUV410P, PIXEL_FORMAT_RGB8) |
        (PIXEL_FORMAT_YUV410P, PIXEL_FORMAT_RGB24) => {
            ndarray_yuv_to_rgb(src, PIXEL_FORMAT_YUV410P)
        }
        (PIXEL_FORMAT_YUV411P, PIXEL_FORMAT_RGB8) |
        (PIXEL_FORMAT_YUV411P, PIXEL_FORMAT_RGB24) => {
            ndarray_yuv_to_rgb(src, PIXEL_FORMAT_YUV411P)
        }
        (PIXEL_FORMAT_YUV420P, PIXEL_FORMAT_RGB8) |
        (PIXEL_FORMAT_YUV420P, PIXEL_FORMAT_RGB24) => {
            ndarray_yuv_to_rgb(src, PIXEL_FORMAT_YUV420P)
        }
        (PIXEL_FORMAT_YUV422P, PIXEL_FORMAT_RGB8) |
        (PIXEL_FORMAT_YUV422P, PIXEL_FORMAT_RGB24) => {
            ndarray_yuv_to_rgb(src, PIXEL_FORMAT_YUV422P)
        }
        (PIXEL_FORMAT_YUV440P, PIXEL_FORMAT_RGB8) |
        (PIXEL_FORMAT_YUV440P, PIXEL_FORMAT_RGB24) => {
            ndarray_yuv_to_rgb(src, PIXEL_FORMAT_YUV440P)
        }
        (PIXEL_FORMAT_YUV444P, PIXEL_FORMAT_RGB8) |
        (PIXEL_FORMAT_YUV444P, PIXEL_FORMAT_RGB24) => {
            ndarray_yuv_to_rgb(src, PIXEL_FORMAT_YUV444P)
        }

        // RGB8/RGB24 to YUV
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV410P) |
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV410P) => {
            ndarray_rgb_to_yuv(src, PIXEL_FORMAT_YUV410P)
        }
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV411P) |
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV411P) => {
            ndarray_rgb_to_yuv(src, PIXEL_FORMAT_YUV411P)
        }
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV420P) |
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV420P) => {
            ndarray_rgb_to_yuv(src, PIXEL_FORMAT_YUV420P)
        }
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV422P) |
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV422P) => {
            ndarray_rgb_to_yuv(src, PIXEL_FORMAT_YUV422P)
        }
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV440P) |
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV440P) => {
            ndarray_rgb_to_yuv(src, PIXEL_FORMAT_YUV440P)
        }
        (PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV444P) |
        (PIXEL_FORMAT_RGB24, PIXEL_FORMAT_YUV444P) => {
            ndarray_rgb_to_yuv(src, PIXEL_FORMAT_YUV444P)
        }

        _ => Err(Error::msg(format!(
            "Unsupported conversion path: {} to {}",
            src_format, dst_format
        ))),
    }
}

/// swap RGB <-> BGR channels
fn swap_rgb_bgr<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Copy + Clone + NumCast + Zero,
    U: Copy + Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::<U>::zeros((height, width, 3));

    for h in 0..height {
        for w in 0..width {
            // RGB -> BGR: swap R and B channels
            dst[[h, w, 0]] = NumCast::from(src[[h, w, 2]].to_f64().unwrap()).unwrap(); // B / R
            dst[[h, w, 1]] = NumCast::from(src[[h, w, 1]].to_f64().unwrap()).unwrap(); // G
            dst[[h, w, 2]] = NumCast::from(src[[h, w, 0]].to_f64().unwrap()).unwrap();
            // R / B
        }
    }

    Ok(dst)
}

/// Helper function to handle alpha channel addition
fn add_alpha_channel<T, U>(src: &Array3<T>, alpha_value: U) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::zeros((height, width, 4));
    let alpha: U = NumCast::from(alpha_value).unwrap();

    for h in 0..height {
        for w in 0..width {
            // Copy
            dst[[h, w, 0]] = NumCast::from(src[[h, w, 0]].to_f64().unwrap()).unwrap(); // B / R
            dst[[h, w, 1]] = NumCast::from(src[[h, w, 1]].to_f64().unwrap()).unwrap(); // G
            dst[[h, w, 2]] = NumCast::from(src[[h, w, 2]].to_f64().unwrap()).unwrap(); // R / B
                                                                                       // Add alpha channel
            dst[[h, w, 3]] = alpha.clone();
        }
    }
    Ok(dst)
}

/// Helper function to remove alpha channel
fn remove_alpha_channel<T, U>(src: &Array3<T>) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::zeros((height, width, 3));

    for h in 0..height {
        for w in 0..width {
            // Copy only RGB channels
            dst[[h, w, 0]] = NumCast::from(src[[h, w, 0]].to_f64().unwrap()).unwrap(); // B / R
            dst[[h, w, 1]] = NumCast::from(src[[h, w, 1]].to_f64().unwrap()).unwrap(); // G
            dst[[h, w, 2]] = NumCast::from(src[[h, w, 2]].to_f64().unwrap()).unwrap();
            // R / B
        }
    }

    Ok(dst)
}

/// RGB to GRAY
fn rgb_to_gray8<T, U>(src: &Array3<T>, src_format: &str) -> Result<Array3<U>>
where
    T: Copy + Clone + NumCast + Zero,
    U: Copy + Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
    let mut dst = Array3::zeros((height, width, 1));

    // 根据输入格式确定归一化因子
    let normalize_factor: f64 = match src_format {
        // 对于 RGB4，每个颜色通道使用 4 位表示，值范围是 0-15 (2^4 - 1 = 15)
        PIXEL_FORMAT_RGB4 => 15.0, // 4-bit 最大值
        // RGB8/RGB24/RGB32，每个颜色通道使用 8 位表示，值范围是 0-255 (2^8 - 1 = 255)
        _ => 255.0, // 8-bit 最大值
    };

    for h in 0..height {
        for w in 0..width {
            let r = src[[h, w, 0]].to_f64().unwrap() / normalize_factor; // R
            let g = src[[h, w, 1]].to_f64().unwrap() / normalize_factor; // G
            let b = src[[h, w, 2]].to_f64().unwrap() / normalize_factor; // B

            // RGB to Grayscale weights (BT.709)
            // 计算灰度值并缩放回 0-255 范围
            let gray = (r * 0.2126 + g * 0.7152 + b * 0.0722) * 255.0;

            dst[[h, w, 0]] = NumCast::from(gray.round().clamp(0.0, 255.0)).unwrap();
        }
    }

    Ok(dst)
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
fn ndarray_yuv_to_rgb<T, U>(src: &Array3<T>, src_format: &str) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();
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
            let y_val = src[[y, x, 0]].to_f64().unwrap();
            let u_val = src[[y / uv_height_ratio, x / uv_width_ratio, 1]]
                .to_f64()
                .unwrap();
            let v_val = src[[y / uv_height_ratio, x / uv_width_ratio, 2]]
                .to_f64()
                .unwrap();

            // YUV to RGB conversion
            let (r, g, b) = yuv_to_rgb(y_val, u_val, v_val);

            // Store RGB values
            dst[[y, x, 0]] = NumCast::from(r).unwrap();
            dst[[y, x, 1]] = NumCast::from(g).unwrap();
            dst[[y, x, 2]] = NumCast::from(b).unwrap();
        }
    }

    Ok(dst)
}

/// Convert RGB (RGB8 or RGB24) to YUV planar format
fn ndarray_rgb_to_yuv<T, U>(src: &Array3<T>, dst_format: &str) -> Result<Array3<U>>
where
    T: Clone + NumCast + Zero,
    U: Clone + NumCast + Zero,
{
    let (height, width, _channels) = src.dim();

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

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            // Convert RGB to Y component
            let r = src[[y, x, 0]].to_f64().unwrap_or(0.0);
            let g = src[[y, x, 1]].to_f64().unwrap_or(0.0);
            let b = src[[y, x, 2]].to_f64().unwrap_or(0.0);
            let (y_val, _, _) = rgb_to_yuv(r, g, b);
            dst[[y, x, 0]] = NumCast::from(y_val).unwrap();

            // Process UV components at subsampling points
            if y % uv_height_ratio == 0 && x % uv_width_ratio == 0 {
                let (u_sum, v_sum, count) = {
                    let mut u_sum = 0.0;
                    let mut v_sum = 0.0;
                    let mut count = 0;

                    // Calculate UV values for the current block
                    for sy in y..height.min(y + uv_height_ratio) {
                        for sx in x..width.min(x + uv_width_ratio) {
                            let r = src[[sy, sx, 0]].to_f64().unwrap_or(0.0);
                            let g = src[[sy, sx, 1]].to_f64().unwrap_or(0.0);
                            let b = src[[sy, sx, 2]].to_f64().unwrap_or(0.0);

                            let (_, u, v) = rgb_to_yuv(r, g, b);
                            u_sum += u;
                            v_sum += v;
                            count += 1;
                        }
                    }
                    (u_sum, v_sum, count)
                };

                // Store averaged UV values if block is not empty
                if count > 0 {
                    let uv_y = y / uv_height_ratio;
                    let uv_x = x / uv_width_ratio;

                    let u_val = (u_sum / count as f64).round().clamp(-128.0, 127.0);
                    let v_val = (v_sum / count as f64).round().clamp(-128.0, 127.0);

                    dst[[uv_y, uv_x, 1]] = NumCast::from(u_val).unwrap();
                    dst[[uv_y, uv_x, 2]] = NumCast::from(v_val).unwrap();
                }
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

        // Convert RGB8 -> YUV444P
        let yuv_result =
            convert_pixel_format::<u8, u8>(&rgb8_img, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_YUV444P)
                .unwrap();

        // Convert back YUV444P -> RGB8
        let back_to_rgb =
            convert_pixel_format::<u8, u8>(&yuv_result, PIXEL_FORMAT_YUV444P, PIXEL_FORMAT_RGB8)
                .unwrap();

        // Check dimensions
        assert_eq!(back_to_rgb.dim(), rgb8_img.dim());

        // Allow small differences due to YUV conversion
        for h in 0..test_size.0 {
            for w in 0..test_size.1 {
                for c in 0..3 {
                    let src = rgb8_img[[h, w, c]];
                    let dst = back_to_rgb[[h, w, c]];
                    let diff = (src as i32 - dst as i32).abs();
                    println!("rgb8_img:{}, back_to_rgb:{}, diff:{}", src, dst, diff);
                    assert!(diff <= 3);
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

    // 辅助函数：创建指定大小的测试图像
    fn create_image<T>(width: usize, height: usize, r: T, g: T, b: T) -> Array3<T>
    where
        T: Copy + Clone + Zero,
    {
        let mut img = Array3::zeros((height, width, 3));
        for i in 0..height {
            for j in 0..width {
                img[[i, j, 0]] = r;
                img[[i, j, 1]] = g;
                img[[i, j, 2]] = b;
            }
        }
        img
    }

    #[test]
    fn test_rgb4_to_gray8_white() -> Result<()> {
        // RGB4 最大值为 15
        let rgb = create_image(2, 2, 15u8, 15u8, 15u8);
        let gray = convert_pixel_format::<u8, u8>(&rgb, PIXEL_FORMAT_RGB4, PIXEL_FORMAT_GRAY8)?;

        // 全白图像，转换后应该是 255
        assert_eq!(gray[[0, 0, 0]], 255);
        assert_eq!(gray[[1, 1, 0]], 255);
        Ok(())
    }

    #[test]
    fn test_rgb4_to_gray8_black() -> Result<()> {
        // RGB4 最小值为 0
        let rgb = create_image(2, 2, 0u8, 0u8, 0u8);
        let gray = convert_pixel_format::<u8, u8>(&rgb, PIXEL_FORMAT_RGB4, PIXEL_FORMAT_GRAY8)?;

        // 全黑图像，转换后应该是 0
        assert_eq!(gray[[0, 0, 0]], 0);
        assert_eq!(gray[[1, 1, 0]], 0);
        Ok(())
    }

    #[test]
    fn test_rgb8_to_gray8_white() -> Result<()> {
        // RGB8 最大值为 255
        let rgb = create_image(2, 2, 255u8, 255u8, 255u8);
        let gray = convert_pixel_format::<u8, u8>(&rgb, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_GRAY8)?;

        // 全白图像，转换后应该是 255
        assert_eq!(gray[[0, 0, 0]], 255);
        assert_eq!(gray[[1, 1, 0]], 255);
        Ok(())
    }

    #[test]
    fn test_rgb8_to_gray8_black() -> Result<()> {
        // RGB8 最小值为 0
        let rgb = create_image(2, 2, 0u8, 0u8, 0u8);
        let gray = convert_pixel_format::<u8, u8>(&rgb, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_GRAY8)?;

        // 全黑图像，转换后应该是 0
        assert_eq!(gray[[0, 0, 0]], 0);
        assert_eq!(gray[[1, 1, 0]], 0);
        Ok(())
    }

    #[test]
    fn test_rgb8_to_gray8_red() -> Result<()> {
        // 纯红色图像
        let rgb = create_image(2, 2, 255u8, 0u8, 0u8);
        let gray = convert_pixel_format::<u8, u8>(&rgb, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_GRAY8)?;

        // R 权重为 0.2126，因此灰度值应该约为 54
        assert_eq!(gray[[0, 0, 0]], 54);
        assert_eq!(gray[[1, 1, 0]], 54);
        Ok(())
    }

    #[test]
    fn test_rgb8_to_gray8_green() -> Result<()> {
        // 纯绿色图像
        let rgb = create_image(2, 2, 0u8, 255u8, 0u8);
        let gray = convert_pixel_format::<u8, u8>(&rgb, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_GRAY8)?;

        // G 权重为 0.7152，因此灰度值应该约为 182
        assert_eq!(gray[[0, 0, 0]], 182);
        assert_eq!(gray[[1, 1, 0]], 182);
        Ok(())
    }

    #[test]
    fn test_rgb8_to_gray8_blue() -> Result<()> {
        // 纯蓝色图像
        let rgb = create_image(2, 2, 0u8, 0u8, 255u8);
        let gray = convert_pixel_format::<u8, u8>(&rgb, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_GRAY8)?;

        // B 权重为 0.0722，因此灰度值应该约为 18
        assert_eq!(gray[[0, 0, 0]], 18);
        assert_eq!(gray[[1, 1, 0]], 18);
        Ok(())
    }

    #[test]
    fn test_mixed_colors() -> Result<()> {
        let mut rgb = Array3::<u8>::zeros((2, 2, 3));
        // 设置一个混合颜色：R=255, G=128, B=64
        rgb[[0, 0, 0]] = 255; // R
        rgb[[0, 0, 1]] = 128; // G
        rgb[[0, 0, 2]] = 64; // B

        let gray = convert_pixel_format::<u8, u8>(&rgb, PIXEL_FORMAT_RGB8, PIXEL_FORMAT_GRAY8)?;

        // 计算期望的灰度值：
        // 255 * 0.2126 + 128 * 0.7152 + 64 * 0.0722 ≈ 150
        assert_eq!(gray[[0, 0, 0]], 150);
        Ok(())
    }

    #[test]
    fn test_hsv_conversion() {
        // Test pure red
        let (h, s, v) = rgb_to_hsv(255.0, 0.0, 0.0);
        assert_eq!(h, 0.0);
        assert_eq!(s, 1.0);
        assert_eq!(v, 1.0);

        // Test conversion back
        let (r, g, b) = hsv_to_rgb(h, s, v);
        assert_eq!(r, 255.0);
        assert_eq!(g, 0.0);
        assert_eq!(b, 0.0);
    }
}
