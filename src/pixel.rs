use num_traits::{NumCast, Zero};
use strum_macros::{Display, EnumString};

/// 像素格式（仅包含本 crate 实际支持的格式）。
///
/// 变体的判别值直接等于 FFmpeg `AVPixelFormat` 的枚举值，作为 `pix_fmt`
/// 映射的**唯一**来源：`pix_fmt()` 通过 `as i32` 得到，反向查找统一走
/// [`PixelFormat::from_av`]，避免在多个模块中重复维护映射表。
#[derive(EnumString, Display, Debug, Clone, Copy, PartialEq, Eq)]
#[strum(serialize_all = "UPPERCASE")]
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
#[repr(i32)]
pub enum PixelFormat {
    YUV420P = 0,
    YUYV422 = 1,
    RGB24 = 2,
    BGR24 = 3,
    YUV422P = 4,
    YUV444P = 5,
    YUV410P = 6,
    YUV411P = 7,
    GRAY8 = 8,
    UYVY422 = 15,
    BGR8 = 17,
    BGR4 = 18,
    RGB8 = 20,
    RGB4 = 21,
    NV12 = 23,
    NV21 = 24,
    RGBA = 26,
    BGRA = 28,
    YUV440P = 31,
}

impl PixelFormat {
    /// 由 FFmpeg `AVPixelFormat` 值反查像素格式。
    pub fn from_av(value: i32) -> Option<PixelFormat> {
        match value {
            0 => Some(Self::YUV420P),
            1 => Some(Self::YUYV422),
            2 => Some(Self::RGB24),
            3 => Some(Self::BGR24),
            4 => Some(Self::YUV422P),
            5 => Some(Self::YUV444P),
            6 => Some(Self::YUV410P),
            7 => Some(Self::YUV411P),
            8 => Some(Self::GRAY8),
            15 => Some(Self::UYVY422),
            17 => Some(Self::BGR8),
            18 => Some(Self::BGR4),
            20 => Some(Self::RGB8),
            21 => Some(Self::RGB4),
            23 => Some(Self::NV12),
            24 => Some(Self::NV21),
            26 => Some(Self::RGBA),
            28 => Some(Self::BGRA),
            31 => Some(Self::YUV440P),
            _ => None,
        }
    }

    /// 返回 FFmpeg `AVPixelFormat` 值。
    pub fn pix_fmt(&self) -> i32 {
        *self as i32
    }

    /// 像素格式所属的族，用于指导转换路径。
    pub fn family(&self) -> PixelFamily {
        match self {
            Self::RGB4 | Self::RGB8 | Self::RGB24 | Self::BGR4 | Self::BGR8 | Self::BGR24 => {
                PixelFamily::Rgb
            }
            Self::RGBA | Self::BGRA => PixelFamily::Rgba,
            Self::GRAY8 => PixelFamily::Gray,
            Self::YUV410P
            | Self::YUV411P
            | Self::YUV420P
            | Self::YUV422P
            | Self::YUV440P
            | Self::YUV444P => PixelFamily::PlanarYuv,
            Self::YUYV422 | Self::UYVY422 => PixelFamily::PackedYuv,
            Self::NV12 | Self::NV21 => PixelFamily::SemiPlanarYuv,
        }
    }

    /// 像素格式的通道数（用于 ndarray 的第三维）。
    pub fn channels(&self) -> usize {
        match self {
            Self::GRAY8 => 1,
            Self::RGB4 | Self::RGB8 | Self::RGB24 | Self::BGR4 | Self::BGR8 | Self::BGR24 => 3,
            Self::RGBA | Self::BGRA => 4,
            Self::YUV410P
            | Self::YUV411P
            | Self::YUV420P
            | Self::YUV422P
            | Self::YUV440P
            | Self::YUV444P
            | Self::YUYV422
            | Self::NV12
            | Self::NV21
            | Self::UYVY422 => 3,
        }
    }

    /// 返回 YUV 子采样指数 `(subsample_x, subsample_y)`。
    ///
    /// 每个色度采样点覆盖 `2^subsample_x × 2^subsample_y` 个像素；
    /// 对非 YUV 格式返回 `None`。
    pub fn yuv_params(&self) -> Option<(u32, u32)> {
        match self {
            Self::YUV410P => Some((2, 2)),                 // 4:1:0
            Self::YUV411P => Some((2, 0)),                 // 4:1:1
            Self::YUV420P => Some((1, 1)),                 // 4:2:0
            Self::YUV422P => Some((1, 0)),                 // 4:2:2
            Self::YUV440P => Some((0, 1)),                 // 4:4:0
            Self::YUV444P => Some((0, 0)),                 // 4:4:4
            Self::YUYV422 | Self::UYVY422 => Some((1, 0)), // 4:2:2 打包
            Self::NV12 | Self::NV21 => Some((1, 1)),       // 4:2:0 半平面
            _ => None,
        }
    }
}

/// 像素格式族，作为数据驱动的转换分发依据。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFamily {
    /// 3 通道 RGB/BGR（RGB4/8/24、BGR4/8/24）
    Rgb,
    /// 4 通道 RGBA/BGRA
    Rgba,
    /// 单通道灰度（GRAY8）
    Gray,
    /// 平面 YUV（YUV410P/411P/420P/422P/440P/444P）
    PlanarYuv,
    /// 打包 YUV（YUYV422/UYVY422）
    PackedYuv,
    /// 半平面 YUV（NV12/NV21）
    SemiPlanarYuv,
}

/// 可用于 ndarray 像素数组的数值类型。
///
/// 约束：`NumCast` 用于通道值转换（`to_u8`/`to_f64`），`Zero` 用于
/// 零初始化（`Array3::zeros`），`'static` 用于运行时类型识别（`TypeId`）。
pub trait PixelType: Copy + Clone + NumCast + Zero + 'static {}
impl PixelType for u8 {}
impl PixelType for u16 {}
impl PixelType for u32 {}
impl PixelType for u64 {}
impl PixelType for i8 {}
impl PixelType for i16 {}
impl PixelType for i32 {}
impl PixelType for i64 {}
impl PixelType for f32 {}
impl PixelType for f64 {}
