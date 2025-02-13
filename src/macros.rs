#![allow(unused_macros)]
#![allow(unused_imports)]

/// if_image 宏定义
macro_rules! if_image {
    ($($item:item)*) => {
        $(
            #[cfg(feature = "image")]
            $item
        )*
    };
}
pub(crate) use if_image;

/// has_image 宏定义
macro_rules! has_image {
    ($($item:item)*) => {
        crate::macros::if_image! {
            #[allow(unused_imports)]
            use image as _;
            $($item)*
        }
    };
}
pub(crate) use has_image;

/// if_nalgebra 宏定义
macro_rules! if_nalgebra {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                feature = "nalgebra"
            ))]
            $item
        )*
    };
}
pub(crate) use if_nalgebra;

/// has_nalgebra 宏定义
macro_rules! has_nalgebra {
    ($($item:item)*) => {
        crate::macros::if_nalgebra! {
            #[allow(unused_imports)]
            use nalgebra as _;
            $($item)*
        }
    };
}
pub(crate) use has_nalgebra;

/// if_opencv 宏定义
macro_rules! if_opencv {
    ($($item:item)*) => {
        $(
            #[cfg(feature = "opencv")]
            $item
        )*
    };
}
pub(crate) use if_opencv;

/// has_opencv 宏定义
macro_rules! has_opencv {
    ($($item:item)*) => {
        crate::macros::if_opencv! {
            #[allow(unused_imports)]
            use opencv as _;
            $($item)*
        }
    };
}
pub(crate) use has_opencv;

/// if_rsmpeg 宏定义
macro_rules! if_rsmpeg {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                feature = "rsmpeg"
            ))]
            $item
        )*
    };
}
pub(crate) use if_rsmpeg;

/// has_rsmpeg 宏定义
macro_rules! has_rsmpeg {
    ($($item:item)*) => {
        crate::macros::if_rsmpeg! {
            #[allow(unused_imports)]
            use rsmpeg as _;
            $($item)*
        }
    };
}
pub(crate) use has_rsmpeg;

/// if_ndarray 宏定义
macro_rules! if_ndarray {
    ($($item:item)*) => {
        $(
            #[cfg(feature = "ndarray")]
            $item
        )*
    };
}
pub(crate) use if_ndarray;

/// has_ndarray 宏定义
macro_rules! has_ndarray {
    ($($item:item)*) => {
        crate::macros::if_ndarray! {
            #[allow(unused_imports)]
            use ndarray as _;
            $($item)*
        }
    };
}
pub(crate) use has_ndarray;

/// if_tch 宏定义
macro_rules! if_tch {
    ($($item:item)*) => {
        $(
            #[cfg(feature = "tch")]
            $item
        )*
    };
}
pub(crate) use if_tch;

/// has_tch 宏定义
macro_rules! has_tch {
    ($($item:item)*) => {
        crate::macros::if_tch! {
            #[allow(unused_imports)]
            use tch as _;
            $($item)*
        }
    };
}
pub(crate) use has_tch;

/// if_imageproc 宏定义
macro_rules! if_imageproc {
    ($($item:item)*) => {
        $(
            #[cfg(any(feature = "imageproc"))]
            $item
        )*
    };
}
pub(crate) use if_imageproc;

/// has_imageproc 宏定义
macro_rules! has_imageproc {
    ($($item:item)*) => {
        crate::macros::if_imageproc! {
            #[allow(unused_imports)]
            use imageproc as _;
            $($item)*
        }
    };
}
pub(crate) use has_imageproc;
