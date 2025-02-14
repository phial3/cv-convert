#![allow(unused_imports)]
//! Data conversion among computer vision libraries.
//!
//! # Version Selection
//!
//! In the default setting, up-to-date dependencies are used. The
//! default dependency versions are listed in `[features]` in
//! cv-convert `Cargo.toml`.
//!
//! You can manually select desired dependency versions. The choices
//! of dependency versions are named accordingly as Cargo
//! features. For example, the feature `nalgebra_0-31` enables
//! nalgebra 0.31.x.  It allows to list crate version selections in
//! `Cargo.toml`.
//!
//! ```toml
//! [dependencies.cv-convert]
//! version = 'x.y.z'
//! default-features = false
//! features = [
//!     'image_0-24',
//!     'opencv_0-76',
//!     'tch_0-10',
//!     'nalgebra_0-32',
//!     'ndarray_0-15',
//! ]
//! ```
//!
//! It's impossible to enable two or more versions for a
//! dependency. For example, `nalgebra_0-31` and `nalgebra_0-32` are
//! incompatible.
//!
//! # Traits
//!
//! The traits [FromCv] and [IntoCv] provide `.from_cv()` and `.into_cv()`, and
//! traits [TryFromCv] and [TryIntoCv] provide `.try_from_cv()` and `.try_into_cv()` methods respectively.
//! Just like std's [From], [Into], [TryFromCv] and [TryIntoCv].
//!
//!
//! ```rust,ignore
//! use nalgebra;
//! use cv_convert::{FromCv, IntoCv, TryFromCv, TryIntoCv};
//! use nalgebra as na;
//! use opencv as cv;
//!
//! // FromCv
//! let cv_point = cv::core::Point2d::new(1.0, 3.0);
//! let na_points = na::Point2::<f64>::from_cv(&cv_point);
//!
//! // IntoCv
//! let cv_point = cv::core::Point2d::new(1.0, 3.0);
//! let na_points: na::Point2<f64> = cv_point.into_cv();
//!
//! // TryFromCv
//! let na_mat = na::DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! let cv_mat = cv::core::Mat::try_from_cv(&na_mat).unwrap();
//!
//! // TryIntoCv
//! let na_mat = na::DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! let cv_mat: cv::core::Mat = na_mat.try_into_cv().unwrap();
//! ```
//!
//!
//! # Supported conversions
//!
//! The notations are used for simplicity.
//!
//! - `S -> T` suggests the conversion is defined by non-fallible [FromCv].
//! - `S ->? T` suggests the conversion is defined by fallible [TryFromCv].
//! - `(&)T` means the type can be either owned or borrowed.
//! - `&'a S -> &'a T` suggests that the target type borrows the source type.
//!
//! ## opencv -> opencv
//!
//! - [(&)Mat](opencv::core::Mat) ->? [Point_<T>](opencv::core::Point_)
//! - [(&)Mat](opencv::core::Mat) ->? [Point3_<T>](opencv::core::Point3_)
//! - [(&)Point_<T>](opencv::core::Point_) ->? [Mat](opencv::core::Mat)
//! - [(&)Point3_<T>](opencv::core::Point3_) ->? [Mat](opencv::core::Mat)
//!
//! ## std -> tch
//!
//! - owned/borrowed multi-dimensional [array](array) ->? [Tensor](tch::Tensor)
//!   - [(&)\[T; N\]](array) ->? [Tensor](tch::Tensor)
//!   - [(&)\[\[T; N2\]; N1\]](array) ->? [Tensor](tch::Tensor)
//!   - ... and so on up to 6 dimensions
//!
//! ## tch -> std
//!
//! - &'a [Tensor](tch::Tensor) -> &'a multi-dimensional [array](array)
//!   - &'a [Tensor](tch::Tensor) -> &'a [\[T; N\]](array)
//!   - &'a [Tensor](tch::Tensor) -> &'a [\[\[T; N2\]; N1\]](array)
//!   - ... and so on up to 6 dimensions
//! - [(&)Tensor](tch::Tensor) -> owned multi-dimensional [array](array)
//!   - [(&)Tensor](tch::Tensor) ->? [\[T; N\]](array)
//!   - [(&)Tensor](tch::Tensor) ->? [\[\[T; N2\]; N1\]](array)
//!   - ... and so on up to 6 dimensions
//!
//! ## tch -> ndarray
//!
//! - &[Tensor](tch::Tensor) ->? [Array](ndarray::Array)
//!
//! ## ndarray -> tch
//!
//! - &[Array](ndarray::Array) ->? [Tensor](tch::Tensor)
//!
//! ## ndarray -> opencv
//!
//! - &[Array](ndarray::Array) ->? [Mat](opencv::core::Mat)
//!
//! ## image -> tch
//!
//! - [(&)ImageBuffer](image::ImageBuffer) ->? [TchTensorAsImage]
//! - [(&)DynamicImage](image::DynamicImage) ->? [TchTensorAsImage]
//!
//! ## image -> opencv
//!
//! - [(&)ImageBuffer](image::ImageBuffer) ->? [Mat](opencv::core::Mat)
//! - [(&)DynamicImage](image::DynamicImage) ->? [Mat](opencv::core::Mat)
//!
//! ## opencv -> image 0.23
//!
//! - [Mat](opencv::core::Mat) ->? [(&)ImageBuffer](image::ImageBuffer)
//!
//! ## opencv -> image 0.24
//!
//! - [Mat](opencv::core::Mat) ->? [(&)ImageBuffer](image::ImageBuffer)
//! - [Mat](opencv::core::Mat) ->? [(&)DynamicImage](image::DynamicImage)
//!
//! ## opencv -> imageproc
//! - [(&)Point_<T>](opencv::core::Point_) -> [Point<T>](imageproc::point::Point)
//!
//! ## imageproc -> opencv
//! - [(&)Point<T>](imageproc::point::Point) -> [Point_<T>](opencv::core::Point_)
//!
//! ## opencv -> nalgebra
//!
//! - [(&)Mat](opencv::core::Mat) ->? [OMatrix](nalgebra::OMatrix)
//! - [(&)Point_<T>](opencv::core::Point_) -> [Point2<T>](nalgebra::Point2)
//! - [(&)Point3_<T>](opencv::core::Point3_) -> [Point2<T>](nalgebra::Point3)
//! - [(&)OpenCvPose<(&)Point3d>](OpenCvPose) ->? [Isometry3<f64>](nalgebra::Isometry3)
//! - [(&)OpenCvPose<(&)Mat>](OpenCvPose) ->? [Isometry3<f64>](nalgebra::Isometry3)
//!
//! ## nalgebra -> opencv
//!
//! - [(&)OMatrix](nalgebra::OMatrix) ->? [Mat](opencv::core::Mat)
//! - [(&)Point2<T>](nalgebra::Point2) -> [Point_<T>](opencv::core::Point_)
//! - [(&)Point3<T>](nalgebra::Point3) -> [Point3_<T>](opencv::core::Point3_)
//! - [(&)Translation<N, D>](nalgebra::Translation) ->? [Mat](opencv::core::Mat)
//! - [(&)Isometry3<T>](nalgebra::Isometry3) ->? [OpenCvPose<Point3_<T>>](OpenCvPose)
//! - [(&)Isometry3<f64>](nalgebra::Isometry3) ->? [OpenCvPose<Mat>](OpenCvPose)
//! - [(&)Isometry3<f32>](nalgebra::Isometry3) ->? [OpenCvPose<Mat>](OpenCvPose)
//!
//! ## opencv -> tch
//!
//! - [(&)Mat](opencv::core::Mat) ->? [Tensor](tch::Tensor)
//!
//!   The input [Mat](opencv::core::Mat) is regarded as an n-dimensional array with a m-channel elements.
//!   The output [Tensor](tch::Tensor) have n+1 dimensions, which last additional m-sized dimension is the channel.
//!
//! - [(&)Mat](opencv::core::Mat) ->? [TchTensorAsImage]
//!
//!   The input [Mat](opencv::core::Mat) must be a 2D image.
//!   The output [Tensor](tch::Tensor) within [TchTensorAsImage] has 3 dimensions, which last additional dimension is the channel.
//!
//! - [&Mat](opencv::core::Mat) ->? [OpenCvMatAsTchTensor]
//!
//! ## tch -> opencv
//!
//! - [(&)Tensor](tch::Tensor) ->? [Mat](opencv::core::Mat)
//!
//!  The n-dimensinoal input [Tensor](tch::Tensor) is converted to a [Mat](opencv::core::Mat)
//!  with n dimensions and a channel of size 1.
//!
//! - [(&)TchTensorAsImage](TchTensorAsImage) ->? [Mat](opencv::core::Mat)
//!
//!  The output [Mat](opencv::core::Mat) is a 2D image, which height, width and channel size
//!  are judged from the input [TchTensorAsImage](TchTensorAsImage) shape.
//!
//!
//! ## opencv -> ndarray
//!
//! - [&Mat](opencv::core::Mat) ->? [ArrayView](ndarray::ArrayView)
//! - [(&)Mat](opencv::core::Mat) ->? [Array](ndarray::Array)
//!
//!
//! # Notes for OpenCV
//!
//! For opencv older than 0.66, some systems requires `clang-runtime`
//! feature to build successfully. Otherwise you will get `libclang
//! shared library is not loaded on this thread!` panic. Add `opencv`
//! dependency along side `cv-convert` in your project Cargo.toml to
//! enable this feature.
//!
//! ```toml
//! cv-convert = { version = "0.22.0", default-features = false, features = ["opencv_0-65"] }
//! opencv = { version = "0.65", features = ["clang-runtime"] }
//! ```
//!
//! Most opencv modules, such as `videoio` and `aruco`, are disabled
//! by default to avoid bloating. Add opencv dependency to your
//! project Cargo.toml to enable default modules in your project.

pub mod prelude {
    pub use crate::common::prelude::*;
    pub use crate::traits::{FromCv, IntoCv, TryFromCv, TryIntoCv};
}

mod common;
mod macros;
use macros::*;

pub mod traits;
pub use traits::*;

/// 基础工具库模块
#[cfg(feature = "image")]
pub mod with_image;

#[cfg(feature = "imageproc")]
pub mod with_imageproc;

#[cfg(feature = "nalgebra")]
pub mod with_nalgebra;

#[cfg(feature = "ndarray")]
pub mod with_ndarray;

/// 工具库组合模块
#[cfg(all(feature = "imageproc", feature = "nalgebra"))]
pub mod with_imageproc_nalgebra;

#[cfg(all(feature = "imageproc", feature = "ndarray"))]
pub mod with_imageproc_ndarray;

#[cfg(all(feature = "nalgebra", feature = "image"))]
pub mod with_nalgebra_image;

#[cfg(all(feature = "nalgebra", feature = "ndarray"))]
pub mod with_nalgebra_ndarray;

#[cfg(all(feature = "ndarray", feature = "image"))]
pub mod with_ndarray_image;

/// OpenCV 相关模块
#[cfg(feature = "opencv")]
pub mod with_opencv;

#[cfg(all(feature = "opencv", feature = "image"))]
pub mod with_opencv_image;

#[cfg(all(feature = "opencv", feature = "imageproc"))]
pub mod with_opencv_imageproc;

#[cfg(all(feature = "opencv", feature = "nalgebra"))]
pub mod with_opencv_nalgebra;

#[cfg(all(feature = "opencv", feature = "ndarray"))]
pub mod with_opencv_ndarray;

#[cfg(all(feature = "opencv", feature = "rsmpeg"))]
pub mod with_opencv_rsmpeg;

#[cfg(all(feature = "opencv", feature = "tch"))]
pub mod with_opencv_tch;

/// FFmpeg (rsmpeg) 相关模块
#[cfg(feature = "rsmpeg")]
pub mod with_rsmpeg;

#[cfg(all(feature = "rsmpeg", feature = "image"))]
pub mod with_rsmpeg_image;

#[cfg(all(feature = "rsmpeg", feature = "imageproc"))]
pub mod with_rsmpeg_imageproc;

#[cfg(all(feature = "rsmpeg", feature = "nalgebra"))]
pub mod with_rsmpeg_nalgebra;

#[cfg(all(feature = "rsmpeg", feature = "ndarray"))]
pub mod with_rsmpeg_ndarray;

#[cfg(all(feature = "rsmpeg", feature = "tch"))]
pub mod with_rsmpeg_tch;

/// PyTorch (tch) 相关模块
#[cfg(feature = "tch")]
pub mod with_tch;

#[cfg(all(feature = "tch", feature = "image"))]
pub mod with_tch_image;

#[cfg(all(feature = "tch", feature = "imageproc"))]
pub mod with_tch_imageproc;

#[cfg(all(feature = "tch", feature = "nalgebra"))]
pub mod with_tch_nalgebra;

#[cfg(all(feature = "tch", feature = "ndarray"))]
pub mod with_tch_ndarray;

/// Re-exports for convenience
#[cfg(feature = "image")]
pub use image;

#[cfg(feature = "imageproc")]
pub use imageproc;

#[cfg(feature = "nalgebra")]
pub use nalgebra;

#[cfg(feature = "ndarray")]
pub use ndarray;

#[cfg(feature = "opencv")]
pub use opencv;

#[cfg(feature = "rsmpeg")]
pub use rsmpeg;

#[cfg(feature = "tch")]
pub use tch;
