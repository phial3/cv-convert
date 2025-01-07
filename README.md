# cv-convert: Convert computer vision data types in Rust

Type conversions among famous Rust computer vision libraries. It
supports the following crates:

reference:
https://github.com/jerry73204/rust-cv-convert

- [image](https://crates.io/crates/image)
- [imageproc](https://crates.io/crates/imageproc)
- [nalgebra](https://crates.io/crates/nalgebra)
- [opencv](https://crates.io/crates/opencv)
- [tch](https://crates.io/crates/tch)
- [ndarray](https://crates.io/crates/ndarray)

## Usage

Run `cargo add cv-convert` to add this crate to your project. In the
default setting, up-to-date dependency versions are used.

If you desire to enable specified dependency versions. Add
`default-features = false` and select crate versions as Cargo
features. For example, the feature `nalgebra_0-30` enables nalgebra
0.30.x.

```toml
[dependencies.cv-convert]
version = 'x.y.z'  # Please look up the recent version on crates.io
default-features = false
features = [
    'image',
    'opencv',
    'tch',
    'nalgebra',
    'ndarray',
]
```

The minimum supported `rustc` is 1.81. You may use older versions of
the crate (>=0.6) in order to use `rustc` versions that do not support
const-generics.

## Cargo Features

### opencv

- `opencv 0.93`

### image

- `image 0.25`

### imageproc

- `imageproc 0.25`

### ndarray

- `ndarray 0.16`

### nalgebra

- `nalgebra 0.33`

### tch

- `tch 0.13`

## Usage

The crate provides `FromCv`, `TryFromCv`, `IntoCv`, `TryIntoCv` traits, which are similar to standard library's `From` and `Into`.

```rust
use cv_convert::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use nalgebra as na;
use opencv as cv;

// FromCv
let cv_point = cv::core::Point2d::new(1.0, 3.0);
let na_points = na::Point2::<f64>::from_cv(&cv_point);

// IntoCv
let cv_point = cv::core::Point2d::new(1.0, 3.0);
let na_points: na::Point2<f64> = cv_point.into_cv();

// TryFromCv
let na_mat = na::DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let cv_mat = cv::core::Mat::try_from_cv(&na_mat)?;

// TryIntoCv
let na_mat = na::DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let cv_mat: cv::core::Mat = na_mat.try_into_cv()?;
```

## Contribute to this Project

### Add a new dependency version

To add the new version of nalgebra 0.32 for cv-convert for example,
open `cv-convert-generate/packages.toml` in the source repository. Add
a new version to the list like this.

```toml
[package.nalgebra]
versions = ["0.26", "0.27", "0.28", "0.29", "0.30", "0.31", "0.32"]
use_default_features = true
features = []
```

Run `make generate` at the top-level directory. It modifies Rust
source files automatically. One extra step is to copy the snipplet in
`cv-convert/generated/Cargo.toml.snipplet` and paste it to
`cv-convert/Cargo.toml`.


### Add a new type conversion

To add a new type conversion, take `image::DynamicImage` and
`opencv::Mat` for example. Proceed to `cv-convert/src` and implement
the code in `with_opencv_image.rs` because it is a conversion among
opencv and image crates.


Choose `FromCv` or `TryFromCv` trait and add the trait implementation
on `image::DynamicImage` and `opencv::Mat` types. The choice of
`FromCv` or `TryFromCv` depends on whether the conversion is fallible
or not.

```rust
impl FromCv<&image::DynamicImage> for opencv::Mat { /* omit */ }
impl FromCv<&opencv::Mat> for image::DynamicImage { /* omit */ }

// or

impl TryFromCv<&image::DynamicImage> for opencv::Mat { /* omit */ }
impl TryFromCv<&opencv::Mat> for image::DynamicImage { /* omit */ }

#[cfg(test)]
mod tests {
    // Write a test
}
```

## License

MIT license. See [LICENSE](LICENSE.txt) file.
