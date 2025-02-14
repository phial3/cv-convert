# cv-convert
Convert computer vision data types in Rust

Type conversions among famous Rust computer vision libraries. It
supports the following crates:

reference:
https://github.com/jerry73204/rust-cv-convert

## sys related
- [opencv](https://crates.io/crates/opencv)
- [tch](https://crates.io/crates/tch)
- [rsmpeg](https://crates.io/crates/rsmpeg)

## lib
- [image](https://crates.io/crates/image)
- [imageproc](https://crates.io/crates/imageproc)
- [nalgebra](https://crates.io/crates/nalgebra)
- [ndarray](https://crates.io/crates/ndarray)

## Usage

```toml
[dependencies]
cv-convert = { git = "https://github.com/phial3/cv-convert", branch = "main" }
```

The minimum supported `rustc` is 1.81. You may use older versions of
the crate (>=0.6) in order to use `rustc` versions that do not support
const-generics.

## Cargo Features
- `default`: enable `image` + `imageproc` + `nalgebra` + `ndarray`
- `tch`
- `opencv`
- `rsmpeg`
- `full` : enable `tch` + `opencv` + `rsmpeg`
- `image`
- `imageproc`
- `nalgebra`
- `ndarray`


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
