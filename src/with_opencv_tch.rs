use crate::{TchTensorAsImage, TchTensorImageShape, TryFromCv, TryIntoCv};
use anyhow::{Context, Error, Result};
use opencv::{core as cv, prelude::*};
use std::{
    borrow::{Borrow, Cow},
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    slice,
};
use tch;
use utils::*;

mod utils {
    use super::*;

    pub struct TchImageMeta {
        pub kind: tch::Kind,
        pub width: i64,
        pub height: i64,
        pub channels: i64,
    }

    pub struct TchTensorMeta {
        pub kind: tch::Kind,
        pub shape: Vec<i64>,
    }

    pub fn tch_kind_to_opencv_depth(kind: tch::Kind) -> Result<i32> {
        use tch::Kind as K;

        let typ = match kind {
            K::Uint8 => cv::CV_8U,
            K::Int8 => cv::CV_8S,
            K::Int16 => cv::CV_16S,
            K::Half => cv::CV_16F,
            K::Int => cv::CV_32S,
            K::Float => cv::CV_32F,
            K::Double => cv::CV_64F,
            kind => anyhow::bail!("unsupported tensor kind {:?}", kind),
        };

        Ok(typ)
    }

    pub fn opencv_depth_to_tch_kind(depth: i32) -> Result<tch::Kind> {
        use tch::Kind as K;

        let kind = match depth {
            cv::CV_8U => K::Uint8,
            cv::CV_8S => K::Int8,
            cv::CV_16S => K::Int16,
            cv::CV_32S => K::Int,
            cv::CV_16F => K::Half,
            cv::CV_32F => K::Float,
            cv::CV_64F => K::Double,
            _ => anyhow::bail!("unsupported OpenCV Mat depth {}", depth),
        };
        Ok(kind)
    }

    pub fn opencv_mat_to_tch_meta_2d(mat: &cv::Mat) -> Result<TchImageMeta> {
        let cv::Size { height, width } = mat.size()?;
        let kind = opencv_depth_to_tch_kind(mat.depth())?;
        let channels = mat.channels();
        Ok(TchImageMeta {
            kind,
            width: width as i64,
            height: height as i64,
            channels: channels as i64,
        })
    }

    pub fn opencv_mat_to_tch_meta_nd(mat: &cv::Mat) -> Result<TchTensorMeta> {
        let shape: Vec<_> = mat
            .mat_size()
            .iter()
            .map(|&dim| dim as i64)
            .chain([mat.channels() as i64])
            .collect();
        let kind = opencv_depth_to_tch_kind(mat.depth())?;
        Ok(TchTensorMeta { shape, kind })
    }
}

pub use tensor_from_mat::*;
mod tensor_from_mat {
    use super::*;

    /// A [Tensor](tch::Tensor) which data reference borrows from a [Mat](cv::Mat). It can be dereferenced to a [Tensor](tch::Tensor).
    #[derive(Debug)]
    pub struct OpenCvMatAsTchTensor<'a> {
        pub(super) tensor: ManuallyDrop<tch::Tensor>,
        pub(super) _mat: &'a cv::Mat,
    }

    impl<'a> Drop for OpenCvMatAsTchTensor<'a> {
        fn drop(&mut self) {
            unsafe {
                ManuallyDrop::drop(&mut self.tensor);
            }
        }
    }

    impl<'a> AsRef<tch::Tensor> for OpenCvMatAsTchTensor<'a> {
        fn as_ref(&self) -> &tch::Tensor {
            self.tensor.deref()
        }
    }

    impl<'a> Deref for OpenCvMatAsTchTensor<'a> {
        type Target = tch::Tensor;

        fn deref(&self) -> &Self::Target {
            self.tensor.deref()
        }
    }

    impl<'a> DerefMut for OpenCvMatAsTchTensor<'a> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.tensor.deref_mut()
        }
    }
}

impl<'a> TryFromCv<&'a cv::Mat> for OpenCvMatAsTchTensor<'a> {
    type Error = Error;

    fn try_from_cv(from: &'a cv::Mat) -> Result<Self, Self::Error> {
        anyhow::ensure!(from.is_continuous(), "non-continuous Mat is not supported");

        let TchTensorMeta { kind, shape } = opencv_mat_to_tch_meta_nd(&from)?;
        let strides = {
            let mut strides: Vec<_> = shape
                .iter()
                .rev()
                .cloned()
                .scan(1, |prev, dim| {
                    let stride = *prev;
                    *prev *= dim;
                    Some(stride)
                })
                .collect();
            strides.reverse();
            strides
        };

        let tensor = unsafe {
            let ptr = from.ptr(0)? as *const u8;
            tch::Tensor::f_from_blob(ptr, &shape, &strides, kind, tch::Device::Cpu)?
        };

        Ok(Self {
            tensor: ManuallyDrop::new(tensor),
            _mat: from,
        })
    }
}

impl TryFromCv<&cv::Mat> for TchTensorAsImage {
    type Error = Error;

    fn try_from_cv(mat: &cv::Mat) -> Result<Self, Self::Error> {
        let from = if mat.is_continuous() {
            Cow::Borrowed(mat)
        } else {
            // Mat created from clone() is implicitly continuous
            Cow::Owned(mat.try_clone()?)
        };

        let TchImageMeta {
            kind,
            width,
            height,
            channels,
        } = opencv_mat_to_tch_meta_2d(&*from)?;

        let tensor = unsafe {
            let ptr = from.ptr(0)? as *const u8;
            let slice_size = (height * width * channels) as usize * kind.elt_size_in_bytes();
            let slice = slice::from_raw_parts(ptr, slice_size);
            tch::Tensor::f_from_data_size(slice, &[height, width, channels], kind)?
        };

        Ok(TchTensorAsImage {
            tensor,
            kind: TchTensorImageShape::Hwc,
        })
    }
}

impl TryFromCv<cv::Mat> for TchTensorAsImage {
    type Error = Error;

    fn try_from_cv(from: cv::Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl TryFromCv<&cv::Mat> for tch::Tensor {
    type Error = Error;

    fn try_from_cv(mat: &cv::Mat) -> Result<Self, Self::Error> {
        let from = if mat.is_continuous() {
            Cow::Borrowed(mat)
        } else {
            // Mat created from clone() is implicitly continuous
            Cow::Owned(mat.try_clone()?)
        };

        let TchTensorMeta { kind, shape } = opencv_mat_to_tch_meta_nd(&*from)?;

        let tensor = unsafe {
            let ptr = from.ptr(0)? as *const u8;
            let slice_size = shape.iter().cloned().product::<i64>() as usize * kind.elt_size_in_bytes();
            let slice = slice::from_raw_parts(ptr, slice_size);
            tch::Tensor::f_from_data_size(slice, shape.as_ref(), kind)?
        };

        Ok(tensor)
    }
}

impl TryFromCv<cv::Mat> for tch::Tensor {
    type Error = Error;

    fn try_from_cv(from: cv::Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl TryFromCv<&TchTensorAsImage> for cv::Mat {
    type Error = Error;

    fn try_from_cv(from: &TchTensorAsImage) -> Result<Self, Self::Error> {
        let TchTensorAsImage {
            ref tensor,
            kind: convention,
        } = *from;

        use TchTensorImageShape as S;
        let (tensor, [channels, rows, cols]) = match (tensor.size3()?, convention) {
            ((w, h, c), S::Whc) => (tensor.f_permute(&[1, 0, 2])?, [c, h, w]),
            ((h, w, c), S::Hwc) => (tensor.shallow_clone(), [c, h, w]),
            ((c, w, h), S::Cwh) => (tensor.f_permute(&[2, 1, 0])?, [c, h, w]),
            ((c, h, w), S::Chw) => (tensor.f_permute(&[1, 2, 0])?, [c, h, w]),
        };

        // 将张量移动到CPU并转换为连续存储
        let tensor = tensor.f_contiguous()?.f_to_device(tch::Device::Cpu)?;

        // 获取 OpenCV 需要的深度
        let depth = tch_kind_to_opencv_depth(tensor.f_kind()?)?;
        let typ = cv::CV_MAKE_TYPE(depth, channels as i32);

        // 通用函数处理不同类型的数据
        unsafe fn create_mat_from_tensor<T>(
            tensor: &tch::Tensor,
            _total_size: usize,
            typ: i32,
            rows: i32,
            cols: i32,
        ) -> Result<cv::Mat, Error> {
            // let data_ptr = tensor.data_ptr() as *const T;
            // let data_slice = std::slice::from_raw_parts(data_ptr, total_size);
            Ok(Mat::new_rows_cols_with_data_unsafe_def(
                rows,
                cols,
                typ,
                tensor.data_ptr(),
            )?)
        }

        // 计算总数据量
        let total_size = (rows * cols * channels) as usize;

        // 根据 tensor 的 kind 调用 create_mat_from_tensor
        let mat = unsafe {
            match tensor.kind() {
                tch::Kind::Uint8 => create_mat_from_tensor::<u8>(
                    &tensor,
                    total_size,
                    typ,
                    rows as i32,
                    cols as i32,
                )?,
                tch::Kind::Int8 => create_mat_from_tensor::<i8>(
                    &tensor,
                    total_size,
                    typ,
                    rows as i32,
                    cols as i32,
                )?,
                tch::Kind::Int16 => create_mat_from_tensor::<i16>(
                    &tensor,
                    total_size,
                    typ,
                    rows as i32,
                    cols as i32,
                )?,
                tch::Kind::Float => create_mat_from_tensor::<f32>(
                    &tensor,
                    total_size,
                    typ,
                    rows as i32,
                    cols as i32,
                )?,
                tch::Kind::Double => create_mat_from_tensor::<f64>(
                    &tensor,
                    total_size,
                    typ,
                    rows as i32,
                    cols as i32,
                )?,
                _ => return Err(anyhow::anyhow!("Unsupported tensor kind")),
            }
        };

        Ok(mat)
    }
}

impl TryFromCv<TchTensorAsImage> for cv::Mat {
    type Error = Error;

    fn try_from_cv(from: TchTensorAsImage) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl TryFromCv<&tch::Tensor> for cv::Mat {
    type Error = Error;

    fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
        // 将张量移动到CPU并转换为连续存储
        let tensor = from.f_contiguous()?.f_to_device(tch::Device::Cpu)?;

        // 将张量的尺寸转换为 i32 向量
        let size: Vec<_> = tensor.size().into_iter().map(|dim| dim as i32).collect();

        // 获取 OpenCV 需要的深度
        let depth = tch_kind_to_opencv_depth(tensor.f_kind()?)?;

        // 检查张量类型的通道数
        let typ = cv::CV_MAKETYPE(depth, 1);

        // 使用 unsafe 块从指针生成 Mat
        let mat = unsafe { cv::Mat::new_nd_with_data_unsafe(&size, typ, tensor.data_ptr(), None)? };

        Ok(mat.try_clone()?)
    }
}

impl TryFromCv<tch::Tensor> for cv::Mat {
    type Error = Error;

    fn try_from_cv(from: tch::Tensor) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{self, IndexOp, Tensor};

    // const EPSILON: f64 = 1e-8;
    const ROUNDS: usize = 1000;

    #[test]
    fn tensor_mat_conv() -> Result<()> {
        let size = [2, 3, 4, 5];

        for _ in 0..ROUNDS {
            let before = Tensor::randn(size.as_ref(), tch::kind::FLOAT_CPU);
            let mat = cv::Mat::try_from_cv(&before)?;
            let after = Tensor::try_from_cv(&mat)?.f_view(size)?;

            // compare Tensor and Mat values
            {
                fn enumerate_reversed_index(dims: &[i64]) -> Vec<Vec<i64>> {
                    match dims {
                        [] => vec![vec![]],
                        [dim, remaining @ ..] => {
                            let dim = *dim;
                            let indexes: Vec<_> = (0..dim)
                                .flat_map(move |val| {
                                    enumerate_reversed_index(remaining).into_iter().map(
                                        move |mut tail| {
                                            tail.push(val);
                                            tail
                                        },
                                    )
                                })
                                .collect();
                            indexes
                        }
                    }
                }

                enumerate_reversed_index(&before.size())
                    .into_iter()
                    .map(|mut index| {
                        index.reverse();
                        index
                    })
                    .try_for_each(|tch_index| -> Result<_> {
                        let cv_index: Vec<_> =
                            tch_index.iter().cloned().map(|val| val as i32).collect();
                        let tch_index: Vec<_> = tch_index
                            .iter()
                            .cloned()
                            .map(|val| Some(Tensor::from(val)))
                            .collect();
                        let tch_val: f32 = before.f_index(&tch_index)?.try_into().unwrap();
                        let mat_val: f32 = *mat.at_nd(&cv_index)?;
                        anyhow::ensure!(tch_val == mat_val, "value mismatch");
                        Ok(())
                    })?;
            }

            // compare original and recovered Tensor values
            anyhow::ensure!(before == after, "value mismatch",);
        }

        Ok(())
    }

    #[test]
    fn tensor_as_image_and_mat_conv() -> Result<()> {
        for _ in 0..ROUNDS {
            let channels = 3;
            let height = 16;
            let width = 8;

            let before = Tensor::randn(&[channels, height, width], tch::kind::FLOAT_CPU);
            let mat: cv::Mat = TchTensorAsImage::new(before.shallow_clone(), TchTensorImageShape::Chw)?.try_into_cv()?;
            let after = Tensor::try_from_cv(&mat)?.f_permute(&[2, 0, 1])?; // hwc -> chw

            // compare Tensor and Mat values
            for row in 0..height {
                for col in 0..width {
                    let pixel: &cv::Vec3f = mat.at_2d(row as i32, col as i32)?;
                    let [red, green, blue] = **pixel;
                    println!("pixel[{}*{}]: [{},{},{}]", row, col, red, green, blue);
                    anyhow::ensure!(f32::try_from(before.i((0, row, col))).unwrap() == red,"value mismatch");
                    anyhow::ensure!(f32::try_from(before.i((1, row, col))).unwrap() == green,"value mismatch");
                    anyhow::ensure!(f32::try_from(before.i((2, row, col))).unwrap() == blue,"value mismatch");
                }
            }

            // compare original and recovered Tensor values
            {
                let before_size = before.size();
                let after_size = after.size();
                anyhow::ensure!(
                    before_size == after_size,
                    "size mismatch: {:?} vs. {:?}",
                    before_size,
                    after_size
                );
                anyhow::ensure!(before == after, "value mismatch");
            }
        }
        Ok(())
    }

    #[test]
    fn tensor_from_mat_conv() -> Result<()> {
        for _ in 0..ROUNDS {
            let channel = 3;
            let height = 16;
            let width = 8;

            let before = Tensor::randn(&[channel, height, width], tch::kind::FLOAT_CPU);
            let mat: cv::Mat = TchTensorAsImage::new(before.shallow_clone(), TchTensorImageShape::Chw)?.try_into_cv()?;
            let after = OpenCvMatAsTchTensor::try_from_cv(&mat)?; // in hwc
            println!("after={:?}", &*after);
            // compare original and recovered Tensor values
            {
                anyhow::ensure!(after.size() == [height, width, channel], "size mismatch",);
                anyhow::ensure!(&before.f_permute(&[1, 2, 0])? == &*after, "value mismatch");
            }
        }
        Ok(())
    }
}
