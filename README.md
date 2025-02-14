# cv-convert
Convert computer vision data types in Rust

reference: [jerry73204](https://github.com/jerry73204/rust-cv-convert)

## Concept

```mermaid
graph TD
%% 核心结构定义
AF[AVFrame<br><i>视频原始数据</i>]:::avframe
MA[Mat<br><i>OpenCV视觉数据</i>]:::mat
IM[Image<br><i>通用图像数据</i>]:::image
ND[ndarray<br><i>数值计算核心</i>]:::ndarray

%% 转换路径矩阵
subgraph 以AVFrame为中心
  AF <-.->|sws_scale<br>av_image_alloc| IM
  AF <-.->|planes_to_3darray<br>av_image_copy| ND
  AF <-.->|avframe_to_mat<br>Mat::new_ndarray| MA
end

subgraph 以Mat为中心
  MA <-.->|mat_to_ndarray<br>as_slice| ND
  MA <-.->|mat_to_image<br>imencode/imdecode| IM
  MA <-.->|mat_from_avframe<br>av_image_fill_arrays| AF
end

subgraph 以Image为中心
  IM <-.->|imageproc::ops<br>DynamicImage转换| ND
  IM <-.->|image_to_avframe<br>save_to_memory| AF
  IM <-.->|image_to_mat<br>open/保存临时文件| MA
end

classDef avframe fill:#FFEBEE,stroke:#FF5252;
classDef mat fill:#FFF3E0,stroke:#FFB300;
classDef image fill:#E3F2FD,stroke:#2196F3;
classDef ndarray fill:#E8F5E9,stroke:#4CAF50;
```

> 异常处理矩阵：
> 
| 转换路径 |	可能异常	|解决方案 |
|---------  | ------- | ------ |
AVFrame→Mat	    | 色彩空间不匹配	|自动插入sws_scale转换上下文
Image→ndarray   | 通道顺序差异(RGB vs BGR)	| 提供convert_channels特性方法
Mat→Tensor	    | 内存对齐问题	| 使用aligned_alloc分配器


```mermaid
graph TD
    Start{输入数据类型} --> |视频流| A[优先AVFrame中心]
    Start --> |摄像头采集| B[优先Mat中心]
    Start --> |图片文件| C[优先Image中心]
    A --> D{需要视觉分析?} --> |是| E[转换为Mat]
    A --> F{需要机器学习?} --> |是| G[转换为ndarray]
    B --> H{需要持久化存储?} --> |是| I[转换为Image]
    C --> J{需要视频编码?} --> |是| K[转换为AVFrame]
```

## Usage

```toml
[dependencies]
cv-convert = { git = "https://github.com/phial3/cv-convert", branch = "main" }
```

## Features
- `default`: enable `image` + `imageproc` + `nalgebra` + `ndarray`
- `tch`: optional,  (System Required installation: [tch](https://crates.io/crates/tch))
- `opencv`: optional, (System Required installation: [opencv](https://crates.io/crates/opencv))
- `rsmpeg`: optional, (System Required installation: [rsmpeg](https://crates.io/crates/rsmpeg))
- `full` : enable `tch` + `opencv` + `rsmpeg`
- `image`: optional, enable [image](https://crates.io/crates/image)
- `imageproc`: optional, enable [imageproc](https://crates.io/crates/imageproc)
- `nalgebra`: optional, enable [nalgebra](https://crates.io/crates/nalgebra)
- `ndarray`: optional, enable [ndarray](https://crates.io/crates/ndarray)

## Examples

The crate provides `FromCv`, `TryFromCv`, `IntoCv`, `TryIntoCv` traits, which are similar to standard library's `From` and `Into`.

```rust,ignore,no_run
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
