use crate::ToCv;
use nalgebra as na;

impl<T: na::Scalar + Copy> ToCv<na::Vector3<T>> for image::Rgb<T> {
    fn to_cv(&self) -> na::Vector3<T> {
        na::Vector3::new(self.0[0], self.0[1], self.0[2])
    }
}

impl<T: na::Scalar + Copy> ToCv<image::Rgb<T>> for na::Vector3<T> {
    fn to_cv(&self) -> image::Rgb<T> {
        image::Rgb([self.x, self.y, self.z])
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Vector4<T>> for image::Rgba<T> {
    fn to_cv(&self) -> na::Vector4<T> {
        na::Vector4::new(self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

impl<T: na::Scalar + Copy> ToCv<image::Rgba<T>> for na::Vector4<T> {
    fn to_cv(&self) -> image::Rgba<T> {
        image::Rgba([self.x, self.y, self.z, self.w])
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Vector1<T>> for image::Luma<T> {
    fn to_cv(&self) -> na::Vector1<T> {
        na::Vector1::new(self.0[0])
    }
}

impl<T: na::Scalar + Copy> ToCv<image::Luma<T>> for na::Vector1<T> {
    fn to_cv(&self) -> image::Luma<T> {
        image::Luma([self.x])
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Vector2<T>> for image::LumaA<T> {
    fn to_cv(&self) -> na::Vector2<T> {
        na::Vector2::new(self.0[0], self.0[1])
    }
}

impl<T: na::Scalar + Copy> ToCv<image::LumaA<T>> for na::Vector2<T> {
    fn to_cv(&self) -> image::LumaA<T> {
        image::LumaA([self.x, self.y])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_vector3_roundtrip() {
        let rgb = image::Rgb([1u8, 2, 3]);
        let v: na::Vector3<u8> = rgb.to_cv();
        assert_eq!((v.x, v.y, v.z), (1, 2, 3));

        let back: image::Rgb<u8> = v.to_cv();
        assert_eq!(back, rgb);
    }

    #[test]
    fn rgba_vector4_roundtrip() {
        let rgba = image::Rgba([1u8, 2, 3, 4]);
        let v: na::Vector4<u8> = rgba.to_cv();
        assert_eq!((v.x, v.y, v.z, v.w), (1, 2, 3, 4));

        let back: image::Rgba<u8> = v.to_cv();
        assert_eq!(back, rgba);
    }

    #[test]
    fn luma_vector1_roundtrip() {
        let luma = image::Luma([128u8]);
        let v: na::Vector1<u8> = luma.to_cv();
        assert_eq!(v.x, 128);

        let back: image::Luma<u8> = v.to_cv();
        assert_eq!(back, luma);
    }

    #[test]
    fn luma_alpha_vector2_roundtrip() {
        let luma_alpha = image::LumaA([128u8, 200]);
        let v: na::Vector2<u8> = luma_alpha.to_cv();
        assert_eq!((v.x, v.y), (128, 200));

        let back: image::LumaA<u8> = v.to_cv();
        assert_eq!(back, luma_alpha);
    }
}
