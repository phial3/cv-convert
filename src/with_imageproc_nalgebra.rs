use crate::ToCv;
use nalgebra as na;

impl<T: na::Scalar + Copy> ToCv<na::Point2<T>> for imageproc::point::Point<T> {
    fn to_cv(&self) -> na::Point2<T> {
        na::Point2::new(self.x, self.y)
    }
}

impl<T: na::Scalar + Copy> ToCv<imageproc::point::Point<T>> for na::Point2<T> {
    fn to_cv(&self) -> imageproc::point::Point<T> {
        imageproc::point::Point::new(self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_nalgebra_roundtrip() {
        let p = imageproc::point::Point::new(3i32, 4);
        let na_p: na::Point2<i32> = (&p).to_cv();
        assert_eq!((na_p.x, na_p.y), (3, 4));

        let back: imageproc::point::Point<i32> = (&na_p).to_cv();
        assert_eq!(back, p);
    }
}
