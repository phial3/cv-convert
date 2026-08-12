use crate::ToCv;
use nalgebra as na;

impl<T: na::Scalar + Copy> ToCv<na::Vector2<T>> for na::Point2<T> {
    fn to_cv(&self) -> na::Vector2<T> {
        self.coords
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Point2<T>> for na::Vector2<T> {
    fn to_cv(&self) -> na::Point2<T> {
        na::Point2::new(self.x, self.y)
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Vector3<T>> for na::Point3<T> {
    fn to_cv(&self) -> na::Vector3<T> {
        self.coords
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Point3<T>> for na::Vector3<T> {
    fn to_cv(&self) -> na::Point3<T> {
        na::Point3::new(self.x, self.y, self.z)
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Vector1<T>> for na::Point1<T> {
    fn to_cv(&self) -> na::Vector1<T> {
        self.coords
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Point1<T>> for na::Vector1<T> {
    fn to_cv(&self) -> na::Point1<T> {
        na::Point1::new(self.x)
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Vector4<T>> for na::Point4<T> {
    fn to_cv(&self) -> na::Vector4<T> {
        self.coords
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Point4<T>> for na::Vector4<T> {
    fn to_cv(&self) -> na::Point4<T> {
        na::Point4::new(self.x, self.y, self.z, self.w)
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Vector5<T>> for na::Point5<T> {
    fn to_cv(&self) -> na::Vector5<T> {
        self.coords
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Point5<T>> for na::Vector5<T> {
    fn to_cv(&self) -> na::Point5<T> {
        na::Point5::new(self.x, self.y, self.z, self.w, self.a)
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Vector6<T>> for na::Point6<T> {
    fn to_cv(&self) -> na::Vector6<T> {
        self.coords
    }
}

impl<T: na::Scalar + Copy> ToCv<na::Point6<T>> for na::Vector6<T> {
    fn to_cv(&self) -> na::Point6<T> {
        na::Point6::new(self.x, self.y, self.z, self.w, self.a, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::abs_diff_eq;

    #[test]
    fn point2_vector2() {
        let p = na::Point2::new(1.0f64, 2.0);
        let v: na::Vector2<f64> = (&p).to_cv();
        assert!(abs_diff_eq!(v.x, 1.0) && abs_diff_eq!(v.y, 2.0));

        let back: na::Point2<f64> = (&v).to_cv();
        assert_eq!(back, p);
    }

    #[test]
    fn point3_vector3() {
        let p = na::Point3::new(1.0f64, 2.0, 3.0);
        let v: na::Vector3<f64> = (&p).to_cv();
        assert!(abs_diff_eq!(v.z, 3.0));

        let back: na::Point3<f64> = (&v).to_cv();
        assert_eq!(back, p);
    }
}
