use crate::ToCv;
use opencv::core as cv_core;

impl<T> ToCv<cv_core::Point_<T>> for imageproc::point::Point<T>
where
    T: num_traits::Num + Copy,
{
    fn to_cv(&self) -> cv_core::Point_<T> {
        cv_core::Point_::new(self.x, self.y)
    }
}

impl<T> ToCv<imageproc::point::Point<T>> for cv_core::Point_<T>
where
    T: num_traits::Num + Copy,
{
    fn to_cv(&self) -> imageproc::point::Point<T> {
        imageproc::point::Point::new(self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use crate::ToCv;
    use anyhow::Result;
    use approx::abs_diff_eq;
    use opencv::core as cv_core;
    use rand::prelude::*;
    use std::f64;

    #[test]
    fn convert_opencv_imageproc() -> Result<()> {
        let mut rng = rand::rng();

        for _ in 0..5000 {
            // opencv to imageproc
            {
                let cv_point = cv_core::Point2d::new(rng.random(), rng.random());
                let imageproc_point: imageproc::point::Point<f64> = cv_point.to_cv();
                anyhow::ensure!(
                    abs_diff_eq!(cv_point.x, imageproc_point.x)
                        && abs_diff_eq!(cv_point.y, imageproc_point.y),
                    "point conversion failed"
                );
            }

            // imageproc to opencv
            {
                let imageproc_point =
                    imageproc::point::Point::<f64>::new(rng.random(), rng.random());
                let cv_point: cv_core::Point2d = imageproc_point.to_cv();
                anyhow::ensure!(
                    abs_diff_eq!(imageproc_point.x, cv_point.x)
                        && abs_diff_eq!(imageproc_point.y, cv_point.y),
                    "point conversion failed"
                );
            }
        }
        Ok(())
    }
}
