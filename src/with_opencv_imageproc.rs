use crate::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use opencv::core as cv_core;

impl<T> FromCv<&imageproc::point::Point<T>> for cv_core::Point_<T>
where
    T: num_traits::Num + Copy,
{
    fn from_cv(from: &imageproc::point::Point<T>) -> Self {
        cv_core::Point_::new(from.x, from.y)
    }
}

impl<T> FromCv<imageproc::point::Point<T>> for cv_core::Point_<T>
where
    T: num_traits::Num + Copy,
{
    fn from_cv(from: imageproc::point::Point<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

impl<T> FromCv<&cv_core::Point_<T>> for imageproc::point::Point<T>
where
    T: num_traits::Num + Copy,
{
    fn from_cv(from: &cv_core::Point_<T>) -> Self {
        Self::new(from.x, from.y)
    }
}

impl<T> FromCv<cv_core::Point_<T>> for imageproc::point::Point<T>
where
    T: num_traits::Num + Copy,
{
    fn from_cv(from: cv_core::Point_<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

#[cfg(test)]
mod tests {
    use crate::{FromCv, IntoCv};
    use anyhow::Result;
    use approx::abs_diff_eq;
    use opencv::core as cv_core;
    use rand::prelude::*;
    use std::f64;

    #[test]
    fn convert_opencv_imageproc() -> Result<()> {
        let mut rng = rand::rng();

        for _ in 0..5000 {
            // FromCv
            // opencv to imageproc
            {
                let cv_point = cv_core::Point2d::new(rng.random(), rng.random());
                let imageproc_point = imageproc::point::Point::<f64>::from_cv(&cv_point);
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
                let cv_point = cv_core::Point2d::from_cv(&imageproc_point);
                anyhow::ensure!(
                    abs_diff_eq!(imageproc_point.x, cv_point.x)
                        && abs_diff_eq!(imageproc_point.y, cv_point.y),
                    "point conversion failed"
                );
            }

            // IntoCv
            // opencv to imageproc
            {
                let cv_point = cv_core::Point2d::new(rng.random(), rng.random());
                let imageproc_point: imageproc::point::Point<f64> = cv_point.into_cv();
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
                let cv_point: cv_core::Point2d = imageproc_point.into_cv();
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
