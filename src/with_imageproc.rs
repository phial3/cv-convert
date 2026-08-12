use crate::ToCv;

impl<T: Copy> ToCv<(T, T)> for imageproc::point::Point<T> {
    fn to_cv(&self) -> (T, T) {
        (self.x, self.y)
    }
}

impl<T: Copy> ToCv<imageproc::point::Point<T>> for (T, T) {
    fn to_cv(&self) -> imageproc::point::Point<T> {
        imageproc::point::Point::new(self.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_tuple_roundtrip() {
        let p = imageproc::point::Point::new(3i32, 4);
        let tuple: (i32, i32) = (&p).to_cv();
        assert_eq!(tuple, (3, 4));

        let back: imageproc::point::Point<i32> = (&tuple).to_cv();
        assert_eq!(back, p);
    }
}
