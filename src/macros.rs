#![allow(unused_macros)]
#![allow(unused_imports)]

macro_rules! if_image {
    ($ ($ item : item) *) => {
        $ ( #[cfg (any (feature = "image"))] $ item) *
    };
}
pub(crate) use if_image;

macro_rules! has_image {
    ($ ($ item : item) *) => {
        crate::macros::if_image! {
            # [allow (unused_imports)]
            use crate::image as _ ; $ ($ item) *
        }
    }
}
pub(crate) use has_image;

macro_rules! if_nalgebra { ($ ($ item : item) *) => { $ (# [cfg (any (feature = "nalgebra"))] $ item) *} ; }
pub(crate) use if_nalgebra;

macro_rules! has_nalgebra { ($ ($ item : item) *) => { crate :: macros :: if_nalgebra ! { # [allow (unused_imports)] use crate :: nalgebra as _ ; $ ($ item) * } } }
pub(crate) use has_nalgebra;

macro_rules! if_opencv { ($ ($ item : item) *) => { $ (# [cfg (any (feature = "opencv"))] $ item) * } ; }
pub(crate) use if_opencv;

macro_rules! has_opencv { ($ ($ item : item) *) => { crate :: macros :: if_opencv ! { # [allow (unused_imports)] use crate :: opencv as _ ; $ ($ item) * } } }
pub(crate) use has_opencv;

macro_rules! if_ndarray { ($ ($ item : item) *) => { $ (# [cfg (any (feature = "ndarray"))] $ item) * } ; }
pub(crate) use if_ndarray;

macro_rules! has_ndarray { ($ ($ item : item) *) => { crate :: macros :: if_ndarray ! { # [allow (unused_imports)] use crate :: ndarray as _ ; $ ($ item) * } } }
pub(crate) use has_ndarray;

macro_rules! if_tch { ($ ($ item : item) *) => { $ (# [cfg (any (feature = "tch"))] $ item) * } ; }
pub(crate) use if_tch;

macro_rules! has_tch { ($ ($ item : item) *) => { crate :: macros :: if_tch ! { # [allow (unused_imports)] use crate :: tch as _ ; $ ($ item) * } } }
pub(crate) use has_tch;

macro_rules! if_imageproc { ($ ($ item : item) *) => { $ (# [cfg (any (feature = "imageproc"))] $ item) * } ; }
pub(crate) use if_imageproc;

macro_rules! has_imageproc { ($ ($ item : item) *) => { crate :: macros :: if_imageproc ! { # [allow (unused_imports)] use crate :: imageproc as _ ; $ ($ item) * } } }
pub(crate) use has_imageproc;
