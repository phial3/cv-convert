//! Extension traits for viewing a slice as a slice of arrays or vice versa.
//!
//! Provides the following methods on `[T]`:
//!
//!  * **[`nest`]**: `&[T] -> &[[T; n]]`
//!  * **[`flat`]**: `&[[T; n]] -> &[T]`
//!  * **[`as_array`]**: `&[T] -> &[T; n]` (the reverse is
//!    already provided by a coercion)
//!  * **`nest_mut`, `flat_mut`, `as_mut_array`** for `&mut [_]`.
//!
//! Altogether, these let you swap between arbitrary representations
//! of contiguous, `T`-aligned streams of `T` data.  For instance,
//! to view a `[[i32; 6]; 5]` as a `&[[[i32; 3]; 2]; 5]`,
//! one could write
//!
//! ```rust,ignore
//! // FIXME: Dumb/confusing example. I actually wrote it wrong
//! //        the first time, calling `flat()` twice because it
//! //        didn't occur to me that the outer '; 5' is already
//! //        automatically eliminated by coercion.
//! //
//! //        Almost makes a case for providing `.as_slice()`
//! //        as an explicit form of this coercion.
//!
//! use cv_convert::prelude::*;
//! let _ = || {
//!     let x: [[i32; 6]; 5] = unimplemented!();
//!     let _: &[[[i32; 3]; 2]; 5] = x.flat().nest().nest().as_array_ref();
//! };
//! ```
//!
//! Type inference generally works quite well, and as long as the
//! final shape is unambiguous there is no need to annotate types
//! in the middle of the method chain.
//!
//! In cases where type inference is unable to determine the target
//! array size, one can use a turbofish: e.g .`x.nest::<[_; 3]>()`.
//!
//! ```rust,ignore
//! use ::cv_convert::prelude::*;
//!
//! let vec = vec![[2i32, 2, 2], [7, 7, 7], [4, 4, 4], [1, 1, 1]];
//! assert_eq!(vec.flat(), &[2, 2, 2, 7, 7, 7, 4, 4, 4, 1, 1, 1]);
//!
//! // note: this requires an annotation only due to polymorphism in PartialEq
//! let slc = vec.nest::<[_; 2]>();
//! assert_eq!(slc, &[[[2i32, 2, 2], [7, 7, 7]], [[4, 4, 4], [1, 1, 1]]]);
//! ```
//!
//! [`nest`] and [`as_array`] panic on failure rather than returning options.
//! The rationale is that it is believed that these these conversions are
//! seldom needed on arbitrary user data which may be the wrong size; rather,
//! they are most likely used when bridging the gap between APIs that work
//! with flattened slices and APIs that work with slices of arrays.
//!
//! Zero-cost conversions in owned data (e.g. between `Vec<T>`
//! and `Vec<[T; n]>`) are not provided, and are probably impossible
//! in consideration of e.g. custom allocators. If you need to
//! convert between such types, you can use these traits in tandem
//! with `<[T]>::to_vec` to perform a copy:
//!
//! ```rust,ignore
//! use ::cv_convert::prelude::*;
//! let vec = vec![[2i32, 2, 2], [7, 7, 7]];
//!
//! // copying into a Vec<i32>
//! let flattened = vec.flat().to_vec();
//! assert_eq!(flattened, vec![2i32, 2, 2, 7, 7, 7]);
//! ```
//!
//! [`nest`]: SliceNestExt::nest
//! [`flat`]: SliceFlatExt::flat
//! [`as_array`]: SliceArrayExt::as_array

use core::slice;

pub mod prelude {
    //! This module contains extension traits from `slice_of_array`.
    //!
    //! It is meant to be glob imported, by users who may find it obnoxious to remember
    //! the precise names of the traits that each method belongs to.
    //!
    //! ```rust,ignore
    //! use cv_convert::prelude::*;
    //! ```
    //!
    //! `slice_of_array` follows an opinionated policy on what preludes should and should
    //! not contain. This prelude will never contain anything that the user will likely
    //! want to refer to by name.

    pub use super::SliceArrayExt;
    pub use super::SliceFlatExt;
    pub use super::SliceNestExt;
}

/// Marker trait used in bounds of `Slice{Flat,Nest,Array}Ext`.
///
/// This marks the array types approved for use with `slice_of_array`.
///
/// # Safety
///
/// For any implementation, `Self` must have the same size and
/// alignment as `[Self::Element; Self::LEN]`.  Furthermore, you
/// must be comfortable with the possibility of `[Self]` being
/// reinterpreted bitwise as `[[Self::Element; Self::LEN]]` (or
/// vice versa) in any possible context.
///
/// # Notice
///
/// **Please do NOT use this trait in public interfaces in your code.**
///
/// `slice_of_array` is not yet 1.0, is not ready (or even designed)
/// to be used as a public dependency.
///
/// However, feel free to implement this trait on your own private
/// wrapper types around arrays and/or `#[repr(C)]` structs. (these use
/// cases are explicitly supported because the author does it himself,
/// and quite frankly, it's pretty convenient!)
pub unsafe trait IsSliceomorphic: Sized {
    type Element;
    const LEN: usize;
}

unsafe impl<T, const N: usize> IsSliceomorphic for [T; N] {
    type Element = T;
    const LEN: usize = N;
}

// Validate some known assumptions of IsSliceomorphic "at runtime,"
//  in a manner which should get optimized into thin air.
fn validate_alignment_and_size<V: IsSliceomorphic>() {
    use core::mem::{align_of, size_of};

    assert_eq!(align_of::<V::Element>(), align_of::<V>(),);

    assert_eq!(V::LEN * size_of::<V::Element>(), size_of::<V>(),);
}

/// Permits viewing a slice of arrays as a flat slice.
///
/// # Panics
///
/// Will panic if the new length exceeds `usize::MAX`.
/// (in practice, this can only happen with zero-sized types)
///
/// # Implementors
///
/// The methods are available on `&[[T; n]]` and `&mut [[T; n]]`
/// for all `T` and `n`. Of course, they are also available on
/// `Vec<[T; n]>` and any other type that derefs or unsizes to `[[T; n]]`.
///
/// `&[[T; 0]]` does support being flattened into an empty slice, however,
/// please do mind that the inverse operation [`SliceNestExt::nest`] will panic
/// (as it cannot possibly recover the original length of the slice).
///
/// # Notice
///
/// The existence of this trait is an implementation detail.  Future versions may
/// split it up, merge or rename it.
/// Therefore, **please do NOT use this trait as a generic bound in your code.**
///
/// (Prefer `[V] where V: `[`IsSliceomorphic`]`<Element=T>` instead)
pub trait SliceFlatExt<T> {
    /// View `&[[T; n]]` as `&[T]`.
    fn flat(&self) -> &[T];

    /// View `&mut [[T; n]]` as `&mut [T]`
    fn flat_mut(&mut self) -> &mut [T];
}

/// Permits viewing a slice as a slice of arrays.
///
/// The new array dimension can often be inferred.
/// When it is not, a turbofish can be used: `.nest::<[_; 3]>()`.
///
/// # Panics
///
/// All methods panic if the input length is not divisible by `n`.
///
/// # Implementors
///
/// The methods are available on `&[T]` and `&mut [T]` for all `T`.
/// Of course, they are also available on `Vec<T>` and any other type
/// that derefs or unsizes to `[T]`.
///
/// **The implementation for `N=0` panics!** (even if the length of the slice is
/// zero, as in this case the length of the nested slice would be degenerate)
///
/// # Notice
///
/// The existence of this trait is an implementation detail.  Future versions may
/// split it up, merge or rename it.
/// Therefore, **please do NOT use this trait as a generic bound in your code.**
///
/// (Prefer `<V> where V: `[`IsSliceomorphic`]`<Element=T>` instead)
pub trait SliceNestExt<T> {
    /// View `&[T]` as `&[[T; n]]` without copying.
    fn nest<V: IsSliceomorphic<Element = T>>(&self) -> &[V];

    /// View `&mut [T]` as `&mut [[T; n]]` without copying.
    fn nest_mut<V: IsSliceomorphic<Element = T>>(&mut self) -> &mut [V];
}

/// Permits viewing a slice as an array.
///
/// The output array length can often be inferred.
/// When it is not, a turbofish can be used: `.as_array::<[_; 3]>()`.
///
/// # Panics
///
/// All methods panic if the slice is not exactly the requested length.
///
/// # Implementors
///
/// The methods are available on `&[T]` and `&mut [T]` for all `T`.
/// Of course, they are also available on `Vec<T>` and any other type
/// that derefs or unsizes to `[T]`.
///
/// # Notice
///
/// The existence of this trait is an implementation detail.  Future versions may
/// split it up, merge or rename it.
/// Therefore, **please do NOT use this trait as a generic bound in your code.**
///
/// (Prefer `V where V: `[`IsSliceomorphic`]`<Element=T>` instead)
pub trait SliceArrayExt<T> {
    /// View `&[T]` as `&[T; n]`.
    fn as_array_ref<V: IsSliceomorphic<Element = T>>(&self) -> &V;

    /// View `&mut [T]` as `&mut [T; n]`.
    fn as_mut_array_ref<V: IsSliceomorphic<Element = T>>(&mut self) -> &mut V;

    /// Clone `&[T]` to `[T; n]`.
    ///
    /// This is provided because `.as_array().clone()` tends to cause trouble for
    /// type inference.
    fn to_array<V>(&self) -> V
    where
        V: Clone + IsSliceomorphic<Element = T>,
    {
        self.as_array_ref::<V>().clone()
    }
}

impl<V: IsSliceomorphic> SliceFlatExt<V::Element> for [V] {
    fn flat(&self) -> &[V::Element] {
        let new_len = checked_compute_flattened_len::<V>(self.len());

        // UNSAFETY: (::core::slice::from_raw_parts)
        // - pointer must be non-null (even for zero-length)
        // - pointer must be aligned
        // - pointer must be valid for given size
        // - lifetimes are unchecked
        unsafe { slice::from_raw_parts(self.as_ptr() as *const _, new_len) }
    }

    fn flat_mut(&mut self) -> &mut [V::Element] {
        let new_len = checked_compute_flattened_len::<V>(self.len());

        // UNSAFETY: (::core::slice::from_raw_parts_mut)
        // - pointer must be non-null (even for zero-length)
        // - pointer must be aligned
        // - pointer must be valid for given size
        // - lifetimes are unchecked
        // - aliasing guarantees of &mut are unchecked
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr() as *mut _, new_len) }
    }
}

#[inline(always)]
fn checked_compute_flattened_len<V: IsSliceomorphic>(len: usize) -> usize {
    validate_alignment_and_size::<V>();

    if core::mem::size_of::<V::Element>() == 0 {
        usize::checked_mul(len, V::LEN).expect("overflow when computing length of flattened array")
    } else {
        // Given that each value occupies at least one byte, the mere existence
        // of the slice ensures that this will not overflow.
        len * V::LEN
    }
}

impl<T> SliceNestExt<T> for [T] {
    fn nest<V: IsSliceomorphic<Element = T>>(&self) -> &[V] {
        let new_len = checked_compute_nested_len::<V>(self.len(), "&");

        // UNSAFETY: (core::slice::from_raw_parts)
        // - pointer must be non-null (even for zero-length)
        // - pointer must be aligned
        // - pointer must be valid for given size
        // - lifetimes are unchecked
        unsafe { slice::from_raw_parts(self.as_ptr() as *const _, new_len) }
    }

    fn nest_mut<V: IsSliceomorphic<Element = T>>(&mut self) -> &mut [V] {
        let new_len = checked_compute_nested_len::<V>(self.len(), "&mut ");

        // UNSAFETY: (core::slice::from_raw_parts_mut)
        // - pointer must be non-null (even for zero-length)
        // - pointer must be aligned
        // - pointer must be valid for given size
        // - lifetimes are unchecked
        // - aliasing guarantees of &mut are unchecked
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr() as *mut _, new_len) }
    }
}

#[inline(always)]
fn checked_compute_nested_len<V: IsSliceomorphic>(len: usize, prefix: &str) -> usize {
    validate_alignment_and_size::<V>();
    assert_ne!(0, V::LEN, "cannot nest arrays of length 0",);
    assert_eq!(
        0,
        len % V::LEN,
        "cannot view slice of length {} as {}[[_; {}]]",
        len,
        prefix,
        V::LEN,
    );

    len / V::LEN
}

impl<T> SliceArrayExt<T> for [T] {
    fn as_array_ref<V: IsSliceomorphic<Element = T>>(&self) -> &V {
        validate_as_array_assumptions::<V>(self.len(), "&");

        // &self.nest()[0]  // <-- would not work for V::LEN = 0

        // UNSAFETY: (<*const T>::as_ref)
        // - pointer must be aligned
        // - pointer must be valid for given size
        // - lifetimes are unchecked
        unsafe { (self.as_ptr() as *const V).as_ref().unwrap() }
    }

    fn as_mut_array_ref<V: IsSliceomorphic<Element = T>>(&mut self) -> &mut V {
        validate_as_array_assumptions::<V>(self.len(), "&mut ");

        // &mut self.nest_mut()[0]  // <-- would not work for V::LEN = 0

        // UNSAFETY: (<*mut T>::as_mut)
        // - pointer must be aligned
        // - pointer must be valid for given size
        // - lifetimes are unchecked
        // - aliasing guarantees of &mut are unchecked
        unsafe { (self.as_mut_ptr() as *mut V).as_mut().unwrap() }
    }
}

#[inline(always)]
fn validate_as_array_assumptions<V: IsSliceomorphic>(len: usize, prefix: &str) {
    validate_alignment_and_size::<V>();
    assert_eq!(
        len,
        V::LEN,
        "cannot view slice of length {} as {}[_; {}]",
        len,
        prefix,
        V::LEN,
    );
}

#[cfg(test)]
mod tests {
    pub use super::prelude::*;

    #[test]
    fn inference_lattice() {
        // Checks that chaining nest().nest() or nest().as_array()
        // can be done without explicit annotations on the first method call.
        let v: &mut [()] = &mut [(); 9];

        {
            let _: &[[(); 3]; 3] = v.nest().as_array_ref();
        }
        {
            let _: &[[[(); 3]; 3]] = v.nest().nest();
        }
        {
            let _: &mut [[(); 3]; 3] = v.nest_mut().as_mut_array_ref();
        }
        {
            let _: &mut [[[(); 3]; 3]] = v.nest_mut().nest_mut();
        }
        {
            let _: [[(); 3]; 3] = v.nest().to_array();
        }
        {
            let _: Vec<[(); 3]> = v.nest().to_vec();
        }
    }

    #[test]
    fn test_flat_zst_and_non_zst() {
        let v: &mut [_] = &mut [[(); 234]; 456];
        assert_eq!(v.flat(), &[(); 234 * 456] as &[()]);
        assert_eq!(v.flat_mut(), &[(); 234 * 456] as &[()]);

        let v: &mut [_] = &mut [[1; 23]; 45];
        assert_eq!(v.flat(), &[1; 23 * 45] as &[i32]);
        assert_eq!(v.flat_mut(), &[1; 23 * 45] as &[i32]);
    }

    #[test]
    fn test_flat_zero() {
        let v: &mut [[(); 0]] = &mut [[(); 0]; 6];
        assert_eq!(v.flat(), &[] as &[()]);
        assert_eq!(v.flat_mut(), &[] as &[()]);
    }

    #[test]
    fn test_array_zero() {
        let v: &mut [[(); 0]] = &mut [[], [], [], []];
        assert_eq!(v.flat(), &[] as &[()]);
        assert_eq!(v.flat_mut(), &[] as &[()]);
    }

    mod failures {
        use super::super::*;

        // Two usizes that overflow when multiplied together.
        const BIG_1: usize = 0x30;
        const BIG_2: usize = usize::MAX >> 4;

        #[test]
        #[should_panic(expected = "overflow when computing length")]
        fn flat_zst_overflow() {
            let v: &[_] = &[[(); BIG_1]; BIG_2];
            let _: &[()] = v.flat();
        }

        #[test]
        #[should_panic(expected = "overflow when computing length")]
        fn flat_mut_zst_overflow() {
            let v: &mut [_] = &mut [[(); BIG_1]; BIG_2];
            let _: &mut [()] = v.flat_mut();
        }

        #[test]
        #[should_panic(expected = "cannot view slice of length 8")]
        fn nest_not_multiple() {
            let v: &[_] = &[(); 8];
            let _: &[[(); 3]] = v.nest();
        }

        #[test]
        #[should_panic(expected = "cannot view slice of length 8")]
        fn nest_mut_not_multiple() {
            let v: &mut [_] = &mut [(); 8];
            let _: &mut [[(); 3]] = v.nest_mut();
        }

        #[test]
        #[should_panic(expected = "cannot nest arrays of length 0")]
        fn nest_zero() {
            let v: &[_] = &[(); 0];
            let _: &[[(); 0]] = v.nest();
        }

        #[test]
        #[should_panic(expected = "cannot nest arrays of length 0")]
        fn nest_mut_zero() {
            let v: &mut [_] = &mut [(); 0];
            let _: &mut [[(); 0]] = v.nest_mut();
        }

        // bad array size tests;
        //  we try converting slices of length 1 or 6 into a length 3 array.
        //  These sizes were chosen to catch accidental acceptance in
        //    the case of sizes that divide evenly
        #[test]
        #[should_panic(expected = "cannot view slice of length 1")]
        fn as_array_too_small() {
            let v: &[_] = &[(); 1];
            let _: &[(); 3] = v.as_array_ref();
        }

        #[test]
        #[should_panic(expected = "cannot view slice of length 6")]
        fn as_array_too_large() {
            let v: &[_] = &[(); 6];
            let _: &[(); 3] = v.as_array_ref();
        }

        #[test]
        #[should_panic(expected = "cannot view slice of length 6")]
        fn as_array_bad_zero() {
            let v: &[_] = &[(); 6];
            let _: &[(); 0] = v.as_array_ref();
        }

        #[test]
        #[should_panic(expected = "cannot view slice of length 1")]
        fn as_mut_array_too_small() {
            let v: &mut [_] = &mut [(); 1];
            let _: &mut [(); 3] = v.as_mut_array_ref();
        }

        #[test]
        #[should_panic(expected = "cannot view slice of length 6")]
        fn as_mut_array_too_large() {
            let v: &mut [_] = &mut [(); 6];
            let _: &mut [(); 3] = v.as_mut_array_ref();
        }

        #[test]
        #[should_panic(expected = "cannot view slice of length 6")]
        fn as_mut_array_bad_zero() {
            let v: &mut [_] = &mut [(); 6];
            let _: &[(); 0] = v.as_mut_array_ref();
        }
    }
}
