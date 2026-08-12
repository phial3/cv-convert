use anyhow::{Context, Result};
use rsmpeg::avutil::AVFrame;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;

/// Convert an AVFrame
pub fn convert_avframe(
    src_frame: &AVFrame,
    dst_width: i32,
    dst_height: i32,
    dst_pix_fmt: i32,
) -> Result<AVFrame> {
    /*
     * Scaler selection options. Only one may be active at a time.
     */
    // SWS_FAST_BILINEAR = 1 <<  0, ///< fast bilinear filtering
    // SWS_BILINEAR      = 1 <<  1, ///< bilinear filtering
    // SWS_BICUBIC       = 1 <<  2, ///< 2-tap cubic B-spline
    // SWS_X             = 1 <<  3, ///< experimental
    // SWS_POINT         = 1 <<  4, ///< nearest neighbor
    // SWS_AREA          = 1 <<  5, ///< area averaging
    // SWS_BICUBLIN      = 1 <<  6, ///< bicubic luma, bilinear chroma
    // SWS_GAUSS         = 1 <<  7, ///< gaussian approximation
    // SWS_SINC          = 1 <<  8, ///< unwindowed sinc
    // SWS_LANCZOS       = 1 <<  9, ///< 3-tap sinc/sinc
    // SWS_SPLINE        = 1 << 10, ///< cubic Keys spline

    /*
     * Return an error on underspecified conversions. Without this flag,
     * unspecified fields are defaulted to sensible values.
     */
    // SWS_STRICT        = 1 << 11,

    /*
     * Emit verbose log of scaling parameters.
     */
    // SWS_PRINT_INFO    = 1 << 12,

    /*
     * Perform full chroma upsampling when upscaling to RGB.
     *
     * For example, when converting 50x50 yuv420p to 100x100 rgba, setting this flag
     * will scale the chroma plane from 25x25 to 100x100 (4:4:4), and then convert
     * the 100x100 yuv444p image to rgba in the final output step.
     *
     * Without this flag, the chroma plane is instead scaled to 50x100 (4:2:2),
     * with a single chroma sample being re-used for both of the horizontally
     * adjacent RGBA output pixels.
     */
    // SWS_FULL_CHR_H_INT = 1 << 13,

    /*
     * Perform full chroma interpolation when downscaling RGB sources.
     *
     * For example, when converting a 100x100 rgba source to 50x50 yuv444p, setting
     * this flag will generate a 100x100 (4:4:4) chroma plane, which is then
     * downscaled to the required 50x50.
     *
     * Without this flag, the chroma plane is instead generated at 50x100 (dropping
     * every other pixel), before then being downscaled to the required 50x50
     * resolution.
     */
    // SWS_FULL_CHR_H_INP = 1 << 14,

    /*
     * Force bit-exact output. This will prevent the use of platform-specific
     * optimizations that may lead to slight difference in rounding, in favor
     * of always maintaining exact bit output compatibility with the reference
     * C code.
     *
     * Note: It is recommended to set both of these flags simultaneously.
     */
    // SWS_ACCURATE_RND   = 1 << 18,
    // SWS_BITEXACT       = 1 << 19,

    // 考虑性能和质量平衡
    let flags =
        ffi::SWS_BICUBIC | ffi::SWS_FULL_CHR_H_INT | ffi::SWS_ACCURATE_RND | ffi::SWS_BITEXACT;

    // 创建转换上下文
    let mut sws_ctx = SwsContext::get_context(
        src_frame.width,
        src_frame.height,
        src_frame.format,
        dst_width,
        dst_height,
        dst_pix_fmt,
        flags,
        None,
        None,
        None,
    )
    .context("Failed to create a swscale context.")?;

    // 创建目标缓冲区
    let mut dst_frame = AVFrame::new();
    dst_frame.set_width(dst_width);
    dst_frame.set_height(dst_height);
    dst_frame.set_format(dst_pix_fmt);
    dst_frame.set_pts(src_frame.pts);
    dst_frame.set_time_base(src_frame.time_base);
    dst_frame.set_pict_type(src_frame.pict_type);
    dst_frame.set_ch_layout(src_frame.ch_layout);
    dst_frame.set_nb_samples(src_frame.nb_samples);
    dst_frame.set_sample_rate(src_frame.sample_rate);
    dst_frame
        .alloc_buffer()
        .context("Failed to allocate the buffer for the frame.")?;

    sws_ctx
        .scale_frame(src_frame, 0, src_frame.height, &mut dst_frame)
        .context(format!(
            "Failed to scale frame from [pix:{}, size:{}x{}] to [pix:{}, size:{}x{}]",
            src_frame.format, src_frame.width, src_frame.height, dst_pix_fmt, dst_width, dst_height
        ))?;

    Ok(dst_frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsmpeg::avutil::AVFrame;

    fn make_gray_frame(width: i32, height: i32, val: u8) -> AVFrame {
        let mut frame = AVFrame::new();
        frame.set_width(width);
        frame.set_height(height);
        frame.set_format(ffi::AV_PIX_FMT_GRAY8);
        frame.alloc_buffer().unwrap();

        let linesize = frame.linesize[0] as usize;
        for y in 0..height as usize {
            let row = unsafe {
                std::slice::from_raw_parts_mut(frame.data[0].add(y * linesize), width as usize)
            };
            row.fill(val);
        }
        frame
    }

    #[test]
    fn convert_gray_to_gray() {
        let src = make_gray_frame(4, 3, 100);
        let dst = convert_avframe(&src, 4, 3, ffi::AV_PIX_FMT_GRAY8).unwrap();
        assert_eq!(dst.width, 4);
        assert_eq!(dst.height, 3);
        assert_eq!(dst.format, ffi::AV_PIX_FMT_GRAY8);

        let linesize = dst.linesize[0] as usize;
        let data =
            unsafe { std::slice::from_raw_parts(dst.data[0], linesize * dst.height as usize) };
        for y in 0..dst.height as usize {
            for x in 0..dst.width as usize {
                assert_eq!(data[y * linesize + x], 100);
            }
        }
    }

    #[test]
    fn convert_yuv420p_to_rgb24() {
        // 构造一个 2x2 YUV420P（全白：Y=235, U=V=128），手动设置数据指针与 linesize
        let mut src = AVFrame::new();
        src.set_width(2);
        src.set_height(2);
        src.set_format(ffi::AV_PIX_FMT_YUV420P);

        unsafe {
            let mut_frame = src.as_mut_ptr();
            let y_ptr = Box::new(vec![235u8; 4]);
            (*mut_frame).data[0] = Box::leak(y_ptr).as_mut_ptr();
            (*mut_frame).linesize[0] = 2;
            let u_ptr = Box::new(vec![128u8; 1]);
            (*mut_frame).data[1] = Box::leak(u_ptr).as_mut_ptr();
            (*mut_frame).linesize[1] = 1;
            let v_ptr = Box::new(vec![128u8; 1]);
            (*mut_frame).data[2] = Box::leak(v_ptr).as_mut_ptr();
            (*mut_frame).linesize[2] = 1;
        }

        let dst = convert_avframe(&src, 2, 2, ffi::AV_PIX_FMT_RGB24).unwrap();
        assert_eq!(dst.format, ffi::AV_PIX_FMT_RGB24);
        // 全白在 RGB24 下应为 (255,255,255)（ffmpeg 有限->全范围扩展）
        let linesize = dst.linesize[0] as usize;
        let data =
            unsafe { std::slice::from_raw_parts(dst.data[0], linesize * dst.height as usize) };
        for y in 0..dst.height as usize {
            for x in 0..dst.width as usize {
                let pos = y * linesize + x * 3;
                assert_eq!(&data[pos..pos + 3], &[255, 255, 255]);
            }
        }
    }
}
