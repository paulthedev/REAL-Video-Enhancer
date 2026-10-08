"""
Regression tests for the bundled imageio-ffmpeg binary.

The app pipes rawvideo into and out of ffmpeg (FFmpegRead / FFmpegWrite);
these tests lock that contract against the static ffmpeg build that
imageio-ffmpeg ships, so a bundling regression can't go unnoticed.
"""

import imageio_ffmpeg
import numpy as np
import pytest
import subprocess

from apps.backend.utils.Util import subprocess_popen_without_terminal

pytestmark = pytest.mark.skipif(
    imageio_ffmpeg.get_ffmpeg_exe() is None,
    reason="imageio-ffmpeg has no bundled binary for this platform",
)


@pytest.fixture
def ffmpeg():
    return imageio_ffmpeg.get_ffmpeg_exe()


def _random_frames(count, width, height, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (count, height, width, 3), dtype=np.uint8)


def _run(ffmpeg, args, stdin_bytes):
    process = subprocess_popen_without_terminal(
        [ffmpeg, "-loglevel", "error", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = process.communicate(stdin_bytes)
    assert process.returncode == 0, err.decode()
    return out


def test_rawvideo_passthrough_is_lossless(ffmpeg):
    """rgb24 stdin -> rawvideo stdout round-trips byte-for-byte."""
    frames = _random_frames(3, 32, 32)
    out = _run(
        ffmpeg,
        [
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x32", "-r", "8",
            "-i", "-",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        frames.tobytes(),
    )
    assert out == frames.tobytes()


def test_rawvideo_scale_filter(ffmpeg):
    """The scale filter path works: 32x32 rgb24 in -> 64x64 rgb24 out."""
    frames = _random_frames(2, 32, 32)
    out = _run(
        ffmpeg,
        [
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x32", "-r", "8",
            "-i", "-",
            "-vf", "scale=64:64",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        frames.tobytes(),
    )
    assert len(out) == 2 * 64 * 64 * 3


def test_image2pipe_rawvideo_is_headerless(ffmpeg):
    """FFmpegRead reads fixed-size chunks from `image2pipe`+rawvideo; the
    muxer must emit raw pixels with no per-frame PPM headers."""
    frames = _random_frames(3, 8, 8)
    out = _run(
        ffmpeg,
        [
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "8x8", "-r", "8",
            "-i", "-",
            "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo",
            "-",
        ],
        frames.tobytes(),
    )
    assert out == frames.tobytes()


def test_encode_h264_then_decode_frame_count(ffmpeg, tmp_path):
    """A rawvideo -> libx264 -> mp4 encode decodes back to the same frames."""
    count, width, height = 8, 64, 64
    frames = _random_frames(count, width, height, seed=7)
    video = tmp_path / "in.mp4"
    process = subprocess_popen_without_terminal(
        [
            ffmpeg, "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", "8", "-i", "-",
            "-pix_fmt", "yuv420p", "-c:v", "libx264",
            "-movflags", "+faststart", "-y", str(video),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _, err = process.communicate(frames.tobytes())
    assert process.returncode == 0, err.decode()

    out = _run(
        ffmpeg,
        ["-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        b"",
    )
    assert len(out) == count * width * height * 3
