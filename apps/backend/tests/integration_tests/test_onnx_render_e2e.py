"""
End-to-end render test with a synthetic ONNX model (migration Phase 5).

Instead of downloading a real model, a tiny 2x upscaler is exported in-test
with torch.onnx.export (nearest-neighbor resize, a few KB). A generated
1-second clip is then pushed through the real render path — the rve-backend
CLI, Render -> FFmpegRead -> UpscaleONNX -> FFmpegWrite — and the output is
decoded and compared against the expected nearest-neighbor 2x of the input.

This proves the ONNX inference path actually renders, without any network
or model downloads.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

import imageio_ffmpeg

from apps.backend.utils.Util import subprocess_popen_without_terminal

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir
    )
)
BACKEND_SCRIPT = os.path.join(REPO_ROOT, "apps", "backend", "rve-backend.py")

WIDTH, HEIGHT, FPS, FRAME_COUNT = 64, 64, 8, 8
SCALE = 2


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A minimal 2x upscaler: nearest-neighbor resize, exported to ONNX.

    The filename must contain "2x" — Render derives the model scale from it.
    """
    model_dir = tmp_path_factory.mktemp("model")
    model_path = model_dir / "2x_tiny_nearest_upscale.onnx"

    class Tiny2x(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(
                x, scale_factor=SCALE, mode="nearest"
            )

    dummy = torch.zeros(1, 3, HEIGHT, WIDTH, dtype=torch.float32)
    torch.onnx.export(
        Tiny2x(),
        (dummy,),
        str(model_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,  # the dynamo exporter mangles scalar/time inputs
    )
    assert model_path.exists()
    return model_path


@pytest.fixture(scope="module")
def input_video(tmp_path_factory):
    """A 1-second solid-color clip; each frame a different flat color.

    Solid colors survive yuv420p chroma subsampling losslessly, which keeps
    the output comparison tight.
    """
    render_dir = tmp_path_factory.mktemp("render")
    rng = np.random.default_rng(42)
    frames = np.zeros(
        (FRAME_COUNT, HEIGHT, WIDTH, 3), dtype=np.uint8
    )
    for i in range(FRAME_COUNT):
        frames[i, :, :, :] = rng.integers(30, 226, size=3, dtype=np.uint8)

    raw_path = render_dir / "frames.raw"
    raw_path.write_bytes(frames.tobytes())
    video_path = render_dir / "input.mp4"

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess_popen_without_terminal(
        [
            ffmpeg, "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
            "-i", str(raw_path),
            "-pix_fmt", "yuv420p", "-c:v", "libx264",
            "-movflags", "+faststart", "-y", str(video_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _, err = process.communicate()
    assert process.returncode == 0, err.decode()
    raw_path.unlink()
    return render_dir, video_path, frames


def decode_rawvideo(ffmpeg, video_path, width, height, frame_count):
    """Decode a video to raw rgb24 frames, mirroring FFmpegRead."""
    process = subprocess_popen_without_terminal(
        [
            ffmpeg, "-loglevel", "error",
            "-i", str(video_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo", "-s", f"{width}x{height}", "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = process.communicate(b"")
    assert process.returncode == 0, err.decode()
    expected = frame_count * height * width * 3
    assert len(out) == expected, (
        f"expected {expected} bytes of raw frames, got {len(out)}"
    )
    return np.frombuffer(out, dtype=np.uint8).reshape(
        frame_count, height, width, 3
    )


def test_onnx_upscale_render_end_to_end(tiny_model, input_video):
    render_dir, video_path, source_frames = input_video
    output_path = render_dir / "output.mp4"

    result = subprocess.run(
        [
            sys.executable, BACKEND_SCRIPT,
            "-i", str(video_path),
            "-o", str(output_path),
            "-b", "onnx",
            "--upscale_model", str(tiny_model),
            "--device", "cpu",
            "--precision", "float32",
            "--video_encoder_preset", "libx264",
            "--overwrite",
            "--ffmpeg_path", imageio_ffmpeg.get_ffmpeg_exe(),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=render_dir,
    )
    assert result.returncode == 0, (
        f"backend failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout[-3000:]}\n"
        f"stderr:\n{result.stderr[-3000:]}"
    )
    assert output_path.exists() and output_path.stat().st_size > 1024

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # the reader side: the input clip decodes to the frame count/size RVE
    # was told about
    input_frames = decode_rawvideo(
        ffmpeg, video_path, WIDTH, HEIGHT, FRAME_COUNT
    )
    assert input_frames.shape[0] == FRAME_COUNT

    # the writer side: output is the input nearest-2x'd by the model
    output_frames = decode_rawvideo(
        ffmpeg, output_path, WIDTH * SCALE, HEIGHT * SCALE, FRAME_COUNT
    )
    expected = source_frames.repeat(SCALE, axis=1).repeat(SCALE, axis=2)
    # the input is re-encoded once (read) and the output once (write), so
    # allow a little codec slop around an otherwise exact pixel match
    max_diff = int(np.abs(output_frames.astype(int) - expected.astype(int)).max())
    assert max_diff <= 5, (
        f"output differs from nearest-neighbor 2x input (max diff {max_diff})"
    )


@pytest.fixture(scope="module")
def tiny_interp_model(tmp_path_factory):
    """A minimal 2-input+timestep blender: out = (1-t)*img0 + t*img1.

    At t=0.5 (2x interpolation) the mid frame is the exact 50/50 blend.
    The filename contains no scale tag — interpolation needs none.
    """
    model_dir = tmp_path_factory.mktemp("interp_model")
    model_path = model_dir / "tiny_blend_interp.onnx"

    class Blend(torch.nn.Module):
        def forward(self, img0, img1, timestep):
            t = timestep.view(1, 1, 1, 1)
            return (1.0 - t) * img0 + t * img1

    dummy = torch.zeros(1, 3, HEIGHT, WIDTH, dtype=torch.float32)
    torch.onnx.export(
        Blend(),
        (dummy, dummy, torch.zeros(1, 1)),
        str(model_path),
        input_names=["img0", "img1", "timestep"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,  # the dynamo exporter folds the timestep away
    )
    assert model_path.exists()
    return model_path


def test_onnx_interpolate_render_end_to_end(tiny_interp_model, input_video):
    """2x interpolation through the real CLI: frame count doubles and the
    in-between frames are the 50/50 blends of their neighbours."""
    render_dir, video_path, source_frames = input_video
    output_path = render_dir / "output_interp.mp4"

    result = subprocess.run(
        [
            sys.executable, BACKEND_SCRIPT,
            "-i", str(video_path),
            "-o", str(output_path),
            "-b", "onnx",
            "--interpolate_model", str(tiny_interp_model),
            "--interpolate_factor", "2",
            "--scene_detect_method", "none",
            "--device", "cpu",
            "--precision", "float32",
            "--overwrite",
            "--ffmpeg_path", imageio_ffmpeg.get_ffmpeg_exe(),
        ],
        capture_output=True, text=True, timeout=300, cwd=render_dir,
    )
    assert result.returncode == 0, (
        f"backend failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout[-2500:]}\nstderr:\n{result.stderr[-2500:]}"
    )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # the engine yields the in-betweens; the loop pushes each source frame —
    # the first input has no in-between, so expect 2*N - 1 frames
    out_frames = decode_rawvideo(
        ffmpeg, output_path, WIDTH, HEIGHT, FRAME_COUNT * 2 - 1
    )
    # every even output frame is a source frame; odd frames are blends of
    # their two neighbours (codec slop allowed)
    sources = out_frames[0::2].astype(int)
    max_even = int(np.abs(sources - source_frames.astype(int)).max())
    assert max_even <= 6, f"source frames drifted (max diff {max_even})"
    blends = out_frames[1::2].astype(int)
    a = source_frames[:-1].astype(int)
    b = source_frames[1:].astype(int)
    expected = (a + b) // 2
    max_blend = int(np.abs(blends - expected).max())
    assert max_blend <= 8, f"blends wrong (max diff {max_blend})"
