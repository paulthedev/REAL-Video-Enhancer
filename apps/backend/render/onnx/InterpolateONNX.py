"""ONNX RIFE-style frame interpolation engine.

Consumes and produces raw rgb24 frame bytes (the render loop converts at
the boundary). Stateless per pair except the cached previous frame.

RIFE ONNX models take img0/img1 (NCHW, 0-1 float) and, on most exports,
a scalar timestep input; the output is the interpolated frame.
"""

import numpy as np

from .session import create_session, prepare_model, resolve_precision


class InterpolateONNX:
    def __init__(
        self,
        modelPath: str,
        interpolateFactor: int = 2,
        width: int = 1920,
        height: int = 1080,
        deviceID: int = 0,
        precision: str = "auto",
        hdr_mode: bool = False,
    ):
        self.width = width
        self.height = height
        self.factor = max(int(interpolateFactor), 1)
        self.hdr_mode = hdr_mode
        self.precision = resolve_precision(precision)
        self.model = prepare_model(modelPath, self.precision)
        self.session = create_session(self.model, deviceID)

        inputs = self.session.get_inputs()
        self.input_names = [i.name for i in inputs]
        # convention: img0, img1 [, timestep]
        self.has_timestep = len(self.input_names) > 2
        self.prev = None  # cached HWC uint8 frame

    def _to_tensor(self, hwc: np.ndarray) -> np.ndarray:
        tensor = hwc.astype(np.float32) / 255.0
        return tensor.transpose(2, 0, 1)[None]

    def _run(self, img0: np.ndarray, img1: np.ndarray, timestep: float) -> bytes:
        feed = {
            self.input_names[0]: self._to_tensor(img0),
            self.input_names[1]: self._to_tensor(img1),
        }
        if self.has_timestep:
            feed[self.input_names[2]] = np.array([[timestep]], dtype=np.float32)
        out = self.session.run(None, feed)[0]
        if out.ndim == 4:
            out = out[0]
        frame = (
            (np.clip(out, 0.0, 1.0).transpose(1, 2, 0) * 255.0)
            .astype(np.uint8)
        )
        return np.ascontiguousarray(frame).tobytes()

    def __call__(self, img1, transition=False):
        """Yield only the IN-BETWEEN frames for the pair (prev, img1) —
        the render loop pushes img1 itself afterwards. On a scene change
        (or the first frame) there is nothing to blend across, so an empty
        iterator is returned (kept truthy: an empty list would end the
        render — the old torch engine returned a generator for this)."""
        fb = img1 if isinstance(img1, bytes) else img1.get_frame_bytes()
        img = np.frombuffer(fb, dtype=np.uint8).reshape(self.height, self.width, 3)
        prev = self.prev
        self.prev = img.copy()
        if prev is None or transition or self.factor <= 1:
            return iter(())
        return iter(
            self._run(prev, img, i / self.factor) for i in range(1, self.factor)
        )
