import numpy as np
import os
import cv2
from collections import deque
import sys
from .PySceneDetectUtils import ContentDetector
from ..utils.Frame import Frame
class BaseDetector:
    def __init__(self, threshold: int = 0):
        pass

    def sceneDetect(self, frame: Frame) -> bool:
        return False

class ModelDetector(BaseDetector):
    def __init__(self, threshold: int = 0, model_path: str = None, model_dtype: str = "float32", model_device: str = "cpu"):
        super().__init__(threshold)
    
class PySceneDetect(BaseDetector):
    def __init__(self, threshold=2):
        self.detector = ContentDetector(
            threshold=threshold * 10, min_scene_len=1
        )  # has to be 1 to stay synced
        self.frameNum = 0

    def sceneDetect(self, frame: Frame):
        frame = cv2.resize(frame.get_np_sdr(), (640, 360))
        frameList = self.detector.process_frame(self.frameNum, frame)
        self.frameNum += 1
        if len(frameList) > 0:
            if self.frameNum != frameList[0] + 1:
                print(
                    f"Transition Mismatch {self.frameNum} is not equal to {frameList[0] + 1}, skipping",
                    file=sys.stderr,
                )
                return False

        return len(frameList) > 0

class ONNXSudoSceneDetect(ModelDetector):
    def __init__(self, threshold=0, model_path="", model_dtype="float32", model_device="cpu", **kwargs):
        from ..models.scene_detect.onnx import ONNXSceneDetectModel, ONNXSceneDetectConfig
        config = ONNXSceneDetectConfig(
            model_path=model_path,
            threshold=threshold,
        )
        self.model = ONNXSceneDetectModel(config=config)
        self.model.load("onnx")
        self.i0 = None
    def sceneDetect(self, frame: Frame):
        frame_np = frame.get_frame_np()
        if self.i0 is None:
            self.i0 = frame_np
            return False
        out = self.model.detect(frame_np, self.i0)
        self.i0 = frame_np
        return out

class SceneDetect:
    """
    Class to detect scene changes based on a few parameters
    sceneChangeSsensitivity: This dictates the sensitivity where a scene detect between frames is activated
        - Lower means it is more suseptable to triggering a scene change
        -
    """

    def __init__(
        self,
        sceneChangeMethod: str = "pyscenedetect",
        sceneChangeSensitivity: float = 2.0,
        width: int = 1920,
        height: int = 1080,
        model_path: str = None,
        model_backend: str = "pytorch",
        model_dtype: str = "float32",
        model_device: str = "cpu",
        model_gpu_id: int = 0,
    ):
        self.width = width
        self.height = height
        self.sceneChangeMethod = sceneChangeMethod.lower()
        scmethoddict = {
            "pyscenedetect": PySceneDetect,
            "none": BaseDetector,
        }

        assert self.sceneChangeMethod in scmethoddict or model_path is not None, "Invalid Scene Change Method"
        if self.sceneChangeMethod in scmethoddict:
            self.detector: BaseDetector = scmethoddict[self.sceneChangeMethod](
                threshold=sceneChangeSensitivity
            )
        else:
            assert model_path is not None and os.path.exists(model_path),  "Model path must be provided for model-based scene detection. Please pass --scene_detect_model parameter"
            assert model_backend == "onnx", (
                "Model-based scene detection is ONNX-only"
            )
            self.detector: ModelDetector = ONNXSudoSceneDetect(
                threshold=sceneChangeSensitivity,
                model_path=model_path,
                model_dtype=model_dtype,
                model_device=model_device,
                model_backend=model_backend,
                model_gpu_id=model_gpu_id,
            )

    def detect(self, frame: Frame) -> bool:
        return self.detector.sceneDetect(frame)
