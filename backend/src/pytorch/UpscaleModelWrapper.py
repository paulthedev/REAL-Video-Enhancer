import torch
from ..utils.Util import log
from .TorchUtils import TorchUtils
from .VSRArchs.AnimeSR.vsr_inference_helper import AnimeSRInferenceHelper
from .VSRArchs.AnimeSR.animesr_arch import AnimeSR
from .VSRArchs.TSPAN.vsr_inference_helper import TemporalSPANInferenceHelper
from .VSRArchs.TSPAN.tspan import TemporalSPAN
class UpscaleModelWrapper:
    def __init__(self, model_path: torch.nn.Module, device: torch.device, precision: torch.dtype):
        self.__model_path = model_path
        self.__device = device
        self.__precision = precision
        self.__dummy_input_pre_channels = None
        self.__channels = 3
        self.__inference_mode = None
        self.load_model()
        self.set_precision(self.__precision)
        self.__test_model_precision()

    def set_precision(self, precision: torch.dtype):
        self.__precision = precision
        self.__model.to(self.__device, dtype=precision)

    def get_model(self):
        return self.__model
    
    def get_scale(self):
        return self.__scale

    def load_state_dict(self, state_dict):
        self.__model.load_state_dict(state_dict)

    def __test_inference(self, test_input:torch.Tensor):
        # inference and get re-load state dict due to issue with span.
        with torch.inference_mode():
            model = self.inference_helper
            model.eval()
            model(test_input)
            output = model(test_input)
            self.__model.load_state_dict(model.state_dict()) # reload state dict to fix span
            TorchUtils.clear_cache()
            del model

    def __test_model_precision(self):
        test_input = torch.randn(1, 3, 64, 64).to(self.__device, dtype=self.__precision)
        with torch.inference_mode():
            try:
                self.__test_inference(test_input)
            except Exception as e:
                log(f"Model precision {self.__precision} not supported, falling back to float32: {e}")
                self.set_precision(torch.float32)
                test_input = test_input.to(dtype=torch.float32)  # must match the model's new dtype
                self.__test_inference(test_input)
    
    def get_dummy_input(self, width: int, height: int) -> torch.Tensor:
        assert self.__dummy_input_pre_channels is not None, "Dummy input pre channels not set."
        dummy_input = self.__dummy_input_pre_channels.copy()
        dummy_input.append(self.__channels)
        dummy_input.append(height)
        dummy_input.append(width)
        return torch.zeros(dummy_input, dtype=self.__precision, device=self.__device)
    
    def load_model(self, model=None) -> torch.nn.Module:
        # NOTE: This must NOT be run under inference_mode, otherwise the
        # freshly built model's parameters become inference tensors and the
        # load_state_dict calls below fail on GPU with
        # "Inference tensors do not track version counter".
        # Inference itself is wrapped in inference_mode at call sites.
        if not model:
            from .spandrel import ModelLoader, ImageModelDescriptor, UnsupportedModelError
            try:
                model = ModelLoader().load_from_file(self.__model_path)
                assert isinstance(model, ImageModelDescriptor)
                self.__scale = model.scale
                model = model.model
                model.eval()  # Fix BatchNorm training mode issue for neosr-trained models
                self.__model = model
                self.inference_helper = self.__model
                self.__dummy_input_pre_channels = [1]
                self.__inference_mode = 'spandrel'

            except (UnsupportedModelError) as e:
                try:
                    model = AnimeSR()
                    # dummy attributes
                    self.__scale = 4
                    state_dict = torch.load(self.__model_path, map_location=self.__device)
                    model.load_state_dict(state_dict=state_dict)
                    self.__model = model.to(self.__device, dtype=self.__precision)
                    self.inference_helper = AnimeSRInferenceHelper(model=self.__model, scale=self.__scale)
                    self.__dummy_input_pre_channels = [3]
                    self.__inference_mode = 'animesr'
                except Exception as e:
                    try:
                        self.__scale = 2
                        self.__dummy_input_pre_channels = [1, 5,]
                        model = TemporalSPAN(upscale=self.__scale)
                        state_dict = torch.load(self.__model_path, map_location=self.__device)
                        model.load_state_dict(state_dict=state_dict['params_ema'], strict=False)
                        self.__model = model.to(self.__device, dtype=self.__precision)
                        self.inference_helper = TemporalSPANInferenceHelper(model=self.__model, scale=self.__scale)
                        self.__inference_mode = 'tspan'
                    except Exception as e:
                        log(f"Model at {self.__model_path} is not supported: {e}")
                        raise e
        else:
            if self.__inference_mode == 'spandrel':
                self.inference_helper = model
            elif self.__inference_mode == 'animesr':
                self.inference_helper = AnimeSRInferenceHelper(model=model, scale=self.__scale)
            elif self.__inference_mode == 'tspan':
                self.inference_helper = TemporalSPANInferenceHelper(model=model, scale=self.__scale)
    def enable_compile(self) -> torch.nn.Module | object | None:
        """Wrap the inference helper in torch.compile for faster steady-state
        inference.

        Only plain spandrel models are eligible: they are stateless nn.Modules
        whose forward is fully traceable at eval time. The AnimeSR/TSPAN helpers
        hold temporal buffers and custom Python control flow that don't trace
        reliably, so they keep running eager.

        Returns the previous helper (call restore_inference_helper with it to
        fall back), or None if this mode was not eligible for compilation."""
        if self.__inference_mode != 'spandrel':
            return None
        prev = self.inference_helper
        # Static shapes: the render pipeline feeds fixed tile/full-frame sizes.
        # Graph breaks are allowed (some archs have data-dependent control
        # flow); they still get kernel fusion for the traced regions.
        self.inference_helper = torch.compile(prev, dynamic=False)
        return prev

    def restore_inference_helper(self, helper):
        """Put the inference helper back (e.g. after a failed compile warmup)."""
        if helper is not None:
            self.inference_helper = helper

    def __call__(self, *args, **kwargs):
        assert self.inference_helper is not None, "Inference helper is not initialized."
        return self.inference_helper(*args, **kwargs).clone()