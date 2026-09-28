"""
Base classes for backend-agnostic model definitions.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple


class ModelTask(Enum):
    """Types of model tasks."""
    INTERPOLATE = "interpolate"
    UPSCALE = "upscale"
    RESTORATION = "restoration"
    SCENE_DETECT = "scene_detect"


class ModelFormat(Enum):
    """Model file formats."""
    SAFETENSORS = "safetensors"
    PT = "pt"
    ONNX = "onnx"
    NCNN = "ncnn"


@dataclass
class ModelInfo:
    """Metadata about a model."""
    name: str
    architecture: str
    task: ModelTask
    scale: Optional[int] = None  # For upscale models
    supported_formats: list[ModelFormat] = None
    supported_backends: list[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [ModelFormat.SAFETENSORS, ModelFormat.PT]
        if self.supported_backends is None:
            self.supported_backends = ["pytorch"]


class BaseModel(ABC):
    """
    Abstract base class for all models.
    
    Models should be backend-agnostic - they define the architecture
    and weights, but not how they're executed.
    """
    
    def __init__(self, name: str, architecture: str, task: ModelTask):
        self.name = name
        self.architecture = architecture
        self.task = task
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass of the model.
        
        Returns:
            Model output(s)
        """
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape."""
        pass
    
    @abstractmethod
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return the output shape for a given input shape."""
        pass
    
    def get_info(self) -> ModelInfo:
        """Return model metadata."""
        return ModelInfo(
            name=self.name,
            architecture=self.architecture,
            task=self.task,
        )


class BaseInterpolateModel(BaseModel):
    """Base class for interpolation models."""
    
    def __init__(self, name: str, architecture: str):
        super().__init__(name, architecture, ModelTask.INTERPOLATE)
    
    @abstractmethod
    def interpolate(
        self,
        frame1: Any,
        frame2: Any,
        timestep: float = 0.5
    ) -> Any:
        """
        Interpolate between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            timestep: Interpolation factor (0.0 to 1.0)
        
        Returns:
            Interpolated frame
        """
        pass


class BaseUpscaleModel(BaseModel):
    """Base class for upscaling models."""
    
    def __init__(self, name: str, architecture: str, scale: int = 2):
        self.scale = scale
        super().__init__(name, architecture, ModelTask.UPSCALE)
    
    @abstractmethod
    def upscale(self, frame: Any) -> Any:
        """
        Upscale a frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Upscaled frame
        """
        pass
    
    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return the output shape for a given input shape."""
        if len(input_shape) == 3:
            # H, W, C
            return (input_shape[0] * self.scale, input_shape[1] * self.scale, input_shape[2])
        elif len(input_shape) == 4:
            # N, C, H, W
            return (input_shape[0], input_shape[1], input_shape[2] * self.scale, input_shape[3] * self.scale)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")


class BaseRestorationModel(BaseModel):
    """Base class for restoration models."""
    
    def __init__(self, name: str, architecture: str):
        super().__init__(name, architecture, ModelTask.RESTORATION)
    
    @abstractmethod
    def restore(self, frame: Any) -> Any:
        """
        Restore a frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Restored frame
        """
        pass


class BaseSceneDetectModel(BaseModel):
    """Base class for scene detection models."""
    
    def __init__(self, name: str, architecture: str):
        super().__init__(name, architecture, ModelTask.SCENE_DETECT)
    
    @abstractmethod
    def detect(self, frame: Any, prev_frame: Optional[Any] = None) -> bool:
        """
        Detect scene change.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame (optional)
        
        Returns:
            True if scene change detected
        """
        pass
