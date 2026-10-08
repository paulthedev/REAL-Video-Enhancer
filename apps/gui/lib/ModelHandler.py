import os
import re

from apps.gui.util import createDirectory, log
from apps.gui.constants import CUSTOM_MODELS_PATH

"""
Key value pairs of the model name in the GUI
Data inside the tuple:
[0] = file in models directory
[1] = file to download
[2] = upscale times
[3] = arch
"""

onnxInterpolateModels = {
    "RIFE 4.22 (Recommended Model)": (
        "rife422_v2_ensembleFalse_op20_clamp.onnx",
        "rife422_v2_ensembleFalse_op20_clamp.onnx",
        1,
        "rife422-lite",
    ),
}
onnxUpscaleModels = {
    "SPAN (Animation) (2X)": (
        "2x_ModernSpanimationV2_clamp_op20.onnx",
        "2x_ModernSpanimationV2_clamp_op20.onnx",
        2,
        "SPAN",
    ),
    "TSPAN (Temporal) (2X) (Slow)": (
        "tspan.onnx",
        "tspan.onnx",
        2,
        "TSPAN",
    ),
}

# detect custom models (ONNX only)
createDirectory(CUSTOM_MODELS_PATH)
customOnnxUpscaleModels = {}
for model in os.listdir(CUSTOM_MODELS_PATH):
    if model.endswith(".onnx"):
        customOnnxUpscaleModels[model] = (model, model, 4, "custom")

onnxUpscaleModels = onnxUpscaleModels | customOnnxUpscaleModels

totalModels = onnxInterpolateModels | onnxUpscaleModels

def getModels():
    """
    Returns the ONNX model tables used to populate the GUI model comboboxes
    [interpolate, upscale, deblur, denoise, decompress, scene change].

    ONNX is the only backend; categories without converted models are empty.
    """
    return (
        onnxInterpolateModels,
        onnxUpscaleModels,
        {},  # deblur
        {},  # denoise
        {},  # decompress
        {},  # scene change (pyscenedetect needs no model)
    )


def getModelDisplayName(model: str):
    try:
        cutPosition = model.index(" (") if " (" in model else len(model)
        onlyName = model[:cutPosition]
        return onlyName.lower().replace(' ', '-')
    except:
        return (model[0] + model[1]) if len(model) >= 2 else ""
