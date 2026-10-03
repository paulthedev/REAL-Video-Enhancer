"""Model download service.

Downloads a model from the RVE model releases and installs it for
pytorch/onnx/ncnn into its backend directory.
"""

import os

from apps.gui.constants import MODELS_PATH, MODELS_RELEASE_BASE_URL
from apps.gui.ui.QTcustom import DownloadProgressPopup
from apps.gui.util import createDirectory, extractTarGZ, networkCheck


class DownloadModel:
    """
    Takes in the name of a model and the name of the backend in the GUI, and downloads it from a URL
    model: any valid model used by RVE
    backend: the backend used (pytorch, ncnn)
    """

    def __init__(
        self,
        modelFile: str,
        downloadModelFile: str,
        modelPath: str = MODELS_PATH,
    ):
        self.modelFile = modelFile
        self.modelPath = modelPath
        self.downloadModelFile = downloadModelFile
        self.downloadModelPath = os.path.join(modelPath, downloadModelFile)
        createDirectory(modelPath)
        # check if internet doesnt exist, if it doesnt dont allow to be added to queue

    def downloadModel(self) -> bool:
        """
        Downloads a model from the github repo
        If the installation is unsucessful, or interent is unavailable to download, it will return False
        else
        returns True if the model exists, or is sucessfully downloaded
        """
        if os.path.isfile(
            os.path.join(self.modelPath, self.modelFile)
        ) or os.path.exists(os.path.join(self.modelPath, self.modelFile)):
            return True
        if not networkCheck():
            return False
        url = MODELS_RELEASE_BASE_URL + self.downloadModelFile
        title = "Downloading: " + self.downloadModelFile
        DownloadProgressPopup(
            link=url, title=title, downloadLocation=self.downloadModelPath
        )
        print("Done")
        if "tar.gz" in self.downloadModelFile:
            print("Extracting File")
            extractTarGZ(self.downloadModelPath)
        return True


__all__ = ["DownloadModel"]
