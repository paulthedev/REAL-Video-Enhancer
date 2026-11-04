"""Models page: custom model import via drag-and-drop or file picker.

- .onnx files are imported as-is into the custom models folder
- .pth / .safetensors are loaded with spandrel (architecture auto-detected
  from the state dict), exported to ONNX on the CPU, then imported — so
  PyTorch checkpoints become first-class upscalers without a backend swap
"""

import os
import shutil

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from apps.gui.constants import CUSTOM_MODELS_PATH, IS_FLATPAK, PLATFORM, USE_LOCAL_BACKEND, CWD
from apps.gui.util import FileHandler, log


class ConvertThread(QThread):
    """Converts a PyTorch checkpoint (.pth/.safetensors) to ONNX off the UI
    thread, using spandrel for architecture detection."""

    finished_ok = Signal(str)   # output path
    failed = Signal(str)

    def __init__(self, source_path, output_path):
        super().__init__()
        self.source_path = source_path
        self.output_path = output_path

    def run(self):
        try:
            import torch
            from spandrel import ModelLoader

            descriptor = ModelLoader().load_from_file(self.source_path)
            model = descriptor.model.eval()
            scale = max(int(getattr(descriptor, "scaling", 2) or 2), 1)
            in_ch = getattr(descriptor, "input_channels", 3) or 3

            stem = os.path.splitext(os.path.basename(self.source_path))[0]
            out = os.path.join(CUSTOM_MODELS_PATH, f"{scale}x_{stem}.onnx")

            torch.onnx.export(
                model,
                (torch.zeros(1, in_ch, 64, 64, dtype=torch.float32),),
                out,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"},
                },
                opset_version=17,
                dynamo=False,
            )
            self.finished_ok.emit(out)
        except Exception as e:
            log(f"Model conversion failed: {e}")
            self.failed.emit(str(e))


class DropZone(QFrame):
    """Drag-and-drop target plus click-to-browse for model files."""

    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(170)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        icon = QLabel()
        icon.setPixmap(QIcon(":/icons/icons/upload-cloud.svg").pixmap(40, 40))
        icon.setAlignment(Qt.AlignCenter)
        lay.addWidget(icon)
        title = QLabel("Drop model files here")
        title.setAlignment(Qt.AlignCenter)
        f = title.font(); f.setPointSize(12); f.setBold(True); title.setFont(f)
        lay.addWidget(title)
        caption = QLabel(
            ".onnx imports directly · .pth / .safetensors are converted to ONNX\n"
            "or click to browse"
        )
        caption.setProperty("caption", True)
        caption.setAlignment(Qt.AlignCenter)
        lay.addWidget(caption)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile()]
        if paths:
            self.filesDropped.emit(paths)

    def mousePressEvent(self, event):
        files, _ = QFileDialog.getOpenFileNameNames(
            self, "Select model files", os.path.expanduser("~"),
            "Model files (*.onnx *.pth *.safetensors)",
        )
        if files:
            self.filesDropped.emit(files)


class DownloadTab:
    """Custom-model import page (drag & drop / browse) + app uninstaller."""

    VALID_EXT = (".onnx", ".pth", ".safetensors")

    def __init__(
        self,
        parent,
        backends: list = None,
        skip_info_popup: bool = False,
    ):
        self.parent = parent
        self.backends = backends or []
        self.skip_info_popup = skip_info_popup

        self.has_enough_space = True
        if IS_FLATPAK:
            minimum_space_required = 7
        else:
            minimum_space_required = 15
        self.parent.low_storage_label.setVisible(False)
        if FileHandler.getFreeSpace() < minimum_space_required:
            self.parent.low_storage_label.setVisible(True)
            self.has_enough_space = False

        if IS_FLATPAK or USE_LOCAL_BACKEND:
            self.parent.uninstallAppBtn.setDisabled(True)
        else:
            self.parent.uninstallAppBtn.clicked.connect(self.uninstallApp)

        self._build_import_ui()
        FileHandler.createDirectory(CUSTOM_MODELS_PATH)

    # ---------- import UI ----------

    def _build_import_ui(self):
        from apps.gui.constants import HOME_PATH

        from PySide6.QtWidgets import QGridLayout

        page = self.parent.downloadPage
        layout = page.layout()
        if layout is None:
            return

        self.drop_zone = DropZone()
        self.drop_zone.filesDropped.connect(self.importModels)
        self.import_status = QLabel("Models are imported into your custom_models folder")
        self.import_status.setProperty("caption", True)

        if isinstance(layout, QGridLayout):
            # shift every existing row (uninstall etc.) down two slots and
            # claim rows 0/1 for the import UI
            layout.setRowMinimumHeight(0, 0)
            layout.addWidget(self.drop_zone, 0, 0)
            layout.addWidget(self.import_status, 1, 0)
        else:
            layout.insertWidget(0, self.drop_zone)
            layout.insertWidget(1, self.import_status)

    def importModels(self, paths):
        from apps.gui.constants import HOME_PATH
        FileHandler.createDirectory(CUSTOM_MODELS_PATH)

        to_convert = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext not in self.VALID_EXT:
                self._notify(f"Unsupported file: {os.path.basename(path)}")
                continue
            if ext == ".onnx":
                dest = os.path.join(CUSTOM_MODELS_PATH, os.path.basename(path))
                try:
                    shutil.copy2(path, dest)
                    self._notify(f"Imported {os.path.basename(path)}")
                except Exception as e:
                    self._notify(f"Import failed: {e}")
            else:
                to_convert.append(path)

        for path in to_convert:
            self._notify(f"Converting {os.path.basename(path)}… (this can take a minute)")
            thread = ConvertThread(path, CUSTOM_MODELS_PATH)
            thread.finished_ok.connect(self._on_converted)
            thread.failed.connect(self._notify)
            thread.start()
            self._threads.append(thread)

    @property
    def _threads(self):
        if not hasattr(self, "_convert_threads"):
            self._convert_threads = []
        return self._convert_threads

    def _on_converted(self, out_path):
        self._notify(f"Converted: {os.path.basename(out_path)}")
        # make the new model available without a restart
        process_tab = getattr(self.parent, "processTab", None)
        if process_tab is not None and hasattr(process_tab, "populateModels"):
            process_tab.populateModels()

    def _notify(self, text):
        from apps.gui.lib.QTcustom import NotificationOverlay

        NotificationOverlay(text, self.parent, timeout=2500)

    # ---------- uninstall ----------

    def uninstallApp(self):
        reply = QMessageBox.question(
            self.parent,
            "",
            "Are you sure you want to uninstall?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.Yes:
            os.chdir(os.path.expanduser("~"))
            FileHandler().removeFolder(CWD)
            os._exit(0)
