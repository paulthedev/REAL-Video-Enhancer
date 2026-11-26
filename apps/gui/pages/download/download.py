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
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from apps.gui.constants import CUSTOM_MODELS_PATH, IS_FLATPAK, IS_SOURCE_LAYOUT, USE_LOCAL_BACKEND, CWD
from apps.gui.util import FileHandler, log, open_folder


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
            out = os.path.join(self.output_path, f"{scale}x_{stem}.onnx")

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

    def __init__(self, parent=None, torch_available=False):
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
        if torch_available:
            caption_text = (
                ".onnx imports directly · .pth / .safetensors are converted to ONNX\n"
                "or click to browse"
            )
        else:
            caption_text = (
                ".onnx imports directly\n"
                "(.pth / .safetensors conversion needs the optional torch extra)\n"
                "or click to browse"
            )
        caption = QLabel(caption_text)
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
        files, _ = QFileDialog.getOpenFileNames(
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

        if IS_FLATPAK or USE_LOCAL_BACKEND or IS_SOURCE_LAYOUT:
            self.parent.uninstallAppBtn.setDisabled(True)
        else:
            self.parent.uninstallAppBtn.clicked.connect(self.uninstallApp)

        self._build_import_ui()
        FileHandler.createDirectory(CUSTOM_MODELS_PATH)

    # ---------- import UI ----------

    def _build_import_ui(self):
        import importlib.util

        from PySide6.QtWidgets import QGridLayout

        page = self.parent.downloadPage
        layout = page.layout()
        if layout is None:
            return

        self.drop_zone = DropZone(
            torch_available=importlib.util.find_spec("torch") is not None
        )
        self.drop_zone.filesDropped.connect(self.importModels)
        self.import_status = QLabel("Models are imported into your custom_models folder")
        self.import_status.setProperty("caption", True)
        self.import_hint = QLabel(
            "Tip: include the scale in custom upscale filenames "
            "(for example 2x_mymodel.onnx) so the renderer picks the right output size."
        )
        self.import_hint.setProperty("caption", True)
        self.import_hint.setWordWrap(True)

        self.models_label = QLabel("Imported models")
        self.models_button = QPushButton("Open Folder")
        self.models_button.setProperty("variant", "ghost")
        self.models_button.clicked.connect(lambda: open_folder(CUSTOM_MODELS_PATH))
        self.models_list = QListWidget()
        self.models_list.setMinimumHeight(120)
        self.models_list.setMaximumHeight(260)
        self.models_list.setSelectionMode(QListWidget.NoSelection)
        self.models_list.itemDoubleClicked.connect(
            lambda _item: open_folder(CUSTOM_MODELS_PATH)
        )

        if isinstance(layout, QGridLayout):
            # row 0: import target, 1: status, 2: hint, 3: list heading,
            # row 4: imported list, row 5: slack, row 6: danger zone
            layout.addWidget(self.drop_zone, 0, 0)
            layout.addWidget(self.import_status, 1, 0)
            layout.addWidget(self.import_hint, 2, 0)
            header = QHBoxLayout()
            header.addWidget(self.models_label)
            header.addStretch(1)
            header.addWidget(self.models_button)
            layout.addLayout(header, 3, 0)
            layout.addWidget(self.models_list, 4, 0)
            container = getattr(self.parent, "backendSelectContainer", None)
            if container is not None:
                layout.addWidget(container, 6, 0)
            layout.setRowStretch(5, 1)
        else:
            layout.insertWidget(0, self.drop_zone)
            layout.insertWidget(1, self.import_status)

        self._refresh_model_list()

        uninstall_btn = getattr(self.parent, "uninstallAppBtn", None)
        if uninstall_btn is not None:
            uninstall_btn.setProperty("variant", "danger")
            uninstall_btn.setToolTip("Uninstall REAL Video Enhancer")
            uninstall_btn.style().unpolish(uninstall_btn)
            uninstall_btn.style().polish(uninstall_btn)

        low_storage = getattr(self.parent, "low_storage_label", None)
        if low_storage is not None:
            low_storage.setText(
                "Low disk space: free up about 15 GB for model downloads and a "
                "GPU provider."
            )

    def importModels(self, paths):
        import importlib.util

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
                    self._refresh_models()
                except Exception as e:
                    self._notify(f"Import failed: {e}")
            else:
                to_convert.append(path)

        if to_convert and importlib.util.find_spec("torch") is None:
            # torch is dev/conversion-only tooling, not part of the bundle
            self._notify(
                "PyTorch checkpoint conversion needs the optional torch extra; "
                "import an ONNX file instead."
            )
            return

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
        self._refresh_models()

    def _refresh_models(self):
        process_tab = getattr(self.parent, "processTab", None)
        if process_tab is not None and hasattr(process_tab, "populateModels"):
            process_tab.populateModels()
        self._refresh_model_list()

    def _refresh_model_list(self):
        if not hasattr(self, "models_list"):
            return
        try:
            files = sorted(
                f for f in os.listdir(CUSTOM_MODELS_PATH)
                if f.lower().endswith(".onnx")
            )
        except OSError:
            files = []
        self.models_list.clear()
        if files:
            self.models_label.setText(f"Imported models ({len(files)})")
            for name in files:
                self.models_list.addItem(QListWidgetItem(name))
        else:
            self.models_label.setText("Imported models")
            item = QListWidgetItem("No custom models imported yet")
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(Qt.NoItemFlags)
            self.models_list.addItem(item)

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
