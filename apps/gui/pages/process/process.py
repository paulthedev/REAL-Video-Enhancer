import subprocess
import os
import cv2
from threading import Thread, Event
import re
from multiprocessing import shared_memory

from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QFontMetrics, QImage
from PySide6.QtCore import Qt, QUrl, Signal, QThread, QObject
from PySide6.QtWidgets import QMessageBox

from apps.gui.lib.RenderQueue import (
    RenderQueue,
    RenderJob,
    QUEUED,
    RENDERING,
    DONE,
    FAILED,
    CANCELLED,
)

from apps.gui.lib.AnimationHandler import AnimationHandler
from apps.gui.lib.QTcustom import (
    RegularQTPopup,
    UpdateGUIThread,
    show_layout_widgets,
    hide_layout_widgets,
)
from apps.gui.pages.process.layout import apply_studio_layout
from apps.gui.widgets.PreviewCompare import PreviewCompare
from apps.gui.constants import (
    BACKEND_PATH,
    PYTHON_EXECUTABLE_PATH,
    MODELS_PATH,
    CUSTOM_MODELS_PATH,
    IMAGE_SHARED_MEMORY_ID,
    PAUSED_STATE_SHARED_MEMORY_ID,
    INPUT_TEXT_FILE,
    FFMPEG_PATH,
    CWD,
    PLATFORM,
)
from apps.gui.util import (
    log,
)
from apps.gui.utils.backends import DownloadModel
from apps.gui.pages.settings.settings import Settings
from apps.gui.lib.DiscordRPC import DiscordRPC
from apps.gui.lib.ModelHandler import getModels
from apps.gui.lib.RenderQueue import RenderOptions


class SourceFrameReader(QThread):
    """Reads original frames from the current job's input for compare modes.

    The render worker drops ``(input_path, output_frame, factor)`` requests
    via :meth:`request`; this thread seeks the matching source frame and
    emits it as a QImage. Comparison wants approximate sync, not exactness,
    so a stale read just skips ahead on the next request.
    """

    frameReady = Signal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._requested_frame = None
        self._input_path = None
        self._capture = None
        self._stop = False

    def request(self, input_path: str, output_frame: int, factor: int):
        self._input_path = input_path
        self._requested_frame = (max(int(output_frame), 0), max(int(factor), 1))

    def set_input(self, input_path: str):
        if input_path != self._input_path:
            self._input_path = input_path
            self._reopen()

    def _reopen(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._input_path and os.path.isfile(self._input_path):
            self._capture = cv2.VideoCapture(self._input_path)

    def available(self, input_path: str) -> bool:
        return bool(input_path) and os.path.isfile(input_path)

    def stop(self):
        self._stop = True

    def run(self):
        last_sent = -1
        while not self._stop:
            req = self._requested_frame
            if req is None or self._input_path is None:
                self.msleep(50)
                continue
            output_frame, factor = req
            src_index = output_frame // factor
            if src_index == last_sent:
                self.msleep(50)
                continue
            if self._capture is None:
                self._reopen()
            if self._capture is None or not self._capture.isOpened():
                self.msleep(100)
                continue
            self._requested_frame = None
            last_sent = src_index
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, src_index)
            ok, frame = self._capture.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(
                    rgb.data, rgb.shape[1], rgb.shape[0],
                    rgb.shape[1] * 3, QImage.Format_RGB888,
                ).copy()
                self.frameReady.emit(image)
            else:
                self.msleep(50)
        if self._capture is not None:
            self._capture.release()


class _OverwriteBridge(QObject):
    """Marshals the per-job overwrite prompt from the worker thread to the
    GUI thread; the worker blocks on an Event until the user answers."""

    askOverwrite = Signal(str)


class ProcessTab:
    def __init__(self, parent, settings: Settings):
        self.parent = parent
        self.renderTextOutputList = None
        self.isOverwrite = False
        self.outputVideoHeight = None
        self.outputVideoWidth = None
        self.currentFrame = 0
        self.fps = 0
        self.eta = 0
        self.max_value = 0
        self.isPreview = False
        self.userKilled = False
        self.currentRenderOptions = None
        self.current_job = None
        self.status = "Idle"
        self.animationHandler = AnimationHandler()
        self.tileUpAnimationHandler = AnimationHandler()
        self.tileDownAnimationHandler = AnimationHandler()
        self.settings = settings
        self.return_codes = []
        self._render_thread = None
        self._overwrite_answer = None
        self._overwrite_event = Event()
        self._bridge = _OverwriteBridge()
        self._bridge.askOverwrite.connect(self.showOverwritePrompt)

        self.qualityToCRF = {
            "Low": "28",
            "Medium": "23",
            "High": "18",
            "Very High": "15",
        }

        # Studio Polish scaffold: preview-on-top + Source/Models/Queue cards.
        # Re-homes the generated widgets; keeps every self.<name> reference.
        self.preview_compare, self.queue_panel, self.preview_stack = (
            apply_studio_layout(parent)
        )
        self.source_reader = SourceFrameReader()
        self.source_reader.frameReady.connect(self.preview_compare.set_source)
        self.source_reader.start(QThread.LowPriority)
        self.preview_compare.modeChanged.connect(self._on_preview_mode_changed)

        # get default backend
        self.QConnect()
        self.populateModels()

    def get_current_backend(self) -> str:
        """ONNX is the only backend."""
        return "onnx"

    def populateModels(self) -> None:
        """
        Populates the model comboboxes from the ONNX model tables.
        """
        interpolateModels, upscaleModels, deblurModels, denoiseModels, decompressModels, _ = getModels()
        self.parent.interpolateModelComboBox.clear()
        self.parent.upscaleModelComboBox.clear()
        self.parent.deblurModelComboBox.clear()
        self.parent.denoiseModelComboBox.clear()
        self.parent.decompressModelComboBox.clear()
        self.parent.interpolateModelComboBox.addItems(
            list(interpolateModels.keys())
        )
        self.parent.interpolateModelComboBox.setCurrentIndex(len(list(interpolateModels.keys()))-1)
        self.parent.upscaleModelComboBox.addItems(list(upscaleModels.keys()))
        self.parent.deblurModelComboBox.addItems(list(deblurModels.keys()))
        self.parent.denoiseModelComboBox.addItems(list(denoiseModels.keys()))
        self.parent.decompressModelComboBox.addItems(list(decompressModels.keys()))
    def onTilingSwitch(self):
        if self.parent.tilingCheckBox.isChecked():
            self.parent.tileSizeContainer.setVisible(True)
            self.tileDownAnimationHandler.dropDownAnimation(
                self.parent.tileSizeContainer
            )
        else:
            self.tileUpAnimationHandler.moveUpAnimation(self.parent.tileSizeContainer)
            self.parent.tileSizeContainer.setVisible(False)

    def openOutputFolderInExplorer(self):
        output_folder = os.path.dirname(self.parent.outputFileText.text())
        if os.path.isdir(output_folder):
            if PLATFORM == "win32":
                os.startfile(output_folder)
            elif PLATFORM == "darwin":
                subprocess.Popen(["open", output_folder])
            else:
                subprocess.Popen(["xdg-open", output_folder])

    def QConnect(self):
        # connect file select buttons
        self.parent.addToRenderQueueButton.clicked.connect(self.parent.addToRenderQueue)
        self.parent.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.parent.batchSelectButton.clicked.connect(self.parent.openBatchFiles)
        self.parent.inputFileText.textChanged.connect(self.parent.loadVideo)
        self.parent.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        self.parent.openOutputFolderButton.clicked.connect(self.openOutputFolderInExplorer)
        # connect render button
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        # set tile size visible to false by default
        self.parent.tileSizeContainer.setVisible(False)
        # set slo mo container visable to false by default

        self.parent.interpolateContainer_2.setVisible(False)
        # connect up tilesize container visiable
        self.parent.tilingCheckBox.stateChanged.connect(self.onTilingSwitch)

        self.parent.interpolationMultiplierSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )

        self.parent.upscaleModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.upscaleScaleSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.interpolateModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.decompressModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.interpolateCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.upscaleCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.deblurCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.denoiseCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.decompressCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.sloMoModeCheckBox.clicked.connect(self.parent.updateVideoGUIDetails)
        self.parent.EncoderCommand.textChanged.connect(lambda: self.parent.EncoderCommand.setFixedWidth(max(50, QFontMetrics(self.parent.EncoderCommand.font()).horizontalAdvance(self.parent.EncoderCommand.text()) + 10)))
        # connect up pausing
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.pauseRenderButton.clicked.connect(self.pauseRender)
        self.parent.killRenderButton.clicked.connect(self.killRenderProcess)

    def _on_preview_mode_changed(self, mode: str):
        """Compare modes need original frames; nudge/quiet the reader."""
        self.preview_compare.set_hint(
            "Start a render to see the live preview"
            if mode == "enhanced"
            else "Waiting for frames…"
        )

    def _stop_source_reader(self):
        self.source_reader.stop()
        self.source_reader.wait()
        self.source_reader = SourceFrameReader()
        self.source_reader.frameReady.connect(self.preview_compare.set_source)
        self.source_reader.start(QThread.LowPriority)

    def killRenderProcess(self):
        try:  # kills  render process if necessary
            self.userKilled = True
            self.renderProcess.terminate()
        except AttributeError:
            log("No render process!")

    def pauseRender(self):
        shm = getattr(self, "pausedSharedMemory", None)
        if shm is None:
            return  # render thread has not created the shared memory yet
        shmbuf = shm.buf
        shmbuf[0] = 1  # 1 = True
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(True)
        self.parent.startRenderButton.setEnabled(True)

    def resumeRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 0  # 0 = False
        show_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.onRenderButtonsContiainer.setEnabled(True)
        self.parent.startRenderButton.setVisible(False)

    def startGUIUpdate(self):
        
        self.workerThread = UpdateGUIThread(
            parent=self,
            imagePreviewSharedMemoryID=IMAGE_SHARED_MEMORY_ID,
        )
        self.workerThread.latestPreviewPixmap.connect(self.updateProcessTab)
        self.workerThread.finished.connect(self.guiChangesOnRenderCompletion)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.workerThread.start()

    def splitListIntoStringWithNewLines(self, string_list: list[str]):
        # Join the strings with newline characters
        return "\n".join(string_list)
        # Set the text to the QTextEdit

    def questionToOverride(self, path) -> bool:
        reply = QMessageBox.question(
            self.parent,
            "",
            f"{os.path.basename(path)} already exists.\nOverwrite it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        return reply == QMessageBox.Yes  # type: ignore

    def showOverwritePrompt(self, path):
        """GUI-thread slot: ask the user, store the answer, release the worker."""
        self._overwrite_answer = self.questionToOverride(path)
        self._overwrite_event.set()

    def run(
        self,
        renderQueue: RenderQueue,
    ):
        if getattr(self, "_render_thread", None) is not None and self._render_thread.is_alive():
            return  # a render session is already consuming the queue

        self.return_codes = []  # reset return codes
        self.userKilled = False  # reset userkilled
        # gui changes
        show_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(False)
        self.parent.startRenderButton.clicked.disconnect()
        self.parent.startRenderButton.clicked.connect(self.resumeRender)

        self.startDiscordRPC()
        self.settings.readSettings()

        self._overwrite_answer = None
        self._overwrite_event = Event()

        self.startGUIUpdate()
        self._render_thread = Thread(target=lambda: self.renderToPipeThread(renderQueue))
        self._render_thread.start()

    def startDiscordRPC(self):
        if self.settings.settings["discord_rich_presence"] == "True":
            try:
                self.discordRPC = DiscordRPC()
                self.discordRPC.start_discordRPC()
            except Exception:
                pass

    def createPausedSharedMemory(self):
        try:
            self.pausedSharedMemory = shared_memory.SharedMemory(
                name=PAUSED_STATE_SHARED_MEMORY_ID, create=True, size=1
            )
        except FileExistsError:
            log("FileExistsError! Using existing paused shared memory")
            self.pausedSharedMemory = shared_memory.SharedMemory(
                name=PAUSED_STATE_SHARED_MEMORY_ID
            )

    def _ask_overwrite_from_worker(self, output_path: str) -> bool:
        """Worker-thread side: prompt on the GUI thread, block for the answer."""
        if not os.path.isfile(output_path):
            return True
        self._overwrite_answer = None
        self._overwrite_event.clear()
        self._bridge.askOverwrite.emit(output_path)
        self._overwrite_event.wait()
        return bool(self._overwrite_answer)

    def renderToPipeThread(
        self,
        renderQueue: RenderQueue,
    ):
        self.createPausedSharedMemory()

        # pull jobs dynamically: anything enqueued while this loop runs is
        # picked up; draining the queue ends the session
        while True:
            job = renderQueue.take_next()
            if job is None:
                break
            if self.userKilled:
                renderQueue.update_job(job, status=CANCELLED)
                break

            self.isPreview = job.options.isPreview
            self.currentRenderOptions = job.options
            self.current_job = job

            overwrite_ok = self._ask_overwrite_from_worker(job.output_path)
            if not overwrite_ok:
                renderQueue.update_job(job, status=CANCELLED, detail="Skipped")
                continue

            self.workerThread.setOutputVideoRes(
                job.options.videoWidth * job.options.overrideUpscaleScale,
                job.options.videoHeight * job.options.overrideUpscaleScale,
            )
            self.workerThread.createNewSharedMemory(channels=6 if job.options.hdrMode else 3)
            self.max_value = job.options.videoFrameCount * job.options.interpolateTimes
            renderQueue.set_job_progress(job, 0, max_value=self.max_value)
            renderQueue.update_job(job, status=RENDERING, detail="Starting…")
            command = self.build_command(job.options)
            log(str(command))

            kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "universal_newlines": True,
                "text": True,                 # return str instead of bytes
                "encoding": "utf-8",          # decode using utf-8
                "errors": "replace",
            }

            if PLATFORM == "win32":
                kwargs["startupinfo"] = subprocess.STARTUPINFO()
                kwargs["startupinfo"].dwFlags |= subprocess.STARTF_USESHOWWINDOW


            self.renderProcess = subprocess.Popen(
                command,
                **kwargs,
            )
            textOutput = []
            for line in iter(self.renderProcess.stdout.readline, ""):
                if self.renderProcess.poll() is not None:
                    break  # Exit the loop if the process has terminated

                try:
                    line = str(line.strip())
                    if "it/s" in line:
                        textOutput = textOutput[:-1]
                    if "FPS" in line:
                        textOutput = textOutput[
                            :-1
                        ]  # slice the list to only get the last updated data
                        frame_match = re.search(r"Current Frame: (\d+)", line)
                        fps_match = re.search(r"FPS: (\d+)", line)
                        eta_match = re.search(r"ETA: (.+)", line)
                        if frame_match:
                            self.currentFrame = int(frame_match.group(1))
                        if fps_match:
                            self.fps = int(fps_match.group(1))
                        if eta_match:
                            self.eta = eta_match.group(1)
                        self.status = "Rendering"
                        renderQueue.set_job_progress(job, self.currentFrame, max_value=self.max_value)

                    if "this may take a while" in line.lower():
                        self.status = "Building Engine, this may take a while."
                        renderQueue.update_job(job, detail="Building engine…")


                    if any(char.isalpha() for char in line):
                        textOutput.append(line)
                    # self.setRenderOutputContent(textOutput)
                    self.renderTextOutputList = textOutput.copy()
                except Exception as e:
                    pass
                if "Time to complete render" in line:
                    break
            return_code = self.renderProcess.wait()
            self.return_codes.append(return_code)
            for line in textOutput:
                if len(line) > 2:
                    log(line)

            final_status = DONE if return_code == 0 else (
                CANCELLED if self.userKilled else FAILED
            )
            renderQueue.update_job(
                job,
                status=final_status,
                detail=f"exit {return_code}" if final_status == FAILED else "",
                return_code=return_code,
            )
            self.parent.OutputFilesListWidget.addItem(
                job.output_path
            )  # add the file to the list widget
            self.workerThread.deleteSharedMemory()

        self.current_job = None
        try:
            self.pausedSharedMemory.close()
            self.pausedSharedMemory.unlink()
        except Exception: # too lazy to patch why this errors maybe on exit
            pass

        # done jobs stay in the queue (status chips + clear-finished); the
        # queue is no longer wiped so the user can see what happened
        self.onRenderCompletion()

    def guiChangesOnRenderCompletion(self):
        if all(return_code == 0 or return_code == 3221225477 for return_code in self.return_codes) or self.userKilled: # 3221225477 comes up when using ncnn on windows for some reason, but no error in output itself.
            log("All render processes completed successfully")
        else:
            log("Some render processes failed: Error code: " + str(self.return_codes))
            RegularQTPopup("Rendering failed! Please check the logs tab!")
        # Have to swap the visibility of these here otherwise crash for some reason
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setEnabled(True)
        self.parent.outputFileText.setEnabled(True)
        self.parent.inputFileText.setEnabled(True)
        self.preview_compare.clear()
        self.parent.startRenderButton.clicked.disconnect()
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        self.parent.enableProcessPage()
        self.parent.startRenderButton.setVisible(True)
        self.parent.FPS.setText("FPS: ")
        self.parent.ETA.setText("ETA: ")
        self.parent.STATUS.setText("Status: ")
        # NOTE: no renderQueue.clear() here — the render loop clears the queue
        # on completion, and clearing on the overwrite-declined path would wipe
        # the user's queued items.
        if self.currentRenderOptions is not None and self.currentRenderOptions.isPreview:
            from PySide6.QtMultimedia import QMediaPlayer
            try:
                def onScroll(preview:QMediaPlayer, value):
                    preview.setPosition(value)

                

                player = QMediaPlayer()
                player.setSource(QUrl.fromLocalFile(self.currentRenderOptions.outputPath))
                player.setVideoOutput(self.parent.VideoPreview)
                player.play()
                player.pause()
                player.setPosition(5)
                # show the playback page (video + seek controls) of the preview card
                self.preview_stack.setCurrentIndex(1)
                self.parent.timeInVideoScrollBar.setRange(0, (((self.currentRenderOptions.endTime-self.currentRenderOptions.startTime))*10)-1) # convert to ms
                try:
                    self.parent.timeInVideoScrollBar.valueChanged.disconnect()
                except RuntimeError:
                    pass  # nothing connected yet
                self.parent.timeInVideoScrollBar.valueChanged.connect(lambda: onScroll(player, int(self.parent.timeInVideoScrollBar.value()*100)))
                self._preview_player = player  # keep a reference so Qt doesn't GC it

            except Exception as e:
                log(f"Error: {e}")
        else:
            self.preview_stack.setCurrentIndex(0)


    def onRenderCompletion(self):
        self.eta = 0
        self.fps = 0
        try:
            self.renderProcess.wait()
        except Exception:
            pass
        # Have to swap the visibility of these here otherwise crash for some reason
        if (
            self.settings.settings["discord_rich_presence"] == "True"
            and getattr(self, "discordRPC", None) is not None
        ):  # only close if it actually started
            self.discordRPC.closeRPC()
        try:
            self.workerThread.stop()
            self.workerThread.quit()
            self.workerThread.wait()
        except Exception:
            pass  # pass just incase internet error caused a skip

    def getRoundedPixmap(self, pixmap, corner_radius):
        size = pixmap.size()
        mask = QPixmap(size)
        mask.fill(Qt.transparent)  # type: ignore

        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore

        path = QPainterPath()
        path.addRoundedRect(
            0, 0, size.width(), size.height(), corner_radius, corner_radius
        )

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        rounded_pixmap = QPixmap(size)
        rounded_pixmap.fill(Qt.transparent)  # type: ignore

        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore
        painter.drawPixmap(0, 0, mask)
        painter.end()

        return rounded_pixmap

    def updateProcessTab(self, qimage: QtGui.QImage):
        """
        Called by the worker QThread, and updates the GUI elements: Progressbar, Preview, FPS
        """

        if self.renderTextOutputList is not None:
            self.parent.progressBar.setRange(
                0,
                # only set the range to multiply the frame count if the method is interpolate
                self.max_value
            )
            # print(self.renderTextOutputList)
            self.parent.renderOutput.setPlainText(
                self.splitListIntoStringWithNewLines(self.renderTextOutputList)
            )
            scrollbar = self.parent.renderOutput.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            self.parent.progressBar.setValue(self.currentFrame)
            if self.fps != 0:
                self.parent.FPS.setVisible(True)
                self.parent.FPS.setText(f"FPS: {self.fps}")
            else:
                self.parent.FPS.setVisible(False)
            if self.eta != 0:
                self.parent.ETA.setVisible(True)
                self.parent.ETA.setText(f"ETA: {self.eta}")
            else:
                self.parent.ETA.setVisible(False)
            self.parent.STATUS.setText(f"Status: {self.status}")
            # live progress on the active queue row
            if self.current_job is not None:
                self.parent.renderQueue.set_job_progress(
                    self.current_job, self.currentFrame, max_value=self.max_value
                )
        if not qimage.isNull():
            self.preview_compare.set_enhanced(qimage)
            # compare modes want the matching original frame
            if self.preview_compare.mode != PreviewCompare.MODE_ENHANCED:
                opts = self.currentRenderOptions
                if opts is not None:
                    self.source_reader.set_input(opts.inputFile)
                    self.source_reader.request(
                        opts.inputFile,
                        self.currentFrame,
                        max(int(getattr(opts, "interpolateTimes", 1) or 1), 1),
                    )

    def build_command(self, renderOptions: RenderOptions):
        if (
            renderOptions.backend == "pytorch (cuda)"
            or renderOptions.backend == "pytorch (rocm)"
            or renderOptions.backend == "pytorch (xpu)"
            or renderOptions.backend == "pytorch (mps)"
        ):
            renderOptions.backend = (
                "pytorch"  # pytorch is the same for both cuda and rocm
            )

        command = [
            f"{PYTHON_EXECUTABLE_PATH}",
            "-W",
            "ignore",
            os.path.join(BACKEND_PATH, "rve-backend.py"),
            "-i",
            renderOptions.inputFile,
            "-o",
            renderOptions.outputPath,
            "-b",
            f"{renderOptions.backend}",
            "--precision",
            f"{self.settings.settings['precision']}",
            "--custom_encoder",
            f" {renderOptions.encoderCommand} ",
            "--pause_shared_memory_id",
            f"{PAUSED_STATE_SHARED_MEMORY_ID}",
            "--gpu_id",
            f"{Settings.gpu_index(self.parent.fullOutput, 'onnx', self.settings.settings['gpu_id'])}",
            "--cwd",
            f"{CWD}",
            "--ffmpeg_path",
            f"{FFMPEG_PATH}",

        ]

        if renderOptions.upscaleModelFile:
            modelPath = os.path.join(MODELS_PATH, renderOptions.upscaleModelFile)
            if renderOptions.upscaleModelArch == "custom":
                modelPath = os.path.join(
                    CUSTOM_MODELS_PATH, renderOptions.upscaleModelFile
                )
            command += [
                "--upscale_model",
                modelPath,
            ]
            command += ["--override_upscale_scale", f"{renderOptions.overrideUpscaleScale}"]
            
        if renderOptions.tilingEnabled:
            command += [
                "--tilesize",
                f"{renderOptions.tilesize}",
            ]
            

        if renderOptions.interpolateModelFile:
            command += [
                "--interpolate_model",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.interpolateModelFile,
                ),
                "--interpolate_factor",
                f"{renderOptions.interpolateTimes}",
            ]
            if renderOptions.sloMoMode:
                command += [
                    "--slomo_mode",
                ]
            if renderOptions.dyanmicScaleOpticalFlow:
                command += [
                    "--dynamic_scaled_optical_flow",
                ]
            if renderOptions.ensemble:
                command += [
                    "--ensemble",
                ]
        
        if renderOptions.deblurModelFile:
            command += [
                "--extra_restoration_models",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.deblurModelFile,
                ),
            ]
        if renderOptions.denoiseModelFile:
            command += [
                "--extra_restoration_models",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.denoiseModelFile,
                ),
            ]
        if renderOptions.decompressModelFile:
            command += [
                "--extra_restoration_models",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.decompressModelFile,
                ),
            ]

        if self.settings.settings["auto_border_cropping"] == "True":
            command += [
                "--border_detect",
            ]

        if self.settings.settings["preview_enabled"] == "True":
            command += [
                "--preview_shared_memory_id",
                f"{IMAGE_SHARED_MEMORY_ID}",
            ]

        # scene detection is always on: pyscenedetect costs ~1ms/frame and
        # prevents interpolation artifacts across cuts
        command += [
            "--scene_detect_method",
            "pyscenedetect",
            "--scene_detect_threshold",
            self.settings.settings["scene_change_detection_threshold"],
        ]
        if renderOptions.sceneChangeModelFile:
            command += [
                "--scene_detect_model",
                os.path.join(
                    MODELS_PATH,
                    renderOptions.sceneChangeModelFile,
                ),
            ]

        if renderOptions.benchmarkMode:
            command += ["--benchmark"]

        if self.settings.settings["uhd_mode"] == "True":
            if renderOptions.videoWidth > 1920 or renderOptions.videoHeight > 1080:
                command += ["--UHD_mode"]
                log("UHD mode enabled")

        if self.isOverwrite:
            command += ["--overwrite"]

        if renderOptions.hdrMode:
            command += ["--hdr_mode"]



        return command