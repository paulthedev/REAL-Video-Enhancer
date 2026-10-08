from PySide6.QtWidgets import QListWidget
from dataclasses import dataclass


@dataclass
class RenderOptions:
    def __init__(
        self,
        inputFile: str,
        outputPath: str,
        videoWidth: int,
        videoHeight: int,
        videoFps: int,
        tilingEnabled: bool,
        tilesize: str,
        videoFrameCount: int,
        backend: str,
        modelScale: int,
        interpolateTimes: int,
        benchmarkMode: bool,
        sloMoMode: bool,
        dyanmicScaleOpticalFlow: bool,
        ensemble: bool,
        interpolateModelFile: str | None = None,
        upscaleModelFile: str | None = None,
        upscaleModelArch: str | None = None,
        deblurModelFile: str | None = None,
        denoiseModelFile: str | None = None,
        decompressModelFile: str | None = None,
        sceneChangeModelFile: str | None = None,
        hdrMode: bool = False,
        startTime: float | None = None,
        endTime: float | None = None,
        isPreview: bool = False,
        overrideUpscaleScale: int | None = None,
        encoderCommand: str | None = None,
    ):
        self._inputFile = inputFile
        self._outputPath = outputPath
        self._videoWidth = videoWidth
        self._videoHeight = videoHeight
        self._videoFps = videoFps
        self._tilingEnabled = tilingEnabled
        self._tilesize = tilesize
        self._videoFrameCount = videoFrameCount
        self._backend = backend
        self._interpolateModelFile = interpolateModelFile
        self._deblurModelFile = deblurModelFile
        self._upscaleModelFile = upscaleModelFile
        self._upscaleModelArch = upscaleModelArch
        self._denoiseModelFile = denoiseModelFile
        self._decompressModelFile = decompressModelFile
        self._sceneChangeModelFile = sceneChangeModelFile
        self._modelScale = modelScale
        self._interpolateTimes = interpolateTimes
        self._benchmarkMode = benchmarkMode
        self._sloMoMode = sloMoMode
        self._dyanmicScaleOpticalFlow = dyanmicScaleOpticalFlow
        self._ensemble = ensemble
        self._hdrMode = hdrMode
        self._startTime = startTime
        self._endTime = endTime
        self._isPreview = isPreview
        self._overrideUpscaleScale = overrideUpscaleScale
        self._encoderCommand = encoderCommand

    @property
    def inputFile(self):
        return self._inputFile

    @inputFile.setter
    def inputFile(self, value: str):
        self._inputFile = value

    @property
    def outputPath(self):
        return self._outputPath

    @outputPath.setter
    def outputPath(self, value: str):
        self._outputPath = value

    @property
    def videoWidth(self):
        return self._videoWidth

    @videoWidth.setter
    def videoWidth(self, value: int):
        self._videoWidth = value

    @property
    def videoHeight(self):
        return self._videoHeight

    @videoHeight.setter
    def videoHeight(self, value: int):
        self._videoHeight = value

    @property
    def videoFps(self):
        return self._videoFps

    @videoFps.setter
    def videoFps(self, value: int):
        self._videoFps = value

    @property
    def tilingEnabled(self):
        return self._tilingEnabled

    @tilingEnabled.setter
    def tilingEnabled(self, value: bool):
        self._tilingEnabled = value

    @property
    def tilesize(self):
        return self._tilesize

    @tilesize.setter
    def tilesize(self, value: str):
        self._tilesize = value

    @property
    def videoFrameCount(self):
        return self._videoFrameCount

    @videoFrameCount.setter
    def videoFrameCount(self, value: int):
        self._videoFrameCount = value

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value: str):
        self._backend = value
    
    @property
    def modelScale(self):
        return self._modelScale
    @modelScale.setter
    def modelScale(self, value: int):
        self._modelScale = value

    @property
    def denoiseModelFile(self):
        return self._denoiseModelFile
    @denoiseModelFile.setter
    def denoiseModelFile(self, value: str):
        self._denoiseModelFile = value
    
    @property
    def decompressModelFile(self):
        return self._decompressModelFile
    @decompressModelFile.setter
    def decompressModelFile(self, value: str):
        self._decompressModelFile = value
    
    @property
    def sceneChangeModelFile(self):
        return self._sceneChangeModelFile
    @sceneChangeModelFile.setter
    def sceneChangeModelFile(self, value: str):
        self._sceneChangeModelFile = value

    @property
    def upscaleModelArch(self):
        return self._upscaleModelArch
    @upscaleModelArch.setter
    def upscaleModelArch(self, value: str):
        self._upscaleModelArch = value

    @property
    def interpolateModelFile(self):
        return self._interpolateModelFile

    @interpolateModelFile.setter
    def interpolateModelFile(self, value: str):
        self._interpolateModelFile = value
    
    @property
    def deblurModelFile(self):
        return self._deblurModelFile
    
    @deblurModelFile.setter
    def deblurModelFile(self, value: str):
        self._deblurModelFile = value

    @property
    def upscaleModel(self):
        return self._upscaleModel

    @upscaleModel.setter
    def upscaleModel(self, value: str):
        self._upscaleModel = value

   
   
    @property
    def interpolateTimes(self):
        return self._interpolateTimes

    @interpolateTimes.setter
    def interpolateTimes(self, value: int):
        self._interpolateTimes = value

    @property
    def benchmarkMode(self):
        return self._benchmarkMode

    @benchmarkMode.setter
    def benchmarkMode(self, value: bool):
        self._benchmarkMode = value

    @property
    def sloMoMode(self):
        return self._sloMoMode

    @sloMoMode.setter
    def sloMoMode(self, value: bool):
        self._sloMoMode = value

    @property
    def dyanmicScaleOpticalFlow(self):
        return self._dyanmicScaleOpticalFlow

    @dyanmicScaleOpticalFlow.setter
    def dyanmicScaleOpticalFlow(self, value: bool):
        self._dyanmicScaleOpticalFlow = value

    @property
    def ensemble(self):
        return self._ensemble

    @ensemble.setter
    def ensemble(self, value: bool):
        self._ensemble = value

    @property
    def upscaleModelFile(self):
        return self._upscaleModelFile

    @upscaleModelFile.setter
    def upscaleModelFile(self, value: str):
        self._upscaleModelFile = value
    
    @property
    def hdrMode(self):
        return self._hdrMode
    @hdrMode.setter
    def hdrMode(self, value: bool):
        self._hdrMode = value
    
    @property
    def startTime(self):
        return self._startTime
    @startTime.setter
    def startTime(self, value: float):
        self._startTime = value

    @property
    def endTime(self):
        return self._endTime
    @endTime.setter
    def endTime(self, value: float):
        self._endTime = value

    @property
    def isPreview(self):
        return self._isPreview
    @isPreview.setter
    def isPreview(self, value: bool):
        self._isPreview = value
    
    @property
    def overrideUpscaleScale(self):
        return self._overrideUpscaleScale
    @overrideUpscaleScale.setter
    def overrideUpscaleScale(self, value: int):
        self._overrideUpscaleScale = value
    
    @property
    def encoderCommand(self):
        return self._encoderCommand
    @encoderCommand.setter
    def encoderCommand(self, value: str):
        self._encoderCommand = value
        
        


from PySide6.QtWidgets import QListWidget
from PySide6.QtCore import QObject, Signal
from dataclasses import dataclass, field


#: lifecycle of a :class:`RenderJob`
QUEUED = "Queued"
RENDERING = "Rendering"
DONE = "Done"
FAILED = "Failed"
CANCELLED = "Cancelled"

#: statuses a job can be removed from the queue in
REMOVABLE = (QUEUED, CANCELLED, FAILED, DONE)


class RenderJob:
    """One queue entry: its render options plus runtime status/progress."""

    def __init__(self, options: RenderOptions):
        self.options = options
        self.status = QUEUED
        self.progress = 0  # 0..max_value
        self.max_value = 0
        self.status_detail = ""  # e.g. "Building Engine…"
        self.return_code = None

    @property
    def input_file(self) -> str:
        return self.options.inputFile

    @property
    def output_path(self) -> str:
        return self.options.outputPath

    def summary(self) -> str:
        """One-line model summary for the queue row, e.g. ``2x SPAN · RIFE 4.22``."""
        parts = []
        o = self.options
        if o.upscaleModelFile:
            scale = f"{o.overrideUpscaleScale}x" if o.overrideUpscaleScale else ""
            parts.append(f"{scale} {o.upscaleModelArch or o.upscaleModelFile}".strip())
        if o.interpolateModelFile:
            parts.append(f"interp {o.interpolateTimes}x")
        if o.denoiseModelFile:
            parts.append("denoise")
        if o.decompressModelFile:
            parts.append("decompress")
        if o.deblurModelFile:
            parts.append("deblur")
        if o.benchmarkMode:
            parts.append("benchmark")
        return " · ".join(parts) if parts else "no models"


class RenderQueue(QObject):
    """The render queue model.

    Jobs are added at any time — including while a render is running; the
    render worker pulls with :meth:`take_next`, so a job enqueued mid-render
    is picked up automatically. All mutations emit :attr:`queueChanged`, and
    progress updates go through :meth:`update_job` /
    :meth:`set_job_progress` (which emit :attr:`jobUpdated`).
    """

    queueChanged = Signal()
    jobUpdated = Signal(int)  # index of the changed job

    def __init__(self, qlistwidget: QListWidget | None = None):
        super().__init__()
        self.jobs: list[RenderJob] = []
        # legacy handle: code that did renderQueue.queue[i] keeps working
        # (list of RenderOptions, same order as jobs)
        self.qlistwidget = qlistwidget

    # ---- legacy compat ----
    @property
    def queue(self) -> list[RenderOptions]:
        return [job.options for job in self.jobs]

    @property
    def inputNameList(self) -> list[str]:
        return [job.input_file for job in self.jobs]

    # ---- queries ----
    def getQueue(self) -> list[RenderOptions]:
        return self.queue

    def __len__(self):
        return len(self.jobs)

    def counts(self) -> dict:
        out = {"queued": 0, "rendering": 0, "done": 0, "failed": 0, "cancelled": 0}
        for job in self.jobs:
            key = job.status.lower()
            out[key] = out.get(key, 0) + 1
        return out

    def take_next(self) -> RenderJob | None:
        """Pop the oldest queued job (marks it RENDERING). None when drained."""
        for job in self.jobs:
            if job.status == QUEUED:
                job.status = RENDERING
                self._emit_job(self.jobs.index(job))
                return job
        return None

    def current_job(self) -> RenderJob | None:
        return next((j for j in self.jobs if j.status == RENDERING), None)

    # ---- mutations ----
    def add(self, renderable: RenderOptions):
        self.jobs.append(RenderJob(renderable))
        self.queueChanged.emit()

    def addToStart(self, renderable: RenderOptions):
        self.jobs.insert(0, RenderJob(renderable))
        self.queueChanged.emit()

    def remove(self, index: int | None = None):
        if index is None and self.qlistwidget is not None:
            index = self.qlistwidget.currentRow()
        if index is None or not (0 <= index < len(self.jobs)):
            return
        if self.jobs[index].status not in REMOVABLE:
            return  # can't remove the job currently rendering
        del self.jobs[index]
        self.queueChanged.emit()

    def retry(self, index: int):
        job = self.jobs[index]
        if job.status in (FAILED, CANCELLED):
            job.status = QUEUED
            job.progress = 0
            job.status_detail = ""
            self._emit_job(index)

    def clear(self):
        """Drop every job (used on full-queue completion)."""
        self.jobs.clear()
        self.queueChanged.emit()

    def clear_finished(self):
        self.jobs = [j for j in self.jobs if j.status not in (DONE, FAILED, CANCELLED)]
        self.queueChanged.emit()

    def move(self, index: int, direction="up"):
        """Move the job at ``index`` (never the currently-rendering one)."""
        new_index = index + (1 if direction == "down" else -1)
        if not (0 <= index < len(self.jobs)) or not (0 <= new_index < len(self.jobs)):
            return
        if RENDERING in (self.jobs[index].status, self.jobs[new_index].status):
            return
        self.jobs[index], self.jobs[new_index] = (
            self.jobs[new_index],
            self.jobs[index],
        )
        if self.qlistwidget is not None:
            item = self.qlistwidget.takeItem(index)
            self.qlistwidget.insertItem(new_index, item)
            self.qlistwidget.setCurrentRow(new_index)
        self.queueChanged.emit()

    def moveitem(self, direction="up"):
        """Legacy entry point: move the currently-selected row."""
        if self.qlistwidget is None:
            return
        self.move(self.qlistwidget.currentRow(), direction)

    # ---- progress/status (called from the render worker thread) ----
    def update_job(self, job: RenderJob, status=None, detail=None, return_code=None):
        if job not in self.jobs:
            return
        if status is not None:
            job.status = status
        if detail is not None:
            job.status_detail = detail
        if return_code is not None:
            job.return_code = return_code
        self._emit_job(self.jobs.index(job))

    def set_job_progress(self, job: RenderJob, value: int, max_value=None):
        if job not in self.jobs:
            return
        job.progress = value
        if max_value is not None:
            job.max_value = max_value
        self._emit_job(self.jobs.index(job))

    def _emit_job(self, index: int):
        self.jobUpdated.emit(index)
