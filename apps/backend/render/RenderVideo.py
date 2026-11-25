from threading import Thread
import os
import math
from time import sleep
from typing import Optional
import sys
from multiprocessing import shared_memory
import cv2

from .FFmpegBuffers import FFmpegRead, FFmpegWrite, MPVOutput
from .FFmpeg import InformationWriteOut
from ..utils.Encoders import EncoderSettings
from ..utils.SceneDetect import SceneDetect
from ..utils.Util import log, resize_image_bytes
from ..utils.BorderDetect import BorderDetect
from ..utils.VideoInfo import OpenCVInfo
import numpy as np


def remove_shared_memory_block(name):
    try:
        existing_shm = shared_memory.SharedMemory(name=name)
        existing_shm.close()
        existing_shm.unlink()
        print(f"Shared memory block '{name}' removed.")
    except FileNotFoundError:
        print(f"Shared memory block '{name}' does not exist.")
    except Exception as e:
        print(f"Error removing shared memory block '{name}': {e}")


class Render:
    """
    Subclass of FFmpegRender
    FFMpegRender options:
    inputFile: str, The path to the input file.
    outputFile: str, The path to the output file.
    interpolateTimes: int, this sets the multiplier for the framerate when interpolating, when only upscaling this will be set to 1.
    encoder: str, The exact name of the encoder ffmpeg will use (default=libx264)
    pixelFormat: str, The pixel format ffmpeg will use, (default=yuv420p)

    interpolateOptions:
    interpolationMethod
    upscaleModel
    backend (onnx only)
    device (cpu,cuda)
    precision (float16,float32)

    NOTE:
    Everything in here has to happen in a specific order:
    Get the video properties (res,fps,etc)
    set up upscaling/interpolation, this gets the scale for upscaling if upscaling is the current task
    assign framechunksize to a value, as this is needed to catch bytes and set up shared memory
    set up shared memory
    """

    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        # backend settings
        backend="onnx",
        device="default",
        precision="auto",
        gpu_id: int = 0,
        cwd: str = os.getcwd(),
        # model settings
        upscaleModel=None,
        interpolateModel=None,
        interpolateFactor: int = 1,
        extraRestorationModels=None,
        sceneDetectModel: str = None,
        tile_size=None,
        # ffmpeg settings
        ffmpeg_path: str = "./bin/ffmpeg",
        start_time=None,
        end_time=None,
        custom_encoder: str = "libx264",
        pixelFormat: str = "yuv420p",
        benchmark: bool = False,
        overwrite: bool = False,
        crf: str = "18",
        video_encoder_preset: str = "libx264",
        audio_encoder_preset: str = "aac",
        subtitle_encoder_preset: str = "srt",
        audio_bitrate: str = "192k",
        border_detect: bool = False,
        hdr_mode: bool = False,
        merge_subtitles: bool = True,
        # misc
        pause_shared_memory_id=None,
        sceneDetectMethod: str = "pyscenedetect",
        sceneDetectSensitivity: float = 3.0,
        sharedMemoryID: Optional[str] = None,
        override_upscale_scale: int | None = None,
        slomo_mode: bool = False,
        ensemble: bool = False,
        output_to_mpv: bool = False,
    ):
        self.inputFile = inputFile
        self.backend = backend
        self.upscaleModel = upscaleModel
        self.interpolateModel = interpolateModel
        self.tilesize = tile_size
        self.device = device
        self.precision = precision
        self.interpolateFactor = interpolateFactor
        self.ceilInterpolateFactor = math.ceil(self.interpolateFactor)
        self.setupFrame0 = None
        self.interpolateOption = None
        self.upscaleOption = None
        self.isPaused = False
        self.sceneDetectModelPath = sceneDetectModel
        self.sceneDetectMethod = sceneDetectMethod
        self.sceneDetectSensitivty = sceneDetectSensitivity
        self.sharedMemoryID = sharedMemoryID
        self.uncacheNextFrame = False
        self.gpu_id = gpu_id
        self.outputFrameChunkSize = None
        self.hdr_mode = hdr_mode
        self.ensemble = ensemble
        self.override_upscale_scale = override_upscale_scale
        self.extraRestorationModels = []
        
        if cwd:
            log("Working Directory: " + cwd)
        else:
            cwd = os.getcwd()
            log("No Working Directory specified, using current directory: " + cwd)
        videoInfo = OpenCVInfo(input_file=inputFile, start_time=start_time, end_time=end_time, ffmpeg_path=ffmpeg_path)
        
        if not videoInfo.is_valid_video:
            log("Input video is not valid!")
        
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = videoInfo.get_duration_seconds()

        self.width, self.height = videoInfo.get_width_x_height()
        self.originalWidth = self.width
        self.originalHeight = self.height
        input_pix_fmt = videoInfo.get_pixel_format()
        self.borderX = 0
        self.borderY = 0  # set borders for cropping automatically to 0, will be overwritten if borders are detected
        self.totalInputFrames = videoInfo.get_total_frames()
        self.totalOutputFrames = int(
            self.totalInputFrames * self.ceilInterpolateFactor
        )
        self.fps = videoInfo.get_fps()
        color_space = videoInfo.get_color_space()
        color_primaries = videoInfo.get_color_primaries()
        color_transfer = videoInfo.get_color_transfer()

        video_encoder = EncoderSettings(video_encoder_preset)
        audio_encoder = EncoderSettings(audio_encoder_preset, type="audio")
        subtitle_encoder = EncoderSettings(subtitle_encoder_preset, type="subtitle")

        if border_detect:  # border detect has to be put before everything, to overwrite the width and height
            print("Detecting borders", file=sys.stderr)
            borderDetect = BorderDetect(inputFile=self.inputFile, ffmpeg_path=ffmpeg_path)
            self.width, self.height, self.borderX, self.borderY = (
                borderDetect.getBorders()
            )
            log(
                f"Detected borders: Width,Height:{self.width}x{self.height}, X,Y: {self.borderX}x{self.borderY}"
            )

        log("Using backend: " + self.backend)
        # upscale has to be called first to get the scale of the upscale model
        if upscaleModel:
            self.setupUpscale()
            self.upscaleOption.hotUnload()  # unload model to free up memory for trt enging building
            log("Using Upscaling Model: " + self.upscaleModel)
        else:
            self.upscaleTimes = 1  # if no upscaling, it will default to 1
            self.modelScale = 1
            

        
        if extraRestorationModels:
            for model in extraRestorationModels:
                extraRestoration = self.setupExtraRestoration(model)
                if extraRestoration:
                    log("Using Extra Restoration Model: " + model)
                    self.extraRestorationModels.append(extraRestoration)
                    extraRestoration.hotUnload()  # unload model to free up memory for trt enging building
        

        if interpolateModel:
            self.setupInterpolate()
            log("Using Interpolation Model: " + self.interpolateModel)

        if upscaleModel: # load model after interpolation model is loaded, this saves on vram if the user builds 2 separate engines
            self.upscaleOption.hotReload()

        for extraRestoration in self.extraRestorationModels:
            extraRestoration.hotReload()
        
        if self.modelScale and self.override_upscale_scale:
            if int(self.modelScale) == int(self.override_upscale_scale):
                log("Override upscale scale is set to the same value as the model scale, this will not change the output resolution.")
                self.override_upscale_scale = False
        
        log(f"Upscale Times: {self.override_upscale_scale if self.override_upscale_scale else self.upscaleTimes}")
        log(f"Interpolate Factor: {self.interpolateFactor}")
        log(f"Total Output Frames: {self.totalOutputFrames}")
        log("Model Scale: " + str(self.modelScale))
        log("HDR Mode: " + str(hdr_mode))

        self.readBuffer = FFmpegRead(  # input width
            inputFile=inputFile,
            width=self.width,
            height=self.height,
            start_time=start_time,
            end_time=end_time,
            borderX=self.borderX,
            borderY=self.borderY,
            hdr_mode=hdr_mode,
            backend=self.backend,
            device=self.device,
            gpu_id=self.gpu_id,
            dtype=self.precision,
            color_space=color_space,
            color_primaries=color_primaries,
            color_transfer=color_transfer,
            input_pixel_format=input_pix_fmt,
            ffmpeg_path=ffmpeg_path,
        )

        self.writeBuffer = FFmpegWrite(
            inputFile=inputFile,
            outputFile=outputFile,
            width=self.width,
            height=self.height,
            start_time=start_time,
            end_time=end_time,
            fps=self.fps,
            crf=crf,
            audio_bitrate=audio_bitrate,
            pixelFormat=pixelFormat,
            overwrite=overwrite,
            custom_encoder=custom_encoder,
            benchmark=benchmark,
            slowmo_mode=slomo_mode,
            upscaleTimes=self.upscaleTimes if not self.override_upscale_scale else self.override_upscale_scale,
            interpolateFactor=self.interpolateFactor,
            ceilInterpolateFactor=self.ceilInterpolateFactor,
            video_encoder=video_encoder,
            audio_encoder=audio_encoder,
            subtitle_encoder=subtitle_encoder,
            mpv_output=output_to_mpv,
            hdr_mode=hdr_mode,
            merge_subtitles=merge_subtitles,
            color_space=color_space,
            color_primaries=color_primaries,
            color_transfer=color_transfer,
            ffmpeg_path=ffmpeg_path,
            ffmpeg_log_file=os.path.join(cwd, "ffmpeg_log.txt"),
        )

        shm_mul = self.override_upscale_scale if self.override_upscale_scale else self.upscaleTimes
        hdr_mul = 6 if hdr_mode else 3

        self.informationHandler = InformationWriteOut(
            sharedMemoryID=sharedMemoryID,
            sharedMemoryChunkSize=self.originalHeight*self.originalWidth*shm_mul*shm_mul*hdr_mul,
            paused_shared_memory_id=pause_shared_memory_id,
            outputWidth=self.originalWidth*shm_mul,
            outputHeight=self.originalHeight*shm_mul,
            croppedOutputWidth=self.width*shm_mul,
            croppedOutputHeight=self.height*shm_mul,
            totalOutputFrames=self.totalOutputFrames,
            border_detect=border_detect,
            hdr_mode=hdr_mode,
            
        )
        

        self.renderThread = Thread(target=self._render_thread_main)
        self.ffmpegReadThread = Thread(target=self.readBuffer.read_frames_into_queue)
        self.ffmpegWriteThread = Thread(target=self.writeBuffer.write_out_frames)
        self.sharedMemoryThread = Thread(
            target=self.informationHandler.writeOutInformation
        )

        self.sharedMemoryThread.start()
        self.ffmpegReadThread.start()
        self.ffmpegWriteThread.start()
        self.renderThread.start()

        if output_to_mpv:
            MPVOut = MPVOutput(self.writeBuffer, width=self.width*self.upscaleTimes, height=self.height*self.upscaleTimes,fps=self.fps*self.interpolateFactor, outputFrameChunkSize=self.outputFrameChunkSize)
            MPVoutThread = Thread(target=MPVOut.write_out_frames)
            MPVoutThread.start()

    def _render_thread_main(self):
        """Run the render loop; never leave the process half-alive on error.

        The reader/writer threads block on queues, so an exception in this
        thread used to keep the backend running forever (the GUI waited for
        output that would never come). Exit non-zero instead.
        """
        try:
            self.render()
        except Exception:
            import traceback

            log("Render thread crashed:\n" + traceback.format_exc())
            try:
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(1)

    def write_bytes_to_cv2_frame_debug(self, frame):
        # Convert the byte array to a numpy array
        frame_array = np.frombuffer(frame, dtype=np.uint8)
        # Reshape the array to the correct dimensions
        frame_array = frame_array.reshape((self.height, self.width, 3))
        # Convert the BGR image to RGB
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        cv2.imwrite("frame.jpg", frame_array)

    @staticmethod
    def _as_bytes(frame):
        """Engines consume/produce raw frame bytes; the read buffer yields
        Frame objects — convert at the boundary, keep bytes untouched."""
        return frame if isinstance(frame, bytes) else frame.get_frame_bytes()

    def render(self):
        frames_rendered = 0
        """
        from viztracer import VizTracer
        tracer = VizTracer()
        tracer.start()
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()
        """

        while True:
            if not self.informationHandler.get_is_paused():
                frame = self.readBuffer.get()
                #self.write_bytes_to_cv2_frame_debug(frame)
                if frame is None:
                    self.informationHandler.stopWriting()
                    break

                for extraRestoration in self.extraRestorationModels:
                    frame = extraRestoration(self._as_bytes(frame))

                if self.interpolateModel:
                    sceneDetect = self.sceneDetect.detect(frame)
                    interpolated_frames = self.interpolateOption(
                        img1=frame,
                        transition=sceneDetect,
                    )
                    if not interpolated_frames:
                        return
                    
                    for interpolated_frame in interpolated_frames:

                        if self.upscaleModel:
                            interpolated_frame = self.upscaleOption(
                                self._as_bytes(interpolated_frame)
                            )
                        if self.override_upscale_scale:
                            interpolated_frame = resize_image_bytes(self._as_bytes(interpolated_frame),
                                               width=self.width*self.modelScale,
                                               height=self.height*self.modelScale,
                                               target_width=self.width*self.override_upscale_scale,
                                               target_height=self.height*self.override_upscale_scale,)
                        self.informationHandler.setPreviewFrame(interpolated_frame.get_frame_bytes() if type(interpolated_frame) != bytes else interpolated_frame)
                        self.informationHandler.setFramesRendered(frames_rendered)
                        self.writeBuffer.writeQueue.put(interpolated_frame.get_frame_bytes() if type(interpolated_frame) != bytes else interpolated_frame)
                
                

                if self.upscaleModel:
                    frame = self.upscaleOption(
                        self._as_bytes(frame)
                    )



                if self.override_upscale_scale:
                    frame = resize_image_bytes(self._as_bytes(frame),
                                               width=self.width*self.modelScale,
                                               height=self.height*self.modelScale,
                                               target_width=self.width*self.override_upscale_scale,
                                               target_height=self.height*self.override_upscale_scale,)

                
                self.informationHandler.setFramesRendered(frames_rendered)
                self.informationHandler.setPreviewFrame(frame.get_frame_bytes() if type(frame) != bytes else frame)
                
                self.writeBuffer.writeQueue.put(frame.get_frame_bytes() if type(frame) != bytes else frame)
                frames_rendered += int(self.ceilInterpolateFactor)
            else:
                sleep(1)
        self.writeBuffer.writeQueue.put(None)
        """
        tracer.stop()
        tracer.save()
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        """
    
    def upscaleONNXObject(self, scale=None, modelPath=None):
        from .onnx.UpscaleONNX import UpscaleONNX
        return UpscaleONNX(
            modelPath=modelPath,
            device=self.device,
            gpu_id=self.gpu_id,
            precision=self.precision,
            width=self.width,
            height=self.height,
            scale=scale if scale is not None else self.modelScale,
            hdr_mode=self.hdr_mode,
        )

    def setupExtraRestoration(self, modelPath):
        log("Setting up Extra Restoration")
        # restoration models are 1x — reuse the upscale engine
        return self.upscaleONNXObject(scale=1, modelPath=modelPath)

    def setupUpscale(self):
        log("Setting up Upscale")
        from .onnx.UpscaleONNX import UpscaleONNX

        self.modelScale = UpscaleONNX.getModelScale(self.upscaleModel)
        self.upscaleOption = self.upscaleONNXObject(
            scale=self.modelScale, modelPath=self.upscaleModel
        )
        self.upscaleTimes = (
            self.modelScale
            if not self.override_upscale_scale
            else self.override_upscale_scale
        )

    def setupInterpolate(self):
        log("Setting up Interpolation")
        self.sceneDetect = SceneDetect(
            sceneChangeMethod=self.sceneDetectMethod,
            sceneChangeSensitivity=self.sceneDetectSensitivty,
            width=self.width,
            height=self.height,
            model_path=self.sceneDetectModelPath,
            model_backend=self.backend,
            model_dtype=self.precision,
            model_device=self.device,
            model_gpu_id=self.gpu_id,
        )
        if self.sceneDetectMethod != "none":
            log("Scene Detection Enabled")
        else:
            log("Scene Detection Disabled")

        from .onnx.InterpolateONNX import InterpolateONNX

        self.interpolateOption = InterpolateONNX(
            modelPath=self.interpolateModel,
            interpolateFactor=self.ceilInterpolateFactor,
            width=self.width,
            height=self.height,
            deviceID=self.gpu_id,
            precision=self.precision,
            hdr_mode=self.hdr_mode,
            device=self.device,
            ensemble=self.ensemble,
        )
