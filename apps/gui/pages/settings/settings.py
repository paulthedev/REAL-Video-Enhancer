import os
import re

from PySide6.QtWidgets import QMainWindow, QFileDialog
from apps.gui.constants import PLATFORM, HOME_PATH
from apps.gui.util import currentDirectory, checkForWritePermissions, open_folder, log, FileHandler
from apps.gui.lib.QTcustom import RegularQTPopup
from apps.gui.GenerateFFMpegCommand import FFMpegCommand
from apps.gui.VideoInfo import VideoLoader

class SettingsTab:
    def __init__(
        self,
        parent: QMainWindow,
        halfPrecisionSupport,
        backend_output,
    ):
        self.parent = parent
        self.input_file = None
        self.ffmpegInfoWrapper = None
        self.color_space = None
        self.color_primaries = None
        self.color_transfer = None
        self.in_pix_fmt = ""
        self.hdr_mode = False
        self.ffmpeg_settings_dict = {
            "encoder": self.parent.encoder,
            "audio_encoder": self.parent.audio_encoder,
            "audio_bitrate": self.parent.audio_bitrate,
            "video_pixel_format": self.parent.video_pixel_format,
            "video_quality": self.parent.video_quality,

        }
        self.settings = Settings()
        log("Settings: " + str(self.settings.settings))
        self.connectSettingText() # has to be in this order, otherwise settings wont stick because they get reset.
        self.connectWriteSettings()
        

        # disable half option if its not supported
        if not halfPrecisionSupport:
            self.parent.precision.removeItem(1)

        # populate the GPU dropdown with the friendly names of the GPUs
        # reported by the backend info output
        self._populate_gpu_combo(
            self.parent.gpu_id,
            self._gpu_names(backend_output, "pytorch")
            or self._gpu_names(backend_output, "ncnn"),
        )
        self.parent.openRVEFolderBtn.clicked.connect(lambda:open_folder(currentDirectory()))

    @staticmethod
    def _gpu_names(backend_output: str, gpu_type: str) -> list:
        """The names of the GPUs reported in the backend info output
        (lines like 'PyTorch GPU 1: NVIDIA GeForce ...')."""
        names = []
        for line in backend_output.split("\n"):
            match = re.match(rf"^{gpu_type} gpu \d+: (.+)$", line.strip(), re.IGNORECASE)
            if match:
                names.append(match.group(1).strip())
        return names

    @staticmethod
    def gpu_index(backend_output: str, gpu_type: str, name: str) -> int:
        """The index of a GPU name in the backend info output, 0 when it
        is not found (e.g. 'Auto' or a GPU that is no longer present)."""
        names = SettingsTab._gpu_names(backend_output, gpu_type)
        for i, n in enumerate(names):
            if n == name:
                return i
        return 0

    def _populate_gpu_combo(self, combo, gpu_names: list):
        combo.clear()
        combo.addItem("Auto")
        for name in gpu_names:
            combo.addItem(name)
        

        self.updateFFMpegCommand()

    def updateFFMpegCommand(self):
        """
        Updates the FFMpegCommand object with the current settings.
        """
        for key, value in self.ffmpeg_settings_dict.items():
            self.settings.writeSetting(key, value.currentText())
        
        self.out_pixel_fmt = self.settings.settings['video_pixel_format']
        pxfmtDict = {
                "yuv420p": "yuv420p",
                "yuv422p": "yuv422p",
                "yuv444p": "yuv444p",
                "yuv420p (10 bit)": "yuv420p10le",
                "yuv422p (10 bit)": "yuv422p10le",
                "yuv444p (10 bit)": "yuv444p10le",
            }
        self.out_pixel_fmt = pxfmtDict[self.out_pixel_fmt]

        input_file = self.parent.inputFileText.text()
        if input_file and len(input_file) > 1: # caching is nice
            if self.input_file != input_file:
                self.input_file = input_file
                self.ffmpegInfoWrapper = VideoLoader(self.input_file)
                self.ffmpegInfoWrapper.loadVideo()
                self.ffmpegInfoWrapper.getData()
                self.hdr_mode = (self.ffmpegInfoWrapper.is_hdr) and self.settings.settings['auto_hdr_mode'] == "True"
                self.color_space = self.ffmpegInfoWrapper.color_space
                self.color_primaries = self.ffmpegInfoWrapper.color_primaries
                self.color_transfer = self.ffmpegInfoWrapper.color_transfer
                self.in_pix_fmt = self.ffmpegInfoWrapper.pixel_format


        if self.hdr_mode or ("10" in self.in_pix_fmt and self.settings.settings['auto_hdr_mode'] == "True"):
            pxfmtDict = {
                        "yuv420p": "yuv420p10le",
                        "yuv422p": "yuv422p10le",
                        "yuv444p": "yuv444p10le",
                    }

            if self.out_pixel_fmt in pxfmtDict:
                self.out_pixel_fmt = pxfmtDict[self.out_pixel_fmt]


        command = FFMpegCommand(
            self.settings.settings['encoder'].replace(' (experimental)', '').replace(' (40 series and up)', ''),
            self.settings.settings['video_encoder_speed'],
            self.settings.settings['video_quality'],
            self.out_pixel_fmt,
            self.settings.settings['audio_encoder'],
            self.settings.settings['audio_bitrate'],
            self.settings.settings['subtitle_encoder'],  
            self.hdr_mode,
            self.color_space if self.in_pix_fmt != "yuv420p" else None,
            self.color_primaries,
            self.color_transfer,  
        ).build_command()
        self.parent.EncoderCommand.setText(" ".join(command))
        self.parent.updateVideoGUIText()
         

    def connectWriteSettings(self):
        
        self.parent.subtitle_encoder.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "subtitle_encoder", self.parent.subtitle_encoder.currentText()
            )
        )

        self.parent.precision.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "precision", self.parent.precision.currentText()
            )
        )
        self.parent.preview_enabled.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "preview_enabled",
                "True" if self.parent.preview_enabled.isChecked() else "False",
            )
        )
        self.parent.scene_change_detection_enabled.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_change_detection_enabled",
                "True"
                if self.parent.scene_change_detection_enabled.isChecked()
                else "False",
            )
        )
        self.parent.discord_rich_presence.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "discord_rich_presence",
                "True" if self.parent.discord_rich_presence.isChecked() else "False",
            )
        )
        self.parent.scene_change_detection_method.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_change_detection_method",
                self.parent.scene_change_detection_method.currentText(),
            )
        )
        self.parent.scene_change_detection_threshold.valueChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_change_detection_threshold",
                str(self.parent.scene_change_detection_threshold.value()),
            )
        )
        self.parent.video_quality.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_quality",
                str(self.parent.video_quality.currentText()),
            )
        )
        self.parent.output_folder_location.textChanged.connect(
            lambda: self.writeOutputFolder()
        )
        self.parent.use_same_output_folder_as_input_file_enabled.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "use_same_output_folder_as_input_file_enabled",
                "True" if self.parent.use_same_output_folder_as_input_file_enabled.isChecked() else "False",
            )
        )

        self.parent.resetSettingsBtn.clicked.connect(self.resetSettings)

        self.parent.uhd_mode.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "uhd_mode",
                "True" if self.parent.uhd_mode.isChecked() else "False",
            )
        )
        self.parent.torch_compile_enabled.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "torch_compile_enabled",
                "True" if self.parent.torch_compile_enabled.isChecked() else "False",
            )
        )
        self.parent.gpu_id.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "gpu_id", self.parent.gpu_id.currentText()
            )
        )
        self.parent.auto_border_cropping.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "auto_border_cropping",
                "True" if self.parent.auto_border_cropping.isChecked() else "False",
            )
        )
        self.parent.video_container.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_container", self.parent.video_container.currentText()
            )
        )
        self.parent.video_pixel_format.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_pixel_format", self.parent.video_pixel_format.currentText()
            )
        )
        self.parent.pytorch_version.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "pytorch_version", self.parent.pytorch_version.currentText()
            )
        )
        self.parent.pytorch_backend.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "pytorch_backend", self.parent.pytorch_backend.currentText()
            )
        )
        
        self.parent.auto_hdr_mode.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "auto_hdr_mode",
                "True" if self.parent.auto_hdr_mode.isChecked() else "False"
            )
        )
        self.parent.video_encoder_speed.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "video_encoder_speed", self.parent.video_encoder_speed.currentText()
            )
        )
        self.parent.inputFileText.textChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.encoder.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.audio_encoder.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.video_pixel_format.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.audio_bitrate.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.auto_hdr_mode.stateChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.video_quality.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.video_encoder_speed.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )
        self.parent.subtitle_encoder.currentIndexChanged.connect(
            self.updateFFMpegCommand
        )

    def writeOutputFolder(self):
        outputlocation = self.parent.output_folder_location.text()
        if not (os.path.exists(outputlocation) and os.path.isdir(outputlocation)):
            RegularQTPopup(
                "No videos folder\nSetting default output folder to home directory."
            )
            outputlocation = HOME_PATH
        if not checkForWritePermissions(outputlocation):
            RegularQTPopup(
                "No permissions to export here!\nSetting default output folder to home directory."
            )
            outputlocation = HOME_PATH

        self.settings.writeSetting(
            "output_folder_location",
            str(outputlocation),
        )

    def resetSettings(self):
        for i in range(10): # idk why, but settings wont fully reset until like 5 button presses.
            self.settings.writeDefaultSettings()
            self.settings.readSettings()
            self.connectSettingText()
            self.parent.switchToSettingsPage()

    def connectSettingText(self):
        if PLATFORM == "darwin":
            index = self.parent.encoder.findText("av1")
            self.parent.encoder.removeItem(index)

        self.parent.precision.setCurrentText(self.settings.settings["precision"])
        self.parent.encoder.setCurrentText(self.settings.settings["encoder"])
        self.parent.audio_encoder.setCurrentText(
            self.settings.settings["audio_encoder"]
        )
        self.parent.audio_bitrate.setCurrentText(
            self.settings.settings["audio_bitrate"]
        )
        self.parent.preview_enabled.setChecked(
            self.settings.settings["preview_enabled"] == "True"
        )
        self.parent.discord_rich_presence.setChecked(
            self.settings.settings["discord_rich_presence"] == "True"
        )
        self.parent.scene_change_detection_enabled.setChecked(
            self.settings.settings["scene_change_detection_enabled"] == "True"
        )
        self.parent.scene_change_detection_method.setCurrentText(
            self.settings.settings["scene_change_detection_method"]
        )
        self.parent.scene_change_detection_threshold.setValue(
            float(self.settings.settings["scene_change_detection_threshold"])
        )
        self.parent.video_quality.setCurrentText(
            self.settings.settings["video_quality"]
        )
        self.parent.output_folder_location.setText(
            self.settings.settings["output_folder_location"]
        )
        self.parent.select_output_folder_location_btn.clicked.connect(
            self.selectOutputFolder
        )
        self.parent.use_same_output_folder_as_input_file_enabled.setChecked(
            self.settings.settings["use_same_output_folder_as_input_file_enabled"] == "True"
        )
        self.parent.uhd_mode.setChecked(self.settings.settings["uhd_mode"] == "True")
        self.parent.torch_compile_enabled.setChecked(
            self.settings.settings["torch_compile_enabled"] == "True"
        )
        self.parent.gpu_id.setCurrentText(
            self.settings.settings["gpu_id"]
        )
        self.parent.auto_border_cropping.setChecked(
            self.settings.settings["auto_border_cropping"] == "True"
        )
        self.parent.video_container.setCurrentText(
            self.settings.settings["video_container"]
        )
        self.parent.video_pixel_format.setCurrentText(
            self.settings.settings["video_pixel_format"]
        )
        self.parent.pytorch_version.setCurrentText(
            self.settings.settings["pytorch_version"]
        )
        self.parent.pytorch_backend.setCurrentText(
            self.settings.settings["pytorch_backend"]
        )
        self.parent.auto_hdr_mode.setChecked(
            self.settings.settings["auto_hdr_mode"] == "True"
        )
        self.parent.video_encoder_speed.setCurrentText(
            self.settings.settings["video_encoder_speed"]
        )
        self.parent.subtitle_encoder.setCurrentText(
            self.settings.settings["subtitle_encoder"]
        )

    def selectOutputFolder(self):
        outputFile = QFileDialog.getExistingDirectory(
            parent=self.parent,
            caption="Select Folder",
            dir=os.path.expanduser("~"),
        )
        outputlocation = outputFile
        if os.path.exists(outputlocation) and os.path.isdir(outputlocation):
            if checkForWritePermissions(outputlocation):
                self.settings.writeSetting(
                    "output_folder_location",
                    str(outputlocation),
                )
                self.parent.output_folder_location.setText(outputlocation)
            else:
                RegularQTPopup("No permissions to export here!")


class Settings:
    def __init__(self):
        self.settingsFile = os.path.join(currentDirectory(), "settings.txt")

        """
        The default settings are set here, and are overwritten by the settings in the settings file if it exists and the legnth of the settings is the same as the default settings.
        The key is equal to the name of the widget of the setting in the settings tab.
        """
        output_folder_default = FileHandler.getDefaultOutputFolder()
        self.defaultSettings = {
            "precision": "auto",
            "encoder": "libx264",
            "video_encoder_speed": "medium",
            "audio_encoder": "copy_audio",
            "subtitle_encoder": "copy_subtitle",
            "audio_bitrate": "192k",
            "preview_enabled": "True",
            "scene_change_detection_method": "sudo_scene_detect",
            "scene_change_detection_enabled": "True",
            "scene_change_detection_threshold": "3.5",
            "discord_rich_presence": "False",
            "video_quality": "High",
            "output_folder_location": output_folder_default,
            "use_same_output_folder_as_input_file_enabled": "False",
            "last_input_folder_location": output_folder_default,
            "uhd_mode": "True",
            "torch_compile_enabled": "True",
            "gpu_id": "Auto",
            "auto_border_cropping": "False",
            "video_container": "mkv",
            "video_pixel_format": "yuv420p",
            "pytorch_version": "2.14.0",
            "pytorch_backend": "CUDA",
            "auto_hdr_mode": "True",
        }
        self.allowedSettings = {
            "precision": ("auto", "float32", "float16"),
            "encoder": (
                "libx264",
                "libx265",
                "vp9",
                "av1",
                "prores",
                "ffv1",
                "utvideo",
                "x264_nvenc",
                "x265_nvenc",
                "av1_nvenc (40 series and up)",
            ),
            "video_encoder_speed": ("placebo","slow", "medium", "fast", "fastest"),
            "audio_encoder": ("aac", "libmp3lame", "opus", "copy_audio"),
            "audio_bitrate": "ANY",
            "subtitle_encoder": ("copy_subtitle","srt","ass","webvtt"),
            "preview_enabled": ("True", "False"),
            "scene_change_detection_method": (
                "mean",
                "mean_segmented",
                "pyscenedetect",
                "sudo_scene_detect",
            ),
            "scene_change_detection_enabled": ("True", "False"),
            "scene_change_detection_threshold": [
                str(num / 10) for num in range(1, 100)
            ],
            "discord_rich_presence": ("True", "False"),
            "video_quality": ("Low", "Medium", "High", "Very_High", "Ultra", "Lossless"),
            "output_folder_location": "ANY",
            "use_same_output_folder_as_input_file_enabled": ("True", "False"),
            "last_input_folder_location": "ANY",
            "uhd_mode": ("True", "False"),
            "torch_compile_enabled": ("True", "False"),
            "gpu_id": "ANY",
            "auto_border_cropping": ("True", "False"),
            "video_container": ("mkv", "mp4", "mov", "webm", "avi"),
            "video_pixel_format": "ANY",
            "pytorch_version": ("2.15.0.dev", "2.14.0"),
            "pytorch_backend": "ANY",
            "auto_hdr_mode": ("True", "False"),
        }
        self.settings = self.defaultSettings.copy()
        if not os.path.isfile(self.settingsFile):
            self.writeDefaultSettings()
        self.readSettings()
        # check if the settings file is corrupted
        if len(self.defaultSettings) != len(self.settings):
            self.writeDefaultSettings()
        
    def readSettings(self):
        """
        Reads the settings from the 'settings.txt' file and stores them in the 'settings' dictionary.

        Returns:
            None
        """
        with open(self.settingsFile, "r") as file:
            try:
                for line in file:
                    key, value = line.strip().split(",")
                    self.settings[key] = value
            except (
                ValueError
            ):  # writes and reads again if the settings file is corrupted
                self.writeDefaultSettings()
                self.readSettings()

    def writeSetting(self, setting: str, value: str):
        """
        Writes the specified setting with the given value to the settings dictionary.

        Parameters:
        - setting (str): The name of the setting to be written, this will be equal to the widget name in the settings tab if set correctly.
        - value (str): The value to be assigned to the setting.

        Returns:
        None
        """
        self.settings[setting] = value
        self.writeOutCurrentSettings()

    def writeDefaultSettings(self):
        """
        Writes the default settings to the settings file if it doesn't exist.

        Parameters:
            None

        Returns:
            None
        """
        self.settings = self.defaultSettings.copy()
        self.writeOutCurrentSettings()

    def writeOutCurrentSettings(self):
        """
        Writes the current settings to a file.

        Parameters:
            self (SettingsTab): The instance of the SettingsTab class.

        Returns:
            None
        """
        with open(self.settingsFile, "w") as file:
            for key, value in self.settings.items():
                if key in self.defaultSettings:  # check if the key is valid
                    if (
                        value in self.allowedSettings[key]
                        or self.allowedSettings[key] == "ANY"
                    ):  # check if it is in the allowed settings dict
                        file.write(f"{key},{value}\n")
                else:
                    self.writeDefaultSettings()
