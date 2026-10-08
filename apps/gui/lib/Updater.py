"""Application self-update via GitHub releases."""

import os
import re
import stat
import urllib.request

from apps.gui.version import version

from apps.gui.constants import (
    HAS_NETWORK_ON_STARTUP,
    PLATFORM,
    RELEASES_LATEST_API_URL,
    RELEASE_DOWNLOAD_URL_TEMPLATE,
)
from apps.gui.util import FileHandler, log


class ApplicationUpdater:
    def __init__(self):
        if HAS_NETWORK_ON_STARTUP:
            self.tag = self.get_latest_version_tag()
            self.clean_tag = self.get_latest_version_tag(clean_tag=True)
            if PLATFORM == "win32":
                platform_name = "Windows"
            elif PLATFORM == "darwin":
                platform_name = "MacOS"
            else:
                platform_name = "Linux"

            self.file_name = f"REAL-Video-Enhancer-{self.clean_tag}-{platform_name}.zip"
            self.download_url = self.build_download_url()

    def check_for_updates(self):
        if networkCheck():
            tag = self.get_latest_version_tag(clean_tag=True)
            return not tag == version and int(tag.replace(".", "")) > int(
                version.replace(".", "")
            )  # returns true if there is a new version
        return False

    def download_new_version(self):
        
        full_download_path = os.path.join(TEMP_DOWNLOAD_PATH, self.file_name)
        DownloadProgressPopup(
            link=self.download_url,
            downloadLocation=full_download_path,
            title=f"Downloading {self.tag}",
        )
        FileHandler.unzipFile(
            os.path.join(TEMP_DOWNLOAD_PATH, self.file_name), TEMP_DOWNLOAD_PATH
        )

    def remove_old_files(self):
        FileHandler.removeFolder(BACKEND_PATH)
        FileHandler.removeFolder(LIBS_PATH)
        FileHandler.removeFile(EXE_PATH)

    def move_new_files(self):
        if PLATFORM == "linux":
            folder_to_copy_from = "bin"
        elif PLATFORM == "win32":
            folder_to_copy_from = "REAL-Video-Enhancer"
        else:
            folder_to_copy_from = "REAL-Video-Enhancer"

        FileHandler.moveFolder(
            os.path.join(TEMP_DOWNLOAD_PATH, folder_to_copy_from, "backend"),
            os.path.join(BACKEND_PATH),
        )
        FileHandler.moveFolder(
            os.path.join(TEMP_DOWNLOAD_PATH, folder_to_copy_from, EXE_NAME),
            os.path.join(EXE_PATH),
        )
        FileHandler.moveFolder(
            os.path.join(TEMP_DOWNLOAD_PATH, folder_to_copy_from, LIBS_NAME),
            os.path.join(LIBS_PATH),
        )

    def make_exe_executable(self):
        FileHandler.makeExecutable(EXE_PATH)

    def build_download_url(self):
        return RELEASE_DOWNLOAD_URL_TEMPLATE.format(tag=self.tag, file_name=self.file_name)

    def get_latest_version_tag(self, clean_tag=False) -> str:
        try:
            resp = urllib.request.urlopen(RELEASES_LATEST_API_URL, timeout=15)
            tag_name = json.load(resp)["tag_name"]
        except Exception as e:
            print(f"Failed to fetch latest version: {e}")
            tag_name = version  # return current version if it failed to get the latest versions

        if clean_tag:
            tag_name = re.sub(r"[^0-9.]", "", tag_name)

        return tag_name

    def install_new_update(self):
        if NetworkCheckPopup():
            if self.check_for_updates():
                self.download_new_version()
                self.remove_old_files()
                self.move_new_files()
                self.make_exe_executable()
                RegularQTPopup("Update complete! Please restart the app.")
            else:
                RegularQTPopup("No update available!")
