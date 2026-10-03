"""
File (and archive) handling utilities.

Part of the `apps.gui.util` package (Log, FileHandler, SystemInfo,
Networking, Permissions, Subprocess, Misc).
"""

import os
import stat
import shutil
import subprocess
import tarfile
import zipfile

from apps.gui.constants import CWD, HOME_PATH, PLATFORM
from apps.gui.util.Logging import log


class FileHandler:
    @staticmethod
    def getFreeSpace() -> int:
        """
        Returns the available disk space in GB.
        """
        try:
            total, used, free = shutil.disk_usage(CWD)
            available_space = free / (1024**3)
            return available_space
        except Exception as e:
            log(f"An error occurred while getting available disk space: {e}")
            return 0

    @staticmethod
    def moveFolder(prev: str, new: str):
        """
        moves a folder from prev to new
        """
        if not os.path.exists(new):
            if not os.path.isfile(new):
                shutil.move(prev, new)
            else:
                print("WARN tried to rename a file to a file that already exists")
        else:
            print("WARN tried to rename a folder to a folder that already exists")

    @staticmethod
    def unzipFile(file, outputDirectory):
        """
        Extracts a zip file in the same directory as the zip file and deletes it after extraction.
        """
        origCWD = os.getcwd()
        dir_path = os.path.dirname(os.path.realpath(file))
        os.chdir(dir_path)
        log("Extracting: " + file)
        with zipfile.ZipFile(file, "r") as f:
            f.extractall(outputDirectory)
        removeFile(file)
        os.chdir(origCWD)

    @staticmethod
    def removeFolder(folder):
        """
        Removes the folder of the current working directory
        """
        if os.path.exists(folder):
            shutil.rmtree(folder)

    @staticmethod
    def removeFile(file):
        """
        Removes the file of the current working directory
        """
        if os.path.isfile(file):
            os.remove(file)

    @staticmethod
    def copy(prev: str, new: str):
        """
        moves a folder from prev to new
        """
        if not os.path.exists(new):
            if not os.path.isfile(new):
                shutil.copytree(prev, new)
            else:
                print("WARN tried to rename a file to a file that already exists")
        else:
            print("WARN tried to rename a folder to a folder that already exists")

    @staticmethod
    def copyFile(prev: str, new: str):
        """
        moves a file from prev to a new directory (new)
        """
        if not os.path.isfile(new):
            shutil.copy(prev, new)
        else:
            print("WARN tried to rename a file to a file that already exists")

    @staticmethod
    def moveFile(prev: str, new: str):
        """
        moves a file from prev to new
        """
        if not os.path.exists(new):
            if not os.path.isfile(new):
                os.rename(prev, new)
            else:
                print("WARN tried to rename a file to a file that already exists")
        else:
            print("WARN tried to rename a folder to a folder that already exists")

    @staticmethod
    def makeExecutable(file_path):
        st = os.stat(file_path)
        os.chmod(file_path, st.st_mode | stat.S_IEXEC)

    @staticmethod
    def createDirectory(dir: str):
        if not os.path.exists(dir):
            os.mkdir(dir)

    @staticmethod
    def getUnusedFileName(base_file_name: str, outputDirectory: str, extension: str):
        """
        Returns an unused file name by adding an iteration number to the file name.
        """
        iteration = 0
        output_file = base_file_name
        while os.path.isfile(base_file_name):
            output_file = os.path.join(
                outputDirectory,
                f"{base_file_name}_({iteration}).{extension}",
            )
            iteration += 1
        return output_file

    @staticmethod
    def getDefaultOutputFolder() -> str:
        """
        Returns the default output folder based on the operating system.
        """
        videos_folder = os.path.join(f"{HOME_PATH}", "Videos")

        if PLATFORM == "linux":
            try:
                result = subprocess.run(
                    ["xdg-user-dir", "VIDEOS"], capture_output=True, text=True
                ).stdout.strip()
                if os.path.isdir(result):
                    videos_folder = result
            except Exception as e:
                log(f"An error occurred while getting the Videos folder on Linux: {e}")

        if PLATFORM == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                CSIDL_MYVIDEO = 0x000e
                SHGFP_TYPE_CURRENT = 0

                buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
                ctypes.windll.shell32.SHGetFolderPathW(
                    None, CSIDL_MYVIDEO, None, SHGFP_TYPE_CURRENT, buf
                )
                if os.path.isdir(buf.value):
                    videos_folder = buf.value
            except Exception as e:
                log(f"An error occurred while getting the Videos folder on Windows: {e}")

        if PLATFORM == "darwin":
            videos_folder = os.path.join(f"{HOME_PATH}", "Desktop")

        return videos_folder


def removeFolder(folder):
    """
    Removes the folder of the current working directory
    """
    shutil.rmtree(folder)


def copy(prev: str, new: str):
    """
    moves a folder from prev to new
    """
    if not os.path.exists(new):
        if not os.path.isfile(new):
            shutil.copytree(prev, new)
        else:
            print("WARN tried to rename a file to a file that already exists")
    else:
        print("WARN tried to rename a folder to a folder that already exists")


def copyFile(prev: str, new: str):
    """
    moves a file from prev to a new directory (new)
    """
    if not os.path.isfile(new):
        shutil.copy(prev, new)
    else:
        print("WARN tried to rename a file to a file that already exists")


def move(prev: str, new: str):
    """
    moves a file from prev to new
    """
    if not os.path.exists(new):
        if not os.path.isfile(new):
            os.rename(prev, new)
        else:
            print("WARN tried to rename a file to a file that already exists")
    else:
        print("WARN tried to rename a folder to a folder that already exists")


def makeExecutable(file_path):
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)


def createDirectory(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)


def currentDirectory():
    return CWD


def removeFile(file):
    try:
        os.remove(file)
    except Exception:
        print("Failed to remove file!")


def downloadFile(link, downloadLocation):
    import urllib.request

    with urllib.request.urlopen(link) as response, open(
        downloadLocation, "wb"
    ) as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def extractTarGZ(file):
    """
    Extracts a tar gz in the same directory as the tar file and deleted it after extraction.
    """
    origCWD = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(file))
    os.chdir(dir_path)
    log("Extracting: " + file)
    with tarfile.open(file, "r:gz") as f:
        f.extractall()
    removeFile(file)
    os.chdir(origCWD)
