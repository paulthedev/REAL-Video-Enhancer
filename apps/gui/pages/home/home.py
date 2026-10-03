import json
import urllib.request

from apps.gui.constants import APP_URL, RELEASES_API_URL
from apps.gui.util import openLink, networkCheck


class HomeTab:
    def __init__(self, parent):
        self.parent = parent
        self.QConnect()

    def getChangelog(self):
        changeLog = ""
        try:
            with urllib.request.urlopen(RELEASES_API_URL, timeout=15) as response:
                releases = json.load(response)
            releaseTags = [release["tag_name"] for release in releases][:5]
            releaseBodies = [release["body"].replace(r"\r\n", "") for release in releases][:5]
            for releaseTag, releaseBody in zip(releaseTags, releaseBodies):
                changeLog += "\n# " + releaseTag
                changeLog += "\n" + releaseBody
        except Exception:
            pass
        return changeLog

    def QConnect(self):
        self.parent.githubBtn.clicked.connect(lambda: openLink(APP_URL))
        self.parent.kofiBtn.clicked.connect(
            lambda: openLink("https://ko-fi.com/tntwise")
        )
        if networkCheck():
            self.parent.changeLogText.setVisible(True)
            changelog = self.getChangelog()
            self.parent.changeLogText.setMarkdown(changelog)
