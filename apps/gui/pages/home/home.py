import json
import urllib.request

from apps.gui.constants import APP_URL, RELEASES_API_URL
from apps.gui.util import openLink, networkCheck


class HomeTab:
    def __init__(self, parent):
        self.parent = parent
        self._build_hero_cta()
        self.QConnect()

    def _build_hero_cta(self):
        """Primary action: jump straight into the process page."""
        from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

        hero = self.parent.stackedWidget.findChild(QWidget, "homePage")
        if hero is None:
            return
        cta = QPushButton("Start Enhancing")
        cta.setProperty("variant", "primary")
        cta.setCursor(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.PointingHandCursor)
        cta.clicked.connect(self.parent.processBtn.click)
        layout = hero.layout()
        if layout is not None:
            layout.addWidget(cta)
            layout.setSpacing(14)

    def getChangelog(self):
        changeLog = ""
        try:
            with urllib.request.urlopen(RELEASES_API_URL, timeout=15) as response:
                releases = json.load(response)
            releases = releases[:1]  # latest release only; full history on GitHub
            releaseTags = [release["tag_name"] for release in releases]
            releaseBodies = [release["body"].replace(r"\r\n", "") for release in releases]
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
