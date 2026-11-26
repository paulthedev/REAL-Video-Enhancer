import json
import urllib.request

from PySide6.QtCore import QThread, Signal

from apps.gui.constants import APP_URL, RELEASES_API_URL
from apps.gui.util import openLink


class _ChangelogThread(QThread):
    """Fetch the latest release body without blocking the main window."""

    loaded = Signal(str)

    def run(self):
        self.loaded.emit(HomeTab.fetch_changelog())


class HomeTab:
    def __init__(self, parent):
        self.parent = parent
        self._build_hero_cta()
        self.QConnect()

    def _build_hero_cta(self):
        """Primary action: jump straight into the process page."""
        from PySide6.QtWidgets import QPushButton, QWidget
        from PySide6.QtGui import QIcon

        hero = self.parent.stackedWidget.findChild(QWidget, "homePage")
        if hero is None:
            return
        cta = QPushButton(" Start Enhancing")
        cta.setIcon(QIcon(":/icons/icons/zap.svg"))
        cta.setProperty("variant", "primary")
        cta.setCursor(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.PointingHandCursor)
        cta.clicked.connect(self.parent.processBtn.click)
        layout = hero.layout()
        if layout is not None:
            layout.addWidget(cta)
            layout.setSpacing(14)

    @staticmethod
    def fetch_changelog() -> str:
        change_log = ""
        try:
            with urllib.request.urlopen(RELEASES_API_URL, timeout=15) as response:
                releases = json.load(response)
            releases = releases[:1]  # latest release only; full history on GitHub
            for release in releases:
                change_log += "\n# " + release["tag_name"]
                change_log += "\n" + release["body"].replace("\r\n", "")
        except Exception:
            pass
        return change_log

    def _set_changelog(self, changelog: str):
        if changelog.strip():
            self.parent.changeLogText.setMarkdown(changelog)
        else:
            self.parent.changeLogText.setPlainText(
                "Changelog unavailable right now.\n"
                "Open the GitHub page for the latest release notes."
            )

    def QConnect(self):
        self.parent.githubBtn.clicked.connect(lambda: openLink(APP_URL))

        self.parent.changeLogText.setPlainText("Loading changelog…")
        self._changelog_thread = _ChangelogThread(self.parent)
        self._changelog_thread.loaded.connect(self._set_changelog)
        self._changelog_thread.finished.connect(self._changelog_thread.deleteLater)
        self._changelog_thread.start()
