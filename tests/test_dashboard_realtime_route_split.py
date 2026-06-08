"""Source-level guard for the dedicated Realtime dashboard page."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DashboardRealtimeRouteSplitTest(unittest.TestCase):
    def test_realtime_has_its_own_route_and_nav_item(self) -> None:
        app = (ROOT / "web/src/App.tsx").read_text()
        self.assertIn('import RealtimePage from "@/pages/RealtimePage";', app)
        self.assertIn('path: "/realtime"', app)
        self.assertIn('"/realtime": RealtimePage', app)

    def test_chat_page_does_not_mount_realtime_voice_ui(self) -> None:
        chat_page = (ROOT / "web/src/pages/ChatPage.tsx").read_text()
        self.assertNotIn("RealtimeVoiceOverlay", chat_page)

    def test_realtime_page_owns_voice_session_controls(self) -> None:
        realtime_page = (ROOT / "web/src/pages/RealtimePage.tsx").read_text()
        self.assertIn('const SDP_ENDPOINT = "/api/realtime/voice/sdp";', realtime_page)
        self.assertIn('const TOOL_ENDPOINT = "/api/realtime/voice/tool";', realtime_page)
        self.assertIn("resetSession", realtime_page)
        self.assertIn("setMicMuted", realtime_page)
        self.assertIn("setSpeakerMuted", realtime_page)


if __name__ == "__main__":
    unittest.main()
