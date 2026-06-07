"""Contract tests for the dashboard Realtime voice SDP endpoint."""

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server


pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")

SDP_ENDPOINT = "/api/realtime/voice/sdp"
OFFER_SDP = (
    "v=0\r\n"
    "o=- 46117327 2 IN IP4 127.0.0.1\r\n"
    "s=-\r\n"
    "t=0 0\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
    "a=rtpmap:111 opus/48000/2\r\n"
)
ANSWER_SDP = (
    "v=0\r\n"
    "o=- 46117328 2 IN IP4 127.0.0.1\r\n"
    "s=-\r\n"
    "t=0 0\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
    "a=rtpmap:111 opus/48000/2\r\n"
)


@pytest.fixture
def client():
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_auth_required = getattr(web_server.app.state, "auth_required", None)

    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    web_server.app.state.auth_required = False

    try:
        yield TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    finally:
        web_server.app.state.bound_host = prev_host
        web_server.app.state.bound_port = prev_port
        web_server.app.state.auth_required = prev_auth_required


def _auth_headers() -> dict[str, str]:
    return {
        web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN,
        "Content-Type": "application/sdp",
        "Accept": "application/sdp",
    }


def test_realtime_voice_sdp_requires_dashboard_session_token(client):
    response = client.post(
        SDP_ENDPOINT,
        content=OFFER_SDP,
        headers={"Content-Type": "application/sdp"},
    )

    assert response.status_code == 401


def test_realtime_voice_sdp_without_openai_key_fails_before_network(
    client,
    monkeypatch,
):
    monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(web_server, "load_env", lambda: {})

    def fail_if_called(**_kwargs):
        raise AssertionError("Realtime SDP endpoint must not contact OpenAI without a key")

    monkeypatch.setattr(web_server, "_post_openai_realtime_call", fail_if_called)

    response = client.post(SDP_ENDPOINT, content=OFFER_SDP, headers=_auth_headers())

    assert response.status_code in {400, 503}
    assert "OPENAI_API_KEY" in response.text or "VOICE_TOOLS_OPENAI_KEY" in response.text


def test_realtime_voice_sdp_accepts_sdp_and_returns_answer_text(
    client,
    monkeypatch,
):
    captured: dict[str, object] = {}

    monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "sk-test-realtime")
    monkeypatch.setattr(web_server, "load_env", lambda: {})

    def fake_post_openai_realtime_call(**kwargs):
        captured.update(kwargs)
        return ANSWER_SDP, "call_test_123"

    monkeypatch.setattr(
        web_server,
        "_post_openai_realtime_call",
        fake_post_openai_realtime_call,
    )

    response = client.post(SDP_ENDPOINT, content=OFFER_SDP, headers=_auth_headers())

    assert response.status_code == 200
    assert response.text == ANSWER_SDP
    assert response.headers["content-type"].startswith("application/sdp")
    assert response.headers["x-hermes-realtime-call"] == "call_test_123"
    assert captured["sdp"] == OFFER_SDP
    assert captured["api_key"] == "sk-test-realtime"
    assert captured["session"]["type"] == "realtime"
    assert captured["session"]["audio"]["output"]["voice"] == "marin"
    assert captured["session"]["tools"] == []


def test_openai_realtime_call_uses_ga_multipart_shape(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        headers = {"Location": "call_ga_123"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return ANSWER_SDP.encode("utf-8")

    def fake_request(url, *, data, method, headers):
        captured["url"] = url
        captured["data"] = data
        captured["method"] = method
        captured["headers"] = headers
        return object()

    def fake_urlopen(_request, timeout):
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(web_server.urllib.request, "Request", fake_request)
    monkeypatch.setattr(web_server.urllib.request, "urlopen", fake_urlopen)

    answer, call_id = web_server._post_openai_realtime_call(
        sdp=OFFER_SDP,
        session=web_server._default_realtime_session_config(),
        api_key="sk-test-realtime",
    )

    headers = captured["headers"]
    data = captured["data"].decode("utf-8")

    assert answer == ANSWER_SDP
    assert call_id == "call_ga_123"
    assert captured["url"] == "https://api.openai.com/v1/realtime/calls"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 30
    assert headers["Authorization"] == "Bearer sk-test-realtime"
    assert headers["Accept"] == "application/sdp"
    assert "OpenAI-Beta" not in headers
    assert 'name="sdp"' in data
    assert 'Content-Type: application/sdp' in data
    assert OFFER_SDP in data
    assert 'name="session"' in data
    assert '"type":"realtime"' in data
    assert '"audio":{"output":{"voice":"marin"}}' in data
