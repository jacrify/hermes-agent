"""Contract tests for the dashboard Realtime voice SDP endpoint."""

import logging
import sys
from types import SimpleNamespace

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

TOOL_ENDPOINT = "/api/realtime/voice/tool"
FAKE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
FAKE_REALTIME_TOOL_DEF = {
    "type": "function",
    "name": "web_search",
    "description": "Search the web",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


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


def _json_auth_headers() -> dict[str, str]:
    return {
        web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN,
        "Content-Type": "application/json",
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
    monkeypatch.setattr(
        web_server,
        "_realtime_tool_definitions",
        lambda config=None: ([FAKE_TOOL_DEF], ["web"], ["memory"]),
    )

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
    assert captured["session"]["tool_choice"] == "auto"
    assert captured["session"]["tools"] == [FAKE_REALTIME_TOOL_DEF]
    assert "memory" in captured["session"]["instructions"]


def test_realtime_tool_definitions_flatten_chat_completion_tool_shape(monkeypatch):
    monkeypatch.setattr(web_server, "_realtime_enabled_toolsets", lambda config=None: ["web"])
    monkeypatch.setattr(
        web_server,
        "wait_for_mcp_discovery",
        lambda timeout: None,
        raising=False,
    )

    fake_model_tools = SimpleNamespace(
        get_tool_definitions=lambda **_kwargs: [FAKE_TOOL_DEF],
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    tools, enabled_toolsets, skipped_tools = web_server._realtime_tool_definitions()

    assert tools == [FAKE_REALTIME_TOOL_DEF]
    assert enabled_toolsets == ["web"]
    assert skipped_tools == []


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
    monkeypatch.setattr(
        web_server,
        "_realtime_tool_definitions",
        lambda config=None: ([FAKE_TOOL_DEF], ["web"], []),
    )

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
    assert '"tool_choice":"auto"' in data
    assert '"name":"web_search"' in data


def test_realtime_voice_tool_endpoint_executes_authenticated_call(
    client,
    monkeypatch,
):
    captured: dict[str, object] = {}

    def fake_execute_realtime_tool_call(**kwargs):
        captured.update(kwargs)
        return '{"ok": true, "result": "hello"}'

    monkeypatch.setattr(
        web_server,
        "_execute_realtime_tool_call",
        fake_execute_realtime_tool_call,
    )

    response = client.post(
        TOOL_ENDPOINT,
        json={
            "name": "web_search",
            "arguments": {"query": "Hermes"},
            "call_id": "call_123",
        },
        headers=_json_auth_headers(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "name": "web_search",
        "call_id": "call_123",
        "output": '{"ok": true, "result": "hello"}',
    }
    assert captured == {
        "name": "web_search",
        "arguments": {"query": "Hermes"},
        "call_id": "call_123",
    }


def test_realtime_tool_execution_uses_scoped_model_tools(monkeypatch):
    captured: dict[str, object] = {}

    def fake_get_tool_definitions(**kwargs):
        captured["tool_defs_kwargs"] = kwargs
        return [FAKE_TOOL_DEF]

    def fake_handle_function_call(**kwargs):
        captured["handle_kwargs"] = kwargs
        return '{"result": "ok"}'

    fake_model_tools = SimpleNamespace(
        get_tool_definitions=fake_get_tool_definitions,
        handle_function_call=fake_handle_function_call,
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
    monkeypatch.setattr(web_server, "_realtime_enabled_toolsets", lambda: ["web", "mcp-files"])

    result = web_server._execute_realtime_tool_call(
        name="web_search",
        arguments={"query": "Hermes"},
        call_id="call_456",
    )

    assert result == '{"result": "ok"}'
    assert captured["tool_defs_kwargs"] == {
        "enabled_toolsets": ["web", "mcp-files"],
        "quiet_mode": True,
    }
    assert captured["handle_kwargs"]["function_name"] == "web_search"
    assert captured["handle_kwargs"]["function_args"] == {"query": "Hermes"}
    assert captured["handle_kwargs"]["tool_call_id"] == "call_456"
    assert captured["handle_kwargs"]["enabled_toolsets"] == ["web", "mcp-files"]


def test_realtime_tool_execution_bridges_configured_terminal_cwd(monkeypatch, tmp_path):
    service_cwd = tmp_path / "hermes-agent"
    workspace_cwd = tmp_path / "workspace"
    service_cwd.mkdir()
    workspace_cwd.mkdir()
    captured: dict[str, object] = {}

    def fake_get_tool_definitions(**_kwargs):
        return [FAKE_TOOL_DEF]

    def fake_handle_function_call(**kwargs):
        captured["terminal_cwd"] = web_server.os.environ.get("TERMINAL_CWD")
        captured["process_cwd"] = web_server.os.getcwd()
        captured["handle_kwargs"] = kwargs
        return '{"result": "ok"}'

    fake_model_tools = SimpleNamespace(
        get_tool_definitions=fake_get_tool_definitions,
        handle_function_call=fake_handle_function_call,
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
    monkeypatch.setattr(web_server, "_realtime_enabled_toolsets", lambda: ["web"])
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {"terminal": {"cwd": str(workspace_cwd)}},
    )
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(service_cwd)

    result = web_server._execute_realtime_tool_call(
        name="web_search",
        arguments={"query": "Hermes"},
        call_id="call_cwd",
    )

    assert result == '{"result": "ok"}'
    assert captured["process_cwd"] == str(service_cwd)
    assert captured["terminal_cwd"] == str(workspace_cwd)
    assert captured["handle_kwargs"]["function_name"] == "web_search"


def test_realtime_tool_execution_logs_cwd_context(monkeypatch, caplog, tmp_path):
    captured: dict[str, object] = {}

    def fake_get_tool_definitions(**_kwargs):
        return [FAKE_TOOL_DEF]

    def fake_handle_function_call(**kwargs):
        captured["handle_kwargs"] = kwargs
        return '{"result": "ok"}'

    fake_model_tools = SimpleNamespace(
        get_tool_definitions=fake_get_tool_definitions,
        handle_function_call=fake_handle_function_call,
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
    monkeypatch.setattr(web_server, "_realtime_enabled_toolsets", lambda: ["web"])
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
        result = web_server._execute_realtime_tool_call(
            name="web_search",
            arguments={"query": "Hermes"},
            call_id="call_logged",
        )

    assert result == '{"result": "ok"}'
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Realtime voice tool call started name=web_search call_id=call_logged"
        in message
        and f"terminal_cwd={tmp_path}" in message
        for message in messages
    )
    assert any(
        "Realtime voice tool call finished name=web_search call_id=call_logged"
        in message
        and "output_chars=16" in message
        for message in messages
    )


def test_realtime_voice_tool_endpoint_wraps_unexpected_failures(client, monkeypatch):
    def fail_execute_realtime_tool_call(**_kwargs):
        raise RuntimeError("tool transport exploded")

    monkeypatch.setattr(
        web_server,
        "_execute_realtime_tool_call",
        fail_execute_realtime_tool_call,
    )

    response = client.post(
        TOOL_ENDPOINT,
        json={
            "name": "web_search",
            "arguments": {"query": "Hermes"},
            "call_id": "call_failure",
        },
        headers=_json_auth_headers(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["name"] == "web_search"
    assert payload["call_id"] == "call_failure"
    assert "tool transport exploded" in payload["output"]
    assert '"status": "error"' in payload["output"]


def test_realtime_tool_execution_rejects_agent_loop_tool():
    result = web_server._execute_realtime_tool_call(
        name="memory",
        arguments={},
        call_id="call_memory",
    )

    assert "not available in realtime voice yet" in result
