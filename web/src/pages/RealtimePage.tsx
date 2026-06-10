import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  AlertTriangle,
  Loader2,
  Mic,
  MicOff,
  PhoneOff,
  Plus,
  Radio,
  Send,
  Volume2,
  VolumeX,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";

import { ToolCall, type ToolEntry } from "@/components/ToolCall";
import { authedFetch } from "@/lib/api";
import { cn } from "@/lib/utils";

type RealtimeState = "idle" | "connecting" | "live" | "error";
type TranscriptRole = "user" | "assistant" | "system";

interface TranscriptItem {
  id: string;
  role: TranscriptRole;
  text: string;
  at: number;
}

interface RealtimeFunctionCallItem {
  id?: string;
  type?: string;
  name?: string;
  arguments?: string;
  call_id?: string;
}

const SDP_ENDPOINT = "/api/realtime/voice/sdp";
const TOOL_ENDPOINT = "/api/realtime/voice/tool";
const TRANSCRIPT_EVENT_ENDPOINT = "/api/realtime/voice/transcript-event";
const MAX_EVENTS = 18;

function autoStartRequested(pathname: string, search: string): boolean {
  const normalizedPath = pathname.replace(/\/$/, "") || "/";
  if (normalizedPath === "/voice") return true;

  const params = new URLSearchParams(search);
  const value = params.get("start") ?? params.get("autostart");
  return value === "1" || value === "true" || value === "yes";
}

function voicePrerequisiteError(): string | null {
  if (typeof window === "undefined" || typeof navigator === "undefined") {
    return "Voice is unavailable while the page is loading";
  }
  if (!window.isSecureContext) {
    return "Voice needs HTTPS or localhost.";
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return "Microphone access is unavailable in this browser context.";
  }
  if (typeof RTCPeerConnection === "undefined") {
    return "This browser does not support WebRTC voice sessions.";
  }
  return null;
}

function stateLabel(state: RealtimeState): string {
  if (state === "connecting") return "connecting";
  if (state === "live") return "live";
  if (state === "error") return "error";
  return "idle";
}

function stateTone(state: RealtimeState): string {
  if (state === "connecting") return "text-warning";
  if (state === "live") return "text-success";
  if (state === "error") return "text-destructive";
  return "text-text-secondary";
}

function parseRealtimeMessage(message: MessageEvent<string>): Record<string, unknown> | null {
  if (typeof message.data !== "string") return null;
  try {
    const parsed = JSON.parse(message.data) as unknown;
    return parsed && typeof parsed === "object"
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function eventType(event: Record<string, unknown> | null): string {
  return typeof event?.type === "string" ? event.type : "message";
}

function functionCallFromRealtimeEvent(
  event: Record<string, unknown>,
): RealtimeFunctionCallItem | null {
  const item = event.item;
  if (item && typeof item === "object") {
    const candidate = item as RealtimeFunctionCallItem;
    if (candidate.type === "function_call" && candidate.name) return candidate;
  }
  return null;
}

function realtimeFunctionCallKeys(item: RealtimeFunctionCallItem): string[] {
  return [item.call_id, item.id].filter(
    (value, index, values): value is string =>
      typeof value === "string" && value.trim() !== "" && values.indexOf(value) === index,
  );
}

function parseFunctionArguments(raw: string | undefined): Record<string, unknown> {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

function stringFromEvent(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function contentText(item: unknown): string | null {
  if (!item || typeof item !== "object") return null;
  const content = (item as { content?: unknown }).content;
  if (!Array.isArray(content)) return null;
  const parts = content
    .map((part) => {
      if (!part || typeof part !== "object") return null;
      const p = part as { text?: unknown; transcript?: unknown };
      return stringFromEvent(p.text) ?? stringFromEvent(p.transcript);
    })
    .filter((part): part is string => !!part);
  return parts.length ? parts.join("\n") : null;
}

function shouldShadowRealtimeTranscriptEvent(type: unknown): boolean {
  return (
    type === "conversation.item.input_audio_transcription.completed" ||
    type === "conversation.item.created" ||
    type === "response.audio_transcript.delta" ||
    type === "response.audio_transcript.done" ||
    type === "response.output_audio_transcript.delta" ||
    type === "response.output_audio_transcript.done" ||
    type === "response.text.delta" ||
    type === "response.text.done" ||
    type === "response.output_text.delta" ||
    type === "response.output_text.done" ||
    type === "response.done"
  );
}

function toolContext(name: string, args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (!entries.length) return name;
  const compact = entries
    .slice(0, 3)
    .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join(" ");
  return compact.length > 140 ? `${compact.slice(0, 137)}...` : compact;
}

export default function RealtimePage() {
  const location = useLocation();
  const shouldAutoStart = autoStartRequested(location.pathname, location.search);
  const [state, setState] = useState<RealtimeState>("idle");
  const [prerequisiteError, setPrerequisiteError] = useState<string | null>(() =>
    voicePrerequisiteError(),
  );
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [transcript, setTranscript] = useState<TranscriptItem[]>([]);
  const [tools, setTools] = useState<ToolEntry[]>([]);
  const [text, setText] = useState("");
  const [micMuted, setMicMuted] = useState(false);
  const [speakerMuted, setSpeakerMuted] = useState(false);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dataChannelRef = useRef<RTCDataChannel | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const remoteStreamRef = useRef<MediaStream | null>(null);
  const runIdRef = useRef(0);
  const startInFlightRef = useRef(false);
  const autoStartAttemptedRef = useRef(false);
  const executingToolCallsRef = useRef<Set<string>>(new Set());
  const completedToolCallsRef = useRef<Set<string>>(new Set());
  const pendingFunctionCallsRef = useRef<Map<string, RealtimeFunctionCallItem>>(new Map());
  const functionArgumentBuffersRef = useRef<Map<string, string>>(new Map());
  const sidebandToolsRef = useRef(false);
  const callIdRef = useRef<string | null>(null);
  const micMutedRef = useRef(micMuted);
  const speakerMutedRef = useRef(speakerMuted);

  useEffect(() => {
    micMutedRef.current = micMuted;
    localStreamRef.current
      ?.getAudioTracks()
      .forEach((track) => {
        track.enabled = !micMuted;
      });
  }, [micMuted]);

  useEffect(() => {
    speakerMutedRef.current = speakerMuted;
    if (audioRef.current) {
      audioRef.current.muted = speakerMuted;
    }
  }, [speakerMuted]);

  const pushEvent = useCallback((name: string) => {
    setEvents((current) => [name, ...current].slice(0, MAX_EVENTS));
  }, []);

  const appendTranscript = useCallback((item: Omit<TranscriptItem, "at">) => {
    setTranscript((current) => {
      const index = current.findIndex((existing) => existing.id === item.id);
      if (index >= 0) {
        const next = [...current];
        next[index] = {
          ...next[index],
          text: `${next[index].text}${item.text}`,
        };
        return next;
      }
      return [...current, { ...item, at: Date.now() }];
    });
  }, []);

  const addTranscript = useCallback((role: TranscriptRole, textValue: string, id?: string) => {
    const trimmed = textValue.trim();
    if (!trimmed) return;
    setTranscript((current) => [
      ...current,
      {
        id: id ?? `${role}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        role,
        text: trimmed,
        at: Date.now(),
      },
    ]);
  }, []);

  const teardown = useCallback(() => {
    runIdRef.current += 1;
    startInFlightRef.current = false;

    const channel = dataChannelRef.current;
    if (channel) {
      channel.onopen = null;
      channel.onmessage = null;
      channel.onerror = null;
      channel.onclose = null;
      if (channel.readyState === "open" || channel.readyState === "connecting") {
        channel.close();
      }
    }
    dataChannelRef.current = null;
    executingToolCallsRef.current.clear();
    completedToolCallsRef.current.clear();
    pendingFunctionCallsRef.current.clear();
    functionArgumentBuffersRef.current.clear();
    sidebandToolsRef.current = false;
    callIdRef.current = null;

    const pc = pcRef.current;
    if (pc) {
      pc.ontrack = null;
      pc.onconnectionstatechange = null;
      pc.oniceconnectionstatechange = null;
      pc.getSenders().forEach((sender) => sender.track?.stop());
      pc.close();
    }
    pcRef.current = null;

    localStreamRef.current?.getTracks().forEach((track) => track.stop());
    localStreamRef.current = null;
    remoteStreamRef.current?.getTracks().forEach((track) => track.stop());
    remoteStreamRef.current = null;

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.srcObject = null;
    }
  }, []);

  const stop = useCallback(() => {
    teardown();
    setState("idle");
  }, [teardown]);

  const updateTool = useCallback((tool: ToolEntry) => {
    setTools((current) => {
      const index = current.findIndex((entry) => entry.id === tool.id);
      if (index < 0) return [tool, ...current].slice(0, 30);
      const next = [...current];
      next[index] = { ...next[index], ...tool };
      return next;
    });
  }, []);

  const executeRealtimeToolCall = useCallback(async (item: RealtimeFunctionCallItem) => {
    const channel = dataChannelRef.current;
    if (!channel || channel.readyState !== "open") return;
    const name = item.name?.trim();
    if (!name) return;

    const callId = item.call_id || item.id || `realtime-${Date.now()}`;
    const dedupeKey = `${callId}:${name}`;
    if (
      executingToolCallsRef.current.has(dedupeKey) ||
      completedToolCallsRef.current.has(dedupeKey)
    ) {
      return;
    }
    executingToolCallsRef.current.add(dedupeKey);

    const args = parseFunctionArguments(item.arguments);
    const startedAt = Date.now();
    updateTool({
      kind: "tool",
      id: callId,
      tool_id: callId,
      name,
      context: toolContext(name, args),
      preview: JSON.stringify(args, null, 2),
      status: "running",
      startedAt,
    });

    try {
      const response = await authedFetch(TOOL_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          arguments: args,
          call_id: item.call_id,
          item_id: item.id,
        }),
      });
      if (!response.ok) {
        const body = await response.text().catch(() => response.statusText);
        throw new Error(`${response.status}: ${body || response.statusText}`);
      }

      const payload = (await response.json()) as { output?: unknown; call_id?: string };
      const output =
        typeof payload.output === "string"
          ? payload.output
          : JSON.stringify(payload.output ?? "", null, 2);

      updateTool({
        kind: "tool",
        id: callId,
        tool_id: callId,
        name,
        context: toolContext(name, args),
        summary: output,
        status: "done",
        startedAt,
        completedAt: Date.now(),
      });

      if (channel.readyState === "open") {
        channel.send(JSON.stringify({
          type: "conversation.item.create",
          item: {
            type: "function_call_output",
            call_id: payload.call_id || callId,
            output,
          },
        }));
        channel.send(JSON.stringify({ type: "response.create" }));
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Tool call failed";
      updateTool({
        kind: "tool",
        id: callId,
        tool_id: callId,
        name,
        context: toolContext(name, args),
        error: message,
        status: "error",
        startedAt,
        completedAt: Date.now(),
      });
      if (channel.readyState === "open") {
        channel.send(JSON.stringify({
          type: "conversation.item.create",
          item: {
            type: "function_call_output",
            call_id: callId,
            output: JSON.stringify({ error: message }),
          },
        }));
        channel.send(JSON.stringify({ type: "response.create" }));
      }
    } finally {
      completedToolCallsRef.current.add(dedupeKey);
      executingToolCallsRef.current.delete(dedupeKey);
    }
  }, [updateTool]);

  const shadowRealtimeTranscriptEvent = useCallback((event: Record<string, unknown>) => {
    const callId = callIdRef.current;
    if (!callId || !shouldShadowRealtimeTranscriptEvent(event.type)) return;
    void authedFetch(TRANSCRIPT_EVENT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ call_id: callId, event }),
    }).catch(() => undefined);
  }, []);

  const handleRealtimeMessage = useCallback((message: MessageEvent<string>) => {
    const event = parseRealtimeMessage(message);
    pushEvent(eventType(event));
    if (!event) return;
    shadowRealtimeTranscriptEvent(event);

    const type = event.type;
    const functionCall = functionCallFromRealtimeEvent(event);
    if (functionCall) {
      if (sidebandToolsRef.current) {
        return;
      }
      const keys = realtimeFunctionCallKeys(functionCall);
      for (const key of keys) {
        pendingFunctionCallsRef.current.set(key, functionCall);
        if (functionCall.arguments) {
          functionArgumentBuffersRef.current.set(key, functionCall.arguments);
        }
      }

      if (type === "response.output_item.done") {
        const bufferedArguments = keys
          .map((key) => functionArgumentBuffersRef.current.get(key))
          .find((value): value is string => typeof value === "string");
        void executeRealtimeToolCall({
          ...functionCall,
          arguments: functionCall.arguments ?? bufferedArguments,
        });
      }
      return;
    }

    if (type === "response.function_call_arguments.delta") {
      if (sidebandToolsRef.current) {
        return;
      }
      const keys = [
        stringFromEvent(event.call_id),
        stringFromEvent(event.item_id),
      ].filter((value): value is string => !!value);
      const delta = stringFromEvent(event.delta) ?? "";
      if (delta) {
        for (const key of keys) {
          functionArgumentBuffersRef.current.set(
            key,
            `${functionArgumentBuffersRef.current.get(key) ?? ""}${delta}`,
          );
        }
      }
      return;
    }

    if (type === "response.function_call_arguments.done") {
      if (sidebandToolsRef.current) {
        return;
      }
      const keys = [
        stringFromEvent(event.call_id),
        stringFromEvent(event.item_id),
      ].filter((value): value is string => !!value);
      const pending = keys
        .map((key) => pendingFunctionCallsRef.current.get(key))
        .find((item): item is RealtimeFunctionCallItem => !!item);
      const argumentsText =
        stringFromEvent(event.arguments) ??
        keys
          .map((key) => functionArgumentBuffersRef.current.get(key))
          .find((value): value is string => typeof value === "string");
      const name = stringFromEvent(event.name) ?? pending?.name;
      if (name) {
        void executeRealtimeToolCall({
          ...pending,
          id: pending?.id ?? stringFromEvent(event.item_id) ?? undefined,
          call_id: stringFromEvent(event.call_id) ?? pending?.call_id,
          name,
          type: "function_call",
          arguments: argumentsText,
        });
      }
      return;
    }

    const responseId =
      stringFromEvent(event.response_id) ??
      stringFromEvent(event.item_id) ??
      "assistant-live";

    const assistantDelta =
      stringFromEvent(event.delta) ??
      stringFromEvent(event.text) ??
      stringFromEvent(event.transcript);
    if (
      assistantDelta &&
      (type === "response.audio_transcript.delta" ||
        type === "response.output_audio_transcript.delta" ||
        type === "response.text.delta" ||
        type === "response.output_text.delta")
    ) {
      appendTranscript({ id: responseId, role: "assistant", text: assistantDelta });
      return;
    }

    if (type === "conversation.item.input_audio_transcription.completed") {
      const transcriptText = stringFromEvent(event.transcript);
      if (transcriptText) addTranscript("user", transcriptText, stringFromEvent(event.item_id) ?? undefined);
      return;
    }

    if (type === "conversation.item.created") {
      const item = event.item;
      const textValue = contentText(item);
      const role = item && typeof item === "object"
        ? stringFromEvent((item as { role?: unknown }).role)
        : null;
      if (textValue && (role === "user" || role === "assistant")) {
        addTranscript(role, textValue, stringFromEvent((item as { id?: unknown }).id) ?? undefined);
      }
    }
  }, [addTranscript, appendTranscript, executeRealtimeToolCall, pushEvent, shadowRealtimeTranscriptEvent]);

  const start = useCallback(async () => {
    if (startInFlightRef.current || state === "connecting" || state === "live") return;
    const prereq = voicePrerequisiteError();
    setPrerequisiteError(prereq);
    if (prereq) {
      setState("error");
      setError(prereq);
      return;
    }

    startInFlightRef.current = true;
    setState("connecting");
    setError(null);
    pushEvent("microphone.request");
    const runId = ++runIdRef.current;
    let pc: RTCPeerConnection | null = null;
    const isCurrentRun = () => runIdRef.current === runId && pcRef.current === pc;

    try {
      const localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (runIdRef.current !== runId) {
        localStream.getTracks().forEach((track) => track.stop());
        return;
      }
      localStream.getAudioTracks().forEach((track) => {
        track.enabled = !micMutedRef.current;
      });
      localStreamRef.current = localStream;

      pc = new RTCPeerConnection();
      pcRef.current = pc;

      const remoteStream = new MediaStream();
      remoteStreamRef.current = remoteStream;
      if (audioRef.current) {
        audioRef.current.srcObject = remoteStream;
        audioRef.current.muted = speakerMutedRef.current;
        audioRef.current.volume = 1;
      }

      pc.ontrack = (event) => {
        pushEvent(`audio.track.${event.track.kind}`);
        event.streams[0]?.getAudioTracks().forEach((track) => remoteStream.addTrack(track));
        if (!event.streams[0]) remoteStream.addTrack(event.track);
        void audioRef.current?.play().catch((err: unknown) => {
          pushEvent(err instanceof Error ? err.message : "audio.play.failed");
        });
      };
      pc.onconnectionstatechange = () => {
        pushEvent(`pc.${pc?.connectionState ?? "unknown"}`);
        if (pc?.connectionState === "failed" || pc?.connectionState === "disconnected") {
          setState("error");
          setError(`WebRTC ${pc.connectionState}`);
        }
      };
      pc.oniceconnectionstatechange = () => {
        if (pc?.iceConnectionState === "failed") {
          setState("error");
          setError("ICE connection failed");
        }
      };

      localStream.getAudioTracks().forEach((track) => pc?.addTrack(track, localStream));

      const dataChannel = pc.createDataChannel("oai-events");
      dataChannelRef.current = dataChannel;
      dataChannel.onopen = () => {
        pushEvent("datachannel.open");
        setState("live");
      };
      dataChannel.onmessage = (message: MessageEvent<string>) => handleRealtimeMessage(message);
      dataChannel.onerror = () => {
        pushEvent("datachannel.error");
        setState("error");
        setError("Realtime data channel error");
      };
      dataChannel.onclose = () => {
        pushEvent("datachannel.close");
        if (pcRef.current === pc) stop();
      };

      const offer = await pc.createOffer();
      if (!isCurrentRun()) return;
      await pc.setLocalDescription(offer);
      if (!isCurrentRun()) return;
      if (!pc.localDescription?.sdp) throw new Error("Failed to create local SDP offer");

      const response = await authedFetch(SDP_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/sdp",
          Accept: "application/sdp",
        },
        body: pc.localDescription.sdp,
      });
      if (!isCurrentRun()) return;
      if (!response.ok) {
        const body = await response.text().catch(() => response.statusText);
        throw new Error(`${response.status}: ${body || response.statusText}`);
      }
      sidebandToolsRef.current = response.headers.get("X-Hermes-Realtime-Sideband") === "1";
      callIdRef.current = response.headers.get("X-Hermes-Realtime-Call");
      pushEvent(sidebandToolsRef.current ? "sideband.tools" : "browser.tools");
      const answerSdp = await response.text();
      if (!isCurrentRun()) return;
      await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
      pushEvent("remote-answer.set");
    } catch (err) {
      if (!isCurrentRun()) return;
      const message = err instanceof Error ? err.message : "Unable to start realtime";
      setState("error");
      setError(message);
      teardown();
      setState("error");
    } finally {
      startInFlightRef.current = false;
    }
  }, [handleRealtimeMessage, pushEvent, state, stop, teardown]);

  const resetSession = useCallback(() => {
    stop();
    autoStartAttemptedRef.current = false;
    setTranscript([]);
    setTools([]);
    setEvents([]);
    setError(null);
    setText("");
  }, [stop]);

  const sendText = useCallback(() => {
    const value = text.trim();
    if (!value) return;
    const channel = dataChannelRef.current;
    if (!channel || channel.readyState !== "open") {
      setError("Realtime session is not live");
      return;
    }
    addTranscript("user", value);
    channel.send(JSON.stringify({
      type: "conversation.item.create",
      item: {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: value }],
      },
    }));
    channel.send(JSON.stringify({ type: "response.create" }));
    setText("");
  }, [addTranscript, text]);

  useEffect(() => {
    if (!shouldAutoStart || autoStartAttemptedRef.current) return;
    autoStartAttemptedRef.current = true;
    pushEvent("shortcut.autostart");
    void start();
  }, [pushEvent, shouldAutoStart, start]);

  useEffect(() => teardown, [teardown]);

  const canStart = state === "idle" || state === "error";
  const canStop = state === "connecting" || state === "live";
  const sendDisabled = state !== "live" || !text.trim();
  const statusIcon = state === "connecting"
    ? <Loader2 className="h-4 w-4 animate-spin" />
    : state === "live"
      ? <Radio className="h-4 w-4" />
      : state === "error"
        ? <AlertTriangle className="h-4 w-4" />
        : <Mic className="h-4 w-4" />;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <audio ref={audioRef} autoPlay playsInline className="hidden" />

      <div className="flex min-w-0 flex-col gap-2 border-b border-current/15 pb-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 items-center gap-2">
          <span className={cn("shrink-0", stateTone(state))}>{statusIcon}</span>
          <div className="min-w-0">
            <h2 className="font-mondwest text-display text-lg font-bold leading-tight tracking-[0.08em] text-midground">
              Realtime
            </h2>
            <div className="mt-1 flex min-w-0 items-center gap-2">
              <Badge tone={state === "live" ? "success" : state === "error" ? "destructive" : "secondary"}>
                {stateLabel(state)}
              </Badge>
              {prerequisiteError && (
                <span className="truncate text-xs text-warning">{prerequisiteError}</span>
              )}
              {error && !prerequisiteError && (
                <span className="truncate text-xs text-destructive">{error}</span>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <Button ghost onClick={resetSession} className="h-8 rounded border border-current/20 px-2 text-xs normal-case">
            <Plus className="mr-1 h-3.5 w-3.5" />
            new session
          </Button>
          <Button
            ghost
            onClick={() => setMicMuted((value) => !value)}
            className="h-8 rounded border border-current/20 px-2 text-xs normal-case"
            aria-pressed={micMuted}
          >
            {micMuted ? <MicOff className="mr-1 h-3.5 w-3.5" /> : <Mic className="mr-1 h-3.5 w-3.5" />}
            {micMuted ? "mic off" : "mic on"}
          </Button>
          <Button
            ghost
            onClick={() => setSpeakerMuted((value) => !value)}
            className="h-8 rounded border border-current/20 px-2 text-xs normal-case"
            aria-pressed={speakerMuted}
          >
            {speakerMuted ? <VolumeX className="mr-1 h-3.5 w-3.5" /> : <Volume2 className="mr-1 h-3.5 w-3.5" />}
            {speakerMuted ? "speaker off" : "speaker on"}
          </Button>
          <Button
            ghost
            onClick={() => void start()}
            disabled={!canStart || !!prerequisiteError}
            className="h-8 rounded border border-current/20 px-2 text-xs normal-case"
          >
            <Mic className="mr-1 h-3.5 w-3.5" />
            start
          </Button>
          <Button
            ghost
            onClick={stop}
            disabled={!canStop}
            className="h-8 rounded border border-current/20 px-2 text-xs normal-case"
          >
            <PhoneOff className="mr-1 h-3.5 w-3.5" />
            stop
          </Button>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <section className="flex min-h-0 min-w-0 flex-col border border-current/15 bg-black/25">
          <div className="min-h-0 flex-1 overflow-y-auto p-3">
            {transcript.length === 0 ? (
              <div className="flex h-full items-center justify-center text-sm text-text-secondary">
                {state === "live" ? "listening" : "idle"}
              </div>
            ) : (
              <div className="space-y-3">
                {transcript.map((item) => (
                  <div
                    key={item.id}
                    className={cn(
                      "max-w-[48rem] border border-current/15 px-3 py-2 text-sm leading-relaxed",
                      item.role === "user" ? "ml-auto bg-midground/5" : "mr-auto bg-black/20",
                    )}
                  >
                    <div className="mb-1 font-mondwest text-[0.65rem] uppercase tracking-[0.12em] text-text-tertiary">
                      {item.role}
                    </div>
                    <p className="whitespace-pre-wrap">{item.text}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="border-t border-current/15 p-2">
            <div className="flex min-w-0 items-end gap-2">
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendText();
                  }
                }}
                rows={2}
                className="min-h-12 flex-1 resize-none border border-current/15 bg-black/30 px-2 py-1.5 text-sm outline-none focus:border-current/40"
              />
              <Button
                ghost
                onClick={sendText}
                disabled={sendDisabled}
                className="h-12 rounded border border-current/20 px-3"
                aria-label="Send text"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </section>

        <aside className="flex min-h-0 flex-col gap-3">
          <section className="min-h-0 flex-1 border border-current/15 bg-black/20">
            <div className="border-b border-current/15 px-3 py-2 font-mondwest text-xs uppercase tracking-[0.12em] text-text-secondary">
              Tools
            </div>
            <div className="min-h-0 space-y-2 overflow-y-auto p-2">
              {tools.length === 0 ? (
                <div className="py-8 text-center text-xs text-text-tertiary">no tool calls yet</div>
              ) : (
                tools.map((tool) => <ToolCall key={tool.id} tool={tool} />)
              )}
            </div>
          </section>

          <section className="max-h-44 border border-current/15 bg-black/20">
            <div className="border-b border-current/15 px-3 py-2 font-mondwest text-xs uppercase tracking-[0.12em] text-text-secondary">
              Events
            </div>
            <div className="space-y-1 overflow-y-auto p-2 font-mono text-[0.7rem] text-text-tertiary">
              {events.length === 0 ? (
                <div>idle</div>
              ) : (
                events.map((eventName, index) => <div key={`${eventName}-${index}`}>{eventName}</div>)
              )}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
