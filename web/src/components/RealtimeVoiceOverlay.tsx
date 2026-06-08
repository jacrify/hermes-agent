import { Button } from "@nous-research/ui/ui/components/button";
import { AlertTriangle, Loader2, Mic, PhoneOff, Radio } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { authedFetch } from "@/lib/api";
import { cn } from "@/lib/utils";

type VoiceState = "idle" | "connecting" | "live" | "error";

interface VoiceLogItem {
  id: number;
  tone: "event" | "error";
  text: string;
}

interface RealtimeFunctionCallItem {
  id?: string;
  type?: string;
  name?: string;
  arguments?: string;
  call_id?: string;
  status?: string;
}

const SDP_ENDPOINT = "/api/realtime/voice/sdp";
const TOOL_ENDPOINT = "/api/realtime/voice/tool";
const MAX_LOG_ITEMS = 5;

function voicePrerequisiteError(): string | null {
  if (typeof window === "undefined" || typeof navigator === "undefined") {
    return "Voice is unavailable while the page is loading";
  }

  if (!window.isSecureContext) {
    return "Voice needs HTTPS or localhost. Open Hermes through a Tailscale HTTPS URL, not http://192.168.20.162:9119.";
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    return "This browser does not expose microphone access here. Try Safari/Chrome over the Tailscale HTTPS URL.";
  }

  if (typeof RTCPeerConnection === "undefined") {
    return "This browser does not support WebRTC voice sessions.";
  }

  return null;
}

function stateLabel(state: VoiceState): string {
  switch (state) {
    case "connecting":
      return "connecting";
    case "live":
      return "live";
    case "error":
      return "error";
    case "idle":
    default:
      return "idle";
  }
}

function stateTone(state: VoiceState): string {
  switch (state) {
    case "connecting":
      return "text-warning";
    case "live":
      return "text-success";
    case "error":
      return "text-destructive";
    case "idle":
    default:
      return "text-text-secondary";
  }
}

function eventTypeFromMessage(message: MessageEvent<string>): string {
  if (typeof message.data !== "string") return "binary-event";
  try {
    const data = JSON.parse(message.data) as { type?: unknown; error?: unknown };
    if (typeof data.type === "string" && data.type.trim()) {
      return data.type;
    }
    if (data.error) return "error";
  } catch {
    /* fall through */
  }
  return "message";
}

function parseRealtimeMessage(message: MessageEvent<string>): Record<string, unknown> | null {
  if (typeof message.data !== "string") return null;
  try {
    const parsed = JSON.parse(message.data) as unknown;
    return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

function functionCallFromRealtimeEvent(
  event: Record<string, unknown>,
): RealtimeFunctionCallItem | null {
  const item = event.item;
  if (item && typeof item === "object") {
    const candidate = item as RealtimeFunctionCallItem;
    if (candidate.type === "function_call" && candidate.name) {
      return candidate;
    }
  }
  if (event.type === "conversation.item.done" && event.item && typeof event.item === "object") {
    const candidate = event.item as RealtimeFunctionCallItem;
    if (candidate.type === "function_call" && candidate.name) {
      return candidate;
    }
  }
  return null;
}

function parseFunctionArguments(raw: string | undefined): Record<string, unknown> {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : {};
  } catch {
    return {};
  }
}

export function RealtimeVoiceOverlay({ active = true }: { active?: boolean }) {
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [prerequisiteError, setPrerequisiteError] = useState<string | null>(() =>
    voicePrerequisiteError(),
  );
  const [logs, setLogs] = useState<VoiceLogItem[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dataChannelRef = useRef<RTCDataChannel | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const remoteStreamRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const logIdRef = useRef(0);
  const startRunIdRef = useRef(0);
  const startInFlightRef = useRef(false);
  const executingToolCallsRef = useRef<Set<string>>(new Set());

  const pushLog = useCallback((text: string, tone: VoiceLogItem["tone"] = "event") => {
    setLogs((current) => [
      { id: ++logIdRef.current, tone, text },
      ...current,
    ].slice(0, MAX_LOG_ITEMS));
  }, []);

  const teardownVoice = useCallback(() => {
    startRunIdRef.current += 1;
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

    const pc = pcRef.current;
    if (pc) {
      pc.ontrack = null;
      pc.oniceconnectionstatechange = null;
      pc.onconnectionstatechange = null;
      pc.getSenders().forEach((sender) => {
        if (sender.track) sender.track.stop();
      });
      pc.close();
    }
    pcRef.current = null;

    localStreamRef.current?.getTracks().forEach((track) => track.stop());
    localStreamRef.current = null;

    remoteStreamRef.current?.getTracks().forEach((track) => track.stop());
    remoteStreamRef.current = null;

    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.srcObject = null;
    }
  }, []);

  const executeRealtimeToolCall = useCallback(async (item: RealtimeFunctionCallItem) => {
    const channel = dataChannelRef.current;
    if (!channel || channel.readyState !== "open") return;

    const name = item.name?.trim();
    if (!name) return;

    const callId = item.call_id || item.id || `voice-${Date.now()}`;
    const dedupeKey = `${callId}:${name}`;
    if (executingToolCallsRef.current.has(dedupeKey)) return;
    executingToolCallsRef.current.add(dedupeKey);

    pushLog(`tool.${name}`);
    try {
      const response = await authedFetch(TOOL_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name,
          arguments: parseFunctionArguments(item.arguments),
          call_id: item.call_id,
          item_id: item.id,
        }),
      });

      if (!response.ok) {
        const text = await response.text().catch(() => response.statusText);
        throw new Error(`${response.status}: ${text || response.statusText}`);
      }

      const payload = await response.json() as { output?: unknown; call_id?: string };
      const output = typeof payload.output === "string"
        ? payload.output
        : JSON.stringify(payload.output ?? "", null, 2);

      if (channel.readyState !== "open") return;
      channel.send(JSON.stringify({
        type: "conversation.item.create",
        item: {
          type: "function_call_output",
          call_id: payload.call_id || callId,
          output,
        },
      }));
      channel.send(JSON.stringify({ type: "response.create" }));
      pushLog(`tool.${name}.done`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Tool call failed";
      pushLog(`tool.${name}.error`, "error");
      setErrorMessage(message);
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
      executingToolCallsRef.current.delete(dedupeKey);
    }
  }, [pushLog]);

  const handleRealtimeMessage = useCallback((message: MessageEvent<string>) => {
    pushLog(eventTypeFromMessage(message));
    const event = parseRealtimeMessage(message);
    if (!event) return;
    const functionCall = functionCallFromRealtimeEvent(event);
    if (functionCall) {
      void executeRealtimeToolCall(functionCall);
    }
  }, [executeRealtimeToolCall, pushLog]);

  const stopVoice = useCallback(() => {
    teardownVoice();
    setVoiceState("idle");
  }, [teardownVoice]);

  const startVoice = useCallback(async () => {
    if (startInFlightRef.current || voiceState === "connecting" || voiceState === "live") {
      return;
    }

    const prereq = voicePrerequisiteError();
    setPrerequisiteError(prereq);
    if (prereq) {
      setVoiceState("error");
      setErrorMessage(prereq);
      setLogs([]);
      pushLog("secure-context-required", "error");
      return;
    }

    startInFlightRef.current = true;
    setVoiceState("connecting");
    setErrorMessage(null);
    setLogs([]);
    pushLog("requesting microphone");

    let pc: RTCPeerConnection | null = null;
    const runId = ++startRunIdRef.current;
    const isCurrentRun = () => startRunIdRef.current === runId && pcRef.current === pc;

    try {
      const localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (startRunIdRef.current !== runId) {
        localStream.getTracks().forEach((track) => track.stop());
        return;
      }
      localStreamRef.current = localStream;

      pc = new RTCPeerConnection();
      pcRef.current = pc;

      const remoteStream = new MediaStream();
      remoteStreamRef.current = remoteStream;

      const audio = audioRef.current;
      if (audio) {
        audio.srcObject = remoteStream;
        audio.muted = false;
        audio.volume = 1;
      }

      pc.ontrack = (event) => {
        event.streams[0]?.getAudioTracks().forEach((track) => {
          remoteStream.addTrack(track);
        });
        if (!event.streams[0]) {
          remoteStream.addTrack(event.track);
        }
        void audioRef.current?.play().catch((err: unknown) => {
          pushLog(err instanceof Error ? err.message : "remote audio play failed", "error");
        });
      };

      pc.onconnectionstatechange = () => {
        pushLog(`pc.${pc?.connectionState ?? "unknown"}`);
        if (pc?.connectionState === "failed" || pc?.connectionState === "disconnected") {
          setVoiceState("error");
          setErrorMessage(`WebRTC ${pc.connectionState}`);
        }
      };

      pc.oniceconnectionstatechange = () => {
        if (pc?.iceConnectionState === "failed") {
          setVoiceState("error");
          setErrorMessage("ICE connection failed");
          pushLog("ice.failed", "error");
        }
      };

      localStream.getAudioTracks().forEach((track) => pc?.addTrack(track, localStream));

      const dataChannel = pc.createDataChannel("oai-events");
      dataChannelRef.current = dataChannel;
      dataChannel.onopen = () => {
        pushLog("datachannel.open");
        setVoiceState("live");
      };
      dataChannel.onmessage = (message: MessageEvent<string>) => {
        handleRealtimeMessage(message);
      };
      dataChannel.onerror = () => {
        pushLog("datachannel.error", "error");
        setVoiceState("error");
        setErrorMessage("Realtime data channel error");
      };
      dataChannel.onclose = () => {
        pushLog("datachannel.close");
        if (pcRef.current === pc) stopVoice();
      };

      const offer = await pc.createOffer();
      if (!isCurrentRun()) return;
      await pc.setLocalDescription(offer);
      if (!isCurrentRun()) return;

      if (!pc.localDescription?.sdp) {
        throw new Error("Failed to create local SDP offer");
      }

      pushLog("posting offer");
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
        const text = await response.text().catch(() => response.statusText);
        throw new Error(`${response.status}: ${text || response.statusText}`);
      }

      const answerSdp = await response.text();
      if (!isCurrentRun()) return;
      await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
      pushLog("remote-answer.set");
    } catch (err) {
      if (!isCurrentRun()) return;
      const message = err instanceof Error ? err.message : "Unable to start voice";
      setErrorMessage(message);
      setVoiceState("error");
      pushLog(message, "error");
      teardownVoice();
      setVoiceState("error");
    } finally {
      startInFlightRef.current = false;
    }
  }, [handleRealtimeMessage, pushLog, stopVoice, teardownVoice, voiceState]);

  useEffect(() => teardownVoice, [teardownVoice]);
  useEffect(() => {
    if (!active) stopVoice();
  }, [active, stopVoice]);

  if (!active) return null;

  const statusIcon =
    voiceState === "connecting" ? (
      <Loader2 className="h-3.5 w-3.5 animate-spin" />
    ) : voiceState === "error" ? (
      <AlertTriangle className="h-3.5 w-3.5" />
    ) : voiceState === "live" ? (
      <Radio className="h-3.5 w-3.5" />
    ) : (
      <Mic className="h-3.5 w-3.5" />
    );

  const canStart = voiceState === "idle" || voiceState === "error";
  const startDisabled = !canStart || !!prerequisiteError;
  const canStop = voiceState === "connecting" || voiceState === "live";
  const visibleMessage = prerequisiteError || errorMessage;
  const latestLog = logs[0]?.text;

  return (
    <div
      data-hermes-realtime-overlay
      aria-label={latestLog ? `Realtime voice. Last event: ${latestLog}` : "Realtime voice"}
      title={latestLog}
      className={cn(
        "pointer-events-none absolute right-2 top-2 z-20",
        "w-[min(18rem,calc(100%-1rem))]",
        "sm:right-3 sm:top-3 sm:w-[18rem]",
      )}
    >
      <audio ref={audioRef} autoPlay playsInline className="hidden" />

      <div
        className={cn(
          "pointer-events-auto rounded border border-current/25 bg-black/55 p-2 text-midground shadow-lg backdrop-blur-md",
          "font-mondwest antialiased",
        )}
      >
        <div className="flex min-w-0 items-center justify-between gap-1.5">
          <div className="flex min-w-0 items-center gap-2">
            <span className={cn("shrink-0", stateTone(voiceState))}>{statusIcon}</span>
            <div className="min-w-0">
              <p className="truncate text-xs font-medium leading-tight tracking-wide text-midground">
                realtime voice
              </p>
              <p className={cn("truncate text-[0.7rem] leading-tight", stateTone(voiceState))}>
                {stateLabel(voiceState)}
              </p>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-1">
            <Button
              ghost
              onClick={() => void startVoice()}
              disabled={startDisabled}
              className={cn(
                "h-7 rounded border border-current/25 px-2 text-xs normal-case tracking-wide",
                "bg-black/20 hover:border-current/50 disabled:opacity-40",
              )}
            >
              <Mic className="mr-1 h-3 w-3" />
              start
            </Button>
            <Button
              ghost
              onClick={stopVoice}
              disabled={!canStop}
              className={cn(
                "h-7 rounded border border-current/25 px-2 text-xs normal-case tracking-wide",
                "bg-black/20 hover:border-current/50 disabled:opacity-40",
              )}
            >
              <PhoneOff className="mr-1 h-3 w-3" />
              stop
            </Button>
          </div>
        </div>

        {visibleMessage && (
          <div className="mt-1.5 border-t border-current/15 pt-1.5 text-[0.65rem] leading-tight">
            <p
              className={cn(
                "line-clamp-2 break-words",
                prerequisiteError ? "text-warning" : "text-destructive",
              )}
              title={visibleMessage}
            >
              {visibleMessage}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
