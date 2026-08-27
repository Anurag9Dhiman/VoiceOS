"""Turn management: barge-in (cut speech fast, mark the interrupted
utterance undelivered) and the RealtimeModel's voice/turn-detection knobs.

Barge-in is Gemini Live's own server-side turn detection underneath --
AgentSession already cancels in-flight speech the moment it fires. This
module is the one place that pins our numbers into that mechanism (plan:
"cut speech < 200ms") and tracks which spoken utterances actually landed
versus got cut off, via SpeechHandle.interrupted, rather than
re-implementing audio cancellation from scratch.

min_endpointing_delay/max_endpointing_delay are deliberately gone: those
were STT+VAD endpointing concepts from the cascaded pipeline, and
RealtimeModel's turn-detection config surface (LiveKit only documents
voice= today) doesn't expose an equivalent -- see the migration plan's
risk list. min_interruption_duration IS wired, via RealtimeModel's
realtime_input_config -- not documented by LiveKit, but confirmed
directly against the installed google-genai types:
AutomaticActivityDetection.prefix_padding_ms is described by the SDK
itself as "the required duration of detected speech before start-of-speech
is committed... the lower this value the more sensitive... this also
increases the probability of false positives" -- exactly this setting's
semantics, in milliseconds where ours is seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from google.genai import types as genai_types


@dataclass(frozen=True)
class TurnManagerSettings:
    # Barge-in: anything under this is noise, not intent to interrupt.
    allow_interruptions: bool = True
    min_interruption_duration: float = 0.2

    # Gemini Live's voice for spoken output.
    voice: str = "Puck"

    def as_realtime_model_kwargs(self) -> dict[str, object]:
        return {
            "voice": self.voice,
            "realtime_input_config": genai_types.RealtimeInputConfig(
                automatic_activity_detection=genai_types.AutomaticActivityDetection(
                    prefix_padding_ms=round(self.min_interruption_duration * 1000)
                )
            ),
        }


class _InterruptibleSpeech(Protocol):
    interrupted: bool

    def add_done_callback(self, callback: object) -> None: ...


@dataclass
class UndeliveredTracker:
    """Records utterances a barge-in cut off mid-speech, so the rest of the
    system can treat them as "the user didn't hear this" instead of
    silently assuming every spoken line landed."""

    _undelivered: list[str] = field(default_factory=list)

    def track(self, handle: _InterruptibleSpeech, text: str) -> None:
        def _on_done(h: _InterruptibleSpeech) -> None:
            if h.interrupted:
                self._undelivered.append(text)

        handle.add_done_callback(_on_done)

    def drain(self) -> list[str]:
        undelivered, self._undelivered = self._undelivered, []
        return undelivered
