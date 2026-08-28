"""Local wake-word gating: decides WHEN to even open the expensive Gemini
Live realtime session, as opposed to agent.py's WakeState/_after_phrase,
which decide whether to act on what a session that's *already* open and
streaming heard. Today, without this module wired in, the realtime
session opens the moment a call connects and stays open for the whole
call -- billed and streaming continuously -- even while WakeState.awake
is False. A production always-on voice device would instead run a cheap
local detector continuously and only open the realtime connection once
it fires, the same way Alexa/Google Home devices work.

No microphone hardware or vendor engine license (Picovoice/Porcupine or
similar) exists in this environment, so WakeWordDetector below is a
Protocol plus a documented integration point, not a shipped real
detector -- see _build_wake_detector() in agent.py, which returns None
today. Everything here is exercised by tests except LiveKitAudioFrameSource
specifically, which needs a real room with a real participant publishing
real audio to verify live; its shape is confirmed directly against the
installed livekit-rtc package (AudioStream.from_participant,
TrackSource.SOURCE_MICROPHONE, AudioFrame.data), not merely assumed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from livekit import rtc


class WakeWordDetector(Protocol):
    """One process() call per fixed-size audio frame; True means the wake
    phrase was just detected. Modeled after Porcupine's actual shape (a
    synchronous, per-frame call, not an async stream) without depending
    on it, so a real Porcupine-backed implementation is a thin adapter
    over this, not a rewrite. frame is raw little-endian 16-bit PCM,
    mono, at sample_rate -- the format every common on-device engine
    (Porcupine, openWakeWord) expects."""

    sample_rate: int
    frame_length: int  # samples per frame this detector expects

    def process(self, frame: bytes) -> bool: ...


class NullWakeWordDetector:
    """Never fires. Not meant to be used -- entrypoint() skips the whole
    gate when no real detector is configured, rather than running frames
    through a detector that can't ever say yes. Exists so tests and
    callers have a trivial WakeWordDetector to construct without needing
    a real engine."""

    sample_rate = 16000
    frame_length = 512

    def process(self, frame: bytes) -> bool:
        return False


class AudioFrameSource(Protocol):
    """A stream of raw PCM frames, already resampled to whatever the
    detector expects. Narrow on purpose: the gating loop below is fully
    testable against a scripted fake, no LiveKit room required."""

    def __aiter__(self) -> AsyncIterator[bytes]: ...


class LiveKitAudioFrameSource:
    """Real adapter: reads one participant's microphone track via
    rtc.AudioStream, resampled to the detector's required sample_rate and
    a fixed frame length. Not live-tested -- see module docstring."""

    def __init__(
        self, participant: rtc.RemoteParticipant, *, sample_rate: int, frame_length: int
    ) -> None:
        self._participant = participant
        self._sample_rate = sample_rate
        self._frame_length = frame_length

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._frames()

    async def _frames(self) -> AsyncIterator[bytes]:
        frame_size_ms = round(self._frame_length / self._sample_rate * 1000)
        stream = rtc.AudioStream.from_participant(
            participant=self._participant,
            track_source=rtc.TrackSource.SOURCE_MICROPHONE,
            sample_rate=self._sample_rate,
            num_channels=1,
            frame_size_ms=frame_size_ms,
        )
        try:
            async for event in stream:
                yield bytes(event.frame.data)
        finally:
            await stream.aclose()


async def wait_for_wake_word(frames: AudioFrameSource, detector: WakeWordDetector) -> None:
    """Consumes frames one at a time until detector.process() returns
    True, then returns. Runs until woken or cancelled (e.g. the job
    ending) if the detector never fires -- "listen until woken", not a
    timeout-bounded operation."""
    async for frame in frames:
        if detector.process(frame):
            return
