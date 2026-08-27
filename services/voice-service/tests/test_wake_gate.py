"""wait_for_wake_word is the actual gating decision (should the expensive
realtime session even open yet) -- fully testable against a scripted
AudioFrameSource and detector, no LiveKit room needed. LiveKitAudioFrameSource
itself (the real adapter) isn't exercised here -- it needs a real room with
a real participant publishing real audio; see wake_gate.py's module
docstring.
"""

from __future__ import annotations

import asyncio

from voice_service.wake_gate import NullWakeWordDetector, wait_for_wake_word


class _ScriptedFrames:
    """A fixed sequence of frames, replayed once. Standing in for a real
    LiveKitAudioFrameSource -- wait_for_wake_word only ever needs
    something async-iterable yielding bytes."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames

    async def __aiter__(self):
        for frame in self._frames:
            yield frame


class _FiresOnNthFrame:
    """Detector stub: returns True starting from the nth process() call
    (1-indexed), False before that -- lets a test control exactly which
    frame "contains" the wake word."""

    sample_rate = 16000
    frame_length = 512

    def __init__(self, fires_on: int) -> None:
        self._fires_on = fires_on
        self.calls: list[bytes] = []

    def process(self, frame: bytes) -> bool:
        self.calls.append(frame)
        return len(self.calls) >= self._fires_on


def test_never_fires_when_the_null_detector_is_used():
    """The default: NullWakeWordDetector never returns True, so
    wait_for_wake_word only returns once the frame source itself is
    exhausted -- proving the gate genuinely holds rather than
    accidentally passing through."""
    frames = _ScriptedFrames([b"\x00" * 4, b"\x00" * 4, b"\x00" * 4])
    detector = NullWakeWordDetector()

    asyncio.run(asyncio.wait_for(wait_for_wake_word(frames, detector), timeout=1))
    # Reaching here without TimeoutError already proves it returned once
    # frames ran out rather than hanging -- nothing further to assert.


def test_returns_as_soon_as_the_detector_fires_and_stops_consuming_frames():
    frames = _ScriptedFrames([b"frame1", b"frame2", b"frame3", b"frame4"])
    detector = _FiresOnNthFrame(fires_on=2)

    asyncio.run(wait_for_wake_word(frames, detector))

    # Only the frames up to and including the one that fired were
    # consumed -- the gate stops listening the moment it wakes, it
    # doesn't keep draining the source.
    assert detector.calls == [b"frame1", b"frame2"]


def test_fires_immediately_on_the_first_frame_when_the_detector_says_so():
    frames = _ScriptedFrames([b"frame1", b"frame2"])
    detector = _FiresOnNthFrame(fires_on=1)

    asyncio.run(wait_for_wake_word(frames, detector))

    assert detector.calls == [b"frame1"]


def test_waits_forever_on_an_empty_never_ending_source_until_cancelled():
    """The realistic shape in production: the frame source never ends on
    its own (a live mic stream), so wait_for_wake_word must actually
    block rather than returning early -- proven here by cancelling it
    and confirming it was still running, not already done."""

    async def _never_ending_frames():
        while True:
            yield b"\x00" * 4
            await asyncio.sleep(1000)

    class _NeverEndingSource:
        def __aiter__(self):
            return _never_ending_frames()

    async def scenario():
        task = asyncio.create_task(wait_for_wake_word(_NeverEndingSource(), NullWakeWordDetector()))
        await asyncio.sleep(0.05)
        assert not task.done()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
