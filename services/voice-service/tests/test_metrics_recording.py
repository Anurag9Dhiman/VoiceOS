"""_record_metric's dispatch (which LiveKit metric type maps to which
LatencyAggregator hop) had zero test coverage -- it's exercised live via
the "metrics_collected" event in a real session, but nothing at the unit
level confirmed the three isinstance branches actually record under the
hop names README.md/summary_line output claims they do.
"""

from __future__ import annotations

from livekit.agents import metrics
from voice_service.agent import _record_metric
from voice_service.latency import LatencyAggregator


def test_stt_metrics_record_under_stt_duration():
    aggregator = LatencyAggregator()
    metric = metrics.STTMetrics(
        label="stt", request_id="r1", timestamp=0.0, duration=0.42,
        audio_duration=1.0, streamed=True,
    )

    _record_metric(aggregator, metric)

    assert aggregator.snapshot()["stt.duration"]["count"] == 1


def test_eou_metrics_record_under_end_of_utterance_delay():
    aggregator = LatencyAggregator()
    metric = metrics.EOUMetrics(
        timestamp=0.0, end_of_utterance_delay=0.2,
        transcription_delay=0.1, on_user_turn_completed_delay=0.05,
    )

    _record_metric(aggregator, metric)

    assert aggregator.snapshot()["eou.end_of_utterance_delay"]["count"] == 1


def test_tts_metrics_record_under_tts_ttfb():
    aggregator = LatencyAggregator()
    metric = metrics.TTSMetrics(
        label="tts", request_id="r1", timestamp=0.0, ttfb=0.15,
        duration=0.5, audio_duration=0.5, cancelled=False,
        characters_count=10, streamed=True,
    )

    _record_metric(aggregator, metric)

    assert aggregator.snapshot()["tts.ttfb"]["count"] == 1


def test_an_unrecognized_metric_type_is_ignored():
    """RealtimeModel sessions don't emit STT/EOU/TTS metrics at all today
    (see _record_metric's own comment) -- whatever it does emit must not
    crash this dispatch."""
    aggregator = LatencyAggregator()

    _record_metric(aggregator, object())

    assert aggregator.snapshot() == {}
