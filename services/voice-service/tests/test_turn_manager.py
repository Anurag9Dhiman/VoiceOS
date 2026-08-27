"""as_realtime_model_kwargs had zero test coverage before this -- and its
min_interruption_duration field was dead config (never reached
RealtimeModel at all) until this change wired it to
realtime_input_config.automatic_activity_detection.prefix_padding_ms, the
field google-genai's own types describe as exactly this setting's
semantics (see turn_manager.py's module docstring).
"""

from __future__ import annotations

from voice_service.turn_manager import TurnManagerSettings


def test_default_settings_produce_the_expected_prefix_padding():
    kwargs = TurnManagerSettings().as_realtime_model_kwargs()

    assert kwargs["voice"] == "Puck"
    aad = kwargs["realtime_input_config"].automatic_activity_detection
    assert aad.prefix_padding_ms == 200  # 0.2s, matching the "cut speech < 200ms" plan target


def test_min_interruption_duration_scales_seconds_to_milliseconds():
    kwargs = TurnManagerSettings(min_interruption_duration=0.35).as_realtime_model_kwargs()

    aad = kwargs["realtime_input_config"].automatic_activity_detection
    assert aad.prefix_padding_ms == 350


def test_custom_voice_is_still_passed_through():
    kwargs = TurnManagerSettings(voice="Charon").as_realtime_model_kwargs()

    assert kwargs["voice"] == "Charon"
