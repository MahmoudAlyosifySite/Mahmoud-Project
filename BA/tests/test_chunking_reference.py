import numpy as np
import pytest

from ba_tts import defaults as D
from ba_tts.audio import join, normalize_loudness, loudness_lufs
from ba_tts.chunking import band_distance, check_chunk, chunk_text, spoken_chars, split_sentences
from ba_tts.reference import ReferenceAudioError, prepare_reference, speech_segments
from ba_tts.voice import Voice

SR = 16000


def speechlike(seconds, sr=SR, seed=0, pauses=(), level=0.3, phrase_s=1.8, gap_s=0.5):
    """Harmonic 'syllables' (4 per second) in phrases of ``phrase_s`` separated by ``gap_s`` breaths,
    plus extra silent pauses at the given (start, end) seconds."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.1 + 0.9 * np.sin(2 * np.pi * 4 * t) ** 2
    y = y * env + 0.001 * rng.standard_normal(len(t))
    period = phrase_s + gap_s
    breaths = [(a, a + gap_s) for a in np.arange(phrase_s, seconds - 0.5, period)]
    for a, b in list(breaths) + list(pauses):
        y[int(a * sr):int(b * sr)] = 0.0005 * rng.standard_normal(int(b * sr) - int(a * sr))
    return (level * y / np.max(np.abs(y))).astype(np.float32)


# ------------------------------------------------------------------ chunking
def test_split_sentences_arabic_and_latin():
    s = split_sentences("أنا رايح. إنت جاي؟ تمام!\nNew paragraph. Second one.")
    assert [c.text for c in s] == ["أنا رايح.", "إنت جاي؟", "تمام!", "New paragraph.", "Second one."]
    assert s[2].pause_after == D.PAUSE_PARAGRAPH and s[0].pause_after == D.PAUSE_SENTENCE


def test_pack_respects_max_and_keeps_text():
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    for mc in (60, 150, 300):
        ch = chunk_text(text, mc)
        assert all(len(c.text) <= mc for c in ch)
        assert " ".join(c.text for c in ch) == text


def test_long_sentence_split_at_clauses_then_spaces():
    s = "، ".join(["كلام كتير قوي من غير ما نقف"] * 12) + "."
    ch = chunk_text(s, 100)
    assert len(ch) > 1 and all(len(c.text) <= 100 for c in ch)
    nopunct = " ".join(["word"] * 100)
    ch2 = chunk_text(nopunct, 120)
    assert all(len(c.text) <= 120 for c in ch2) and " ".join(c.text for c in ch2) == nopunct


def test_paragraph_breaks_end_a_chunk():
    ch = chunk_text("One.\nTwo.", 1000)
    assert [c.text for c in ch] == ["One.", "Two."]


def test_spoken_chars_matches_scorer_normalisation():
    assert spoken_chars("أَنا رايِح، الشُّغْل!", "arz") == len("انارايحالشغل")
    assert spoken_chars("It's a test-case.", "en") == len("itsatestcase")


def test_check_chunk_band():
    sr = 48000
    wav = np.concatenate([speechlike(3.0, sr), np.zeros(sr // 2, np.float32)])
    text = "ا" * 30          # 30 chars over ~3 s -> ~10 chars/s: inside the Egyptian band
    assert check_chunk(wav, sr, text, "arz").ok
    r = check_chunk(wav, sr, "ا" * 200, "arz")     # ~67 chars/s: too fast for the audio
    assert not r.ok and r.reason == "too_fast" and band_distance(r, "arz") > 0
    r = check_chunk(wav, sr, "اا", "arz")
    assert not r.ok and r.reason == "too_slow"
    r = check_chunk(np.zeros(sr, np.float32), sr, text, "arz")
    assert not r.ok and r.reason == "empty"
    clipped = np.clip(wav * 20, -1, 1)
    r = check_chunk(clipped, sr, text, "arz")
    assert not r.ok and r.reason == "clipped"


def test_join_inserts_pauses_and_fades():
    sr = 1000
    a, b = np.ones(100, np.float32), np.ones(100, np.float32)
    out = join([a, b], [0.3, 0.0], sr, fade_s=0.01)
    assert len(out) == 100 + 300 + 100
    assert out[0] == 0.0 and out[99] == 0.0 and np.all(out[100:400] == 0)


def test_loudness_normalisation_target_and_ceiling():
    y = speechlike(5.0, 48000, level=0.05)
    z = normalize_loudness(y, 48000, -20.0)
    assert abs(loudness_lufs(z, 48000) + 20.0) < 0.5 or np.max(np.abs(z)) == pytest.approx(D.PEAK_CEILING, rel=1e-3)
    assert np.max(np.abs(z)) <= D.PEAK_CEILING + 1e-6


# ------------------------------------------------------------------ reference checks
def test_reference_selects_6_to_15_s_window():
    y = speechlike(30.0, pauses=[(8.0, 8.6), (15.0, 15.7), (22.0, 22.5)])
    audio, tr, rep = prepare_reference([(SR, y)])
    secs = len(audio) / SR
    assert D.TARGET_MIN_S <= secs <= D.TARGET_MAX_S + 0.5 and tr is None and rep["mode"] == "window"


def test_reference_too_short_is_rejected_with_message():
    y = speechlike(2.0)
    with pytest.raises(ReferenceAudioError) as e:
        prepare_reference([(SR, y)])
    assert e.value.code == "short" and "seconds" in str(e.value) and "ثواني" in str(e.value)


def test_reference_silence_rejected():
    with pytest.raises(ReferenceAudioError) as e:
        prepare_reference([(SR, np.zeros(SR * 5, np.float32))])
    assert e.value.code in ("silent", "short")


def test_reference_clipped_rejected():
    y = np.clip(speechlike(10.0) * 30, -1, 1)
    with pytest.raises(ReferenceAudioError) as e:
        prepare_reference([(SR, y)])
    assert e.value.code == "clipped"


def test_reference_noisy_rejected():
    rng = np.random.default_rng(1)
    y = speechlike(10.0, pauses=[(4.0, 4.6)]) + 0.2 * rng.standard_normal(SR * 10).astype(np.float32)
    with pytest.raises(ReferenceAudioError) as e:
        prepare_reference([(SR, y)])
    assert e.value.code == "noisy"


def test_reference_clean_recording_preferred_over_noisy_one():
    rng = np.random.default_rng(2)
    noisy = speechlike(10.0, seed=3) + 0.2 * rng.standard_normal(SR * 10).astype(np.float32)
    clean = speechlike(9.0, seed=4, pauses=[(4.0, 4.5)])
    audio, _, rep = prepare_reference([(SR, noisy), (SR, clean)])
    assert rep["used"] == [1] and rep["recordings"][0]["rejected"] == "noisy"


def test_several_short_recordings_are_joined():
    parts = [(SR, speechlike(3.5, seed=i)) for i in range(3)]
    audio, _, rep = prepare_reference(parts)
    assert rep["mode"] == "joined" and len(audio) / SR >= D.TARGET_MIN_S


def test_user_transcript_keeps_whole_recording():
    y = speechlike(9.0, pauses=[(4.0, 4.4)])
    audio, tr, rep = prepare_reference([(SR, y)], ["كلام المستخدم"])
    assert tr == "كلام المستخدم" and rep["mode"] == "transcribed_by_user"
    y_long = speechlike(26.0, pauses=[(8.0, 8.6), (17.0, 17.6)])
    audio, tr, rep = prepare_reference([(SR, y_long)], ["too long to keep whole"])
    assert tr is None and rep["notes"]


def test_joint_transcript_for_several_recordings():
    parts = [(SR, speechlike(4.0, seed=i)) for i in range(2)]
    audio, tr, rep = prepare_reference(parts, joint_transcript="first and second")
    assert tr == "first and second" and rep["used"] == [0, 1]


def test_resampling_and_stereo_input():
    y44 = speechlike(10.0, sr=44100, pauses=[(5.0, 5.5)])
    stereo = np.stack([y44, y44], axis=1)
    audio, _, _ = prepare_reference([(44100, stereo)])
    assert audio.ndim == 1 and D.TARGET_MIN_S <= len(audio) / SR <= D.TARGET_MAX_S + 0.5


def test_int16_input():
    y = (speechlike(9.0, pauses=[(4.0, 4.5)]) * 32767).astype(np.int16)
    audio, _, _ = prepare_reference([(SR, y)])
    assert audio.dtype == np.float32


def test_speech_segments_find_pauses():
    y = speechlike(6.0, pauses=[(2.0, 2.8)], phrase_s=100)
    segs = speech_segments(y)
    assert len(segs) == 2 and segs[0][1] <= 2.1 and segs[1][0] >= 2.7


def test_voice_save_load_roundtrip(tmp_path):
    v = Voice(audio=speechlike(7.0), sample_rate=SR, transcript="أنا", language="arz", report={"a": 1})
    p = v.save(str(tmp_path / "voice"))
    w = Voice.load(p)
    assert np.allclose(w.audio, v.audio) and w.transcript == "أنا" and w.language == "arz" and w.mode == "prompt"
    assert Voice(audio=v.audio, sample_rate=SR, transcript=None, language=None).mode == "reference"
