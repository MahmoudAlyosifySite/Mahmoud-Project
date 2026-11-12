"""BA: the public Space (Gradio). Runs on Hugging Face ZeroGPU, on a plain GPU, or (slowly) on CPU.

Privacy: uploads are copied into a per-request temporary directory that is deleted when the
request ends, and Gradio's own copies of the uploads are deleted at the same time. The voice
built from them lives only in this browser session's memory (so you can generate again without
re-uploading) and disappears with the session. Generated audio is served from Gradio's cache,
which is swept every few minutes. No text or audio is logged; flagging and analytics are off.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HERE = Path(__file__).resolve().parent
try:
    import ba_tts  # noqa: F401   (shipped next to app.py in the Space)
except ImportError:  # running from the repository checkout
    sys.path.insert(0, str(HERE.parent / "src"))

import gradio as gr
import numpy as np
import torch

from ba_tts import BA, ReferenceAudioError, on_zerogpu
from ba_tts import defaults as D
from ba_tts.audio import load_audio
from theme import BATheme, CSS, HEADER_HTML

try:
    import spaces
    GPU = spaces.GPU
except ImportError:      # not on Hugging Face: a no-op decorator
    def GPU(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_REPO = os.environ.get("BA_MODEL_REPO", "openbmb/VoxCPM2")
MAX_TEXT = int(os.environ.get("BA_MAX_TEXT_CHARS", "600"))
MAX_UPLOAD_MB = 20
# All three passed gate G3 (BA/eval/results/engine.md). French is marked: it was tested on fewer voices
# (3 French voices, 2 Egyptian voices speaking French) and is the newest language.
LANGUAGES = [("مصري · Egyptian Arabic", "arz"), ("English", "en"), ("Français · French (beta)", "fr")]

DEVICE = "cuda" if (on_zerogpu() or torch.cuda.is_available()) else "cpu"
ENGINE = BA.from_pretrained(MODEL_REPO, device=DEVICE)   # weights from the Hub at start-up
_ = ENGINE.asr                                              # Whisper for reference transcripts
from ba_tts import watermark as _wm                         # noqa: E402
_wm._get("generator", DEVICE)

EXAMPLES = [
    ["أنا رايح الشغل دلوقتي وهرجع على المغرب إن شاء الله.", "arz"],
    ["الميعاد اتغير للساعة 7:30، والتذكرة بقت ب 150 جنيه.", "arz"],
    ["The meeting has been moved to Thursday afternoon, so please update your calendar.", "en"],
    ["Our train leaves at 6:45 in the morning, and the ticket costs $25.", "en"],
    ["Le musée est ouvert du mardi au dimanche et ferme plus tôt les jours fériés.", "fr"],
    ["Le concert commence à 21h, mais il vaut mieux arriver un peu en avance.", "fr"],
]


def _seconds_estimate(text: str) -> int:
    # A40 without torch.compile (as on ZeroGPU): ~0.055 s per character (bench.json no_optimize passage_arz);
    # 0.12 s/char + 30 s leaves more than 2x headroom for a different GPU and for voice preparation
    return int(min(170, 30 + 0.12 * len(text or "")))


def _gpu_duration(clips, transcript, text, lang, seed, steps, voice):
    return _seconds_estimate(text)


@GPU(duration=_gpu_duration)
def _synthesize(clips, transcript, text, lang, seed, steps, voice):
    """Runs on the GPU. ``clips``: list of (sr, float32 array). Returns (sr, wav, voice)."""
    if voice is None:
        voice = ENGINE.create_voice(clips, transcript=transcript or None)
    sr, wav = ENGINE.speak(text, voice, lang, seed=int(seed), inference_timesteps=int(steps) if steps else None)
    return sr, wav, voice


def _paths(ref, extra):
    out = []
    if ref:
        out.append(ref)
    for f in extra or []:
        out.append(f if isinstance(f, str) else getattr(f, "name", None) or f.get("path"))
    return [p for p in out if p]


def generate(ref, extra, transcript, consent, text, lang, seed, steps, session):
    if not consent:
        raise gr.Error("لازم تأكد إن ده صوتك أو إن معاك إذن صاحب الصوت.\n"
                       "Please confirm that this is your voice or that you have the speaker's explicit permission.")
    text = (text or "").strip()
    if not text:
        raise gr.Error("اكتب النص اللي عايز تسمعه. · Please type the text to speak.")
    if len(text) > MAX_TEXT:
        raise gr.Error(f"النص طويل: الحد {MAX_TEXT} حرف. · The text is too long: the limit is {MAX_TEXT} characters.")
    paths = _paths(ref, extra)
    if not paths:
        raise gr.Error("سجّل أو ارفع تسجيل لصوتك الأول. · Please record or upload a recording of the voice first.")
    key = hashlib.sha256(("\n".join(paths) + "\n" + (transcript or "").strip()).encode()).hexdigest()
    session = session or {}
    voice = session.get("voice") if session.get("key") == key else None
    clips = []
    try:
        if voice is None:
            if any(not os.path.exists(p) for p in paths):
                raise gr.Error("التسجيل اتمسح بعد ما استخدمناه عشان خصوصيتك. سجّل أو ارفعه تاني.\n"
                               "Your recording was deleted after use, for your privacy. Please record or upload it again.")
            with tempfile.TemporaryDirectory(prefix="ba-") as tmp:        # per-request; deleted on exit
                for i, p in enumerate(paths):
                    if os.path.getsize(p) > MAX_UPLOAD_MB * 2**20:
                        raise gr.Error(f"الملف كبير زيادة (الحد {MAX_UPLOAD_MB} ميجا). · A file is too large (limit {MAX_UPLOAD_MB} MB).")
                    local = os.path.join(tmp, f"clip{i}{Path(p).suffix}")
                    shutil.copyfile(p, local)
                    try:
                        y, sr = load_audio(local)
                    except Exception:
                        raise gr.Error("مش قادرين نقرا الملف ده كصوت. · This file could not be read as audio.")
                    clips.append((sr, y))
    finally:
        for p in paths:      # Gradio's cached copies of the uploads, also when the voice came from the session
            try:
                os.remove(p)
            except OSError:
                pass
    try:
        sr, wav, voice = _synthesize(clips, transcript, text, lang, seed, steps, voice)
    except (ReferenceAudioError, ValueError) as e:
        raise gr.Error(str(e))
    return (sr, wav), {"key": key, "voice": voice}


EXTRA_CSS = """
.ba-intro p { margin: 0 0 6px; }
.ba-intro p[lang="ar"] { font-family: var(--ba-font-ar); font-size: 17px; line-height: 1.75; color: var(--ba-ivory); }
.ba-intro p[lang="en"] { font-size: 15px; color: var(--ba-muted); }
.ba-foot { color: var(--ba-muted); font-size: 13px; margin-top: 8px; }
.gradio-container label span, .gradio-container .block-info, .gradio-container button { unicode-bidi: plaintext; }
"""


def build():
    with gr.Blocks(title="BA · Your Voice. Your Identity.", analytics_enabled=False, delete_cache=(120, 600)) as demo:
        gr.HTML(HEADER_HTML)
        gr.HTML('<div class="ba-intro"><p lang="ar" dir="rtl">صوتك إنت، بموافقتك. سجّل نفسك وإنت بتقرا بصوت عالي '
                'في مكان هادي (١٠–١٥ ثانية)، واكتب أي نص يتقال بصوتك. BA مش بيحتفظ بتسجيلاتك.</p>'
                '<p lang="en" dir="ltr">Your own voice, with your consent. Record yourself reading aloud in a quiet room '
                '(10–15 seconds) and type any text to hear it in your voice. BA does not keep your recordings. '
                'Every output carries an inaudible watermark.</p></div>')
        session = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                ref = gr.Audio(sources=["microphone", "upload"], type="filepath",
                               label="١ · سجّل أو ارفع صوتك · Record or upload your voice")
                extra = gr.File(file_count="multiple", file_types=["audio"],
                                label="تسجيلات زيادة (اختياري) · More recordings (optional)")
                transcript = gr.Textbox(lines=2, max_length=600,
                                        label="إنت قلت إيه في التسجيل؟ (اختياري) · What did you say in the recording? (optional)",
                                        info="لو سبته فاضي هنكتبه أوتوماتيك. · Leave empty and it is transcribed automatically.")
                consent = gr.Checkbox(value=False, elem_id="ba-consent",
                                      label="ده صوتي، أو معايا إذن صريح من صاحب الصوت إني أستخدمه · "
                                            "This is my voice, or I have the speaker's explicit permission to use it")
            with gr.Column(scale=1, min_width=320):
                text = gr.Textbox(lines=5, max_length=MAX_TEXT, label="٢ · النص · Text",
                                  placeholder="اكتب هنا اللي عايز تسمعه بصوتك… · Type what you want to hear…")
                lang = gr.Radio(choices=LANGUAGES, value="arz", label="لغة النص · Language of the text")
                with gr.Accordion("إعدادات متقدمة · Advanced", open=False):
                    seed = gr.Number(value=D.DEFAULT_SEED, precision=0, label="Seed")
                    steps = gr.Slider(0, 20, value=0, step=1, label="خطوات · Steps (0 = default for the language)")
                go = gr.Button("٣ · اسمع صوتك · Generate", variant="primary")
                out = gr.Audio(label="النتيجة · Result", type="numpy", interactive=False, buttons=["download"])
        gr.Examples(examples=EXAMPLES, inputs=[text, lang], label="أمثلة للنص · Example texts")
        gr.HTML('<p class="ba-foot" lang="en">Built on VoxCPM2 (OpenBMB, Apache-2.0). Reference transcription: '
                'Whisper large-v3-turbo (MIT). Watermark: AudioSeal (MIT). Use only voices you have the right to use; '
                'do not impersonate or deceive.</p>')
        go.click(generate, inputs=[ref, extra, transcript, consent, text, lang, seed, steps, session],
                 outputs=[out, session], concurrency_limit=1, show_progress="full")
    return demo


demo = build()
demo.queue(max_size=16, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(theme=BATheme(), css=CSS + EXTRA_CSS, show_error=True,
                max_file_size=f"{MAX_UPLOAD_MB}mb", enable_monitoring=False,
                server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))))
