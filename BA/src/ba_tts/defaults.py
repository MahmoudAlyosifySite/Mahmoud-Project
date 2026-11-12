"""Engine defaults. Every value that came from a measurement names the results file it came from.

All values below are final for v0.1; none is provisional.
"""

# ---- generation (VoxCPM2); per-language defaults from the sweep (6 settings, tuning texts only):
# BA/eval/results/sweep.json chose cfg 2.0 / 10 steps for Egyptian and for English
GEN_DEFAULTS = {
    "arz": {"cfg_value": 2.0, "inference_timesteps": 10},
    "en": {"cfg_value": 2.0, "inference_timesteps": 10},
    "fr": {"cfg_value": 2.0, "inference_timesteps": 10},   # French is not swept; it takes the English values
}
DEFAULT_SEED = 1234

# ---- code-switching policy for Latin words in Egyptian text: BA/eval/results/codeswitch_policy.json
CODESWITCH_POLICY = "arabic"   # canonical CER tied (0.0916 keep vs 0.0932), dialect share 0.79 vs 0.71

# ---- long text: maximum characters per generated chunk: BA/eval/results/chunk_length.json
MAX_CHUNK_CHARS = {"arz": 1000, "en": 1000, "fr": 1000}   # rule in analyze_engine.py; French takes English

# pauses inserted between chunks (seconds) and crossfade length
PAUSE_SENTENCE = 0.30
PAUSE_CLAUSE = 0.15
PAUSE_PARAGRAPH = 0.60
CROSSFADE_S = 0.015

# ---- per-chunk check: characters per second of speech must fall inside the band measured on
# real speech in Phase 1 (BA/eval/results/bakeoff.json -> failure_band_chars_per_sec)
CHARS_PER_SEC_BAND = {"arz": (4.006, 19.261), "en": (4.462, 22.389), "fr": (6.55, 31.638)}
MIN_ACTIVE_S = 0.3          # failure rule F2 of the bake-off (aggregate.py)
MIN_PEAK = 0.01             # failure rule F2 of the bake-off
OUTPUT_CLIP_LEVEL = 0.999   # |x| at or above this counts as a clipped sample in generated audio
OUTPUT_CLIP_MAX_RATIO = 0.001   # no raw VoxCPM2 bake-off output has any sample >= 0.999 (results/output_clip_check.json)
MAX_RETRIES = 2

# ---- reference audio: BA/eval/results/reference_thresholds.json
REF_SR = 16000              # VoxCPM2 encodes prompts at 16 kHz (audio_vae_config.sample_rate)
MIN_SPEECH_S = 3.0
TARGET_MIN_S = 6.0
TARGET_MAX_S = 15.0
MAX_WITH_TRANSCRIPT_S = 20.0
MAX_INPUT_S = 300.0         # longer uploads are cut to their first five minutes before analysis
# noise: 95th minus 1st percentile of 25 ms frame RMS (dB); reject below MIN, warn below WARN
REF_SNR_MIN_DB = 25.9
REF_SNR_WARN_DB = 32.7
REF_CLIP_LEVEL = 0.99
REF_CLIP_MAX_RATIO = 0.00625   # share of samples at |x| >= 0.99 before any gain change
MAX_INNER_PAUSE_S = 0.5

# ---- loudness
OUTPUT_LUFS = -20.0
REF_LUFS = -20.0
PEAK_CEILING = 0.89         # -1 dBFS

# ---- ASR used to transcribe references when no transcript is given
ASR_MODEL = "openai/whisper-large-v3-turbo"   # MIT; choice: BA/eval/results/asr_check.json

# ---- text limits
MAX_TEXT_CHARS = 5000
