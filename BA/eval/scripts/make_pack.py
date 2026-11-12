"""Build the owner's listening pack: /home/jovyan/ba-work/listening-pack/
Same 6 Egyptian sentences x 2 voices for every system that ran (from <system>-pack/), MP3 128 kbps,
plus the two reference clips, and index.md. Measurement-only material (FLEURS voices): do not publish.
"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matrix import build, refs, OUT_ROOT, PACK_TEXTS, PACK_VOICES

PACK = "/home/jovyan/ba-work/listening-pack"
SYSTEMS = ["cb-v3", "cb-v2", "voxcpm2", "voxcpm2-prompt", "namaa", "lahgtna", "omnivoice", "voicetut"]
LABEL = {"cb-v3": "Chatterbox V3 (MIT, eligible)", "cb-v2": "Chatterbox V2 (MIT, eligible)",
         "voxcpm2": "VoxCPM2 reference mode (Apache-2.0, eligible)", "voxcpm2-prompt": "VoxCPM2 prompt+transcript mode (Apache-2.0, eligible)",
         "namaa": "NAMAA-Egyptian-TTS (baseline)", "lahgtna": "lahgtna-chatterbox-v1 (baseline)",
         "omnivoice": "OmniVoice (reference only, non-commercial)", "voicetut": "VoiceTut-TTS (reference only, non-commercial)"}


def mp3(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-b:a", "128k", dst], check=True)


def main():
    os.makedirs(PACK, exist_ok=True)
    jobs = build("pack"); R = refs()
    texts = {}
    for j in jobs: texts[j["text_id"]] = j["text"]
    L = ["# BA listening pack (Phase 1)\n",
         "For the owner's ear check. Two reference voices from FLEURS (measurement only; do not publish or share).",
         "Each file: `<sentence>__<voice>__<system>.mp3`. Questions to answer per file: does it sound Egyptian (not MSA)? "
         "are all words there? is it the same voice as the reference?\n",
         "## Reference clips\n", "| voice | file | what the reference says |", "|---|---|---|"]
    for v in PACK_VOICES:
        dst = f"{PACK}/REF__{v}.mp3"; mp3(R[v]["path"], dst)
        L.append(f"| {v} ({R[v]['gender']}) | REF__{v}.mp3 | {R[v]['transcript']} |")
    L += ["\n## Sentences\n", "| id | text |", "|---|---|"]
    for t in PACK_TEXTS: L.append(f"| {t} | {texts[t]} |")
    L += ["\n## Systems\n"]
    ran = []
    for s in SYSTEMS:
        d = f"{OUT_ROOT}/{s}-pack"
        if os.path.isdir(d): ran.append(s); L.append(f"- `{s}`: {LABEL[s]}")
        else: L.append(f"- `{s}`: not run, no files")
    L += ["\n## Files\n", "| sentence | voice | " + " | ".join(ran) + " |", "|---|---|" + "---|" * len(ran)]
    for t in PACK_TEXTS:
        for v in PACK_VOICES:
            cells = []
            for s in ran:
                src = f"{OUT_ROOT}/{s}-pack/pack__{t}__{v}.wav"
                name = f"{t}__{v}__{s}.mp3"
                if os.path.exists(src): mp3(src, f"{PACK}/{name}"); cells.append(name)
                else: cells.append("(failed)")
            L.append(f"| {t} | {v} | " + " | ".join(cells) + " |")
    open(f"{PACK}/index.md", "w").write("\n".join(L) + "\n")
    print("pack:", len(os.listdir(PACK)), "files")


if __name__ == "__main__":
    main()
