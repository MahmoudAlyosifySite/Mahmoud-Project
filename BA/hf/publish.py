"""Build and upload the BA model repository and the BA Space (venv /tmp/ba/venvs/ba).

    source /tmp/ba/env.sh          # unsets the server's HF token
    python BA/hf/publish.py stage                  # build both staging folders under /tmp/ba/publish
    python BA/hf/publish.py upload-model           # create mahmoudalyosify/BA (private) and upload
    python BA/hf/publish.py upload-space           # create the Space (private), upload, set variable, request ZeroGPU
    python BA/hf/publish.py files                  # list both repos' files with sizes and commit
    python BA/hf/publish.py public model|space     # make one of them public
    python BA/hf/publish.py runtime                # Space runtime stage and hardware

The owner's write token is read from ~/.config/ba/hf_token inside this process and passed to
HfApi; it is never printed, logged, stored as a Space secret or written anywhere else.
Prerequisites: the wheel built with ``uv build --wheel --out-dir /tmp/ba/publish/dist BA`` and the
samples from BA/hf/make_samples.py in /home/jovyan/ba-work/publish/samples.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BA_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(BA_DIR, "src"))

OWNER = "mahmoudalyosify"
REPO = f"{OWNER}/BA"
BASE_REPO = "openbmb/VoxCPM2"
BASE_REVISION = "7936080b41163874f38610f46f438e2c0757f9fd"
STAGE = "/tmp/ba/publish"
WHEEL = f"{STAGE}/dist/ba_tts-0.1.0-py3-none-any.whl"
SAMPLES = "/home/jovyan/ba-work/publish/samples"
VOXCPM_LICENSE = "/tmp/ba/venvs/ba/lib/python3.12/site-packages/voxcpm-2.0.3.dist-info/licenses/LICENSE"
SPACE_FILES = ["app.py", "theme.py", "style.css", "header.html", "requirements.txt", "packages.txt", "README.md"]
SPACE_HARDWARE = "zero-a10g"


def api():
    from huggingface_hub import HfApi
    with open(os.path.expanduser("~/.config/ba/hf_token"), encoding="utf-8") as f:
        return HfApi(token=f.read().strip())


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    src = os.path.realpath(src)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


NOTICE = """BA (https://huggingface.co/mahmoudalyosify/BA)
BA's own work (the ba_tts code, ba_config.json, the normaliser lexicons, the model card, the
samples and the brand assets) is licensed under the BA Community License 1.0 (LICENSE).

VoxCPM2 model files (unmodified)
  BA ships the unmodified VoxCPM2 weights from openbmb/VoxCPM2 at commit {rev}.
  Copyright OpenBMB. Licensed under the Apache License, Version 2.0 (full text: LICENSE-VoxCPM2).
  Source: https://huggingface.co/openbmb/VoxCPM2 at commit {rev}
  Files:  {files}
  These files are byte-for-byte copies of the files at that commit; they were not modified,
  fine-tuned, converted or quantised. Not copied from the source: README.md (this repository's
  README.md is the BA model card) and .gitattributes.
  LICENSE-VoxCPM2 is OpenBMB's Apache-2.0 licence file as shipped in the voxcpm 2.0.3 package
  on PyPI (https://github.com/OpenBMB/VoxCPM); the openbmb/VoxCPM2 model repository declares
  apache-2.0 in its model card and carries no licence file of its own.

Components downloaded at run time (not in this repository)
  Whisper large-v3-turbo (OpenAI), MIT License: openai/whisper-large-v3-turbo at {asr_rev}
  AudioSeal (Meta), MIT License: facebook/audioseal at {wm_rev} (weights); audioseal 0.2.1 on PyPI (code)
"""


def stage():
    from huggingface_hub import snapshot_download
    import ba_tts
    from ba_tts import defaults as D
    from ba_tts.config import BASE_FILES, CONFIG_NAME, build_config
    from ba_tts.watermark import REVISION as WM_REV
    # ---- model repository
    m = f"{STAGE}/model"
    shutil.rmtree(m, ignore_errors=True)
    os.makedirs(m)
    base = snapshot_download(BASE_REPO, revision=BASE_REVISION, allow_patterns=BASE_FILES, token=False)
    for f in BASE_FILES:
        _copy(f"{base}/{f}", f"{m}/{f}")
    cfg = build_config(ba_tts.__version__, BASE_REPO, BASE_REVISION)
    with open(f"{m}/{CONFIG_NAME}", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    _copy(f"{BA_DIR}/LICENSE", f"{m}/LICENSE")
    _copy(VOXCPM_LICENSE, f"{m}/LICENSE-VoxCPM2")
    with open(f"{m}/NOTICE", "w", encoding="utf-8") as f:
        f.write(NOTICE.format(rev=BASE_REVISION, files=", ".join(BASE_FILES), asr_rev=D.ASR_REVISION, wm_rev=WM_REV))
    _copy(f"{HERE}/model_card.md", f"{m}/README.md")
    _copy(f"{BA_DIR}/assets/banner.png", f"{m}/assets/banner.png")
    _copy(f"{BA_DIR}/assets/social-card.png", f"{m}/assets/social-card.png")
    for name in sorted(os.listdir(SAMPLES)):
        if name.endswith(".mp3") and name.startswith("synthetic_"):
            _copy(f"{SAMPLES}/{name}", f"{m}/samples/{name}")
    _copy(WHEEL, f"{m}/{os.path.basename(WHEEL)}")
    # ---- Space
    s = f"{STAGE}/space"
    shutil.rmtree(s, ignore_errors=True)
    os.makedirs(s)
    for f in SPACE_FILES:
        _copy(f"{BA_DIR}/space/{f}", f"{s}/{f}")
    _copy(f"{BA_DIR}/LICENSE", f"{s}/LICENSE")
    _copy(f"{BA_DIR}/assets/social-card.png", f"{s}/social-card.png")
    shutil.copytree(f"{BA_DIR}/src/ba_tts", f"{s}/ba_tts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for root in (m, s):
        for dp, _, fns in os.walk(root):
            for fn in sorted(fns):
                p = os.path.join(dp, fn)
                print(f"{os.path.getsize(p):>12}  {os.path.relpath(p, STAGE)}")


def upload_model():
    a = api()
    a.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    info = a.upload_folder(repo_id=REPO, repo_type="model", folder_path=f"{STAGE}/model",
                           commit_message="BA v0.1.0: unmodified VoxCPM2 weights, ba_config.json, ba_tts wheel, card, "
                                          "licences, synthetic-voice samples")
    print(info)


def upload_space(private: bool = True):
    """Creates the Space with ZeroGPU hardware from the start: on a free account a Gradio Space on
    cpu-basic cannot be created (HTTP 402, PRO required), while up to 2 ZeroGPU Spaces can."""
    a = api()
    a.create_repo(REPO, repo_type="space", space_sdk="gradio", space_hardware=SPACE_HARDWARE, private=private,
                  exist_ok=True)
    a.add_space_variable(REPO, "BA_MODEL_REPO", REPO, description="model repository the app loads")
    info = a.upload_folder(repo_id=REPO, repo_type="space", folder_path=f"{STAGE}/space",
                           commit_message="BA Space: Gradio app, ba_tts source, theme, licence")
    print(info)
    try:
        rt = a.request_space_hardware(REPO, hardware=SPACE_HARDWARE)
        print("hardware requested:", rt.requested_hardware, "current:", rt.hardware, "stage:", rt.stage)
    except Exception as e:      # reported, not hidden: the owner may have to switch it in Settings
        print("HARDWARE REQUEST FAILED:", type(e).__name__, str(e)[:800])


def files():
    a = api()
    for rt in ("model", "space"):
        try:
            info = a.repo_info(REPO, repo_type=rt, files_metadata=True)
        except Exception as e:
            print(rt, "-", type(e).__name__, str(e)[:200])
            continue
        print(f"== {rt} {REPO} sha={info.sha} private={info.private}")
        for s in sorted(info.siblings, key=lambda x: x.rfilename):
            lfs = s.lfs.get("sha256") if isinstance(s.lfs, dict) else (getattr(s.lfs, "sha256", None) if s.lfs else None)
            print(f"   {s.size:>12}  {s.rfilename}" + (f"  lfs:{lfs[:16]}" if lfs else ""))


def public(kind):
    a = api()
    a.update_repo_settings(REPO, repo_type=kind, private=False)
    print(kind, "public:", not a.repo_info(REPO, repo_type=kind).private)


def runtime():
    rt = api().get_space_runtime(REPO)
    print(json.dumps(dict(stage=rt.stage, hardware=rt.hardware, requested_hardware=rt.requested_hardware,
                          raw=rt.raw), indent=1, default=str)[:3000])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "public":
        public(sys.argv[2])
    else:
        {"stage": stage, "upload-model": upload_model, "upload-space": upload_space, "files": files,
         "runtime": runtime}[cmd]()
