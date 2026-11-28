"""Archive BA on Zenodo under one DOI: the engine code, evaluation texts and scripts, per-utterance scores and results,
the paper sources and PDF, documentation, demo code and tests.

    python3 BA/scripts/zenodo_upload.py reserve    # create a DRAFT record with its metadata and reserve the DOI (no files)
    python3 BA/scripts/zenodo_upload.py prepare    # build the archive from the committed HEAD (no network, no token)
    python3 BA/scripts/zenodo_upload.py upload     # upload the files to the draft (creates the draft if needed)
    python3 BA/scripts/zenodo_upload.py verify     # compare the draft's files (size, MD5) with the staged ones
    python3 BA/scripts/zenodo_upload.py publish    # publish the draft: irreversible, the files can no longer be removed

New version of the published record (same concept DOI, which the paper cites): prepare into a new staging folder,
then `newversion` creates the draft of the next version (once) and replaces all its files; verify and publish as above.
    python3 BA/scripts/zenodo_upload.py prepare --staging ~/ba-work/zenodo_v2
    python3 BA/scripts/zenodo_upload.py newversion --staging ~/ba-work/zenodo_v2

Order: reserve -> put the DOI into the paper and README and commit -> prepare -> upload -> verify -> publish.

Not included, on purpose: generated audio and the FLEURS reference clips (third-party voices are never redistributed),
and model weights (BA runs the unchanged VoxCPM2 weights, served with BA's configuration at
https://huggingface.co/mahmoudalyosify/BA).

The token is read from ~/.config/zenodo/token (scopes deposit:write and deposit:actions) and is never printed. The
deposition id and the reserved DOI are kept in <staging>/zenodo_state.json.
"""
import argparse, hashlib, json, os, subprocess, sys, time

BA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # .../BA
API = "https://zenodo.org/api"
TITLE = ("BA: Right Words, Wrong Dialect. Egyptian Arabic voice cloning: engine code, evaluation texts, "
         "per-utterance scores and paper sources")
CREATORS = [{"name": "Alyosify, Mahmoud", "affiliation": "Queen's University", "orcid": "0009-0005-1336-2726"}]
KEYWORDS = ["voice cloning", "zero-shot text-to-speech", "Egyptian Arabic", "dialect identification",
            "speech synthesis evaluation", "audio watermarking"]
HF_MODEL = "https://huggingface.co/mahmoudalyosify/BA"
HF_SPACE = "https://huggingface.co/spaces/mahmoudalyosify/BA"
LICENSE_ID = "other-nc"          # BA Community License 1.0 (non-commercial); the full text is LICENSE in the archive
VERSION = "paper-v2 (BA 0.1.0)"
V1_RECORD = 24517175           # first published version; its concept DOI is 10.5281/zenodo.24517174
CONCEPT_DOI = "10.5281/zenodo.24517174"
ARCHIVE = "BA_code_eval_paper.zip"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    top = os.path.dirname(BA_DIR)
    return subprocess.run(["git", *args], cwd=top, capture_output=True, text=True, check=True).stdout.strip()


def readme(commit, doi):
    return "\n".join([
        f"# {TITLE}", "",
        "Archive of BA, a consent-first voice-cloning system for Egyptian Arabic, and of the evaluation behind the paper "
        "*BA: Right Words, Wrong Dialect. Standard Metrics Pick the Wrong Model for Egyptian Arabic Voice Cloning* "
        "(Mahmoud Alyosify, Queen's University).", "",
        "The paper shows that, for colloquial Egyptian Arabic, intelligibility (ASR character error rate), speaker "
        "similarity and predicted quality do not see the dialect: used to choose a base model among the licence-eligible "
        "open voice-cloning models, each picks a model that fails a dialect bar calibrated on real colloquial speech. BA "
        "is an inference-time engine on the unchanged VoxCPM2 weights (Egyptian number and symbol normalisation, a "
        "loanword policy, automatic reference transcription, reference screening, checked long-text generation and an "
        "AudioSeal watermark on every output).", "",
        f"This archive: the `BA/` folder at commit {commit[:12]} of the author's repository. Model: {HF_MODEL}. "
        f"Demo: {HF_SPACE}." + (f" DOI of this version: {doi}." if doi else "")
        + f" DOI of all versions (cited in the paper): {CONCEPT_DOI}.", "",
        "Version 2 adds the blind listening test with native Egyptian listeners (aggregate results only; the individual "
        "answers stay private, as the consent text promised), a clip-level comparison of the listener ratings with the "
        "automatic measures, and a phonetic analysis of two Egyptian dialect markers (jiim and qaaf) in the outputs.", "",
        "## Contents", "",
        f"`{ARCHIVE}` holds the `BA/` folder of the repository at that commit:", "",
        "- `src/ba_tts/`: the engine (normalisers and lexicons, reference preparation, transcription, chunking, "
        "watermark, CLI); `tests/`: its tests; `pyproject.toml`.",
        "- `eval/`: evaluation texts (`texts/`, `refs.jsonl`, `calib.jsonl`), every evaluation script (`scripts/`), and "
        "the results with per-utterance scores (`results/`: bake-off, dialect classifiers, seeds and uncertainty, "
        "ablations, long-form, selection analysis, watermark robustness, listening test, dialect markers). Each results "
        "file records the command that produced it.",
        "- `paper/`: LaTeX sources, generated numbers, tables and figure, `main.pdf`, the plan, the internal review "
        "(`REVIEW.md`), and the human-evaluation materials (consent form, recording instructions, listening-test protocol). "
        "`python3 BA/paper/scripts/make_assets.py && bash BA/paper/build.sh` rebuilds every number from `eval/results/`.",
        "- `space/`: the demo app; `hf/`: the Hugging Face model-repository files and publishing scripts; `docs/`: "
        "execution plan, decision log (including the dated amendment of the base-selection gate), brand; `assets/`: logo.",
        "- `SHA256SUMS.txt`: checksum of the archive.", "",
        "## Not included", "",
        "- Generated audio and the FLEURS reference clips: third-party voices and their clones are never redistributed.",
        "- Model weights: BA runs the unchanged VoxCPM2 weights (Apache-2.0), served with BA's configuration at "
        f"{HF_MODEL}.", "",
        "## Licence", "",
        "BA Community License 1.0 (`BA/LICENSE`): free for non-commercial use, modification and sharing, with attribution "
        "and conditions against impersonation and use of a voice without consent. Third-party components keep their own "
        "licences. The licence has not been reviewed by a lawyer.", ""])


def token():
    path = os.path.expanduser("~/.config/zenodo/token")
    if not os.path.exists(path):
        sys.exit(f"no token in {path}")
    return open(path).read().strip()


def call(method, url, tok, **kw):
    import requests
    for attempt in range(4):
        try:
            r = requests.request(method, url, headers={"Authorization": f"Bearer {tok}"}, timeout=(60, 3600), **kw)
            if r.status_code < 500:
                return r
        except requests.RequestException as e:
            print(f"  network error ({e.__class__.__name__}), retrying")
        time.sleep(10 * 2 ** attempt)
    sys.exit(f"{method} {url} failed")


def state_path(a):
    return os.path.join(a.staging, "zenodo_state.json")


def metadata(a, doi=None):
    path = os.path.join(a.staging, "README.md")
    desc = open(path, encoding="utf-8").read() if os.path.exists(path) else TITLE
    return {"metadata": {
        "title": TITLE, "upload_type": "software", "creators": CREATORS, "keywords": KEYWORDS,
        "description": "<pre>" + desc.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre>",
        "access_right": "open", "license": LICENSE_ID, "version": VERSION, "prereserve_doi": True,
        "related_identifiers": [
            {"identifier": HF_MODEL, "relation": "isSupplementTo", "resource_type": "software"},
            {"identifier": HF_SPACE, "relation": "isSupplementTo", "resource_type": "software"}]}}


def reserve(a):
    tok = token()
    os.makedirs(a.staging, exist_ok=True)
    st = json.load(open(state_path(a))) if os.path.exists(state_path(a)) else {}
    if "id" not in st:
        r = call("POST", f"{API}/deposit/depositions", tok, json={})
        if r.status_code != 201:
            sys.exit(f"could not create the record: {r.status_code} {r.text[:300]}")
        d = r.json()
        st = {"id": d["id"], "bucket": d["links"]["bucket"], "html": d["links"]["html"]}
        json.dump(st, open(state_path(a), "w"), indent=1)
        print(f"created draft record {st['id']}")
    r = call("PUT", f"{API}/deposit/depositions/{st['id']}", tok, json=metadata(a))
    if r.status_code != 200:
        sys.exit(f"metadata rejected: {r.status_code} {r.text[:800]}")
    m = r.json()["metadata"]
    st["doi"] = m.get("prereserve_doi", {}).get("doi")
    st["license"] = m.get("license")
    json.dump(st, open(state_path(a), "w"), indent=1)
    print(f"draft (not published): {st['html']}\nreserved DOI: {st['doi']}\nlicence stored: {st['license']}")
    return st, tok


def prepare(a):
    os.makedirs(a.staging, exist_ok=True)
    st = json.load(open(state_path(a))) if os.path.exists(state_path(a)) else {}
    if git("status", "--porcelain", "--", "BA"):
        print("note: BA/ has uncommitted changes; the archive is built from the committed HEAD")
    commit = git("rev-parse", "HEAD")
    out = os.path.join(a.staging, ARCHIVE)
    git("archive", "--format=zip", "-o", out, "HEAD", "BA")
    with open(os.path.join(a.staging, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme(commit, st.get("doi")))
    with open(os.path.join(a.staging, "SHA256SUMS.txt"), "w") as f:
        f.write(f"{sha256(out)}  {ARCHIVE}\n")
    print(f"wrote {ARCHIVE} ({os.path.getsize(out) / 1e6:.1f} MB, commit {commit[:7]}), README.md, SHA256SUMS.txt")


def files(a):
    return [n for n in (ARCHIVE, "README.md", "SHA256SUMS.txt") if os.path.exists(os.path.join(a.staging, n))]


def upload(a):
    st, tok = reserve(a)
    dep = f"{API}/deposit/depositions/{st['id']}"
    have = {f["filename"]: f for f in call("GET", f"{dep}/files", tok).json()}
    import requests
    for n in files(a):
        p = os.path.join(a.staging, n)
        if n in have:          # replace: a staged file may have changed since an earlier upload
            call("DELETE", f"{dep}/files/{have[n]['id']}", tok)
        with open(p, "rb") as f:
            r = requests.put(f"{st['bucket']}/{n}", data=f, headers={"Authorization": f"Bearer {tok}"}, timeout=(60, 3600))
        if r.status_code not in (200, 201):
            sys.exit(f"upload of {n} failed: {r.status_code} {r.text[:300]}")
        print(f"uploaded {n} ({os.path.getsize(p) / 1e6:.2f} MB)")
    reserve(a)          # refresh the description from the final README


def verify(a):
    tok = token()
    st = json.load(open(state_path(a)))
    d = call("GET", f"{API}/deposit/depositions/{st['id']}", tok).json()
    remote = {f["filename"]: (f["filesize"], f["checksum"]) for f in d["files"]}
    bad = 0
    for n in files(a):
        p = os.path.join(a.staging, n)
        md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
        ok = n in remote and remote[n] == (os.path.getsize(p), md5)
        bad += not ok
        print(("OK  " if ok else "BAD ") + n)
    extra = set(remote) - set(files(a))
    m = d["metadata"]
    print(f"state {d['state']}; title: {m['title']}\nlicence {m.get('license')}; type {m.get('upload_type')}; "
          f"version {m.get('version')}; DOI {m.get('prereserve_doi', {}).get('doi')}; extra remote files: {sorted(extra)}")
    if bad or extra:
        sys.exit("verify failed")


def publish(a):
    verify(a)
    tok = token()
    st = json.load(open(state_path(a)))
    r = call("POST", f"{API}/deposit/depositions/{st['id']}/actions/publish", tok)
    if r.status_code != 202:
        sys.exit(f"publish failed: {r.status_code} {r.text[:500]}")
    d = r.json()
    print(f"published: {d['links'].get('record_html', d['links'].get('html'))}  DOI {d.get('doi')}")


def newversion(a):
    """Create (once) the draft of the next version of the published record, then replace all of its files."""
    tok = token()
    os.makedirs(a.staging, exist_ok=True)
    st = json.load(open(state_path(a))) if os.path.exists(state_path(a)) else {}
    if "id" not in st:
        r = call("POST", f"{API}/deposit/depositions/{V1_RECORD}/actions/newversion", tok)
        if r.status_code not in (200, 201):
            sys.exit(f"could not create the new version: {r.status_code} {r.text[:500]}")
        d = call("GET", r.json()["links"]["latest_draft"], tok).json()
        st = {"id": d["id"], "bucket": d["links"]["bucket"], "html": d["links"]["html"]}
        json.dump(st, open(state_path(a), "w"), indent=1)
        print(f"created the draft of the new version: {st['html']}")
    upload(a)                    # replaces every staged file; files of the previous version not staged are removed below
    dep = f"{API}/deposit/depositions/{st['id']}"
    for f in call("GET", f"{dep}/files", tok).json():
        if f["filename"] not in files(a):
            call("DELETE", f"{dep}/files/{f['id']}", tok)
            print(f"removed {f['filename']} (previous version)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["reserve", "prepare", "upload", "verify", "publish", "newversion"])
    p.add_argument("--staging", default="/home/jovyan/ba-work/zenodo")
    a = p.parse_args()
    {"reserve": reserve, "prepare": prepare, "upload": upload, "verify": verify, "publish": publish,
     "newversion": newversion}[a.command](a)
