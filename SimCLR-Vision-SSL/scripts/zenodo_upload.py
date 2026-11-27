"""Archive the study on Zenodo under one DOI: code, job files, per-run results and paper, plus every model checkpoint.

    python scripts/zenodo_upload.py prepare     # build the files in --staging (no network, no token)
    python scripts/zenodo_upload.py reserve     # create a DRAFT record with its metadata and reserve the DOI (no files)
    python scripts/zenodo_upload.py upload      # upload the files to the draft (resumes; creates the draft if needed)
    python scripts/zenodo_upload.py publish     # publish the draft: irreversible, the files can no longer be removed

New version of the published record (third revision): the checkpoint archives of version 1.0 are kept as they are,
the checkpoints of the runs added since go into one more archive, and the code archive, README, checksums and
checkpoint index are replaced. Nothing is published.
    python scripts/zenodo_upload.py prepare_v2 --staging <dir>    # build the new and replaced files (no network for the tar)
    python scripts/zenodo_upload.py newversion --staging <dir>    # create the draft of version 2.0 and upload them

Order: reserve -> put the DOI into the paper and commit -> prepare (the code archive is taken from HEAD) -> upload
-> check the draft on zenodo.org -> publish.

The token is read from ~/.config/zenodo/token (Zenodo -> Applications -> Personal access tokens, scopes
deposit:write and deposit:actions) and is never printed. The deposition id and the reserved DOI are kept in
<staging>/zenodo_state.json, so an interrupted upload resumes where it stopped.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # .../SimCLR-Vision-SSL
API = "https://zenodo.org/api"

TITLE = ("Where Do Scarce Labels Belong? Evaluating Contrastive Pretraining under Equal Compute and Realistic Label "
         "Budgets: code, per-run results and model checkpoints")
CREATORS = [
    {"name": "Alyosify, Mahmoud", "affiliation": "School of Computing, Queen's University", "orcid": "0009-0005-1336-2726"},
    {"name": "Imbabi, Mirna", "affiliation": "School of Computing, Queen's University", "orcid": "0009-0000-6699-3170"},
    {"name": "Nashed, Natalie", "affiliation": "School of Computing, Queen's University", "orcid": "0009-0007-5657-7449"},
    {"name": "Abbas, Hazem M.", "affiliation": "School of Computing, Queen's University", "orcid": "0000-0001-9128-3111"},
]
KEYWORDS = ["contrastive learning", "supervised contrastive learning", "self-supervised learning",
            "semi-supervised learning", "evaluation protocol", "compute-matched comparison"]
GITHUB = "https://github.com/MahmoudAlyosify1/scarce-labels-contrastive"
V1_RECORD = 24494138      # version 1.0, published 2026-11-24
VERSION = "1.0"
# paper-relevant part of the repository (the course project, the demo and W&B files stay on GitHub only)
CODE_PATHS = ["dlt", "scripts", "jobs", "results", "paper", "notebooks", "tests", "README.md", "RUNBOOK.md",
              "LICENSE", "environment.yml", "requirements.txt", "Dockerfile"]
CODE_ARCHIVE = "SimCLR-Vision-SSL_code_results_paper.zip"


def group(run):
    """Which checkpoint archive a run belongs to."""
    if run.startswith("tin_"):
        return "tinyimagenet"
    if run.startswith("stl_"):
        return "stl10"
    if run.startswith("c100_"):
        return "cifar100"
    if run.startswith("r50_"):
        return "cifar10_resnet50"
    if re.search(r"_(cluster\d+|kcenter|mixcluster\d+|mixkcenter\d+)_", run):
        return "cifar10_coverage_pools"
    return "cifar10_resnet18"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def prepare(a):
    os.makedirs(a.staging, exist_ok=True)
    runs = sorted(r for r in os.listdir(a.ckpt_dir) if os.path.exists(os.path.join(a.ckpt_dir, r, "model.pt")))
    groups = {}
    for r in runs:
        groups.setdefault(group(r), []).append(r)
    # keep each archive below about 1.2 GB: longer single uploads to Zenodo were cut off from this server
    archives = {}
    for g, rs in groups.items():
        size = sum(os.path.getsize(os.path.join(a.ckpt_dir, r, "model.pt")) for r in rs)
        parts = max(1, -(-size // int(1.2e9)))
        for i in range(parts):
            chunk = rs[i * len(rs) // parts:(i + 1) * len(rs) // parts]
            name = f"checkpoints_{g}" + (f"_part{i + 1}" if parts > 1 else "") + ".tar"
            archives[name] = chunk
    for name, rs in archives.items():
        out = os.path.join(a.staging, name)
        if os.path.exists(out) and not a.force:
            print(f"exists: {name}")
            continue
        with tarfile.open(out + ".tmp", "w") as tar:
            for r in rs:
                tar.add(os.path.join(a.ckpt_dir, r, "model.pt"), arcname=f"checkpoints/{r}/model.pt")
        os.replace(out + ".tmp", out)
        print(f"wrote {name}: {len(rs)} runs, {os.path.getsize(out) / 1e9:.2f} GB")
    # code, job files, per-run results and paper, exactly as committed (HEAD)
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
    sub = os.path.relpath(REPO_DIR, top)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=top, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "archive", "--format=zip", "--prefix=SimCLR-Vision-SSL/", "-o",
                    os.path.join(a.staging, CODE_ARCHIVE), f"HEAD:{sub}", *CODE_PATHS], cwd=top, check=True)
    print(f"wrote {CODE_ARCHIVE} (commit {commit[:7]})")
    manifest = {name: rs for name, rs in archives.items()}
    with open(os.path.join(a.staging, "checkpoint_index.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(a.staging, "README.md"), "w") as f:
        f.write(readme(archives, commit))
    files = sorted(n for n in os.listdir(a.staging) if n.endswith((".tar", ".zip")))
    with open(os.path.join(a.staging, "SHA256SUMS.txt"), "w") as f:
        for n in files:
            f.write(f"{sha256(os.path.join(a.staging, n))}  {n}\n")
    total = sum(os.path.getsize(os.path.join(a.staging, n)) for n in files)
    print(f"ready: {len(files)} archives, {total / 1e9:.2f} GB, in {a.staging}")


def readme(archives, commit):
    lines = [f"# {TITLE}", "",
             "Archive of the study behind the paper *Where Do Scarce Labels Belong? Evaluating Contrastive Pretraining "
             "under Equal Compute and Realistic Label Budgets* (Alyosify, Imbabi, Nashed, Abbas; Queen's University).",
             "", f"Code repository: {GITHUB} (this archive: commit {commit[:12]}).", "",
             "## Contents", "",
             f"- `{CODE_ARCHIVE}`: the code (`dlt/`, `scripts/`, `tests/`), the job file of every run (`jobs/`), the "
             "results of every run (`results/runs/<run>/result.json` with all metrics, `preds.npz` with the test-set "
             "predictions, `probes.json` with the label-limited probes, and the training log), the LaTeX sources and "
             "PDFs of the paper (`paper/`), and a notebook with the complete code (`notebooks/`).",
             "- `checkpoints_*.tar`: the encoder checkpoint of every pretraining run, as "
             "`checkpoints/<run>/model.pt`; `checkpoint_index.json` lists which runs each archive holds:"]
    for name, rs in archives.items():
        lines.append(f"  - `{name}`: {len(rs)} runs")
    lines += ["- `SHA256SUMS.txt`: checksums of the archives.", "",
              "Run names are those of `jobs/*.txt` and `results/runs/`; the suffix `_s<k>` is the seed.", "",
              "## Loading a checkpoint", "", "```python",
              "from dlt.models import load_encoder            # from the code archive",
              "encoder, args = load_encoder('checkpoints/simclr_all_e200_s0/model.pt')   # args: the training arguments",
              "```", "",
              "Every number, table and figure of the paper is rebuilt from `results/` with `python scripts/make_assets.py`; "
              "`RUNBOOK.md` and `README.md` describe how to rerun the experiments.", "",
              "## Licence", "",
              "Code: MIT (`LICENSE`). Results and checkpoints: CC BY 4.0. The datasets (CIFAR-10, CIFAR-100, STL-10, "
              "Tiny-ImageNet) are not included; they are public and are downloaded or converted by the code.", ""]
    return "\n".join(lines)


def token():
    path = os.path.expanduser("~/.config/zenodo/token")
    if not os.path.exists(path):
        sys.exit(f"no token: put a Zenodo personal access token (deposit:write, deposit:actions) in {path}")
    return open(path).read().strip()


def call(method, url, tok, **kw):
    import requests
    for attempt in range(4):
        try:
            r = requests.request(method, url, headers={"Authorization": f"Bearer {tok}"}, timeout=(60, 6 * 3600), **kw)
            if r.status_code < 500:
                return r
        except requests.RequestException as e:
            print(f"  network error ({e.__class__.__name__}), retrying")
        time.sleep(10 * 2 ** attempt)
    sys.exit(f"{method} {url} failed")


def metadata(a):
    path = os.path.join(a.staging, "README.md")
    desc = open(path).read() if os.path.exists(path) else TITLE
    return {"metadata": {
        "title": TITLE, "upload_type": "dataset", "creators": CREATORS, "keywords": KEYWORDS,
        "description": "<pre>" + desc.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre>",
        "access_right": "open", "license": "cc-by-4.0", "version": VERSION, "prereserve_doi": True,
        "related_identifiers": [{"identifier": GITHUB, "relation": "isSupplementTo", "resource_type": "software"}]}}


def reserve(a):
    """Create the draft record (once), set its metadata and return its state with the reserved DOI."""
    tok = token()
    os.makedirs(a.staging, exist_ok=True)
    state_path = os.path.join(a.staging, "zenodo_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    if "id" not in state:
        r = call("POST", f"{API}/deposit/depositions", tok, json={})
        if r.status_code != 201:
            sys.exit(f"could not create the record: {r.status_code} {r.text[:300]}")
        d = r.json()
        state = {"id": d["id"], "bucket": d["links"]["bucket"], "html": d["links"]["html"]}
        json.dump(state, open(state_path, "w"), indent=1)
        print(f"created draft record {state['id']}")
    r = call("PUT", f"{API}/deposit/depositions/{state['id']}", tok, json=metadata(a))
    if r.status_code != 200:
        sys.exit(f"metadata rejected: {r.status_code} {r.text[:500]}")
    state["doi"] = r.json()["metadata"].get("prereserve_doi", {}).get("doi")
    json.dump(state, open(state_path, "w"), indent=1)
    print(f"draft (not published): {state['html']}\nreserved DOI: {state['doi']}")
    return state, tok


def upload(a):
    state, tok = reserve(a)
    dep = f"{API}/deposit/depositions/{state['id']}"
    have = {f["filename"]: f["filesize"] for f in call("GET", f"{dep}/files", tok).json()}
    files = sorted(n for n in os.listdir(a.staging) if n.endswith((".tar", ".zip")) or n in
                   ("README.md", "SHA256SUMS.txt", "checkpoint_index.json"))
    for n in files:
        p = os.path.join(a.staging, n)
        if have.get(n) == os.path.getsize(p):
            print(f"already uploaded: {n}")
            continue
        t0 = time.time()
        r = None
        for attempt in range(4):     # reopen the file for every attempt: a failed attempt consumes the stream
            import requests
            try:
                with open(p, "rb") as f:
                    r = requests.put(f"{state['bucket']}/{n}", data=f, headers={"Authorization": f"Bearer {tok}"},
                                     timeout=(60, 6 * 3600))
                if r.status_code in (200, 201):
                    break
                print(f"  {n}: HTTP {r.status_code}, retrying")
            except requests.RequestException as e:
                print(f"  {n}: network error ({e.__class__.__name__}), retrying")
            time.sleep(30 * 2 ** attempt)
        if r is None or r.status_code not in (200, 201):
            sys.exit(f"upload of {n} failed: " + ("no response" if r is None else f"{r.status_code} {r.text[:300]}"))
        print(f"uploaded {n} ({os.path.getsize(p) / 1e9:.2f} GB, {(time.time() - t0) / 60:.1f} min)")
    reserve(a)        # refresh the description from the final README
    print("all files uploaded; check the draft on zenodo.org, then run: publish")


def publish(a):
    tok = token()
    state = json.load(open(os.path.join(a.staging, "zenodo_state.json")))
    r = call("POST", f"{API}/deposit/depositions/{state['id']}/actions/publish", tok)
    if r.status_code != 202:
        sys.exit(f"publish failed: {r.status_code} {r.text[:500]}")
    d = r.json()
    print(f"published: {d['links'].get('record_html', d['links'].get('html'))}  DOI {d.get('doi')}")


def _v1_files():
    import requests
    r = requests.get(f"{API}/records/{V1_RECORD}", timeout=60).json()
    return {f["key"]: f["links"]["self"] for f in r["files"]}


def prepare_v2(a):
    """Files of version 2.0: one archive with the checkpoints of the runs added since version 1.0, plus the new code
    archive, README, checksums and checkpoint index. The checkpoint archives of version 1.0 stay as they are."""
    import requests
    os.makedirs(a.staging, exist_ok=True)
    v1 = _v1_files()
    index = requests.get(v1["checkpoint_index.json"], timeout=60).json()
    sums = dict(l.split()[::-1] for l in requests.get(v1["SHA256SUMS.txt"], timeout=60).text.splitlines() if l.strip())
    old = {r for rs in index.values() for r in rs}
    runs = sorted(r for r in os.listdir(a.ckpt_dir) if os.path.exists(os.path.join(a.ckpt_dir, r, "model.pt")))
    new = [r for r in runs if r not in old]
    name = "checkpoints_third_revision.tar"
    out = os.path.join(a.staging, name)
    if not os.path.exists(out) or a.force:
        with tarfile.open(out + ".tmp", "w") as tar:
            for r in new:
                tar.add(os.path.join(a.ckpt_dir, r, "model.pt"), arcname=f"checkpoints/{r}/model.pt")
        os.replace(out + ".tmp", out)
    print(f"{name}: {len(new)} runs, {os.path.getsize(out) / 1e9:.2f} GB")
    index[name] = new
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
    sub = os.path.relpath(REPO_DIR, top)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=top, capture_output=True, text=True).stdout.strip()
    paths = [q for q in CODE_PATHS if subprocess.run(["git", "cat-file", "-e", f"HEAD:{sub}/{q}"], cwd=top).returncode == 0]
    subprocess.run(["git", "archive", "--format=zip", "--prefix=SimCLR-Vision-SSL/", "-o",
                    os.path.join(a.staging, CODE_ARCHIVE), f"HEAD:{sub}", *paths], cwd=top, check=True)
    json.dump(index, open(os.path.join(a.staging, "checkpoint_index.json"), "w"), indent=1)
    text = readme(index, commit).replace("## Contents", "Version 2.0 adds the runs of the third revision of the paper "
                                         f"(`{name}`); the other checkpoint archives are those of version 1.0.\n\n## Contents")
    open(os.path.join(a.staging, "README.md"), "w").write(text)
    sums[name] = sha256(out)
    sums[CODE_ARCHIVE] = sha256(os.path.join(a.staging, CODE_ARCHIVE))
    with open(os.path.join(a.staging, "SHA256SUMS.txt"), "w") as f:
        for n in sorted(sums):
            f.write(f"{sums[n]}  {n}\n")
    print(f"ready in {a.staging} (code archive from commit {commit[:7]})")


def newversion(a):
    """Create (once) the draft of version 2.0 of the published record and upload the files of prepare_v2."""
    global VERSION
    VERSION = "2.0"
    tok = token()
    state_path = os.path.join(a.staging, "zenodo_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    if "id" not in state:
        r = call("POST", f"{API}/deposit/depositions/{V1_RECORD}/actions/newversion", tok)
        if r.status_code not in (200, 201):
            sys.exit(f"could not create the new version: {r.status_code} {r.text[:500]}")
        draft = call("GET", r.json()["links"]["latest_draft"], tok).json()
        state = {"id": draft["id"], "bucket": draft["links"]["bucket"], "html": draft["links"]["html"]}
        json.dump(state, open(state_path, "w"), indent=1)
        print(f"created draft of version 2.0: {state['html']}")
    dep = f"{API}/deposit/depositions/{state['id']}"
    replaced = (CODE_ARCHIVE, "README.md", "SHA256SUMS.txt", "checkpoint_index.json")
    for f in call("GET", f"{dep}/files", tok).json():
        if f["filename"] in replaced and f["filesize"] != os.path.getsize(os.path.join(a.staging, f["filename"])):
            call("DELETE", f"{dep}/files/{f['id']}", tok)
            print(f"removed from the draft: {f['filename']} (version 1.0)")
    upload(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "reserve", "upload", "publish", "prepare_v2", "newversion"])
    p.add_argument("--ckpt_dir", default="/home/jovyan/simclr_store/ckpt")
    p.add_argument("--staging", default="/tmp/simclr_zenodo")
    p.add_argument("--force", action="store_true", help="prepare: rebuild archives that already exist")
    a = p.parse_args()
    {"prepare": prepare, "reserve": reserve, "upload": upload, "publish": publish, "prepare_v2": prepare_v2,
     "newversion": newversion}[a.command](a)
