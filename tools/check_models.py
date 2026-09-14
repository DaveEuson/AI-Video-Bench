#!/usr/bin/env python3
"""The daily model check (.github/workflows/new-models.yml). Stdlib only.

1. New models. ComfyUI's own template list (Comfy-Org/workflow_templates) against
   models.json: an open-weights text-to-video template whose model isn't in any entry's
   comfy_models, or in watch_ignore, is a model the lineup doesn't have. One issue each,
   filed once (closing it means "seen").
2. Files changed upstream. Every file models.json pins a sha256 for, against what Hugging
   Face serves now. Any that changed or vanished: one issue, unless one is already open.

    python tools/check_models.py             # print what it finds
    python tools/check_models.py --github    # and file the issues (GH_TOKEN, GITHUB_REPOSITORY)
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_URL = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/index.json"
TITLE_NEW = "New model in ComfyUI: "
TITLE_FILES = "Model files changed on Hugging Face"
HF_URL = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")
UA = {"User-Agent": "videobench-model-check"}


def get_json(url, headers=None, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    hdr = {**UA, **(headers or {}), **({"Content-Type": "application/json"} if data else {})}
    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr, data=data, method=method), timeout=60) as r:
        return json.load(r)


def text_to_video(index):
    """(template, [model labels]) for each open-weights text-to-video template. API-only
    models (openSource false, or tagged API) can't run locally, so they don't count."""
    for cat in index:
        for t in cat.get("templates") or []:
            tags = t.get("tags") or []
            if "Text to Video" in tags and "API" not in tags and t.get("openSource") is not False:
                yield t, [m.strip() for m in t.get("models") or [] if isinstance(m, str) and m.strip()]


def new_models(index, reg):
    """{model label: [its templates]} for labels models.json doesn't know."""
    known = {x.lower() for m in reg.get("models", []) for x in m.get("comfy_models", [])}
    known |= {x.lower() for x in reg.get("watch_ignore", [])}
    found = {}
    for t, labels in text_to_video(index):
        for label in labels:
            if label.lower() not in known:
                found.setdefault(label, []).append(t)
    return found


def changed_files(reg, fetch=get_json):
    """([(file key, what changed)], [warnings]) for files whose pinned sha256 Hugging Face no
    longer serves. A folder that can't be listed right now is a warning, not a change."""
    listed, changed, warnings = {}, [], []
    for k, f in reg.get("files", {}).items():
        m = HF_URL.match(f.get("url", ""))
        if not f.get("sha256") or not m:
            continue
        repo, rev, path = m.groups()
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        if (repo, rev, folder) not in listed:
            try:
                entries = fetch(f"https://huggingface.co/api/models/{repo}/tree/{rev}/{urllib.parse.quote(folder)}")
                listed[(repo, rev, folder)] = {e.get("path"): e for e in entries}
            except (urllib.error.URLError, OSError, ValueError) as e:
                listed[(repo, rev, folder)] = None
                warnings.append(f"couldn't list {repo}/{folder}: {e}")
        there = listed[(repo, rev, folder)]
        if there is None:
            continue
        e = there.get(path)
        if not e:
            changed.append((k, f"{path} is gone from {repo}"))
            continue
        oid = (e.get("lfs") or {}).get("oid") or ""
        if re.fullmatch(r"[0-9a-f]{64}", oid) and oid != f["sha256"]:
            changed.append((k, f"{path} in {repo} now has sha256 {oid}"))
    return changed, warnings


def new_model_issue(label, temps, reg):
    rows = "\n".join(f"| `{t.get('name')}` | {t.get('date', '?')} | {t.get('minComfyUIVersion') or '?'} | "
                     + (f"[tutorial]({t['tutorialUrl']})" if t.get("tutorialUrl") else "-") + " |" for t in temps)
    pin = (reg.get("tested_with") or {}).get("comfyui", "?")
    return f"""ComfyUI has a text-to-video template for **{label}**, which the lineup doesn't have yet.

| template | added | needs ComfyUI | |
|---|---|---|---|
{rows}

To add it:

1. Open the template in ComfyUI and note its files, size, frames, steps and sampler settings.
2. If a builder in `videobench.py` already makes that workflow (a new version of a family it knows), add a
   `models.json` entry that uses it. That ships to everyone through `--update-models`, with no new videobench.
3. If not, it needs a new builder in `videobench.py`, and maybe a newer ComfyUI (videobench pins `{pin}`).
4. Run it on a real GPU for `ref_s`, check its license (`restricted` if it excludes countries), and bump `revision`.

Not worth adding? Put `{label}` in `watch_ignore` in `models.json` and close this.

_Filed by the daily model check._"""


def files_issue(changed):
    rows = "\n".join(f"- `{k}`: {why}" for k, why in changed)
    return f"""Hugging Face no longer serves the exact files `models.json` pins. Anyone downloading these
gets a clear error instead of a file that doesn't match, until `models.json` is updated:

{rows}

Check what changed upstream, then update the `sha256` (and `bytes`, `gb`) or the URL, bump `revision`,
and re-run the models that use them.

_Filed by the daily model check._"""


def gh(method, path, body=None):
    token, repo = os.environ["GH_TOKEN"], os.environ["GITHUB_REPOSITORY"]
    return get_json(f"https://api.github.com/repos/{repo}{path}", method=method, body=body,
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                             "X-GitHub-Api-Version": "2022-11-28"})


def issue_titles():
    """{title: state} for every issue, open or closed."""
    titles, page = {}, 1
    while True:
        batch = gh("GET", f"/issues?state=all&per_page=100&page={page}")
        titles.update({i["title"]: i["state"] for i in batch if "pull_request" not in i})
        if len(batch) < 100:
            return titles
        page += 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--github", action="store_true", help="file issues (needs GH_TOKEN and GITHUB_REPOSITORY)")
    p.add_argument("--models", default=str(ROOT / "models.json"))
    p.add_argument("--index", help="a local copy of ComfyUI's templates index.json, instead of fetching it")
    args = p.parse_args()
    reg = json.loads(Path(args.models).read_text(encoding="utf-8"))
    index = json.loads(Path(args.index).read_text(encoding="utf-8")) if args.index else get_json(INDEX_URL)

    found = new_models(index, reg)
    changed, warnings = changed_files(reg)
    for label, temps in sorted(found.items()):
        print(f"new model: {label}  ({', '.join(t.get('name', '?') for t in temps)})")
    for k, why in changed:
        print(f"changed file: {k}: {why}")
    for w in warnings:
        print(f"warning: {w}")
    if not found and not changed:
        print(f"nothing new: {sum(1 for _ in text_to_video(index))} text-to-video templates, all known")
    if not args.github:
        return
    titles = issue_titles()
    for label, temps in sorted(found.items()):
        title = TITLE_NEW + label
        if title in titles:
            print(f"  already filed: {title} ({titles[title]})")
            continue
        n = gh("POST", "/issues", {"title": title, "body": new_model_issue(label, temps, reg)})["number"]
        print(f"  filed #{n}: {title}")
    if changed:
        if any(t.startswith(TITLE_FILES) and s == "open" for t, s in titles.items()):
            print(f"  already open: {TITLE_FILES}")
        else:
            n = gh("POST", "/issues", {"title": TITLE_FILES, "body": files_issue(changed)})["number"]
            print(f"  filed #{n}: {TITLE_FILES}")


if __name__ == "__main__":
    sys.exit(main())
