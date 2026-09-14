#!/usr/bin/env python3
"""videobench, by Dave Euson: how fast can this computer turn a prompt into a 5-second AI video?

One fixed test, so results compare across machines. Z-Image-Turbo paints a still,
then LTX-Video animates it into a 5 s clip, all locally in ComfyUI. Same models,
prompt, seed and settings on every machine.

    python videobench.py                     # "fast": LTX-Video 2B, 640x480
    python videobench.py --profile quality   # LTX-Video 13B, 768x512
    python videobench.py --list              # show the plan and downloads, change nothing
    python videobench.py --gallery           # rebuild gallery.html from past runs
    python videobench.py --voices            # text to speech: every voice engine reads the same line
    python videobench.py --list-voices       # the voice engines and their licenses
    python videobench.py --list-models       # the video models: does each fit here, and its license
    python videobench.py --update-models     # fetch the latest model list (models.json) from GitHub

Needs Python 3.10+ and ~20 GB of disk (+7 GB for quality). Works with an NVIDIA
GPU (8 GB+) on Windows or Linux, AMD (ROCm) on Linux, Apple Silicon, or a Jetson.
Everything lives in --dir (default ./videobench):

    results/<time>-<profile>.json   machine, settings, seconds per stage, memory, files
    results.jsonl                   every run, one line each
    gallery.html                    every run with its clip; open it in a browser

On Jetson, PyTorch isn't on PyPI: pass --python pointing at an env that already has
CUDA torch (and --comfy to reuse an existing ComfyUI). Unified-memory machines
(Jetson) automatically get the low-memory mode this was built around on an 8 GB Orin.
"""
import argparse
import ctypes
import functools
import io
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

VERSION = "1.4.0"
AUTHOR = "Dave Euson"
HOME = "github.com/DaveEuson/AI-Video-Bench"
COMFY_REPO = ("https://github.com/comfyanonymous/ComfyUI", "e5a38e3f7b91619ff295ffbbeddff35d8e381677")
GGUF_REPO = ("https://github.com/city96/ComfyUI-GGUF", "6ea2651e7df66d7585f6ffee804b20e92fb38b8a")
HF = "https://huggingface.co"

# The benchmark's own files, built in so the benchmark never changes; the lineup's come from
# models.json. key: (models subfolder, filename, url, GB)
FILES = {
    "zimage": ("diffusion_models", "z-image-turbo-Q4_K_M.gguf", f"{HF}/unsloth/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-Q4_K_M.gguf", 5.02),
    "klein": ("diffusion_models", "flux-2-klein-4b-Q5_K_M.gguf", f"{HF}/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q5_K_M.gguf", 3.07),
    "qwen3": ("text_encoders", "Qwen3-4B-Q5_K_M.gguf", f"{HF}/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q5_K_M.gguf", 2.89),
    "ae": ("vae", "ae.safetensors", f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors", 0.34),
    "flux2vae": ("vae", "flux2-vae.safetensors", f"{HF}/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/vae/flux2-vae.safetensors", 0.34),
    "ltxv2b": ("diffusion_models", "ltxv-2b-0.9.8-distilled-fp8.safetensors", f"{HF}/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.8-distilled-fp8.safetensors", 4.46),
    "ltxv13b": ("diffusion_models", "LTXV-13B-0.9.8-distilled-Q3_K_M.gguf", f"{HF}/QuantStack/LTXV-13B-0.9.8-distilled-GGUF/resolve/main/LTXV-13B-0.9.8-distilled-Q3_K_M.gguf", 6.51),
    # Tensor-for-tensor identical to the VAE inside the 2B checkpoint, so both LTX models use it.
    "ltxvae": ("vae", "LTXV-13B-0.9.8-distilled-VAE.safetensors", f"{HF}/QuantStack/LTXV-13B-0.9.8-distilled-GGUF/resolve/main/vae/LTXV-13B-0.9.8-distilled-VAE.safetensors", 2.49),
    "t5": ("text_encoders", "t5-v1_1-xxl-encoder-Q5_K_M.gguf", f"{HF}/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q5_K_M.gguf", 3.39),
}
STILLS = {
    "zimage": {"label": "Z-Image-Turbo 6B (Q4 GGUF)", "files": ["zimage", "qwen3", "ae"]},
    "klein": {"label": "FLUX.2 klein 4B (Q5 GGUF)", "files": ["klein", "qwen3", "flux2vae"]},
}
PROFILES = {
    "fast": {"video": "ltxv2b", "label": "LTX-Video 2B distilled (fp8)", "still": (1024, 768), "clip": (640, 480)},
    "quality": {"video": "ltxv13b", "label": "LTX-Video 13B distilled (Q3 GGUF)", "still": (1152, 768), "clip": (768, 512)},
}
FRAMES, FPS, SEED = 121, 24, 24  # 5.04 s
MAX_SECS = 12   # --secs ceiling


def frames_for(secs, fps, own):
    """--secs as a frame count the video models take: 8k+1, which is LTX's rule and inside
    Wan's and Hunyuan's (4k+1). 5 s at 24 fps is 121, the length everything here was
    benchmarked at. No --secs: the model's own."""
    if not secs:
        return own
    return -(-round(secs * fps) // 8) * 8 + 1
STILL_PROMPT = "A dog running in a field chasing a thrown ball, sunny afternoon."
CLIP_PROMPT = "A dog running in a field chasing a thrown ball, sunny afternoon, the camera tracking alongside."
LTX_NEGATIVE = ("low quality, worst quality, deformed, distorted, disfigured, blurry, watermark, text, "
                "jpeg artifacts, static, motionless")


class Status:
    """status.json in --dir: what a run is doing right now. `--status` and `--watch` read
    it from any terminal, so a long run in another window (or in the background) is
    never a black box."""

    def __init__(self):
        self.path, self.d = None, {}

    def begin(self, root):
        self.path = root / "status.json"
        self.d = {"pid": os.getpid(), "started": time.time(), "state": "running", "run": None}
        self.save()

    def set(self, **kw):
        if not self.path:
            return
        if "stage" in kw and kw["stage"] != self.d.get("stage"):
            self.d["stage_started"] = time.time()
            self.d.pop("progress", None)
        self.d.update(kw, updated=time.time())
        self.save()

    def save(self):
        try:
            tmp = self.path.with_name("status.tmp")
            tmp.write_text(json.dumps(self.d), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # a status write must never break a run


ST = Status()
STAGES = {"setup": "setting up ComfyUI and PyTorch", "download": "downloading models",
          "still": "painting the still", "clip_encode": "encoding the clip (text + still)",
          "clip_sample": "generating the clip", "clip_decode": "decoding the clip to video",
          "lineup": "running the lineup", "voice_setup": "installing a voice engine",
          "voices": "reading the lines aloud", "judge": "Whisper is checking every read"}


def log(msg):
    if _live["on"]:  # finish the in-place progress line first
        print(flush=True)
        _live["on"] = False
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)
    ST.set(message=msg)


def die(msg):
    log("ERROR: " + msg)
    ST.set(state="failed")
    sys.exit(1)


def pid_alive(pid):
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def ago(sec):
    m, s = divmod(int(max(sec, 0)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def part_progress(root, key):
    """'41% of 5.0 GB' for a download in progress, else None."""
    sub, name, _, gb = FILES[key]
    part = root / "models" / sub / (name + ".part")
    return f"{part.stat().st_size * 100 / (gb * 1e9):.0f}% of {gb:.1f} GB" if part.exists() else None


def status_text(root):
    p = root / "status.json"
    if not p.exists():
        return f"No videobench run has started in {root} yet."
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "Status is being written; try again."
    now = time.time()
    state = d.get("state", "?")
    if state == "running" and not pid_alive(d.get("pid", -1)):
        state = "stopped: the process is gone. Re-run the same command; finished work and downloads resume."
    end = now if d.get("state") == "running" else d.get("updated", now)
    out = [f"videobench  {root}",
           f"  state    {state}  |  {ago(end - d.get('started', now))}" + (f"  |  run {d['run']}" if d.get("run") else "")]
    if d.get("state") == "running" and d.get("stage"):
        out.append(f"  now      {STAGES.get(d['stage'], d['stage'])}  ({ago(now - d.get('stage_started', now))})")
        if d.get("progress"):
            out.append(f"           {d['progress']}")
    keys = [k for k in d.get("models") or [] if k in FILES]
    if keys and (root / "models").exists():
        ready = [k for k in keys if find_model(root / "models", FILES[k][1])]
        out.append(f"  models   {len(ready)} of {len(keys)} ready  "
                   f"({sum(FILES[k][3] for k in ready):.1f} of {sum(FILES[k][3] for k in keys):.1f} GB)")
    if d.get("summary"):
        out.append(f"  result   {d['summary']}")
    if d.get("message"):
        out.append(f"  last     {d['message'][:120]}")
    return "\n".join(out)


# ---------------------------------------------------------------- banner and progress

# Figlet "standard" lettering, one letter per ~-separated column so each glyph can be
# checked on its own.
_LETTERS = r"""
       ~ _ ~     _ ~      ~       ~ _     ~      ~       ~      ~ _
__   __~(_)~  __| |~  ___ ~  ___  ~| |__  ~  ___ ~ _ __  ~  ___ ~| |__
\ \ / /~| |~ / _` |~ / _ \~ / _ \ ~| '_ \ ~ / _ \~| '_ \ ~ / __|~| '_ \
 \ V / ~| |~| (_| |~|  __/~| (_) |~| |_) |~|  __/~| | | |~| (__ ~| | | |
  \_/  ~|_|~ \__,_|~ \___|~ \___/ ~|_.__/ ~ \___|~|_| |_|~ \___|~|_| |_|
"""
BANNER = ("\n".join(r.replace("~", "") for r in _LETTERS.strip("\n").splitlines())
          + f"\n\n   how fast can this machine make a 5-second AI video?\n"
            f"   videobench {VERSION} by {AUTHOR}   {HOME}\n")


def bar(frac, width=30):
    """A plain progress bar: [#########---------]."""
    pos = int(min(max(frac, 0.0), 1.0) * width)
    return "[" + "#" * pos + "-" * (width - pos) + "]"


_live = {"on": False, "last": 0.0}


def live(line):
    """One line that updates in place on a terminal; a new line every 5 s when output
    goes to a file or pipe, so logs stay readable."""
    if sys.stdout.isatty():
        w = shutil.get_terminal_size((100, 20)).columns - 1
        print("\r" + line[:w].ljust(w), end="", flush=True)
        _live["on"] = True
    elif time.time() - _live["last"] > 5:
        _live["last"] = time.time()
        print(time.strftime("[%H:%M:%S]   ") + line, flush=True)


def scorecard(rec):
    s, st, m = rec["stages"], rec["settings"], rec["machine"]
    w = 62
    rule = "+" + "-" * (w - 2) + "+"
    row = lambda t: "|  " + t[:w - 4].ljust(w - 4) + "|"
    top = max(s.values(), default=1) or 1
    out = [rule, row(f"videobench {VERSION}   {rec['profile']}   {m.get('gpu') or 'CPU'}"), rule]
    if st.get("custom_prompt"):
        import textwrap
        out += [row(t) for t in textwrap.wrap(f'custom prompt: "{st["clip_prompt"]}"', w - 4)[:3]]
        out += [row("(times don't compare with standard runs)"), rule]
    if (st.get("gpu_busy_at_start_pct") or 0) > 50:
        out += [row(f"heads up: the GPU was {st['gpu_busy_at_start_pct']}% busy before the start."),
                row("Times aren't a fair benchmark; close other GPU apps."), rule]
    for k in ("still", "clip_encode", "clip_sample", "clip_decode"):
        if k in s:
            out.append(row(f"{k.replace('_', ' '):<12}{s[k]:>8.1f} s  " + "#" * max(1, round(s[k] / top * 26))))
    out += [rule, row(f"{'TOTAL':<12}{rec['total_seconds']:>8.1f} s  for {st['seconds_of_video']} s of {st['clip_size']} video"), rule]
    if rec["status"] == "ok":
        out.append("   Done.")
    else:
        out.append(f"   Failed: {str(rec.get('error', ''))[:300]}")
    return "\n".join(out)


def show_status(root, watch):
    if not watch:
        return print(status_text(root))
    if sys.platform == "win32":
        os.system("")  # turns on ANSI escapes in the Windows console
    try:
        while True:
            print("\x1b[2J\x1b[H" + BANNER + "\n" + status_text(root)
                  + "\n\n  (refreshing every 2 s; Ctrl+C stops watching, not the run)", flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        print()


# ---------------------------------------------------------------- machine

def nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        name, mem, drv = [s.strip() for s in out[0].split(",")]
        return {"name": name, "vram_gb": round(int(mem) / 1024, 1), "driver": drv}
    except Exception:
        return None


def ram_gb():
    try:
        if sys.platform == "win32":
            class M(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong * 5)]
            m = M(); m.len = ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.total / 2**30, 1)
        if sys.platform == "darwin":
            return round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30, 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except Exception:
        return None


def ram_available_mb():
    try:
        if sys.platform == "win32":
            class M(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong * 5)]
            m = M(); m.len = ctypes.sizeof(M)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.avail // 2**20
        if sys.platform.startswith("linux"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


PROBE = r"""
import json, torch
d = {"torch": torch.__version__, "backend": "cpu", "gpu": None, "vram_gb": None, "unified": False}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    d.update(backend="rocm" if torch.version.hip else "cuda", gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1),
             unified=bool(getattr(p, "is_integrated", False)), cuda=torch.version.cuda or torch.version.hip)
    d["cc"] = list(torch.cuda.get_device_capability(0))
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    d.update(backend="mps", gpu="Apple Silicon", unified=True)
# One second of fp16 matrix multiplies: this GPU's speed, which scales the time estimates.
if d["backend"] != "cpu":
    import time
    dev = "mps" if d["backend"] == "mps" else "cuda"
    sync = torch.mps.synchronize if dev == "mps" else torch.cuda.synchronize
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    b = torch.randn_like(a)
    for _ in range(3):
        a @ b
    sync()
    n, t = 0, time.time()
    while time.time() - t < 1.0:
        a @ b
        n += 1
        if n % 10 == 0:
            sync()
    sync()
    d["tflops"] = round(n * 2 * 4096 ** 3 / (time.time() - t) / 1e12, 1)
print(json.dumps(d))
"""


def probe(py):
    r = subprocess.run([py, "-c", PROBE], capture_output=True, text=True)
    if r.returncode:
        die(f"could not import torch with {py}:\n{r.stderr[-1500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------- setup

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):
        return None


def hf_meta(url):
    """(size, sha256) Hugging Face publishes for a file, read from the resolve URL's
    redirect headers; (None, None) if they aren't there."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": f"videobench/{VERSION}"})
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    try:
        r = urllib.request.build_opener(_NoRedirect).open(req, timeout=30)
        h = r.headers
    except urllib.error.HTTPError as e:  # the 302 we asked not to follow
        h = e.headers
    except (urllib.error.URLError, OSError):
        return None, None
    size = h.get("X-Linked-Size") or None
    etag = (h.get("X-Linked-ETag") or "").strip('"')
    return (int(size) if size else None), (etag if len(etag) == 64 else None)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def verified(path, size, sha):
    """True when a finished file matches what Hugging Face says it should be."""
    if size and path.stat().st_size != size:
        log(f"  {path.name}: {path.stat().st_size} bytes, expected {size}")
        return False
    if sha:
        live(f"checking {path.name} against its SHA-256")
        if sha256_file(path) != sha:
            log(f"  {path.name}: SHA-256 doesn't match Hugging Face's")
            return False
    return True


def fetch(url, dest, gb=None, want_sha=None):
    """Resumable download to dest via dest.part, checked against Hugging Face's size
    and SHA-256 (or the one models.json pins) before it's kept. A bad file is thrown
    away and fetched again."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    size, sha = hf_meta(url)
    if want_sha:  # models.json pins this file: Hugging Face must still serve that exact one
        if sha and sha != want_sha:
            die(f"Hugging Face now serves a different {dest.name} than models.json lists. Run --update-models; "
                f"if it still happens, the file changed upstream: please say so at {REPO_URL}/issues")
        sha = want_sha
    for attempt in range(1, 6):
        have = part.stat().st_size if part.exists() else 0
        if size and have > size:  # longer than the real file: can't be trusted
            part.unlink()
            have = 0
        if size and have == size:  # all bytes already here; just check them
            if verified(part, size, sha):
                part.replace(dest)
                log(f"  {dest.name} done (verified)")
                return dest
            part.unlink()
            have = 0
        req = urllib.request.Request(url, headers={"User-Agent": f"videobench/{VERSION}"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        if os.environ.get("HF_TOKEN"):
            req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0  # server ignored the range; start over
                total = have + int(r.headers.get("Content-Length", 0))
                t0, last, done = time.time(), 0.0, have
                with open(part, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last > 0.5:
                            last = now
                            rate = (done - have) / max(now - t0, 1e-6) / 1e6
                            frac = done / total if total else 0
                            line = (f"{bar(frac)} {frac * 100:3.0f}%  {done / 1e9:.2f}/{(total or done) / 1e9:.2f} GB"
                                    f"  {rate:.0f} MB/s  {dest.name}")
                            live(line)
                            ST.set(progress=line)
            if not verified(part, size, sha):
                part.unlink()
                raise OSError("the finished file didn't match Hugging Face's checksum")
            part.replace(dest)
            log(f"  {dest.name} done" + (" (verified)" if sha else ""))
            return dest
        except (urllib.error.URLError, OSError) as e:
            log(f"  {dest.name}: attempt {attempt} failed ({e}); retrying")
            time.sleep(5 * attempt)
    die(f"could not download {url}")


def find_model(models, filename):
    for root, _, files in os.walk(models, followlinks=True):
        if filename in files:
            return Path(root) / filename
    return None


def ensure_files(models, keys):
    missing = [k for k in keys if not find_model(models, FILES[k][1])]
    ST.set(stage="download", models=keys)
    if missing:
        log(f"fetching {len(missing)} model file(s), {sum(FILES[k][3] for k in missing):.1f} GB.")
    for i, k in enumerate(missing, 1):
        sub, name, url, gb = FILES[k]
        log(f"downloading {i} of {len(missing)}: {name} ({gb:.1f} GB), which {ROLES[k]}")
        fetch(url, models / sub / name, gb, FILE_SHA.get(k))


def get_repo(url, ref, dest):
    if dest.exists():
        return
    log(f"fetching {url.rsplit('/', 1)[-1]} @ {ref[:7]}")
    if shutil.which("git"):
        subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "-q", ref], check=True)
    else:  # no git: GitHub's zip of the pinned commit
        data = urllib.request.urlopen(f"{url}/archive/{ref}.zip", timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            top = z.namelist()[0].split("/")[0]
            z.extractall(dest.parent)
        (dest.parent / top).rename(dest)


def torch_index(args, gpu):
    if args.torch_index:
        return args.torch_index
    if Path("/etc/nv_tegra_release").exists():
        die("this is a Jetson: PyTorch for it isn't on PyPI. Re-run with --python pointing at a Python "
            "that already has CUDA torch (jetson-containers or NVIDIA's Jetson wheels), or --torch-index.")
    if sys.platform == "darwin":
        return None
    if gpu:
        major = int(gpu["driver"].split(".")[0])
        return "https://download.pytorch.org/whl/" + ("cu130" if major >= 580 else "cu128" if major >= 570 else "cu126")
    if sys.platform.startswith("linux") and shutil.which("rocminfo"):
        return "https://download.pytorch.org/whl/rocm6.4"
    log("no GPU found: installing CPU PyTorch. A clip will take hours; this is only useful as a smoke test.")
    return "https://download.pytorch.org/whl/cpu"


def pip(py, *a):
    subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check", *a], check=True)


def ensure_python(args, root, comfy):
    if args.python:
        py = args.python
    else:
        venv = root / "venv"
        py = str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
        if not Path(py).exists():
            log("creating a Python venv")
            subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        if subprocess.run([py, "-c", "import torch"], capture_output=True).returncode:
            idx = torch_index(args, nvidia_smi())
            log(f"installing PyTorch ({idx or 'PyPI'}); this is the biggest download of setup")
            pip(py, "torch", "torchvision", "torchaudio", *(["--index-url", idx] if idx else []))
    if subprocess.run([py, "-c", "import comfyui_frontend_package, av"], capture_output=True).returncode:
        log("installing ComfyUI requirements")
        pip(py, "-r", str(comfy / "requirements.txt"))
    if subprocess.run([py, "-c", "import gguf, sentencepiece, google.protobuf"], capture_output=True).returncode:
        pip(py, "gguf", "sentencepiece", "protobuf")
    return py


def patch(path, marker, old_lines, new_lines):
    with open(path, encoding="utf-8", newline="") as f:  # newline="": keep upstream's CRLF
        s = f.read()
    if marker in s:
        return
    nl = "\r\n" if "\r\n" in s else "\n"
    old, new = nl.join(old_lines + [""]), nl.join(new_lines + [""])
    if s.count(old) != 1:
        log(f"warning: couldn't apply the low-memory patch to {path.name}; big models may run out of memory")
        return
    lines = s.replace(old, new).split(nl)
    if "import os" not in lines:
        lines.insert(next(i for i, l in enumerate(lines) if l.startswith(("import ", "from "))), "import os")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(nl.join(lines))


def patch_gguf(node):
    """Only active with GGUF_KEEP_MMAP=1, which videobench sets on unified-memory machines.
    1. GGUFModelPatcher.load() "releases" its mmap by copying every weight into RAM,
       which OOM-killed ComfyUI on an 8 GB Jetson. 2. umt5/T5 embeddings dequantize in
       row chunks instead of one multi-GB pass."""
    patch(node / "nodes.py", "GGUF_KEEP_MMAP",
          ["    def load(self, *args, force_patch_weights=False, **kwargs):", "        if not self.mmap_released:"],
          ["    def load(self, *args, force_patch_weights=False, **kwargs):",
           "        # videobench: on unified memory, releasing the mmap copies every weight into RAM.",
           "        if os.environ.get(\"GGUF_KEEP_MMAP\"):", "            self.mmap_released = True",
           "        if not self.mmap_released:"])
    patch(node / "loader.py", "dequantize_rows",
          ["            logging.warning(f\"Dequantizing {temb_key} to prevent runtime OOM.\")",
           "            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)",
           "        sd = sd_map_replace(sd, T5_SD_MAP)"],
          ["            logging.warning(f\"Dequantizing {temb_key} to prevent runtime OOM.\")",
           "            if os.environ.get(\"GGUF_KEEP_MMAP\"):",
           "                sd[temb_key] = dequantize_rows(sd[temb_key], torch.float16)",
           "            else:", "                sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)",
           "        sd = sd_map_replace(sd, T5_SD_MAP)"])
    patch(node / "loader.py", "def dequantize_rows",
          ["def gguf_clip_loader(path):"],
          ["def dequantize_rows(t, dtype, chunk=16384):",
           "    # videobench: same result as dequantize_tensor, a slice of rows at a time.",
           "    from .dequant import dequantize", "    n, dim = t.tensor_shape",
           "    raw = torch.Tensor.as_subclass(t, torch.Tensor).reshape(n, -1)",
           "    out = torch.empty((n, dim), dtype=dtype)", "    for i in range(0, n, chunk):",
           "        rows = raw[i:i + chunk]",
           "        out[i:i + len(rows)] = dequantize(rows, t.tensor_type, (len(rows), dim), dtype=dtype)",
           "    return out", "", "def gguf_clip_loader(path):"])


NODES = '''"""videobench helper nodes (written by videobench.py).
StashConditioning/StashLatent and their Unstash pairs hand results between prompts
through disk, so a video job can run as encode -> sample -> decode with models
unloaded in between (core SaveLatent drops the noise mask image-to-video needs).
VAEKeepLoaded stops a video decode re-copying the VAE for every chunk of frames."""
import os
import torch
import folder_paths

STASH = os.path.join(folder_paths.get_output_directory(), "stash")


def _path(key):
    return os.path.join(STASH, f"{key}.pt")


class _Stash:
    CATEGORY = "videobench"
    FUNCTION = "stash"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (cls.TYPE,), "key": ("STRING", {"default": "job"})}}

    def stash(self, value, key):
        os.makedirs(STASH, exist_ok=True)
        torch.save(value, _path(key))
        return {}


class _Unstash:
    CATEGORY = "videobench"
    FUNCTION = "unstash"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"key": ("STRING", {"default": "job"})}}

    def unstash(self, key):
        return (torch.load(_path(key), weights_only=False),)


class VAEKeepLoaded:
    CATEGORY = "videobench"
    FUNCTION = "keep"
    RETURN_TYPES = ("VAE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",)}}

    def keep(self, vae):
        vae.disable_offload = True
        return (vae,)


NODE_CLASS_MAPPINGS = {"VAEKeepLoaded": VAEKeepLoaded}
for _t in ("CONDITIONING", "LATENT"):
    _n = _t.title()
    NODE_CLASS_MAPPINGS[f"Stash{_n}"] = type(f"Stash{_n}", (_Stash,), {"TYPE": _t})
    NODE_CLASS_MAPPINGS[f"Unstash{_n}"] = type(f"Unstash{_n}", (_Unstash,), {"RETURN_TYPES": (_t,)})
'''


def setup(args, root):
    for d in ("models", "input", "output", "user", "custom_nodes", "results"):
        (root / d).mkdir(parents=True, exist_ok=True)
    ST.set(stage="setup")
    here = Path(__file__).resolve()
    if here.parent != root:  # a copy lives with the install, so `python videobench.py` works from there
        try:
            shutil.copy2(here, root / here.name)
            mine = here.with_name("models.json")
            if mine.exists() and _revision(mine) > _revision(root / "models.json"):  # never over a newer --update-models
                shutil.copy2(mine, root / "models.json")
        except OSError:
            pass
    log("checking for ComfyUI and PyTorch (anything already installed is skipped)")
    comfy = Path(args.comfy).expanduser().resolve() if args.comfy else root / "ComfyUI"
    get_repo(*COMFY_REPO, comfy)
    py = ensure_python(args, root, comfy)
    node = root / "custom_nodes" / "ComfyUI-GGUF"
    get_repo(*GGUF_REPO, node)
    patch_gguf(node)
    (root / "custom_nodes" / "videobench_nodes.py").write_text(NODES, encoding="utf-8")
    m = (root / "models").as_posix()
    (root / "videobench_paths.yaml").write_text(
        f"videobench:\n  base_path: {root.as_posix()}/\n  is_default: true\n"
        f"  diffusion_models: |\n    models/diffusion_models/\n    models/unet/\n"
        f"  text_encoders: |\n    models/text_encoders/\n    models/clip/\n"
        f"  vae: models/vae/\n  loras: models/loras/\n  checkpoints: models/checkpoints/\n"
        f"  custom_nodes: custom_nodes/\n", encoding="utf-8")
    return comfy, py


# ---------------------------------------------------------------- ComfyUI server

class Server:
    def __init__(self, root, comfy, py, hw, port):
        self.root, self.comfy, self.py, self.hw, self.port = root, comfy, py, hw, port
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None
        self.lowmem = hw["unified"] and hw["backend"] == "cuda"

    def start(self):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", self.port)) == 0:
                die(f"something is already listening on port {self.port}; pass --port")
        r = self.root
        cmd = [self.py, str(self.comfy / "main.py"), "--listen", "127.0.0.1", "--port", str(self.port),
               "--extra-model-paths-config", str(r / "videobench_paths.yaml"),
               "--output-directory", str(r / "output"), "--input-directory", str(r / "input"),
               "--user-directory", str(r / "user"), "--database-url", f"sqlite:///{(r / 'user' / 'comfyui.db').as_posix()}"]
        env = dict(os.environ)
        if self.lowmem:
            # Unified memory (Jetson): CPU and GPU share one pool, so ComfyUI's discrete-GPU
            # offloading only duplicates weights. Keep them memory-mapped, stream per layer.
            cmd += ["--disable-dynamic-vram", "--novram", "--disable-smart-memory", "--disable-pinned-memory", "--cache-none"]
            env["GGUF_KEEP_MMAP"] = "1"
            if shutil.which("systemd-run") and subprocess.run(["systemd-run", "--user", "--scope", "-q", "true"],
                                                              capture_output=True).returncode == 0:
                cap = max(int((ram_gb() - 2) * 1024), 3072)
                cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={cap}M", "-p", "MemorySwapMax=1G"] + cmd
        self.logf = open(r / "comfy.log", "ab")
        self.proc = subprocess.Popen(cmd, cwd=self.comfy, env=env, stdout=self.logf, stderr=subprocess.STDOUT)
        log(f"starting ComfyUI{' (low-memory mode)' if self.lowmem else ''}; log: {r / 'comfy.log'}")
        for _ in range(600):
            if self.proc.poll() is not None:
                die("ComfyUI exited while starting:\n" + self.tail())
            try:
                urllib.request.urlopen(self.url + "/system_stats", timeout=2)
                return
            except Exception:
                time.sleep(1)
        die("ComfyUI didn't come up in 10 minutes:\n" + self.tail())

    def tail(self, n=25):
        try:
            return "\n".join((self.root / "comfy.log").read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def alive(self):
        return self.proc and self.proc.poll() is None

    def stop(self):
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def call(self, path, body=None):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected {path}: {e.read().decode()[:2000]}")
        except (urllib.error.URLError, ConnectionError) as e:
            if not self.alive():
                raise RuntimeError("ComfyUI died, most likely out of memory. Last log lines:\n" + self.tail(8))
            raise RuntimeError(f"can't reach ComfyUI: {e}")

    def step(self, since):
        """Latest sampler progress ("4/8") ComfyUI printed to its log after byte `since`."""
        logp = self.root / "comfy.log"
        try:
            with open(logp, "rb") as f:
                f.seek(max(since, logp.stat().st_size - 4096))
                tail = f.read().decode(errors="replace")
        except OSError:
            return None
        m = re.findall(r"(\d+)/(\d+) \[", tail)
        return f"{m[-1][0]}/{m[-1][1]}" if m else None

    def run(self, graph, stage):
        """Queue one prompt, wait for it. Returns (output file paths, seconds)."""
        t0 = time.time()
        logp = self.root / "comfy.log"
        since = logp.stat().st_size if logp.exists() else 0
        ST.set(stage=stage)
        pid = self.call("/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})["prompt_id"]
        shown = None
        while True:
            time.sleep(1)
            step = self.step(since)
            if step:
                k, n = map(int, step.split("/"))
                line = f"{bar(k / n)} step {step}  {time.time() - t0:.0f} s"
                live(line)
                if step != shown:
                    shown = step
                    ST.set(progress=line)
            h = self.call(f"/history/{pid}").get(pid)
            st = (h or {}).get("status", {})
            if st.get("completed") is not None or st.get("status_str") == "error":
                break
        if st.get("status_str") != "success":
            msgs = [m for kind, m in st.get("messages", []) if kind == "execution_error"]
            raise RuntimeError(f"{msgs[0].get('node_type')}: {msgs[0].get('exception_message')}" if msgs else "prompt failed")
        outs = [self.root / "output" / f.get("subfolder", "") / f["filename"]
                for node in h.get("outputs", {}).values() for v in node.values() if isinstance(v, list)
                for f in v if isinstance(f, dict) and "filename" in f]
        self.call("/free", {"unload_models": True, "free_memory": True})
        return outs, round(time.time() - t0, 1)


class Watch(threading.Thread):
    """Lowest free system RAM and highest GPU memory in use while the benchmark runs."""

    def __init__(self, discrete_nvidia):
        super().__init__(daemon=True)
        self.nv, self.ram_min, self.gpu_max = discrete_nvidia, None, None
        self.halt = threading.Event()

    def run(self):
        while not self.halt.is_set():
            a = ram_available_mb()
            if a is not None:
                self.ram_min = a if self.ram_min is None else min(self.ram_min, a)
            if self.nv:
                try:
                    used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                              capture_output=True, text=True, timeout=10).stdout.split()[0])
                    self.gpu_max = used if self.gpu_max is None else max(self.gpu_max, used)
                except Exception:
                    pass
            self.halt.wait(1)


# ---------------------------------------------------------------- workflows

def gguf_or_unet(name):
    if name.endswith(".gguf"):
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": name}}
    return {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "fp8_e4m3fn"}}


def still_graph(kind, prompt, w, h, prefix, seed=SEED):
    qwen = {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": FILES["qwen3"][1], "type": "lumina2" if kind == "zimage" else "flux2"}}
    if kind == "zimage":
        return {
            "1": gguf_or_unet(FILES["zimage"][1]), "2": qwen,
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": FILES["ae"][1]}},
            "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
            "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
            "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0],
                  "latent_image": ["7", 0], "seed": seed, "steps": 8, "cfg": 1, "sampler_name": "res_multistep",
                  "scheduler": "simple", "denoise": 1}},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
        }
    return {
        "1": gguf_or_unet(FILES["klein"][1]), "2": qwen,
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": FILES["flux2vae"][1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": 1}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": w, "height": h}},
        "9": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["6", 0], "sampler": ["7", 0],
               "sigmas": ["8", 0], "latent_image": ["9", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }


def clip_phases(video, image, prompt, w, h, key, prefix, lowmem, seed=SEED, frames=FRAMES):
    """LTX-Video image-to-video as three prompts that hand off through disk:
    encode (T5 + VAE encode of the still), sample (the DiT alone), decode (the VAE alone).
    Only one big model is ever loaded, and each phase is timed on its own."""
    tiles = ({"tile_size": 256, "overlap": 32, "temporal_size": 32, "temporal_overlap": 4} if lowmem else
             {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8})
    vae = {"class_type": "VAELoader", "inputs": {"vae_name": FILES["ltxvae"][1]}}
    stash = lambda t, ref, part: {"class_type": f"Stash{t}", "inputs": {"value": ref, "key": f"{key}-{part}"}}
    unstash = lambda t, part: {"class_type": f"Unstash{t}", "inputs": {"key": f"{key}-{part}"}}
    encode = {
        "t5": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": FILES["t5"][1], "type": "ltxv"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["t5", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["t5", 0], "text": LTX_NEGATIVE}},
        "vae": vae,
        "img": {"class_type": "LoadImage", "inputs": {"image": image}},
        "i2v": {"class_type": "LTXVImgToVideo", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
                "image": ["img", 0], "width": w, "height": h, "length": frames, "batch_size": 1, "strength": 1.0}},
        "cond": {"class_type": "LTXVConditioning", "inputs": {"positive": ["i2v", 0], "negative": ["i2v", 1], "frame_rate": FPS}},
        "s1": stash("Conditioning", ["cond", 0], "pos"), "s2": stash("Conditioning", ["cond", 1], "neg"),
        "s3": stash("Latent", ["i2v", 2], "lat"),
    }
    sample = {
        "dit": gguf_or_unet(FILES[video][1]),
        "u1": unstash("Conditioning", "pos"), "u2": unstash("Conditioning", "neg"), "u3": unstash("Latent", "lat"),
        "ks": {"class_type": "KSampler", "inputs": {"model": ["dit", 0], "positive": ["u1", 0], "negative": ["u2", 0],
               "latent_image": ["u3", 0], "seed": seed, "steps": 8, "cfg": 1, "sampler_name": "euler",
               "scheduler": "simple", "denoise": 1}},
        "s4": stash("Latent", ["ks", 0], "out"),
    }
    decode = {
        "vae": vae, "vk": {"class_type": "VAEKeepLoaded", "inputs": {"vae": ["vae", 0]}},
        "u4": unstash("Latent", "out"),
        "dec": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["u4", 0], "vae": ["vk", 0], **tiles}},
        "mk": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "fps": FPS}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["mk", 0], "filename_prefix": prefix, "format": "mp4", "codec": "h264"}},
    }
    return [("clip_encode", encode), ("clip_sample", sample), ("clip_decode", decode)]


def sizes(args):
    """(clip, still) sizes: the profile's own, or --size's, with a still big enough to animate from."""
    prof = PROFILES[args.profile]
    if not getattr(args, "size_wh", None):
        return prof["clip"], prof["still"]
    w, h = args.size_wh
    k = 1024 / max(w, h)
    return (w, h), (max(64, round(w * k / 64) * 64), max(64, round(h * k / 64) * 64))


def resize(graph, w, h):
    """--size for a lineup model: every node that sets the frame size gets w x h instead of the
    model's own. In these workflows that is only the empty latent (or image-to-video) node."""
    for node in graph.values():
        i = node.get("inputs", {})
        if isinstance(i.get("width"), int) and isinstance(i.get("height"), int):
            i["width"], i["height"] = w, h
    return graph


def relength(graph, frames):
    """--secs for a lineup model: every node that sets the clip's length gets `frames` instead
    of the model's own -- the empty latent (or image-to-video) node, and LTX-2's audio latent."""
    for node in graph.values():
        i = node.get("inputs", {})
        for k in ("length", "frames_number"):
            if isinstance(i.get(k), int):
                i[k] = frames
    return graph


# ---------------------------------------------------------------- the benchmark

def machine(hw):
    nv = nvidia_smi()
    return {"os": f"{platform.system()} {platform.release()}", "cpu": cpu_name(), "ram_gb": ram_gb(),
            "gpu": hw["gpu"] or (nv or {}).get("name"), "vram_gb": hw["vram_gb"], "unified_memory": hw["unified"],
            "driver": (nv or {}).get("driver"), "backend": hw["backend"], "torch": hw["torch"], "cuda": hw.get("cuda"),
            "jetson": Path("/etc/nv_tegra_release").exists()}


ROLES = {"zimage": "paints the still", "klein": "paints the still", "qwen3": "reads the prompt for the still",
         "ae": "turns the still into pixels", "flux2vae": "turns the still into pixels",
         "ltxv2b": "animates the still into the clip", "ltxv13b": "animates the still into the clip",
         "t5": "reads the prompt for the clip", "ltxvae": "packs the still in, unpacks the video out"}


def plan_keys(args):
    return STILLS[args.still]["files"] + [PROFILES[args.profile]["video"], "t5", "ltxvae"]


def plan_text(args, root, prompt=None, out=None):
    """What a run will download and test, shown before it starts (and by --list)."""
    prof, nv = PROFILES[args.profile], nvidia_smi()
    gpu = f"{nv['name']} {nv['vram_gb']} GB" if nv else "GPU checked once PyTorch loads (no nvidia-smi)"
    lines = ["  ---- THE PLAN " + "-" * 60, f"  machine   {cpu_name()} | {ram_gb()} GB RAM | {gpu}", ""]
    todo = 0.0
    for i, k in enumerate(plan_keys(args)):
        sub, name, _, gb = FILES[k]
        have = find_model(root / "models", name) if (root / "models").exists() else None
        part = None if have else part_progress(root, k)
        todo += 0 if have else gb
        tag = "have" if have else "part" if part else "get"
        lines.append(f"  {'download' if i == 0 else '':<9} {tag:<4} {gb:5.2f} GB  {name:<42} {ROLES[k]}"
                     + (f"  ({part} so far)" if part else ""))
    lines.append(f"  {'':<9} " + (f"{todo:.1f} GB to download" if todo else "nothing to download, all here"))
    (w, h), (sw, sh) = sizes(args)
    frames = frames_for(getattr(args, "secs", None), FPS, FRAMES)
    p = prompt or CLIP_PROMPT
    lines += ["",
              f"  test      [1/4] paint the still     {STILLS[args.still]['label']}, {sw}x{sh}, {8 if args.still == 'zimage' else 4} steps",
              "            [2/4] encode              T5 reads the prompt, the VAE packs the still in",
              f"            [3/4] generate the clip   {prof['label']}, {w}x{h}, {frames} frames, 8 steps",
              f"            [4/4] decode              the VAE unpacks it into a {frames / FPS:.2f} s mp4 at {FPS} fps",
              "",
              f'  prompt    "{p}"' + ("" if p == CLIP_PROMPT else "   (custom: times won't compare)"),
              f"  saves to  {out or root / 'output' / 'videobench'}",
              "  " + "-" * 74]
    return "\n".join(lines)


def gpu_busy():
    """Average NVIDIA GPU load over ~3 s before a run (None without nvidia-smi). A game
    or render running alongside makes a benchmark meaningless, so it gets flagged."""
    if not shutil.which("nvidia-smi"):
        return None
    vals = []
    for _ in range(3):
        try:
            vals.append(int(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                           capture_output=True, text=True, timeout=10).stdout.split()[0]))
        except Exception:
            return None
        time.sleep(1)
    return round(sum(vals) / len(vals))


def save_outputs(root, rec, out):
    """Copy the clip and still to where the user asked. A folder gets a name made from
    the prompt; a path ending in .mp4 is used as-is, with the still beside it."""
    slug = re.sub(r"[^a-z0-9]+", "-", rec["settings"]["clip_prompt"].lower())[:48].strip("-") or "clip"
    dest = Path(out).expanduser()
    clip = dest if dest.suffix.lower() == ".mp4" else dest / f"{slug}-{rec['run']}.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / rec["files"]["clip"], clip)
    still = clip.with_suffix(".png")
    shutil.copy2(root / rec["files"]["still"], still)
    return {"clip_saved": str(clip), "still_saved": str(still)}


def bench(args, root, comfy, py):
    prof = PROFILES[args.profile]
    clip_wh, still_wh = sizes(args)
    frames = frames_for(getattr(args, "secs", None), FPS, FRAMES)
    hw = steady_tflops(root, probe(py))
    if hw["backend"] == "cpu":
        log("warning: PyTorch sees no GPU; this will be extremely slow")
    ensure_files(root / "models", STILLS[args.still]["files"] + [prof["video"], "t5", "ltxvae"])

    stamp = time.strftime("%Y%m%d-%H%M%S")
    key = f"{stamp}-{args.profile}"
    rec = {"videobench": VERSION, "run": key, "profile": args.profile, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "machine": machine(hw),
           "settings": {"still_model": STILLS[args.still]["label"], "video_model": prof["label"],
                        "still_prompt": args.still_prompt, "clip_prompt": args.clip_prompt,
                        "custom_prompt": args.clip_prompt != CLIP_PROMPT, "seed": args.seed,
                        "still_size": "x".join(map(str, still_wh)), "clip_size": "x".join(map(str, clip_wh)),
                        "frames": frames, "fps": FPS, "seconds_of_video": round(frames / FPS, 2),
                        "still_steps": 8 if args.still == "zimage" else 4, "clip_steps": 8},
           "stages": {}, "status": "running", "files": {}}
    srv = Server(root, comfy, py, hw, args.port)
    rec["settings"]["low_memory_mode"] = srv.lowmem
    busy = None if hw["unified"] else gpu_busy()
    rec["settings"]["gpu_busy_at_start_pct"] = busy
    if busy is not None and busy > 50:
        log(f"heads up: something else is using the GPU ({busy}% busy before we even started). "
            "The video will be fine, but these times won't be a fair benchmark.")
    watch = Watch(bool(nvidia_smi()) and not hw["unified"])
    srv.start()
    watch.start()
    t0 = time.time()
    headers = {"clip_encode": "[2/4] encoding the prompt and the still",
               "clip_sample": "[3/4] generating the clip", "clip_decode": "[4/4] decoding it into an mp4"}
    try:
        ST.set(run=key)
        log(f"[1/4] painting the still: {STILLS[args.still]['label']} at {rec['settings']['still_size']}")
        outs, rec["stages"]["still"] = srv.run(still_graph(args.still, args.still_prompt, *still_wh, f"videobench/{key}-still",
                                                           seed=args.seed), "still")
        still = outs[0]
        rec["files"]["still"] = still.relative_to(root).as_posix()
        shutil.copy(still, root / "input" / f"{key}-still.png")
        log(f"      done in {rec['stages']['still']} s")
        for name, g in clip_phases(prof["video"], f"{key}-still.png", args.clip_prompt, *clip_wh, key,
                                   f"videobench/{key}-clip", srv.lowmem, seed=args.seed, frames=frames):
            log(f"{headers[name]}: {prof['label']} at {rec['settings']['clip_size']}")
            outs, rec["stages"][name] = srv.run(g, name)
            log(f"      done in {rec['stages'][name]} s")
        rec["files"]["clip"] = next(o for o in outs if o.suffix == ".mp4").relative_to(root).as_posix()
        rec["status"] = "ok"
    except RuntimeError as e:
        rec["status"], rec["error"] = "failed", str(e)
        log("FAILED: " + str(e))
    finally:
        watch.halt.set()
        srv.stop()
    rec["total_seconds"] = round(time.time() - t0, 1)
    rec["memory"] = {"ram_available_min_mb": watch.ram_min, "gpu_used_max_mb": watch.gpu_max}
    video = still = None
    if rec["status"] == "ok":
        video, still = root / rec["files"]["clip"], root / rec["files"]["still"]
        if args.out:
            try:
                rec["files"].update(save_outputs(root, rec, args.out))
                video, still = Path(rec["files"]["clip_saved"]), Path(rec["files"]["still_saved"])
            except OSError as e:
                log(f"couldn't save to {args.out} ({e}); the clip is still in {video.parent}")
    (root / "results" / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    with open(root / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    gallery(root)

    s = rec["stages"]
    ST.set(state="done" if rec["status"] == "ok" else "failed", stage=None, video=str(video) if video else None,
           summary=f"{rec['status']} in {rec['total_seconds']:.0f} s ("
                   + ", ".join(f"{k.replace('clip_', '')} {v:.0f} s" for k, v in s.items()) + ")")
    print()
    print(scorecard(rec))
    if video:
        print(f"\n  your video:  {video}")
        print(f"  the still:   {still}")
    print(f"\n  result:  {root / 'results' / (key + '.json')}")
    print(f"  gallery: {root / 'gallery.html'}")
    if rec["status"] == "ok":
        print(f"  share it: https://{HOME}/issues/new?template=share-result.yml")
    return rec["status"] == "ok"


def paint_only(args, root, comfy, py, hw):
    """--still-only: the still and nothing else, for a picture that never moves."""
    w, h = sizes(args)[1]
    key = time.strftime("%Y%m%d-%H%M%S") + "-still"
    ensure_files(root / "models", STILLS[args.still]["files"])
    srv = Server(root, comfy, py, hw, args.port)
    ST.set(run=key)
    try:
        srv.start()
        log(f"painting the still: {STILLS[args.still]['label']} at {w}x{h}, seed {args.seed}")
        outs, secs = srv.run(still_graph(args.still, args.still_prompt, w, h, f"videobench/{key}", seed=args.seed), "still")
    except RuntimeError as e:
        log("FAILED: " + str(e))
        ST.set(state="failed", stage=None)
        return False
    finally:
        srv.stop()
    still = outs[0]
    if args.out:
        dest = Path(args.out).expanduser()
        dest = dest if dest.suffix.lower() == ".png" else dest / f"{key}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(still, dest)
        still = dest
    log(f"      done in {secs} s: {still}")
    ST.set(state="done", stage=None, summary=f"still in {secs:.0f} s: {still}")
    return True


# ---------------------------------------------------------------- the lineup: many models, one prompt
#
# Each contestant is text-to-video with the settings from ComfyUI's own template for
# that model, trimmed to one pass (no upscaler stage, no AI prompt rewriter) so runs
# are comparable. Times on the menu are estimates until a model has run here.
#
# The contestants live in models.json, not in this file: each one's files, which builder
# below turns it into a ComfyUI workflow, its settings, and its license. A new version of
# a model family is a models.json edit, which `--update-models` fetches. A new family
# needs a new builder here, so a new videobench.py.

REF_TFLOPS = 52.0  # the RTX 4070 the reference times are for (fp16 matmul, the probe above)
LTX2_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"
LTX2_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
                "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
                "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

REPO_URL = "https://github.com/DaveEuson/AI-Video-Bench"
MODELS_URL = "https://raw.githubusercontent.com/DaveEuson/AI-Video-Bench/main/models.json"
SCHEMA = 1  # the models.json layout this videobench reads
MODEL_FOLDERS = ("diffusion_models", "text_encoders", "vae", "loras", "checkpoints")
BUILTIN_FILES = frozenset(FILES)  # the benchmark's own: models.json may use them, never replace them
CONTESTANTS = []                  # filled from models.json by load_registry()
FILE_SHA = {}                     # file key -> the sha256 models.json pins it to
REGISTRY = {"path": None, "revision": None, "updated": None, "problems": [], "newer": [], "raw": {}}
OPTS = {"country": None}          # where the user said they are (two letters), for licenses that exclude countries
# The European Union's members, for licenses that exclude "the EU" (2026).
EU = frozenset("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split())


def _n(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def _tail(g, images, fps, prefix, audio=None):
    g["90"] = _n("CreateVideo", images=images, fps=fps, **({"audio": audio} if audio else {}))
    g["91"] = _n("SaveVideo", video=["90", 0], filename_prefix=prefix, format="mp4", codec="h264")
    return g


def _unet(name, dtype):
    """A diffusion model: GGUF through ComfyUI-GGUF's loader, anything else through ComfyUI's."""
    if name.endswith(".gguf"):
        return _n("UnetLoaderGGUF", unet_name=name)
    return _n("UNETLoader", unet_name=name, weight_dtype=dtype)


def _clip(name, kind):
    """A text encoder, the same way."""
    if name.endswith(".gguf"):
        return _n("CLIPLoaderGGUF", clip_name=name, type=kind)
    return _n("CLIPLoader", clip_name=name, type=kind, device="default")


# ---- builders: models.json names one per model, says which file plays each part ("roles";
# a "?" marks an optional one), and may change any of its settings ("params").

BUILDERS = {}


def builder(name, roles, splittable=False, **params):
    """splittable: the workflow's KSampler is node "8" and its decode node "9", so the
    low-memory mode can run it as encode / sample / decode (split_phases)."""
    def reg(fn):
        BUILDERS[name] = {"fn": fn, "roles": roles, "params": params, "splittable": splittable}
        return fn
    return reg


class _Model:
    """A builder's view of its models.json entry: files by role, settings with defaults."""

    def __init__(self, c):
        self.c = c
        self.p = {**BUILDERS[c["builder"]]["params"], **(c.get("params") or {})}
        self.w, self.h = (int(v) for v in c["size"].split("x"))
        self.frames, self.fps, self.steps = c["frames"], c["fps"], c["steps"]

    def f(self, role):
        """The file name of the file playing this role."""
        return FILES[self.c["roles"][role]][1]

    def has(self, role):
        return role in self.c["roles"]


def build_graph(c, prompt, seed, tiles, prefix):
    """The ComfyUI workflow for one contestant, at its own size, length and steps."""
    return BUILDERS[c["builder"]]["fn"](_Model(c), prompt, seed, tiles, prefix)


def _lora(g, m, model, node, role="lora"):
    """The LoRA for `role` on top of `model` when the entry has one; the model to sample with."""
    if not m.has(role):
        return model
    g[node] = _n("LoraLoaderModelOnly", model=model, lora_name=m.f(role), strength_model=m.p["lora_strength"])
    return [node, 0]


@builder("ltxv", ("dit", "te", "vae"), splittable=True, dtype="fp8_e4m3fn", cfg=1, sampler="euler", scheduler="simple")
def _ltxv(m, prompt, seed, tiles, prefix):
    """LTX-Video 0.9.x distilled (2B, 13B): 8 steps at CFG 1."""
    p = m.p
    g = {"1": _unet(m.f("dit"), p["dtype"]), "2": _clip(m.f("te"), "ltxv"),
         "3": _n("CLIPTextEncode", clip=["2", 0], text=prompt), "4": _n("CLIPTextEncode", clip=["2", 0], text=LTX_NEGATIVE),
         "5": _n("VAELoader", vae_name=m.f("vae")),
         "6": _n("EmptyLTXVLatentVideo", width=m.w, height=m.h, length=m.frames, batch_size=1),
         "7": _n("LTXVConditioning", positive=["3", 0], negative=["4", 0], frame_rate=m.fps),
         "8": _n("KSampler", model=["1", 0], positive=["7", 0], negative=["7", 1], latent_image=["6", 0], seed=seed,
                 steps=m.steps, cfg=p["cfg"], sampler_name=p["sampler"], scheduler=p["scheduler"], denoise=1),
         "9": _n("VAEDecodeTiled", samples=["8", 0], vae=["5", 0], **tiles)}
    return _tail(g, ["9", 0], m.fps, prefix)


@builder("wan21", ("dit", "te", "vae"), splittable=True, dtype="default", shift=8, cfg=6, sampler="uni_pc",
         scheduler="simple")
def _wan21(m, prompt, seed, tiles, prefix):
    """Wan 2.1 text-to-video (the template's 30 steps at CFG 6): the 1.3B and the 14B."""
    p = m.p
    g = {"1": _unet(m.f("dit"), p["dtype"]),
         "2": _clip(m.f("te"), "wan"),
         "3": _n("VAELoader", vae_name=m.f("vae")),
         "4": _n("ModelSamplingSD3", model=["1", 0], shift=p["shift"]),
         "5": _n("CLIPTextEncode", clip=["2", 0], text=prompt), "6": _n("CLIPTextEncode", clip=["2", 0], text=WAN_NEGATIVE),
         "7": _n("EmptyHunyuanLatentVideo", width=m.w, height=m.h, length=m.frames, batch_size=1),
         "8": _n("KSampler", model=["4", 0], positive=["5", 0], negative=["6", 0], latent_image=["7", 0], seed=seed,
                 steps=m.steps, cfg=p["cfg"], sampler_name=p["sampler"], scheduler=p["scheduler"], denoise=1),
         "9": _n("VAEDecodeTiled", samples=["8", 0], vae=["3", 0], **tiles)}
    return _tail(g, ["9", 0], m.fps, prefix)


@builder("wan22_5b", ("dit", "te", "vae"), splittable=True, dtype="default", shift=8, cfg=5, sampler="uni_pc",
         scheduler="simple")
def _wan22_5b(m, prompt, seed, tiles, prefix):
    """Wan 2.2 TI2V 5B: the official weights (20 steps, CFG 5) or a turbo (4 steps, CFG 1,
    so the negative prompt is zeroed out)."""
    p = m.p
    g = {"1": _unet(m.f("dit"), p["dtype"]),
         "2": _clip(m.f("te"), "wan"),
         "3": _n("VAELoader", vae_name=m.f("vae")),
         "4": _n("ModelSamplingSD3", model=["1", 0], shift=p["shift"]),
         "5": _n("CLIPTextEncode", clip=["2", 0], text=prompt),
         "6": (_n("CLIPTextEncode", clip=["2", 0], text=WAN_NEGATIVE) if p["cfg"] > 1
               else _n("ConditioningZeroOut", conditioning=["5", 0])),
         "7": _n("Wan22ImageToVideoLatent", vae=["3", 0], width=m.w, height=m.h, length=m.frames, batch_size=1),
         "8": _n("KSampler", model=["4", 0], positive=["5", 0], negative=["6", 0], latent_image=["7", 0], seed=seed,
                 steps=m.steps, cfg=p["cfg"], sampler_name=p["sampler"], scheduler=p["scheduler"], denoise=1),
         "9": _n("VAEDecodeTiled", samples=["8", 0], vae=["3", 0], **tiles)}
    return _tail(g, ["9", 0], m.fps, prefix)


@builder("wan22_14b", ("hi", "lo", "lora_hi?", "lora_lo?", "te", "vae"), dtype="default", lora_strength=1.0, shift=5,
         cfg=1, sampler="euler", scheduler="simple")
def _wan22_14b(m, prompt, seed, tiles, prefix):
    """Wan 2.2 A14B, two experts: the high-noise model takes the first half of the steps and
    the low-noise model finishes (with the 4-step LoRAs: steps 0-2, then 2-4)."""
    p, half = m.p, m.steps // 2
    k = lambda model, latent, add, start, end, leftover, s: _n(
        "KSamplerAdvanced", model=model, add_noise=add, noise_seed=s, steps=m.steps, cfg=p["cfg"], sampler_name=p["sampler"],
        scheduler=p["scheduler"], positive=["5", 0], negative=["6", 0], latent_image=latent,
        start_at_step=start, end_at_step=end, return_with_leftover_noise=leftover)
    g = {"hi": _unet(m.f("hi"), p["dtype"]), "lo": _unet(m.f("lo"), p["dtype"])}
    hi, lo = _lora(g, m, ["hi", 0], "lh", "lora_hi"), _lora(g, m, ["lo", 0], "ll", "lora_lo")
    g.update({
        "mh": _n("ModelSamplingSD3", model=hi, shift=p["shift"]), "ml": _n("ModelSamplingSD3", model=lo, shift=p["shift"]),
        "2": _clip(m.f("te"), "wan"),
        "3": _n("VAELoader", vae_name=m.f("vae")),
        "5": _n("CLIPTextEncode", clip=["2", 0], text=prompt), "6": _n("CLIPTextEncode", clip=["2", 0], text=WAN_NEGATIVE),
        "7": _n("EmptyHunyuanLatentVideo", width=m.w, height=m.h, length=m.frames, batch_size=1),
        "k1": k(["mh", 0], ["7", 0], "enable", 0, half, "enable", seed),
        "8": k(["ml", 0], ["k1", 0], "disable", half, m.steps, "disable", 0),
        "9": _n("VAEDecodeTiled", samples=["8", 0], vae=["3", 0], **tiles)})
    return _tail(g, ["9", 0], m.fps, prefix)


@builder("hunyuan15", ("dit", "lora?", "te", "te2", "vae"), splittable=True, dtype="fp8_e4m3fn", lora_strength=1.0,
         shift=7, cfg=1, sampler="euler", scheduler="simple")
def _hunyuan15(m, prompt, seed, tiles, prefix):
    """HunyuanVideo 1.5 480p with the 4-step LoRA: CFG 1, so the negative is zeroed out."""
    p = m.p
    g = {"1": _unet(m.f("dit"), p["dtype"])}
    model = _lora(g, m, ["1", 0], "l")
    g.update({
        "m": _n("ModelSamplingSD3", model=model, shift=p["shift"]),
        "2": _n("DualCLIPLoader", clip_name1=m.f("te"), clip_name2=m.f("te2"), type="hunyuan_video_15", device="default"),
        "3": _n("VAELoader", vae_name=m.f("vae")),
        "5": _n("CLIPTextEncode", clip=["2", 0], text=prompt), "6": _n("ConditioningZeroOut", conditioning=["5", 0]),
        "7": _n("EmptyHunyuanVideo15Latent", width=m.w, height=m.h, length=m.frames, batch_size=1),
        "8": _n("KSampler", model=["m", 0], positive=["5", 0], negative=["6", 0], latent_image=["7", 0], seed=seed,
                steps=m.steps, cfg=p["cfg"], sampler_name=p["sampler"], scheduler=p["scheduler"], denoise=1),
        "9": _n("VAEDecodeTiled", samples=["8", 0], vae=["3", 0], **tiles)})
    return _tail(g, ["9", 0], m.fps, prefix)


@builder("kandinsky5", ("dit", "te", "te2", "vae"), splittable=True, dtype="default", shift=5, cfg=5,
         sampler="euler_ancestral", scheduler="beta")
def _kandinsky5(m, prompt, seed, tiles, prefix):
    """Kandinsky 5 (the template's sampler): Lite sft (50 steps, CFG 5), nocfg (50, CFG 1),
    distilled (16, CFG 1), and Pro, whose 43 GB of bf16 weights load as fp8."""
    p = m.p
    g = {"1": _unet(m.f("dit"), p["dtype"]),
         "2": _n("DualCLIPLoader", clip_name1=m.f("te"), clip_name2=m.f("te2"), type="kandinsky5", device="default"),
         "3": _n("VAELoader", vae_name=m.f("vae")),
         "4": _n("ModelSamplingSD3", model=["1", 0], shift=p["shift"]),
         "5": _n("CLIPTextEncode", clip=["2", 0], text=prompt), "6": _n("CLIPTextEncode", clip=["2", 0], text=""),
         "7": _n("Kandinsky5ImageToVideo", positive=["5", 0], negative=["6", 0], vae=["3", 0], width=m.w, height=m.h,
                 length=m.frames, batch_size=1),
         "8": _n("KSampler", model=["4", 0], positive=["7", 0], negative=["7", 1], latent_image=["7", 2], seed=seed,
                 steps=m.steps, cfg=p["cfg"], sampler_name=p["sampler"], scheduler=p["scheduler"], denoise=1),
         "9": _n("VAEDecodeTiled", samples=["8", 0], vae=["3", 0], **tiles)}
    return _tail(g, ["9", 0], m.fps, prefix)


def _ltx2_av(g, m, model, vae, prompt, seed, tiles, prefix):
    """LTX-2's back half, shared by both builders: video and sound from one latent, on the
    distilled model's fixed sigmas."""
    p = m.p
    g.update({
        "pos": _n("CLIPTextEncode", clip=["te", 0], text=prompt),
        "neg": _n("CLIPTextEncode", clip=["te", 0], text=LTX2_NEGATIVE),
        "cond": _n("LTXVConditioning", positive=["pos", 0], negative=["neg", 0], frame_rate=m.fps),
        "vl": _n("EmptyLTXVLatentVideo", width=m.w, height=m.h, length=m.frames, batch_size=1),
        "al": _n("LTXVEmptyLatentAudio", frames_number=m.frames, frame_rate=m.fps, batch_size=1, audio_vae=["av", 0]),
        "cat": _n("LTXVConcatAVLatent", video_latent=["vl", 0], audio_latent=["al", 0]),
        "gd": _n("CFGGuider", model=model, positive=["cond", 0], negative=["cond", 1], cfg=p["cfg"]),
        "ss": _n("KSamplerSelect", sampler_name=p["sampler"]),
        "sg": _n("ManualSigmas", sigmas=p["sigmas"]),
        "nz": _n("RandomNoise", noise_seed=seed),
        "8": _n("SamplerCustomAdvanced", noise=["nz", 0], guider=["gd", 0], sampler=["ss", 0], sigmas=["sg", 0], latent_image=["cat", 0]),
        "sep": _n("LTXVSeparateAVLatent", av_latent=["8", 0]),
        "9": _n("VAEDecodeTiled", samples=["sep", 0], vae=vae, **tiles),
        "au": _n("LTXVAudioVAEDecode", samples=["sep", 1], audio_vae=["av", 0])})
    return _tail(g, ["9", 0], m.fps, prefix, audio=["au", 0])


@builder("ltx2_ckpt", ("ckpt", "te", "lora?"), lora_strength=1.0, cfg=1, sampler="euler_ancestral", sigmas=LTX2_SIGMAS)
def _ltx2_ckpt(m, prompt, seed, tiles, prefix):
    """LTX-2 and 2.3 from one checkpoint (both VAEs inside it). LTX-2's is already distilled;
    2.3's is a dev checkpoint that its distilled LoRA makes 8-step."""
    ck = m.f("ckpt")
    g = {"ck": _n("CheckpointLoaderSimple", ckpt_name=ck),
         "te": _n("LTXAVTextEncoderLoader", text_encoder=m.f("te"), ckpt_name=ck, device="default"),
         "av": _n("LTXVAudioVAELoader", ckpt_name=ck)}
    model = _lora(g, m, ["ck", 0], "dit")
    return _ltx2_av(g, m, model, ["ck", 2], prompt, seed, tiles, prefix)


@builder("ltx2", ("dit", "te", "vae", "avae"), dtype="default", cfg=1, sampler="euler_ancestral", sigmas=LTX2_SIGMAS)
def _ltx2(m, prompt, seed, tiles, prefix):
    """LTX-2.5 on: the transformer, the text encoder and both VAEs as separate files."""
    g = {"dit": _unet(m.f("dit"), m.p["dtype"]),
         "te": _clip(m.f("te"), "ltxv"),
         "vv": _n("VAELoader", vae_name=m.f("vae")),
         "av": _n("VAELoader", vae_name=m.f("avae"))}
    return _ltx2_av(g, m, ["dit", 0], ["vv", 0], prompt, seed, tiles, prefix)


@builder("minimax_h3", ("dit", "lora?", "te", "vae", "avae"), dtype="default", lora_strength=1.0,
         sampler="res_multistep", scheduler="simple")
def _minimax_h3(m, prompt, seed, tiles, prefix):
    """MiniMax H3, video and sound. Its length rule is 17k + 5 frames (the template's 5 s at
    24 fps is 124), so --secs leaves it at its own length."""
    p = m.p
    g = {"dit": _unet(m.f("dit"), p["dtype"])}
    model = _lora(g, m, ["dit", 0], "lo")
    g.update({
        "te": _clip(m.f("te"), "minimax"),
        "vv": _n("VAELoader", vae_name=m.f("vae")), "va": _n("VAELoader", vae_name=m.f("avae")),
        "i2v": _n("MiniMaxH3ImageToVideo", clip=["te", 0], vae=["vv", 0], prompt=prompt, width=m.w, height=m.h, length=m.frames),
        "gd": _n("BasicGuider", model=model, conditioning=["i2v", 0]),
        "sc": _n("BasicScheduler", model=model, scheduler=p["scheduler"], steps=m.steps, denoise=1),
        "ss": _n("KSamplerSelect", sampler_name=p["sampler"]),
        "nz": _n("RandomNoise", noise_seed=seed),
        "8": _n("SamplerCustomAdvanced", noise=["nz", 0], guider=["gd", 0], sampler=["ss", 0], sigmas=["sc", 0], latent_image=["i2v", 1]),
        "9": _n("VAEDecode", samples=["8", 0], vae=["vv", 0]),
        "au": _n("VAEDecodeAudio", samples=["8", 0], vae=["va", 0])})
    return _tail(g, ["9", 0], m.fps, prefix, audio=["au", 0])


@builder("hunyuan10", ("dit", "te", "te2", "vae"), dtype="default", guidance=6, shift=7, sampler="euler",
         scheduler="simple")
def _hunyuan10(m, prompt, seed, tiles, prefix):
    """HunyuanVideo 1.0, per its template: guidance is embedded (FluxGuidance 6), sigmas come
    from the unshifted model and the shift-7 model does the denoising."""
    p = m.p
    g = {"1": _unet(m.f("dit"), p["dtype"]),
         "2": _n("DualCLIPLoader", clip_name1=m.f("te"), clip_name2=m.f("te2"), type="hunyuan_video", device="default"),
         "3": _n("VAELoader", vae_name=m.f("vae")),
         "4": _n("ModelSamplingSD3", model=["1", 0], shift=p["shift"]),
         "5": _n("CLIPTextEncode", clip=["2", 0], text=prompt),
         "fg": _n("FluxGuidance", conditioning=["5", 0], guidance=p["guidance"]),
         "7": _n("EmptyHunyuanLatentVideo", width=m.w, height=m.h, length=m.frames, batch_size=1),
         "gd": _n("BasicGuider", model=["4", 0], conditioning=["fg", 0]),
         "sc": _n("BasicScheduler", model=["1", 0], scheduler=p["scheduler"], steps=m.steps, denoise=1),
         "ss": _n("KSamplerSelect", sampler_name=p["sampler"]),
         "nz": _n("RandomNoise", noise_seed=seed),
         "8": _n("SamplerCustomAdvanced", noise=["nz", 0], guider=["gd", 0], sampler=["ss", 0], sigmas=["sc", 0], latent_image=["7", 0]),
         "9": _n("VAEDecodeTiled", samples=["8", 0], vae=["3", 0], **tiles)}
    return _tail(g, ["9", 0], m.fps, prefix)


@builder("mochi", ("dit", "te", "vae"), splittable=True, dtype="default", cfg=4.5, sampler="euler", scheduler="simple")
def _mochi(m, prompt, seed, tiles, prefix):
    """Mochi 1 with ComfyUI's example settings (30 steps, CFG 4.5) at its native 30 fps."""
    p = m.p
    g = {"1": _unet(m.f("dit"), p["dtype"]),
         "2": _clip(m.f("te"), "mochi"),
         "3": _n("VAELoader", vae_name=m.f("vae")),
         "5": _n("CLIPTextEncode", clip=["2", 0], text=prompt), "6": _n("CLIPTextEncode", clip=["2", 0], text=""),
         "7": _n("EmptyMochiLatentVideo", width=m.w, height=m.h, length=m.frames, batch_size=1),
         "8": _n("KSampler", model=["1", 0], positive=["5", 0], negative=["6", 0], latent_image=["7", 0], seed=seed,
                 steps=m.steps, cfg=p["cfg"], sampler_name=p["sampler"], scheduler=p["scheduler"], denoise=1),
         "9": _n("VAEDecodeTiled", samples=["8", 0], vae=["3", 0], **tiles)}
    return _tail(g, ["9", 0], m.fps, prefix)


# ---- models.json: checked before anything in it is used. One bad entry is left out with a
# reason and the rest still load; a file this videobench can't read at all is ignored.

_KEY = re.compile(r"[a-z0-9][a-z0-9._-]{0,47}")
_FILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]{0,199}\.(?:safetensors|gguf)")
_HF_URL = re.compile(r"https://huggingface\.co/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/resolve/[A-Za-z0-9_.-]+/[A-Za-z0-9_./+%-]+")
_SHA = re.compile(r"[0-9a-f]{64}")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _whole(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


class RegistryError(ValueError):
    pass


def _bad_file(k, f):
    """Why a models.json file entry can't be used, or None. The name becomes a path under
    models/ and the url is downloaded, so both are held to a narrow shape."""
    if not isinstance(k, str) or not _KEY.fullmatch(k):
        return "its key must be lowercase letters, digits, '.', '_' or '-'"
    if k in BUILTIN_FILES:
        return "that key is one of the benchmark's own files"
    if not isinstance(f, dict):
        return "not an object"
    if f.get("folder") not in MODEL_FOLDERS:
        return f"folder must be one of {', '.join(MODEL_FOLDERS)}"
    name, url = f.get("name"), f.get("url")
    if not isinstance(name, str) or not _FILE_NAME.fullmatch(name):
        return "name must be a plain .safetensors or .gguf file name"
    if not isinstance(url, str) or not _HF_URL.fullmatch(url) or "/../" in url or not url.endswith("/" + name):
        return "url must be a huggingface.co/<repo>/resolve/... link ending in the file's name"
    if not _num(f.get("gb")):
        return "gb must be a positive number"
    if "sha256" in f and not (isinstance(f["sha256"], str) and _SHA.fullmatch(f["sha256"])):
        return "sha256 must be 64 lowercase hex digits"
    if "bytes" in f and not _whole(f["bytes"]):
        return "bytes must be a positive whole number"
    return None


def _bad_model(m, files, seen):
    """Why a models.json model entry can't be used, or None."""
    key = m.get("key")
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        return "its key must be lowercase letters, digits, '.', '_' or '-'"
    if key in seen:
        return "a second model with the same key"
    if not isinstance(m.get("name"), str) or not 0 < len(m["name"]) <= 60:
        return "name must be 1 to 60 characters"
    if not isinstance(m.get("released"), str) or not re.fullmatch(r"\d{4}-\d{2}", m["released"]):
        return "released must look like 2026-08"
    b = BUILDERS[m["builder"]]
    roles = m.get("files")
    if not isinstance(roles, dict):
        return "files must map each part (dit, te, vae ...) to a file key"
    need = {r for r in b["roles"] if not r.endswith("?")}
    allowed = {r.rstrip("?") for r in b["roles"]}
    if need - set(roles):
        return f"files is missing {', '.join(sorted(need - set(roles)))} for the {m['builder']} builder"
    if set(roles) - allowed:
        return f"the {m['builder']} builder has no part called {', '.join(sorted(set(roles) - allowed))}"
    for role, fk in roles.items():
        if fk not in files and fk not in BUILTIN_FILES:
            return f"files.{role} is {fk!r}, which isn't in the files list"
    params = m.get("params", {})
    if not isinstance(params, dict):
        return "params must be an object"
    for k, v in params.items():
        if k not in b["params"]:
            return f"the {m['builder']} builder has no setting called {k!r}"
        want_text = isinstance(b["params"][k], str)
        if want_text != isinstance(v, str) or (not want_text and not (isinstance(v, (int, float)) and not isinstance(v, bool))):
            return f"params.{k} must be {'text' if want_text else 'a number'}"
    if not isinstance(m.get("size"), str) or not re.fullmatch(r"\d{2,4}x\d{2,4}", m["size"]):
        return "size must look like 832x480"
    for f in ("frames", "fps", "steps"):
        if not _whole(m.get(f)):
            return f"{f} must be a positive whole number"
    for f in ("dit_gb", "te_gb", "ref_s"):
        if not _num(m.get(f)):
            return f"{f} must be a positive number"
    lic = m.get("license")
    if not isinstance(lic, dict) or not isinstance(lic.get("name"), str) or not str(lic.get("url", "")).startswith("https://"):
        return "license needs a name and an https url"
    for f in ("restricted", "note"):
        if f in lic and not isinstance(lic[f], str):
            return f"license.{f} must be text"
    if "excluded" in lic or "restricted" in lic:  # one says where, in codes; the other says it in words
        ex = lic.get("excluded")
        if not (isinstance(ex, list) and ex and all(isinstance(x, str) and re.fullmatch(r"[A-Z]{2}", x) for x in ex)):
            return "license.excluded must list the excluded countries as two-letter codes (EU for the whole EU)"
        if not lic.get("restricted"):
            return "license.excluded needs license.restricted: the same, in words"
    if "gated" in m and not str(m["gated"]).startswith("https://huggingface.co/"):
        return "gated must be the model's huggingface.co page"
    if "split" in m and (not isinstance(m["split"], bool) or (m["split"] and not b["splittable"])):
        return f"split can only be true for {', '.join(k for k, v in BUILDERS.items() if v['splittable'])}"
    if "comfy_models" in m and not (isinstance(m["comfy_models"], list) and all(isinstance(x, str) for x in m["comfy_models"])):
        return "comfy_models must be a list of names"
    return None


def check_registry(data):
    """(files, models, problems, newer) from a parsed models.json. `newer` are the models
    whose builder this videobench doesn't have yet; RegistryError if none of it is usable."""
    if not isinstance(data, dict) or not _whole(data.get("schema")) or not isinstance(data.get("revision"), int):
        raise RegistryError("it isn't a videobench models.json (no schema or revision)")
    if data["schema"] > SCHEMA:
        raise RegistryError(f"it's written for a newer videobench (schema {data['schema']}; this one reads {SCHEMA}). "
                            f"Get the latest videobench.py from {REPO_URL}")
    problems, newer, files, models, seen = [], [], {}, [], set()
    for k, f in (data.get("files") or {}).items():
        why = _bad_file(k, f)
        if why:
            problems.append(f"file {k}: {why}")
        else:
            files[k] = f
    for m in data.get("models") or []:
        if not isinstance(m, dict):
            problems.append("a model entry that isn't an object")
            continue
        if isinstance(m.get("builder"), str) and m["builder"] not in BUILDERS:
            newer.append(str(m.get("key")))
            continue
        why = "no builder named" if not isinstance(m.get("builder"), str) else _bad_model(m, files, seen)
        if why:
            problems.append(f"model {m.get('key')}: {why}")
            continue
        seen.add(m["key"])
        models.append(m)
    return files, models, problems, newer


def _revision(path):
    try:
        return int(json.loads(Path(path).read_text(encoding="utf-8"))["revision"])
    except (OSError, ValueError, KeyError, TypeError):
        return -1


def registry_paths(root):
    """Where a models.json can be: beside this script (it ships with it), and in --dir,
    where --update-models saves the latest."""
    here = Path(__file__).resolve().with_name("models.json")
    paths = [here]
    if root is not None and (root / "models.json").resolve() != here:
        paths.append(root / "models.json")
    return paths


def load_registry(root=None):
    """Load the newest usable models.json into FILES, FILE_SHA and CONTESTANTS."""
    best, notes = None, []
    for p in registry_paths(root):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            got = check_registry(data)
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as e:  # RegistryError is a ValueError
            notes.append(f"{p} isn't usable: {e}")
            continue
        if best is None or data["revision"] > best[1]["revision"]:
            best = (p, data, got)
    for k in [k for k in FILES if k not in BUILTIN_FILES]:
        del FILES[k]
    FILE_SHA.clear()
    CONTESTANTS.clear()
    REGISTRY.update(path=None, revision=None, updated=None, problems=notes, newer=[], raw={})
    if best is None:
        return REGISTRY
    p, data, (files, models, problems, newer) = best
    for k, f in files.items():
        FILES[k] = (f["folder"], f["name"], f["url"], f["gb"])
        if f.get("sha256"):
            FILE_SHA[k] = f["sha256"]
        ROLES.setdefault(k, "is part of a lineup model")
    for m in models:
        c = dict(m, roles=dict(m["files"]), files=list(dict.fromkeys(m["files"].values())))
        c["build"] = functools.partial(build_graph, c)
        CONTESTANTS.append(c)
    REGISTRY.update(path=str(p), revision=data["revision"], updated=data.get("updated"), problems=notes + problems,
                    newer=newer, raw={m["key"]: m for m in models})
    return REGISTRY


def registry_lines():
    """Where the model list came from, and anything in it that was left out."""
    r = REGISTRY
    if r["path"]:
        out = [f"   Model list: models.json revision {r['revision']} ({r['updated'] or 'undated'}), {r['path']}"]
    else:
        out = ["   No models.json beside videobench.py or in --dir, so there are no lineup models to show.",
               "   Fetch the list:  python videobench.py --update-models"]
    if r["newer"]:
        out.append(f"   {len(r['newer'])} model(s) in it need a newer videobench.py ({', '.join(r['newer'])}): {REPO_URL}")
    out += [f"   left out: {p}" for p in r["problems"][:8]]
    return out


def update_models(root, url=None):
    """--update-models: fetch the latest models.json into --dir. Only the list changes;
    model weights download when a model first runs, as always."""
    url = url or os.environ.get("VIDEOBENCH_MODELS_URL") or MODELS_URL
    log(f"fetching the model list: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"videobench/{VERSION}", "Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read(4 << 20)  # a model list is kilobytes; 4 MB is plenty
    except (urllib.error.URLError, OSError, ValueError) as e:
        die(f"couldn't fetch it ({e}). Nothing changed.")
    try:
        data = json.loads(raw.decode("utf-8"))
        _, models, problems, newer = check_registry(data)
    except (ValueError, UnicodeDecodeError) as e:
        die(f"that isn't a model list this videobench can use: {e}. Nothing changed.")
    have = REGISTRY["revision"]
    if have is not None and data["revision"] <= have:
        log(f"already up to date: revision {have}" + (" (the one online is older)" if data["revision"] < have else ""))
        return True
    old, new = REGISTRY["raw"], {m["key"]: m for m in models}
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "models.json.part"
    tmp.write_bytes(raw)
    os.replace(tmp, root / "models.json")
    load_registry(root)
    dump = lambda m: json.dumps(m, sort_keys=True)
    added = [k for k in new if k not in old]
    changed = [k for k in new if k in old and dump(new[k]) != dump(old[k])]
    gone = [k for k in old if k not in new]
    log(f"models.json revision {have if have is not None else 'none'} -> {data['revision']} ({data.get('updated') or 'undated'})")
    for label, keys in (("new", added), ("changed", changed), ("removed", gone)):
        if keys:
            log(f"  {label + ':':<9}" + ", ".join(f"{k} ({new[k]['name']})" if k in new else k for k in keys))
    for p in problems:
        log(f"  left out: {p}")
    if newer:
        log(f"  {len(newer)} model(s) need a newer videobench.py ({', '.join(newer)}): {REPO_URL}")
    log(f"saved to {root / 'models.json'}. --list-models shows them.")
    return True


def country_code(text):
    """' ca ' -> 'CA', 'uk' -> 'GB'; anything that isn't two letters -> None."""
    cc = str(text or "").strip().upper()
    cc = {"UK": "GB"}.get(cc, cc)
    return cc if re.fullmatch(r"[A-Z]{2}", cc) and cc != "EU" else None


def licensed_here(c):
    """Whether this model's license covers where the user said they are: True / False, or
    None while they haven't said. A license that excludes no countries is always True."""
    ex = c["license"].get("excluded")
    if not ex:
        return True
    if not OPTS["country"]:
        return None
    return OPTS["country"] not in set(ex) | (EU if "EU" in ex else set())


def _settings(root):
    try:
        s = json.loads((root / "settings.json").read_text(encoding="utf-8"))
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def save_country(root, cc):
    """settings.json in --dir: only this computer reads it, and results never carry it.
    "" records a skipped question, so it isn't asked again."""
    s = _settings(root)
    s["country"] = cc
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(json.dumps(s), encoding="utf-8")
    OPTS["country"] = cc or None


def fit(c, hw, ram):
    """(status, words): fits / offload (runs, slower) / token / restricted / no."""
    here = licensed_here(c)
    if here is None:
        return "restricted", "needs --country"
    if here is False:
        return "restricted", f"not licensed in {OPTS['country']}"
    if c.get("gated") and not os.environ.get("HF_TOKEN"):
        return "token", "needs an HF token"
    if hw.get("unified") and hw.get("backend") == "cuda":  # Jetson-style low-memory mode
        if not c.get("split"):
            return "no", "too big (shared RAM)"
        # 90%: GGUF weights stream from disk; LTX-Video 13B (6.5 GB) ran fine on a 7.4 GB Orin
        return ("fits", "fits (shared RAM)") if c["dit_gb"] <= (ram or 8) * 0.9 else ("no", "too big (shared RAM)")
    if hw.get("backend") == "mps":
        return ("fits", "fits") if c["dit_gb"] + c["te_gb"] <= (ram or 16) * 0.7 else ("no", "too big")
    vram = hw.get("vram_gb") or 0
    if hw.get("backend") == "cpu" or vram < 6:
        return "no", "needs a GPU with 6 GB+"
    if c["dit_gb"] <= vram - 1.5:
        return "fits", "fits in VRAM"
    if max(c["dit_gb"], c["te_gb"]) + 4 <= (ram or 0) * 0.85:
        return "offload", "fits with offloading"
    return "no", "too big"


def steady_tflops(root, hw):
    """Something else using the GPU makes the speed test read low, never high, so the
    menu uses the best reading this machine has given (kept in machine.json, forgotten
    if the GPU changes). The first menu shown mid-lineup read 22 TFLOPS on a 52 card."""
    if not hw.get("tflops"):
        return hw
    p = root / "machine.json"
    try:
        seen = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seen = {}
    best = seen.get("tflops", 0) if seen.get("gpu") == hw.get("gpu") else 0
    if hw["tflops"] >= best:
        try:
            root.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"gpu": hw.get("gpu"), "tflops": hw["tflops"]}), encoding="utf-8")
        except OSError:
            pass
        return hw
    return {**hw, "tflops": best, "tflops_now": hw["tflops"]}


def estimate(c, hw, status):
    """Seconds for this machine: the 4070 reference scaled by measured GPU speed."""
    s = c["ref_s"] * (REF_TFLOPS / hw["tflops"] if hw.get("tflops") else 1.0)
    if not hw.get("unified") and hw.get("backend") != "mps":
        # ref_s already includes the 4070's own offloading (12 GB VRAM), so only the
        # difference counts: more VRAM than that is faster, less is slower.
        over = lambda vram: max(0.0, c["dit_gb"] - (vram - 1.5)) / c["dit_gb"]
        s *= (1 + 0.8 * over(hw.get("vram_gb") or 0)) / (1 + 0.8 * over(12.0))
    if hw.get("unified") and hw.get("backend") == "cuda":
        # low-memory mode: three prompts, reloads between them, decode partly on the CPU.
        # Measured on an 8 GB Orin Nano: LTX-Video 2B ~2.9x, 13B ~2.0x the plain scaling.
        s *= 2.5
    return s


def measured_times(root, hw):
    """Latest successful lineup time per model on this hardware (the same GPU and CPU),
    from results.jsonl."""
    got, p, cpu = {}, root / "results.jsonl", cpu_name()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("machine") or {}
            if r.get("lineup_model") and r.get("status") == "ok" and m.get("gpu") == hw.get("gpu") and m.get("cpu") == cpu:
                got[r["lineup_model"]] = r["total_seconds"]
    return got


def dur(s):
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.1f} min" if s < 5400 else f"{s / 3600:.1f} h"


def dur_range(s):
    lo, hi = s * 0.6, s * 1.6
    if hi < 90:
        return f"~{lo:.0f}-{hi:.0f} s"
    if hi < 5400:
        return f"~{max(1, round(lo / 60))}-{max(1, round(hi / 60))} min"
    return f"~{lo / 3600:.1f}-{hi / 3600:.1f} h"


def missing_gb(root, keys):
    return sum(FILES[k][3] for k in dict.fromkeys(keys) if not find_model(root / "models", FILES[k][1]))


MENU_HEAD = f"{'#':>2}  {'model':<34} {'out':<8} {'download':>9}  {'on this machine':<21} {'a 5-second clip':<22} license"


def _menu_rows(root, hw, ram):
    """Each contestant, newest first, with what the menu says about it on this machine."""
    got, rows = measured_times(root, hw), []
    for i, c in enumerate(CONTESTANTS, 1):
        status, words = fit(c, hw, ram)
        if c["key"] in got:
            secs, t = got[c["key"]], f"{dur(got[c['key']])} (measured)"
        elif status in ("no", "token", "restricted"):
            secs, t = None, "-"
        else:
            secs = estimate(c, hw, status)
            t = dur_range(secs) + " (est.)"
        rows.append({"n": i, "c": c, "status": status, "words": words, "need": missing_gb(root, c["files"]),
                     "secs": secs, "t": t})
    return rows


def _menu_line(r, lead="  "):
    dl = "have it" if r["need"] == 0 else f"{r['need']:.1f} GB"
    return f"{lead} {r['n']:>2}  {r['c']['name']:<34} {r['c']['released']:<8} {dl:>9}  {r['words']:<21} {r['t']:<22} {r['c']['license']['name']}"


def _menu_notes(hw, rows):
    tf = f"{hw['tflops']:.0f} TFLOPS measured" if hw.get("tflops") else "speed not measured"
    lines = [f"   Estimates scale from an RTX 4070 by this GPU's speed ({tf}) and include loading the models.",
             "   \"Fits with offloading\" means the model is bigger than VRAM and streams from system RAM: it works, just slower.",
             "   Models share files (text encoders, VAEs), so picking several can download less than the column adds up to."]
    for r in rows:
        if r["status"] == "token":
            lines += [f"   {r['c']['key']} is gated: accept its license at {r['c']['gated']}, make a read token at",
                      "   huggingface.co/settings/tokens, then set HF_TOKEN before running (PowerShell: $env:HF_TOKEN=\"hf_...\")."]
    why = {}
    for r in rows:
        if licensed_here(r["c"]) is not True:
            why.setdefault(r["c"]["license"]["restricted"], []).append(r["c"]["key"])
    if why and not OPTS["country"]:
        lines += [f"   {', '.join(keys)} {'is' if len(keys) == 1 else 'are'} {reason}." for reason, keys in why.items()]
        lines.append("   Say where you are and videobench offers what's licensed there: --country CA "
                     "(two letters; it stays on this computer).")
    elif why:
        lines.append(f"   Not licensed in {OPTS['country']}, so not offered: {', '.join(k for keys in why.values() for k in keys)}."
                     " (Somewhere else? --country XX)")
    return lines


def lineup_menu(root, hw, ram):
    """The contestant table as plain text (--list-models, and typing mode).
    Returns (rows, text); a row is (n, contestant, status)."""
    rows = _menu_rows(root, hw, ram)
    lines = (["  ---- PICK YOUR CONTESTANTS " + "-" * 86, "   " + MENU_HEAD] + [_menu_line(r) for r in rows]
             + [""] + _menu_notes(hw, rows))
    return [(r["n"], r["c"], r["status"]) for r in rows], "\n".join(lines)


def console_input():
    """Is a person at a keyboard? On Windows, NUL also claims to be a terminal, so ask the
    console itself. Scripts and services never get a question."""
    if not sys.stdin.isatty():
        return False
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.GetStdHandle.restype = ctypes.c_void_p
        k.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        return bool(k.GetConsoleMode(k.GetStdHandle(-10), ctypes.byref(ctypes.c_ulong())))  # -10: stdin
    return True


class _Keys:
    """Single keypresses without Enter: 'up', 'down', 'enter', 'esc', or the character."""

    def __enter__(self):
        if sys.platform == "win32":
            import msvcrt
            self.m = msvcrt
        else:
            import termios, tty
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)  # keys arrive one at a time; Ctrl+C still interrupts
        return self

    def __exit__(self, *exc):
        if sys.platform != "win32":
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read(self):
        if sys.platform == "win32":
            ch = self.m.getwch()
            if ch in ("\x00", "\xe0"):  # arrow keys come as a prefix plus a letter
                return {"H": "up", "P": "down"}.get(self.m.getwch(), "")
            if ch == "\x03":
                raise KeyboardInterrupt
            return {"\r": "enter", "\n": "enter", "\x1b": "esc"}.get(ch, ch.lower())
        ch = os.read(self.fd, 1).decode(errors="ignore")
        if ch == "\x1b":  # an escape sequence (arrows), or Esc on its own
            import select
            seq = ""
            while select.select([self.fd], [], [], 0.03)[0]:
                seq += os.read(self.fd, 1).decode(errors="ignore")
            return {"[A": "up", "[B": "down", "OA": "up", "OB": "down"}.get(seq, "" if seq else "esc")
        return {"\r": "enter", "\n": "enter"}.get(ch, ch.lower())


def pick_checkboxes(root, hw, ram, keys=None):
    """The contestant menu as a checklist: arrows move, space checks, Enter goes.
    Returns the picked contestants ([] = the standard benchmark), or None when single
    keypresses aren't available here, so the caller falls back to typing numbers."""
    rows = _menu_rows(root, hw, ram)
    ok = lambda i: rows[i]["status"] in ("fits", "offload")
    checked, cur, msg, drawn = set(), 0, "", 0
    width = shutil.get_terminal_size((140, 40)).columns - 1
    if keys is None:
        try:
            keys = _Keys().__enter__()
        except Exception:
            return None
    if sys.platform == "win32":
        os.system("")  # turns on ANSI escapes in the Windows console
    for line in _menu_notes(hw, rows):  # the notes don't change, so they print once, above
        print(line)
    print()

    def draw():
        nonlocal drawn
        picked = [rows[i] for i in sorted(checked)]
        if picked:
            gb = missing_gb(root, [k for r in picked for k in r["c"]["files"]])
            total = (f"   -->    {len(picked)} picked: {gb:.1f} GB to download, then about "
                     f"{dur(sum(r['secs'] or 0 for r in picked))} of generating")
        else:
            total = f"   -->    nothing picked yet: Enter runs the standard benchmark (Z-Image still + LTX-Video 2B)"
        lines = ["  ---- PICK YOUR CONTESTANTS " + "-" * 86, " " * 9 + " " + MENU_HEAD]
        for i, r in enumerate(rows):
            box = "[x]" if i in checked else "[ ]" if ok(i) else "[-]"
            lines.append(_menu_line(r, f"{'-->' if i == cur else '':<5} {box}"))
        lines += ["", total,
                  "   up/down move   space check   1-9,0 check that one   a all   f fits   n none   enter go   q quit",
                  f"   {msg}"]
        sys.stdout.write(("\x1b[%dA\x1b[J" % drawn if drawn else "") + "\n".join(l[:width] for l in lines) + "\n")
        sys.stdout.flush()
        drawn = len(lines)

    try:
        sys.stdout.write("\x1b[?25l")  # hide the text cursor: the --> marker does the pointing
        while True:
            draw()
            msg, k = "", keys.read()
            if k in ("up", "k"):
                cur = (cur - 1) % len(rows)
            elif k in ("down", "j"):
                cur = (cur + 1) % len(rows)
            elif k == " " or (len(k) == 1 and k.isdigit()):
                if k != " ":
                    n = (int(k) or 10) - 1  # 0 is the 10th
                    if n >= len(rows):
                        continue
                    cur = n
                if ok(cur):
                    checked ^= {cur}
                elif rows[cur]["status"] == "restricted":
                    msg = f"{rows[cur]['c']['name']}: {skip_reason(rows[cur]['c'], 'restricted')}."
                else:
                    msg = f"{rows[cur]['c']['name']} can't run here: {rows[cur]['words']}."
            elif k == "a":
                checked = {i for i in range(len(rows)) if ok(i)}
            elif k == "f":
                checked = {i for i in range(len(rows)) if rows[i]["status"] == "fits"}
            elif k == "n":
                checked = set()
            elif k == "enter":
                break
            elif k in ("q", "esc"):
                raise KeyboardInterrupt
    finally:
        keys.__exit__()
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()
    return [rows[i]["c"] for i in sorted(checked)]


def parse_picks(text, rows):
    """'1,3,5' / '2-4' / 'all' (everything that runs here) / 'fits' (only what fits in VRAM).
    Returns (picks, skipped); skipped holds (contestant, status) for picks that can't run."""
    text, runnable = text.strip().lower(), [r for r in rows if r[2] in ("fits", "offload")]
    if text in ("all", "a"):
        return [r[1] for r in runnable], []
    if text == "fits":
        return [r[1] for r in rows if r[2] == "fits"], []
    picks, skipped = [], []
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        a, _, b = part.partition("-")
        for n in range(int(a), int(b or a) + 1):  # ValueError on nonsense: the caller asks again
            if not 1 <= n <= len(rows):
                raise ValueError(n)
            _, c, status = rows[n - 1]
            if status not in ("fits", "offload"):
                skipped.append((c, status))
            elif c not in picks:
                picks.append(c)
    return picks, skipped


def skip_reason(c, status):
    """Why a picked model isn't running, in a line."""
    if status == "restricted":
        lic = c["license"]
        where = (f"and you're in {OPTS['country']}" if OPTS["country"] else
                 "so videobench needs to know where you are: --country XX (two letters)")
        return f"it's {lic['restricted']} ({lic['name']}, {lic['url']}), {where}"
    if status == "token":
        return f"it's gated: accept its license at {c['gated']} and set HF_TOKEN (see --list-models)"
    return "it can't run on this machine (see --list-models)"


def split_phases(g, key):
    """A KSampler workflow as three prompts handed off through disk (encode, sample,
    decode), for low-memory machines where the models never fit in memory together."""
    ks, dec = g["8"], g["9"]
    model, todo = set(), [ks["inputs"]["model"][0]]
    while todo:
        n = todo.pop()
        if n not in model:
            model.add(n)
            todo += [v[0] for v in g[n]["inputs"].values() if isinstance(v, list)]
    stash = lambda t, ref, part: _n(f"Stash{t}", value=ref, key=f"{key}-{part}")
    unstash = lambda t, part: _n(f"Unstash{t}", key=f"{key}-{part}")
    encode = {k: v for k, v in g.items() if k not in model | {"8", "9", "90", "91"}}
    encode.update(s1=stash("Conditioning", ks["inputs"]["positive"], "pos"),
                  s2=stash("Conditioning", ks["inputs"]["negative"], "neg"),
                  s3=stash("Latent", ks["inputs"]["latent_image"], "lat"))
    sample = {k: g[k] for k in model}
    sample.update(u1=unstash("Conditioning", "pos"), u2=unstash("Conditioning", "neg"), u3=unstash("Latent", "lat"))
    sample["8"] = {**ks, "inputs": {**ks["inputs"], "positive": ["u1", 0], "negative": ["u2", 0], "latent_image": ["u3", 0]}}
    sample["s4"] = stash("Latent", ["8", 0], "out")
    vae = dec["inputs"]["vae"][0]
    decode = {vae: g[vae], "vk": _n("VAEKeepLoaded", vae=[vae, 0]), "u4": unstash("Latent", "out"), "90": g["90"], "91": g["91"]}
    decode["9"] = {**dec, "inputs": {**dec["inputs"], "samples": ["u4", 0], "vae": ["vk", 0]}}
    return [encode, sample, decode]


def leaderboard(board, hw):
    w = 80
    rule = "+" + "-" * (w - 2) + "+"
    row = lambda t: "|  " + t[:w - 4].ljust(w - 4) + "|"
    ok = sorted((r for r in board if r["status"] == "ok"), key=lambda r: r["total_seconds"])
    top = max((r["total_seconds"] for r in ok), default=1) or 1
    import textwrap
    out = [rule, row(f"THE LINEUP   {len(board)} contestant{'s' if len(board) != 1 else ''}   {hw.get('gpu') or 'CPU'}")]
    out += [row(t) for t in textwrap.wrap(f'"{board[0]["settings"]["clip_prompt"]}"', w - 4)[:2]] + [rule]
    for i, r in enumerate(ok, 1):
        st = r["settings"]
        out.append(row(f"{i:>2}. {st['video_model'][:32]:<32} {dur(r['total_seconds']):>8}  {st['clip_size']:<8} "
                       + "#" * max(1, round(r["total_seconds"] / top * 16))))
    for r in board:
        if r["status"] != "ok":
            out.append(row(f" x  {r['settings']['video_model'][:32]:<32} dropped out: {r.get('error', '')}"))
    out.append(rule)
    if ok:
        out.append(f"   Fastest: {ok[0]['settings']['video_model']} in {dur(ok[0]['total_seconds'])}."
                   " Watch them side by side in gallery.html.")
    else:
        out.append("   Nobody finished. The errors above say why.")
    return "\n".join(out)


def run_lineup(args, root, comfy, py, hw, picks):
    """Every picked model makes the same prompt. One dropping out never stops the rest."""
    key = time.strftime("%Y%m%d-%H%M%S") + "-lineup"
    srv = Server(root, comfy, py, hw, args.port)
    busy = None if hw["unified"] else gpu_busy()
    if busy is not None and busy > 50:
        log(f"heads up: something else is using the GPU ({busy}% busy). Times won't be a fair comparison.")
    tiles = ({"tile_size": 256, "overlap": 32, "temporal_size": 32, "temporal_overlap": 4} if srv.lowmem else
             {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8})
    prompt, board = args.clip_prompt, []
    size = "x".join(map(str, args.size_wh)) if args.size_wh else None
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower())[:40].strip("-") or "clip"
    ST.set(run=key)
    try:
        for i, c in enumerate(picks, 1):
            # --secs where the model's length rule allows it: 4k+1 (LTX, Wan, Hunyuan), not H3 or Mochi.
            secs = getattr(args, "secs", None)
            frames = frames_for(secs, c["fps"], c["frames"]) if c["frames"] % 4 == 1 else c["frames"]
            if secs and c["frames"] % 4 != 1:
                log(f"      {c['name']} keeps its own {c['frames']} frames: --secs needs a 4k+1 model")
            log(f"[{i}/{len(picks)}] {c['name']}: {size or c['size']}, {frames} frames at {c['fps']} fps, {c['steps']} steps")
            rec = {"videobench": VERSION, "run": f"{key}-{c['key']}", "lineup": key, "lineup_model": c["key"],
                   "profile": "lineup", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "machine": machine(hw),
                   "settings": {"still_model": "none (text to video)", "video_model": c["name"], "still_prompt": prompt,
                                "clip_prompt": prompt, "custom_prompt": prompt != CLIP_PROMPT, "seed": args.seed,
                                "clip_size": size or c["size"], "frames": frames, "fps": c["fps"],
                                "seconds_of_video": round(frames / c["fps"], 2), "clip_steps": c["steps"],
                                "low_memory_mode": srv.lowmem, "gpu_busy_at_start_pct": busy},
                   "stages": {}, "files": {}, "memory": {}}
            t0 = time.time()
            try:
                try:
                    ensure_files(root / "models", c["files"])
                except SystemExit:
                    raise RuntimeError("a download failed (see above)")
                if not srv.alive():
                    srv.start()
                ST.set(stage="lineup", progress=f"contestant {i} of {len(picks)}: {c['name']}")
                graph = c["build"](prompt, args.seed, tiles, f"videobench/{key}-{c['key']}")
                if args.size_wh:
                    resize(graph, *args.size_wh)
                if frames != c["frames"]:
                    relength(graph, frames)
                parts = split_phases(graph, f"{key}-{c['key']}") if srv.lowmem and c.get("split") else [graph]
                outs, secs = [], 0.0
                for part in parts:
                    o, s = srv.run(part, "lineup")
                    outs, secs = outs + o, secs + s
                clip = next(o for o in outs if o.suffix == ".mp4")
                rec.update(status="ok", total_seconds=round(secs, 1), stages={"total": round(secs, 1)},
                           files={"clip": clip.relative_to(root).as_posix()})
                if args.out:
                    dest = Path(args.out).expanduser()
                    if dest.suffix.lower() == ".mp4" and len(picks) == 1:
                        saved = dest  # one model and an exact name: that name
                    else:
                        folder = dest.parent if dest.suffix.lower() == ".mp4" else dest  # many clips: use the folder
                        saved = folder / f"{slug}-{c['key']}-{key[:15]}.mp4"
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(clip, saved)
                    rec["files"]["clip_saved"] = str(saved)
                log(f"      done in {dur(secs)}" + (f": {rec['files'].get('clip_saved', clip)}"))
            except (RuntimeError, StopIteration) as e:
                rec.update(status="failed", error=str(e) or "no video came out", total_seconds=round(time.time() - t0, 1))
                log(f"      {c['name']} dropped out: {rec['error']}")
            (root / "results" / f"{rec['run']}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            with open(root / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            board.append(rec)
    finally:
        srv.stop()
    gallery(root)
    ok = [r for r in board if r["status"] == "ok"]
    ST.set(state="done" if ok else "failed", stage=None,
           summary=f"lineup: {len(ok)} of {len(board)} finished" + (f", fastest {ok and min(ok, key=lambda r: r['total_seconds'])['settings']['video_model']}" if ok else ""))
    print()
    print(leaderboard(board, hw))
    print(f"\n  gallery: {root / 'gallery.html'}")
    if ok:
        print(f"  share it: https://{HOME}/issues/new?template=share-result.yml")
    return bool(ok)


# ---------------------------------------------------------------- voices

# Text to speech, run the way the video lineup is: every engine reads the same lines, then
# an independent speech-to-text model (Whisper) listens back to each read and checks it
# against the script. Engines pin
# clashing PyTorch versions, so each gets its own venv under <dir>/voices, built with uv
# (fetched on first use if it isn't on PATH). videobench itself stays stdlib-only.

VOICE_LINE = ("Welcome back to videobench! Today three microphones and twenty voices "
              "race to read this line. Only the clearest one wins.")
VOICE_PY = "3.12"  # Kokoro doesn't support 3.13 yet (PyPI, 2026-09)

RUNNER_HEAD = r'''"""videobench voice runner, written by videobench.py. Reads the job file named on the
command line, speaks each line to a wav, and prints one JSON event per line on stdout."""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

job = json.load(open(sys.argv[1], encoding="utf-8"))
dev = job["device"]


def say(**kw):
    print(json.dumps(kw), flush=True)


def seed(s=None):
    """The run's seed, or the line's own: two takes of the same words need two seeds."""
    s = job["seed"] if s is None else s
    torch.manual_seed(s)
    np.random.seed(s)


def as_wav(path):
    """Reference voices can be mp3, and not every engine's loader reads one."""
    if not path or path.lower().endswith(".wav"):
        return path
    out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])),
                       "ref-" + os.path.splitext(os.path.basename(path))[0] + ".wav")
    if not os.path.exists(out):
        a, sr = sf.read(path, dtype="float32")
        sf.write(out, a, sr)
    return out


def done():
    peak = torch.cuda.max_memory_allocated() // 2**20 if dev == "cuda" else None
    say(event="done", vram_peak_mb=peak, torch=torch.__version__)
'''

KOKORO = RUNNER_HEAD + r'''
t = time.time()
from kokoro import KPipeline
try:
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=dev)
except TypeError:  # an older kokoro without the device argument
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
say(event="loaded", secs=round(time.time() - t, 2))
for ln in job["lines"]:
    seed(ln.get("seed"))
    t = time.time()
    parts =[a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
             for _, _, a in pipe(ln["text"], voice=ln.get("preset") or job["params"]["preset"], speed=job["params"]["speed"])
             if a is not None]
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    sf.write(ln["out"], audio, 24000)
    say(event="line", id=ln["id"], secs=round(time.time() - t, 3), audio_secs=round(len(audio) / 24000, 3))
done()
'''

CHATTERBOX = RUNNER_HEAD + r'''
t = time.time()
import copy
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device=dev)
# The voice it ships with. generate(audio_prompt_path=...) replaces model.conds
# for good, so without this a line with no sample would wear the LAST line's
# cloned voice -- which is exactly how a "built-in voice" test once came out
# identical to its cloned twin.
builtin = copy.deepcopy(model.conds)
say(event="loaded", secs=round(time.time() - t, 2))
for ln in job["lines"]:
    seed(ln.get("seed"))
    t = time.time()
    kw ={"exaggeration": job["params"]["exaggeration"], "cfg_weight": job["params"]["cfg_weight"]}
    if ln.get("ref"):
        kw["audio_prompt_path"] = as_wav(ln["ref"])
    else:
        model.conds = copy.deepcopy(builtin)
    wav = model.generate(ln["text"], **kw)
    audio = wav.squeeze(0).detach().cpu().numpy()
    sf.write(ln["out"], audio, model.sr)
    say(event="line", id=ln["id"], secs=round(time.time() - t, 3), audio_secs=round(len(audio) / model.sr, 3))
done()
'''

WHISPER_JUDGE = RUNNER_HEAD + r'''
from math import gcd
from scipy.signal import resample_poly
import whisper
t = time.time()
model = whisper.load_model(job["model"], device=dev, download_root=job["download_root"])
say(event="loaded", secs=round(time.time() - t, 2))
for item in job["items"]:
    t = time.time()
    a, sr = sf.read(item["file"], dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
    r = model.transcribe(a, language="en", temperature=0.0, fp16=(dev == "cuda"), condition_on_previous_text=False)
    say(event="heard", id=item["id"], text=r["text"].strip(), secs=round(time.time() - t, 2))
done()
'''

# weights_gb: roughly what the first run pulls from Hugging Face. Licenses read from each
# model's own LICENSE on 2026-09-12; F5-TTS (non-commercial weights) was left out on purpose.
VOICES = [
    dict(key="kokoro", name="Kokoro 82M", released="2025-01", license="Apache-2.0", clones=False,
         weights_gb=0.4, packages=["kokoro>=0.9.4", "soundfile"],
         params={"preset": "am_michael", "speed": 1.0}, run=KOKORO),
    dict(key="chatterbox", name="Chatterbox 0.5B (Resemble AI)", released="2025-05",
         license="MIT, audio watermarked", clones=True, weights_gb=3.2,
         torch=["torch==2.6.0", "torchaudio==2.6.0"], cuda="cu126",  # chatterbox-tts 0.1.7 pins torch 2.6.0
         packages=["chatterbox-tts==0.1.7", "soundfile"],
         params={"exaggeration": 0.7, "cfg_weight": 0.3}, run=CHATTERBOX),  # its README's "dramatic" setting
]
JUDGE = dict(key="whisper", name="Whisper large-v3", released="2023-11", license="MIT", clones=False,
             weights_gb=2.9, packages=["openai-whisper", "soundfile", "scipy"], params={"model": "large-v3"},
             run=WHISPER_JUDGE)

NUMBER_WORDS = set(("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                    "fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy "
                    "eighty ninety hundred thousand million").split())


def _words(s):
    s = re.sub(r"(\d),(\d)", r"\1\2", str(s or "").lower())  # "3,000" is one token
    s = re.sub(r"([a-z])\1{2,}", r"\1", s)                    # "DOOOOM" and "dom" match
    return re.sub(r"[^a-z0-9\s]", " ", s).split()


def check_read(script, heard, ignore=(), min_overlap=0.9, slack=3):
    """Did the read say the script? At least 90% of the script's distinct words must be
    heard, and the read may not run more than 3 words long. Numbers and made-up names
    (`ignore`) are left out, since speech-to-text spells them its own way."""
    ign = {w for x in ignore for w in _words(x)}
    drop = lambda w: w.isdigit() or w in NUMBER_WORDS or w in ign
    want, got = _words(script), _words(heard)
    if not want:
        return {"ok": True, "overlap": 1.0}
    if not got:
        return {"ok": False, "overlap": 0.0, "reason": "nothing was heard"}
    # A name STT splits ("mud splat" for Mudsplat) is still the name: both halves go with it, or they
    # would count as words the reader added.
    split = {i for i in range(len(got) - 1) if got[i] + got[i + 1] in ign}
    split |= {i + 1 for i in list(split)}
    want_real = [w for w in want if not drop(w)]
    got_real = [w for i, w in enumerate(got) if i not in split and not drop(w)]
    if len(got_real) > len(want_real) + slack:
        return {"ok": False, "overlap": None, "reason": f"spoke {len(got_real)} words for a {len(want_real)}-word script"}
    # Every heard word, the ignored name included: with it filtered out, a name the script
    # splits ("Sprink-ler") could never join back up.
    got_set = set(got) | {got[i] + got[i + 1] for i in range(len(got) - 1)}  # "sunday fest" = "sundayfest"
    joined = set()
    for i in range(len(want) - 1):  # and the reverse: "up stairs" in the script, "upstairs" heard
        if want[i] + want[i + 1] in got_set:
            joined |= {want[i], want[i + 1]}
    distinct = list(dict.fromkeys(want_real))
    overlap = sum(1 for w in distinct if w in got_set or w in joined) / len(distinct) if distinct else 1.0
    if overlap < min_overlap:
        return {"ok": False, "overlap": round(overlap, 3), "reason": f"only {round(overlap * 100)}% of the words were spoken"}
    return {"ok": True, "overlap": round(overlap, 3)}


def _venv_py(venv):
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def uv_bin(root):
    """uv builds each engine's venv: any Python version, fast, and cached. Use the one on
    PATH, or install it into a small venv of its own the first time."""
    if shutil.which("uv"):
        return shutil.which("uv")
    tools = root / "voices" / "_uv"
    exe = tools / ("Scripts/uv.exe" if sys.platform == "win32" else "bin/uv")
    if not exe.exists():
        log("fetching uv (it builds the voice engines' Python environments)")
        subprocess.run([sys.executable, "-m", "venv", str(tools)], check=True)
        pip(str(_venv_py(tools)), "uv")
    return str(exe)


def voice_env(args, root, uv, spec, device):
    """The engine's own venv, built once. Returns its python and its folder."""
    home = root / "voices" / spec["key"]
    venv, marker = home / "venv", home / ".installed"
    py = _venv_py(venv)
    # uv's cache often sits on another drive than --dir, where hardlinks can't reach; copy
    # quietly instead of printing a warning for every install.
    uvenv = dict(os.environ, UV_LINK_MODE="copy")
    want = json.dumps({"torch": spec.get("torch"), "packages": spec["packages"], "device": device})
    if not py.exists():
        home.mkdir(parents=True, exist_ok=True)
        log(f"      building a Python {VOICE_PY} venv for it")
        # --seed puts pip in the venv: Kokoro's text front-end pip-installs a spaCy model on first use.
        subprocess.run([uv, "venv", "--seed", "--python", VOICE_PY, str(venv)], check=True, env=uvenv)
    if not marker.exists() or marker.read_text(encoding="utf-8") != want:
        if device == "cuda":
            idx = f"https://download.pytorch.org/whl/{spec['cuda']}" if spec.get("cuda") else torch_index(args, nvidia_smi())
        else:
            idx = None if sys.platform == "darwin" else "https://download.pytorch.org/whl/cpu"
        tspec = spec.get("torch") or ["torch", "torchaudio"]
        log(f"      installing PyTorch ({idx or 'PyPI'})")
        again = ["--reinstall-package", "torch", "--reinstall-package", "torchaudio"] if marker.exists() else []
        subprocess.run([uv, "pip", "install", "--python", str(py), *again, *tspec, *(["--index-url", idx] if idx else [])],
                       check=True, env=uvenv)
        log(f"      installing {' '.join(spec['packages'])}")
        subprocess.run([uv, "pip", "install", "--python", str(py), *spec["packages"]], check=True, env=uvenv)
        marker.write_text(want, encoding="utf-8")
    (home / "run.py").write_text(spec["run"], encoding="utf-8")
    return str(py), home


def voice_env_vars(root):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env.update(HF_HOME=str(root / "voices" / "hf"), HF_HUB_DISABLE_SYMLINKS_WARNING="1",
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    return env


def voice_run(py, home, job, env, on_event):
    """Run an engine's runner on a job. on_event gets each JSON event; everything else it
    prints goes to run.log, and the tail comes back in the error if it fails."""
    jf = home / "job.json"
    jf.write_text(json.dumps(job, indent=1), encoding="utf-8")
    tail = []
    with open(home / "run.log", "w", encoding="utf-8") as logf:
        p = subprocess.Popen([py, str(home / "run.py"), str(jf)], cwd=home, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace")
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith('{"event"'):
                try:
                    on_event(json.loads(line))
                    continue
                except ValueError:
                    pass
            logf.write(line + "\n")
            if line.strip():
                tail = (tail + [line.strip()])[-6:]
        p.wait()
    if p.returncode:
        raise RuntimeError(f"it stopped (exit {p.returncode}): " + " | ".join(tail[-3:])[:400])


def voice_lines(args):
    """([{id, text, ignore, ref, compare}], custom?) from --lines, --say, or the standard line."""
    if args.lines:
        src = Path(args.lines).expanduser().resolve()
        data = json.loads(src.read_text(encoding="utf-8-sig"))
        out = []
        for k, v in (data.items() if isinstance(data, dict) else enumerate(data, 1)):
            v = {"text": v} if isinstance(v, str) else dict(v)
            for f in ("ref", "compare"):  # paths in the file are relative to the file
                if v.get(f):
                    v[f] = str((src.parent / v[f]).resolve())
            out.append({"id": re.sub(r"[^A-Za-z0-9_.-]+", "-", str(v.get("id", k))), "text": " ".join(str(v["text"]).split()),
                        "ignore": list(v.get("ignore") or []), "ref": v.get("ref"), "compare": v.get("compare"),
                        "preset": v.get("preset"),
                        "seed": int(v["seed"]) if str(v.get("seed", "")).isdigit() else None})
        if not out:
            die(f"{src} has no lines in it")
        return out, True
    text = " ".join((args.say or VOICE_LINE).split())
    return [{"id": "line" if args.say else "standard", "text": text, "ignore": [], "ref": None, "compare": None,
             "preset": None, "seed": None}], bool(args.say)


def pick_voices(want):
    want = (want or "all").strip().lower()
    if want in ("all", "fits"):
        return list(VOICES)
    by = {v["key"]: v for v in VOICES}
    keys = [k.strip() for k in want.split(",") if k.strip()]
    bad = [k for k in keys if k not in by]
    if bad:
        die(f"unknown voice engine(s): {', '.join(bad)}. Engines: {', '.join(by)}")
    return [by[k] for k in keys]


def voice_menu(root):
    out = [f"   {'key':<11} {'engine':<31} {'out':<8} {'license':<24} {'clones a voice':<15} here?"]
    for s in VOICES + [JUDGE]:
        have = "installed" if (root / "voices" / s["key"] / ".installed").exists() else "~%.1f GB + PyTorch" % s["weights_gb"]
        role = "yes" if s.get("clones") else ("the judge" if s is JUDGE else "-")
        out.append(f"   {s['key']:<11} {s['name']:<31} {s['released']:<8} {s['license']:<24} {role:<15} {have}")
    return "\n".join(out)


def voice_plan(root, picks, lines, device, out_dir, ref):
    out = ["  ---- THE VOICE LINEUP " + "-" * 53]
    for i, s in enumerate(picks, 1):
        how = ("clones the reference voice" if s["clones"] and (ref or any(ln["ref"] for ln in lines)) else
               f"preset voice {s['params']['preset']}" if s["params"].get("preset") else "its default voice")
        out.append(f"    {i}. {s['name']:<32} {s['license']:<24} {how}")
    out.append(f"    judge: {JUDGE['name']} ({JUDGE['license']}) listens to every read and checks it against the script")
    new = [s for s in picks + [JUDGE] if not (root / "voices" / s["key"] / ".installed").exists()]
    out.append("\n   installs first:   " + (f"{', '.join(s['name'] for s in new)}: each gets its own Python {VOICE_PY} "
                                          f"venv (uv), PyTorch, and about {sum(s['weights_gb'] for s in new):.1f} GB of weights"
                                          if new else "nothing, all here already"))
    first = lines[0]["text"]
    out.append(f"   lines:            {len(lines)}   first: \"{first[:90]}{'...' if len(first) > 90 else ''}\"")
    out += [f"   device:           {device}", f"   saves to:         {out_dir}", "  " + "-" * 76]
    return "\n".join(out)


def voice_leaderboard(board, compare, hw, n, judged):
    w = 80
    rule = "+" + "-" * (w - 2) + "+"
    row = lambda t: "|  " + t[:w - 4].ljust(w - 4) + "|"
    ok = sorted((r for r in board if r["status"] == "ok"),
                key=lambda r: (-(r.get("passed") or 0), -(r.get("speed_x_realtime") or 0)))
    out = [rule, row(f"THE VOICE LINEUP   {len(board)} engine{'s' if len(board) != 1 else ''}   "
                     f"{n} line{'s' if n != 1 else ''}   "
                     f"{(hw.get('gpu') or 'GPU') if hw.get('backend') == 'cuda' else 'CPU'}"), rule]
    for i, r in enumerate(ok, 1):
        passed = f"{r['passed']}/{n} pass" if r.get("passed") is not None else "unchecked"
        out.append(row(f"{i:>2}. {r['settings']['engine'][:30]:<30} {passed:<10} "
                       f"{r.get('speed_x_realtime') or 0:>6.1f}x real time  {r['settings']['license'][:20]}"))
    for r in board:
        if r["status"] != "ok":
            out.append(row(f" x  {r['settings']['engine'][:30]:<30} dropped out: {r.get('error', '')}"))
    if compare:
        cp = sum(1 for c in compare.values() if c.get("ok"))
        out.append(row(f"    {'the current voice':<30} {f'{cp}/{len(compare)} pass':<10} (for comparison)"))
    out.append(rule)
    tail = f"Every read was checked by {judged}." if judged else "The judge didn't run, so nothing was checked."
    out.append(f"   {tail} Listen side by side in voices.html.")
    return "\n".join(out)


def voices_page(root, key, lines, board, compare, judged):
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")

    def src(p):
        p = Path(p).resolve()
        try:
            return p.relative_to(root.resolve()).as_posix()
        except ValueError:
            return p.as_uri()

    def cell(item):
        if not item or not item.get("file"):
            return '<td class="none">no audio</td>'
        verdict = ""
        if "ok" in item:
            verdict = (f'<span class="ok">passes, {round((item.get("overlap") or 0) * 100)}% of words</span>' if item["ok"]
                       else f'<span class="bad">fails: {esc(item.get("reason", ""))}</span>')
        took = f'made in {item["secs"]:.1f} s · ' if item.get("secs") else ""
        heard = f' title="Whisper heard: {esc(item["heard"])}"' if item.get("heard") is not None else ""
        return (f'<td{heard}><audio controls preload="none" src="{esc(src(item["file"]))}"></audio>'
                f'<small>{took}{verdict}</small></td>')

    ok = sorted((r for r in board if r["status"] == "ok"), key=lambda r: -(r.get("passed") or 0))
    cols = ([("the current voice", "what airs now", None)] if compare else []) + [
        (r["settings"]["engine"], f'{r["settings"]["license"]} · '
         f'{r["passed"] if r.get("passed") is not None else "?"}/{len(lines)} pass · '
         f'{r.get("speed_x_realtime") or 0:.0f}x real time', r) for r in ok]
    head = "<th>line</th>" + "".join(f"<th>{esc(n)}<small>{esc(sub)}</small></th>" for n, sub, _ in cols)
    body = []
    for ln in lines:
        cells = [f'<td class="line"><b>{esc(ln["id"])}</b>{esc(ln["text"])}</td>']
        for _, _, r in cols:
            cells.append(cell(compare.get(ln["id"]) if r is None else
                              next((x for x in r["lines"] if x["id"] == ln["id"]), None)))
        body.append("<tr>" + "".join(cells) + "</tr>")
    dropped = "".join(f'<li>{esc(r["settings"]["engine"])}: {esc(r.get("error", ""))}</li>'
                      for r in board if r["status"] != "ok")
    judge = (f"Every read was checked by {esc(judged)}: hover a cell to see what it heard. A pass means at least "
             "90% of the script's words were spoken and nothing extra was added."
             if judged else "The judge didn't run this time, so nothing was checked.")
    (root / "voices.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>videobench voices</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#101214;--card:#181b1f;--ink:#e9ecef;--muted:#8d959e;--line:#262b31;--acc:#7cc4ff;--ok:#8fe388;--bad:#ff8a80}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1400px;margin:0 auto;padding:40px 20px;display:grid;gap:18px}}
h1{{margin:0;font-size:26px}} header p{{margin:4px 0 0;color:var(--muted);max-width:900px}}
.wrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%;min-width:760px}}
th,td{{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}
th{{font-size:14px;position:sticky;top:0;background:var(--card)}} th small{{display:block;color:var(--muted);font-weight:400;font-size:12px}}
td.line{{max-width:320px;font-size:13.5px}} td.line b{{display:block;color:var(--acc);font-family:ui-monospace,monospace;font-size:12.5px}}
audio{{width:220px;display:block}} td small{{display:block;color:var(--muted);font-size:12px;margin-top:4px}}
.ok{{color:var(--ok)}} .bad{{color:var(--bad)}} .none{{color:var(--muted)}}
ul{{color:var(--bad);margin:0}}
</style>
<main><header><h1>videobench voices</h1><p>Lineup {esc(key)}: {len(lines)} line{'s' if len(lines) != 1 else ''}, {len(ok)} engine{'s' if len(ok) != 1 else ''}. {judge}</p></header>
<div class="wrap"><table><tr>{head}</tr>{''.join(body)}</table></div>
{f'<ul>{dropped}</ul>' if dropped else ''}
<p style="color:var(--muted);font-size:12.5px;margin:0">videobench {VERSION} by {AUTHOR} · {HOME}</p>
</main>""", encoding="utf-8")


def run_voices(args, root):
    """Every picked engine reads the same lines; Whisper checks every read. One engine
    dropping out never stops the rest."""
    key = time.strftime("%Y%m%d-%H%M%S") + "-voices"
    nv = nvidia_smi()
    device = args.voice_device if args.voice_device != "auto" else ("cuda" if nv else "cpu")
    hw = {"gpu": nv["name"] if nv else None, "vram_gb": nv["vram_gb"] if nv else None,
          "unified": Path("/etc/nv_tegra_release").exists(), "backend": device, "torch": None, "cuda": None}
    lines, custom = voice_lines(args)
    picks = pick_voices(args.voices)
    ref = str(Path(args.voice_ref).expanduser().resolve()) if args.voice_ref else None
    out_dir = Path(args.out).expanduser().resolve() if args.out else root / "voices" / "out" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    (root / "results" / "voices").mkdir(parents=True, exist_ok=True)
    print(voice_plan(root, picks, lines, device, out_dir, ref))
    print(f"\n  follow along from any terminal:  python {Path(__file__).resolve()} --watch --dir {root}\n")
    ST.set(run=key)
    uv, env = uv_bin(root), voice_env_vars(root)
    board = []
    for i, s in enumerate(picks, 1):
        rec = {"videobench": VERSION, "kind": "voice", "run": f"{key}-{s['key']}", "lineup": key, "engine": s["key"],
               "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "machine": machine(hw),
               "settings": {"engine": s["name"], "license": s["license"], "device": device, "params": s["params"],
                            "voice_ref": None, "lines": len(lines), "custom_lines": custom, "seed": args.seed},
               "stages": {}, "memory": {}, "lines": []}
        log(f"[{i}/{len(picks)}] {s['name']}  ({s['license']})")
        t0 = time.time()
        try:
            ST.set(stage="voice_setup", progress=f"engine {i} of {len(picks)}: {s['name']}")
            py, home = voice_env(args, root, uv, s, device)
            rec["stages"]["install"] = round(time.time() - t0, 1)
            ST.set(stage="voices", progress=f"engine {i} of {len(picks)}: {s['name']}, {len(lines)} line(s)")
            job_lines = [{"id": ln["id"], "text": ln["text"], "ref": (ln["ref"] or ref) if s["clones"] else None,
                          "preset": ln.get("preset"), "seed": ln.get("seed"),
                          "out": str(out_dir / f"{ln['id']}-{s['key']}.wav")} for ln in lines]
            rec["settings"]["voice_ref"] = sorted({Path(j["ref"]).name for j in job_lines if j["ref"]}) or None
            got = {}

            def event(e, rec=rec, got=got):
                if e["event"] == "loaded":
                    rec["stages"]["load"] = e["secs"]
                    log(f"      loaded in {e['secs']:.1f} s")
                elif e["event"] == "line":
                    got[e["id"]] = e
                    log(f"      {e['id']}: {e['audio_secs']:.1f} s of speech, made in {e['secs']:.1f} s")
                elif e["event"] == "done":
                    rec["memory"]["gpu_peak_mb"] = e.get("vram_peak_mb")
                    rec["settings"]["torch"] = e.get("torch")

            voice_run(py, home, {"device": device, "seed": args.seed, "params": s["params"], "lines": job_lines}, env, event)
            for j in job_lines:
                e = got.get(j["id"])
                rec["lines"].append({"id": j["id"], "text": j["text"], "file": j["out"] if e else None,
                                     "secs": e["secs"] if e else None, "audio_secs": e["audio_secs"] if e else None})
            spoken = [x for x in rec["lines"] if x["file"]]
            if not spoken:
                raise RuntimeError("no audio came out")
            made = sum(x["secs"] for x in spoken)
            rec.update(status="ok", total_seconds=round(made, 1), audio_seconds=round(sum(x["audio_secs"] for x in spoken), 1))
            rec["stages"]["read"] = round(made, 1)
            rec["speed_x_realtime"] = round(rec["audio_seconds"] / made, 2) if made else None
        except (RuntimeError, subprocess.CalledProcessError, OSError) as e:
            rec.update(status="failed", error=str(e)[:500], total_seconds=round(time.time() - t0, 1))
            log(f"      {s['name']} dropped out: {rec['error']}")
        board.append(rec)

    # The judge: one Whisper pass over every read, plus each line's current voice if given.
    items = [{"id": f"{r['engine']}/{x['id']}", "file": x["file"]}
             for r in board if r["status"] == "ok" for x in r["lines"] if x["file"]]
    items += [{"id": f"compare/{ln['id']}", "file": ln["compare"]} for ln in lines if ln.get("compare")]
    heard, judged = {}, None
    if items:
        log(f"the judge: {JUDGE['name']} listens to {len(items)} read{'s' if len(items) != 1 else ''}")
        ST.set(stage="judge", progress=f"{len(items)} reads to check")
        try:
            py, home = voice_env(args, root, uv, JUDGE, device)
            if not (root / "voices" / "whisper" / f"{JUDGE['params']['model']}.pt").exists():
                log(f"      first run: downloading {JUDGE['name']} ({JUDGE['weights_gb']} GB)")

            def judged_one(e):
                if e["event"] == "loaded":
                    log(f"      Whisper loaded in {e['secs']:.1f} s")
                elif e["event"] == "heard":
                    heard[e["id"]] = e["text"]
                    live(f"      heard {len(heard)} of {len(items)}")

            voice_run(py, home, {"device": device, "seed": args.seed, "model": JUDGE["params"]["model"], "params": {},
                                 "download_root": str(root / "voices" / "whisper"), "items": items}, env, judged_one)
            log(f"      checked {len(heard)} read{'s' if len(heard) != 1 else ''}")
            judged = JUDGE["name"]
        except (RuntimeError, subprocess.CalledProcessError, OSError) as e:
            log(f"      the judge couldn't run ({str(e)[:300]}); the reads are saved but unchecked")
    ignore = {ln["id"]: ln["ignore"] for ln in lines}
    for r in board:
        for x in r["lines"]:
            h = heard.get(f"{r['engine']}/{x['id']}")
            if h is not None:
                x["heard"] = h
                x.update(check_read(x["text"], h, ignore[x["id"]]))
        checked = [x for x in r["lines"] if "ok" in x]
        r["judge"] = judged
        r["passed"] = sum(1 for x in checked if x["ok"]) if checked else None
        (root / "results" / "voices" / f"{r['run']}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        with open(root / "voices.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
    compare = {}
    for ln in lines:
        if ln.get("compare"):
            h = heard.get(f"compare/{ln['id']}")
            compare[ln["id"]] = {"file": ln["compare"], "heard": h,
                                 **(check_read(ln["text"], h, ln["ignore"]) if h is not None else {})}
    if args.out:   # the verdicts beside the audio, where a script that named the folder can find them
        (out_dir / "voices.json").write_text(json.dumps(board, indent=2), encoding="utf-8")
    voices_page(root, key, lines, board, compare, judged)
    ok = [r for r in board if r["status"] == "ok"]
    ST.set(state="done" if ok else "failed", stage=None, summary=f"voices: {len(ok)} of {len(board)} engines read")
    print()
    print(voice_leaderboard(board, compare, hw, len(lines), judged))
    print(f"\n  listen: {root / 'voices.html'}\n  files:  {out_dir}")
    return bool(ok)


# ---------------------------------------------------------------- gallery

def gallery(root):
    runs = []
    if (root / "results.jsonl").exists():
        for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    cards = []
    for r in reversed(runs):
        m, st, s = r["machine"], r["settings"], r["stages"]
        media = (f'<video src="{esc(r["files"]["clip"])}" autoplay muted loop playsinline controls></video>'
                 if r["files"].get("clip") else f'<div class="fail">{esc(r.get("error", "no clip"))}</div>')
        rows = "".join(f"<tr><td>{k.replace('_', ' ')}</td><td>{v:.1f} s</td></tr>" for k, v in s.items())
        mem = r.get("memory", {})
        memline = " · ".join(x for x in [
            f"GPU peak {mem['gpu_used_max_mb'] / 1024:.1f} GB" if mem.get("gpu_used_max_mb") else "",
            f"free RAM floor {mem['ram_available_min_mb'] / 1024:.1f} GB" if mem.get("ram_available_min_mb") else ""] if x)
        cards.append(f"""<article>
  <div class="media">{media}</div>
  <div class="meta">
    <h2>{esc(m.get('gpu') or 'CPU')} <span>{esc(r['profile'])}{' · custom prompt' if st.get('custom_prompt') else ''}</span></h2>
    <p class="small">"{esc(st['clip_prompt'])}"</p>
    <p class="big">{r.get('total_seconds', 0):.0f} s <small>for {st['seconds_of_video']} s of {st['clip_size']} video</small></p>
    <table>{rows}</table>
    <p class="small">{esc(st['video_model'])} · still: {esc(st['still_model'])}<br>{esc(m.get('cpu'))} · {m.get('ram_gb')} GB RAM · {esc(m.get('os'))}{' · ' + memline if memline else ''}<br>{esc(r['at'][:16].replace('T', ' '))} · {esc(r['status'])}</p>
  </div>
</article>""")
    prompt = esc(runs[-1]["settings"]["clip_prompt"]) if runs else esc(CLIP_PROMPT)
    (root / "gallery.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>videobench</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#101214;--card:#181b1f;--ink:#e9ecef;--muted:#8d959e;--line:#262b31;--acc:#7cc4ff}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:40px 20px;display:grid;gap:22px}}
header h1{{margin:0;font-size:26px}} header p{{margin:4px 0 0;color:var(--muted)}}
article{{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
.media{{background:#000;display:grid;place-items:center}} video{{width:100%;display:block}}
.fail{{color:#ff8a80;padding:24px;font-family:ui-monospace,monospace;font-size:13px;white-space:pre-wrap}}
.meta{{padding:18px 20px;display:grid;gap:10px;align-content:start}}
h2{{margin:0;font-size:17px}} h2 span{{color:var(--acc);font-weight:500;font-size:13px;text-transform:uppercase;letter-spacing:.06em;margin-left:6px}}
.big{{margin:0;font-size:30px;font-weight:700;font-variant-numeric:tabular-nums}} .big small{{font-size:13px;font-weight:400;color:var(--muted)}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}} td{{padding:3px 18px 3px 0;border-bottom:1px solid var(--line)}} td+td{{text-align:right}}
.small{{margin:0;color:var(--muted);font-size:12.5px}}
@media (max-width:760px){{article{{grid-template-columns:1fr}}}}
</style>
<main><header><h1>videobench</h1><p>{len(runs)} run{'s' if len(runs) != 1 else ''} on this machine. The standard prompt is "{esc(CLIP_PROMPT)}"; runs tagged custom prompt made their own video, so their times don't compare.</p></header>
{''.join(cards) or '<p>No runs yet.</p>'}
<p class="small">videobench {VERSION} by {AUTHOR} · {HOME}</p>
</main>""", encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=PROFILES, default="fast")
    p.add_argument("--still", choices=STILLS, default="zimage", help="model that paints the still (default zimage)")
    p.add_argument("--dir", default="videobench", help="where everything goes (default ./videobench)")
    p.add_argument("--python", help="use this Python (must already have PyTorch) instead of building a venv")
    p.add_argument("--comfy", help="use an existing ComfyUI checkout instead of cloning one")
    p.add_argument("--torch-index", help="pip index URL for PyTorch, overriding auto-detection")
    p.add_argument("--port", type=int, default=8190)
    p.add_argument("--seed", type=int, default=SEED,
                   help=f"the noise seed (default {SEED}: benchmark times only compare at {SEED})")
    p.add_argument("--size", help="the frame size, like 640x480 for 4:3 (default: each model's own). Rounded to "
                                  "multiples of 32; with a still, the still is painted to match")
    p.add_argument("--secs", type=float,
                   help=f"how long the clip runs, in seconds, up to {MAX_SECS} (default: each model's own, about 5). "
                        "Rounded up to a frame count the model takes")
    p.add_argument("--still-only", action="store_true",
                   help="paint just the still (the --still model) and save it as a PNG to --out: a folder, or a .png path")
    p.add_argument("--list", action="store_true", help="show the plan and what would download, then stop")
    p.add_argument("--status", action="store_true", help="show what a run in --dir is doing right now")
    p.add_argument("--watch", action="store_true", help="like --status, refreshing every 2 s (Ctrl+C stops watching)")
    p.add_argument("--gallery", action="store_true", help="only rebuild gallery.html")
    p.add_argument("--verify", action="store_true", help="check downloaded models against Hugging Face's size and SHA-256")
    p.add_argument("--prompt", help="make a video of this instead of the standard dog prompt (its times won't "
                                    "compare across machines). Without it, a terminal run asks.")
    p.add_argument("--out", help="where to save the video: a folder (named from the prompt) or an exact .mp4 path; "
                                 "the still is saved beside it. Without it, a terminal run asks.")
    p.add_argument("--models", help="run the lineup instead of the benchmark: model keys separated by commas "
                                    "(see --list-models), 'all' for every model that runs here, or 'fits' for "
                                    "the ones that fit in VRAM")
    p.add_argument("--voices", nargs="?", const="all", metavar="KEYS",
                   help="run the voice lineup instead: text-to-speech engines separated by commas (see "
                        "--list-voices), or all. Whisper checks every read against the script")
    p.add_argument("--say", help="the line the voices read (default: a standard test line)")
    p.add_argument("--lines", help="a JSON file of lines for the voices: a list of strings, or of "
                                   "{id, text, ignore, ref, compare} (ref: a voice to clone; compare: a reading to judge too)")
    p.add_argument("--voice-ref", help="a recording of a voice, for engines that can clone one (wav or mp3)")
    p.add_argument("--voice-device", choices=("auto", "cuda", "cpu"), default="auto",
                   help="where the voice engines run (default: the NVIDIA GPU if there is one)")
    p.add_argument("--list-voices", action="store_true", help="show the voice engines, their licenses, and what's installed")
    p.add_argument("--list-models", action="store_true",
                   help="show every video model, whether it fits this machine and how long a 5-second clip should take")
    p.add_argument("--update-models", action="store_true",
                   help="fetch the latest list of video models (models.json) from GitHub into --dir. "
                        "Downloads no model weights")
    p.add_argument("--country", metavar="XX",
                   help="where you are, as two letters like US, GB, DE or JP. A few models' licenses exclude some "
                        "countries; this decides whether they're offered. Saved in --dir (a terminal run asks once)")
    p.add_argument("--no-ask", action="store_true",
                   help="never stop to ask anything: use --prompt, --models and --out, or the defaults")
    args = p.parse_args()
    args.size_wh = None
    if args.size:
        m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", args.size)
        if not m:
            p.error(f"--size wants WIDTHxHEIGHT, like 640x480 (not {args.size!r})")
        args.size_wh = tuple(max(64, round(int(v) / 32) * 32) for v in m.groups())
    if args.secs is not None and not 1 <= args.secs <= MAX_SECS:
        p.error(f"--secs wants 1 to {MAX_SECS} seconds (not {args.secs:g})")
    root = Path(args.dir).expanduser().resolve()
    q = lambda p: f'"{p}"' if " " in str(p) else str(p)
    me = q(Path(__file__).resolve())  # full path: the hints must work from any folder
    OPTS["country"] = country_code(_settings(root).get("country"))
    if args.country is not None:
        if not country_code(args.country):
            p.error(f"--country wants two letters, like US, GB, DE or JP (not {args.country!r})")
        save_country(root, country_code(args.country))
    load_registry(root)
    if args.update_models:
        sys.exit(0 if update_models(root) else 1)

    def quick_hw():
        """probe() when PyTorch is already installed; otherwise what nvidia-smi can tell."""
        py = args.python or str(root / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
        if Path(py).exists() and subprocess.run([py, "-c", "import torch"], capture_output=True).returncode == 0:
            return steady_tflops(root, probe(py))
        nv = nvidia_smi()
        return {"backend": "cuda" if nv else "cpu", "gpu": nv and nv["name"], "vram_gb": nv and nv["vram_gb"],
                "unified": Path("/etc/nv_tegra_release").exists(), "tflops": None}

    if args.list_models:
        print(BANNER)
        print("\n".join(registry_lines()) + "\n")
        if not CONTESTANTS:
            return
        print(lineup_menu(root, quick_hw(), ram_gb())[1])
        same = (f" --python {q(args.python)}" if args.python else "") + (f" --comfy {q(args.comfy)}" if args.comfy else "")
        return print(f"\n   That's the list. To pick from it with the arrow keys:  python {me} --dir {q(root)}{same}\n"
                     f"   Or name them:  python {me} --dir {q(root)}{same} --models ltxv-2b,wan-2.2-5b\n"
                     f"   Keys: {', '.join(c['key'] for c in CONTESTANTS)}   (or 'all', or 'fits')")

    if args.list_voices:
        print(BANNER)
        print(voice_menu(root))
        return print(f"\n   Run some:  python {me} --dir {q(root)} --voices kokoro,chatterbox   (or just --voices for all)\n"
                     f"   Your own words: --say \"...\"  or  --lines lines.json   A voice to clone: --voice-ref voice.wav")

    if args.status or args.watch:
        return show_status(root, args.watch)
    if args.gallery:
        gallery(root)
        return print(root / "gallery.html")
    if args.verify:
        print(BANNER)
        found = [(k, find_model(root / "models", FILES[k][1])) for k in FILES]
        found = [(k, p) for k, p in found if p]
        if not found:
            return log(f"no models downloaded in {root / 'models'} yet")
        bad = []
        for k, p in found:
            size, sha = hf_meta(FILES[k][2])
            sha = FILE_SHA.get(k) or sha
            ok = verified(p, size, sha)
            log(f"{'ok ' if ok else 'BAD'}  {p.name}" + ("" if sha else "   (no checksum published; size checked)"))
            if not ok:
                bad.append(p.name)
        if bad:
            log(f"{len(bad)} bad file(s). Delete them and re-run; videobench fetches them again: " + ", ".join(bad))
            sys.exit(1)
        return log(f"all {len(found)} model files match Hugging Face.")
    if args.list:
        print(BANNER)
        print(f"  videobench {VERSION}   profile {args.profile}   ->  {root}\n")
        print(plan_text(args, root, args.prompt, args.out))
        print("  (plus ComfyUI and PyTorch on the first run, if they aren't installed yet)")
        return print(f"\n  watch a run from another terminal:  python {me} --watch --dir {q(root)}")

    if sys.version_info < (3, 10):
        die("needs Python 3.10 or newer")
    root.mkdir(parents=True, exist_ok=True)
    try:  # one run per folder: two would fight over ComfyUI, the GPU and the files
        prev = json.loads((root / "status.json").read_text(encoding="utf-8"))
        if prev.get("state") == "running" and pid_alive(prev.get("pid", -1)):
            die(f"a run is already going in {root} (pid {prev['pid']}). Watch it: python {me} --watch --dir {q(root)}")
    except (OSError, ValueError):
        pass
    if args.voices:  # the voice lineup needs no ComfyUI: straight to it
        print(BANNER)
        ST.begin(root)
        sys.exit(0 if run_voices(args, root) else 1)
    print(BANNER)
    prompt = args.prompt
    ask = console_input() and not args.no_ask  # scripted and background runs never block on a question

    def bye():
        print("\n   Stopped. Nothing ran.")
        sys.exit(130)

    def line():
        """input(), except that no input at all (a pipe, a service) means: take
        the default answer and stop asking. Only Ctrl+C quits."""
        nonlocal ask
        try:
            return input("\n   > ")
        except EOFError:
            ask = False
            print("(nobody to answer: using the defaults)")
            return ""

    try:
        if prompt is None and ask:
            print(f"   -->    What should I make a 5-second video of?")
            print("          Press Enter for the standard benchmark (a dog chasing a thrown ball); its")
            print("          times compare across machines. Your own prompt runs the same way, just not comparably.")
            prompt = line().strip()
    except KeyboardInterrupt:
        bye()
    if prompt:
        args.still_prompt = args.clip_prompt = prompt
    else:
        args.still_prompt, args.clip_prompt = STILL_PROMPT, CLIP_PROMPT
    print()
    ST.begin(root)
    comfy, py = setup(args, root)
    hw = steady_tflops(root, probe(py))
    if args.still_only:
        sys.exit(0 if paint_only(args, root, comfy, py, hw) else 1)

    picks = []
    try:
        if args.models:
            if not CONTESTANTS:
                die("there's no model list (models.json) here yet. Fetch it: python videobench.py --update-models")
            rows, _ = lineup_menu(root, hw, ram_gb())
            want = args.models.strip().lower()
            if want not in ("all", "fits"):
                by_key = {c["key"]: n for n, c, _ in rows}
                unknown = [k.strip() for k in want.split(",") if k.strip() not in by_key]
                if unknown:
                    die(f"unknown model(s): {', '.join(unknown)}. Keys: {', '.join(by_key)}. Newer models: --update-models")
                want = ",".join(str(by_key[k.strip()]) for k in want.split(","))
            picks, skipped = parse_picks(want, rows)
            for c, status in skipped:
                log(f"skipping {c['name']}: {skip_reason(c, status)}")
            if not picks:
                die("none of the picked models can run on this machine")
        elif ask and CONTESTANTS:
            if "country" not in _settings(root) and any(c["license"].get("excluded") for c in CONTESTANTS):
                print(f"\n   -->    Where are you? Two letters, like US, GB, DE or JP.")
                print("          A few models' licenses exclude some countries; this decides whether they're offered.")
                print("          It stays on this computer. Press Enter to skip: those models stay off (--country later).")
                answer = line().strip()
                if answer and not country_code(answer):
                    print("   Didn't catch that, so skipping. Tell me later with --country XX.")
                save_country(root, country_code(answer) or "")
            print()
            picks = pick_checkboxes(root, hw, ram_gb())
            if picks is None:  # no single keypresses in this terminal: type numbers instead
                rows, menu = lineup_menu(root, hw, ram_gb())
                print(menu)
                print(f"\n   -->    Pick your contestants: numbers like 1,3,5 or 2-4, 'all', or 'fits'.")
                print("          Press Enter for the standard benchmark instead (Z-Image still + LTX-Video 2B).")
                while True:
                    try:
                        picks, skipped = parse_picks(line(), rows)
                        break
                    except ValueError:
                        print("   Didn't catch that. Use numbers from the list, like 1,3 or 2-4.")
                for c, status in skipped:
                    print(f"   (skipping {c['name']}: {skip_reason(c, status)})")
        if ask and args.out is None:
            if picks:
                print(f"\n   -->    Where should I save the videos? A folder; each file is named after the prompt and the model.")
            else:
                print(f"\n   -->    Where should I save it? A folder, or a full path ending in .mp4.")
            print(f"          Press Enter to keep {'them' if picks else 'it'} in {root / 'output' / 'videobench'}")
            args.out = line().strip().strip('"') or None
    except KeyboardInterrupt:
        bye()
    print()

    if picks:
        got = measured_times(root, hw)
        for c in picks:
            if c["license"].get("excluded"):
                log(f"{c['name']}: its license ({c['license']['name']}, {c['license']['url']}) is "
                    f"{c['license']['restricted']}; you said you're in {OPTS['country']}")
        need = missing_gb(root, [k for c in picks for k in c["files"]])
        total = sum(got.get(c["key"]) or estimate(c, hw, fit(c, hw, ram_gb())[0]) for c in picks)
        print("  ---- THE LINEUP " + "-" * 60)
        for n, c in enumerate(picks, 1):
            shown = "x".join(map(str, args.size_wh)) if args.size_wh else c["size"]
            print(f"   {n:>2}. {c['name']:<34} {shown:<8} {c['frames']} frames at {c['fps']} fps, {c['steps']} steps")
        print(f"\n   downloads first:  {need:.1f} GB" + ("  (all here already)" if not need else ""))
        print(f"   then generating:  about {dur(total)} for all {len(picks)} (estimated; measured times where known)")
        print(f'   prompt:           "{args.clip_prompt}"' + ("" if args.clip_prompt == CLIP_PROMPT else "   (custom)"))
        print(f"   saves to:         {args.out or root / 'output' / 'videobench'}")
        print("  " + "-" * 76)
        print(f"\n  follow along from any terminal:  python {me} --watch --dir {q(root)}\n")
        sys.exit(0 if run_lineup(args, root, comfy, py, hw, picks) else 1)

    print(plan_text(args, root, args.clip_prompt, args.out))
    print(f"\n  follow along from any terminal:  python {me} --watch --dir {q(root)}\n")
    sys.exit(0 if bench(args, root, comfy, py) else 1)


if __name__ == "__main__":
    main()
