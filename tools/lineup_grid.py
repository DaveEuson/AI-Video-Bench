#!/usr/bin/env python3
"""One lineup's clips side by side, fastest first, as an animated GIF (or WebP): the
picture at the top of the README.

    <videobench's venv python> tools/lineup_grid.py --dir D:\\videobench --out docs/lineup.gif

Needs PyAV and Pillow. videobench's own venv already has both (ComfyUI uses them):
<dir>/venv/Scripts/python.exe on Windows, <dir>/venv/bin/python elsewhere.
"""
import argparse
import json
import sys
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import videobench as vb  # noqa: E402


def font(size):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def lineup(results, prompt_has, skip):
    """(prompt, [record]) for the lineup to show: the newest good run of each model on one prompt."""
    by_prompt = {}
    for line in results.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("lineup_model") and r.get("status") == "ok" and r["lineup_model"] not in skip:
            by_prompt.setdefault(r["settings"]["clip_prompt"], {})[r["lineup_model"]] = r
    if prompt_has:
        by_prompt = {p: m for p, m in by_prompt.items() if prompt_has.lower() in p.lower()}
    if not by_prompt:
        sys.exit("no lineup matches")
    prompt, runs = max(by_prompt.items(), key=lambda kv: len(kv[1]))
    return prompt, sorted(runs.values(), key=lambda r: r["total_seconds"])


def tiles(clip, times, w, h):
    """The clip's frame at each time, cropped to fill w x h."""
    with av.open(str(clip)) as c:
        s = c.streams.video[0]
        rate = float(s.average_rate or 24)
        want = {min(int(t * rate), 10**6): i for i, t in enumerate(times)}
        got, last = {}, None
        for k, frame in enumerate(c.decode(s)):
            if k in want or k == 0:
                img = frame.to_image()
                scale = max(w / img.width, h / img.height)
                img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
                x, y = (img.width - w) // 2, (img.height - h) // 2
                last = img.crop((x, y, x + w, y + h))
                if k in want:
                    got[k] = last
            if k >= max(want):
                break
    out, prev = [], None
    for k in sorted(want):  # past the clip's end: hold its last frame
        prev = got.get(k, prev or last)
        out.append(prev)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dir", required=True, help="the videobench --dir with the lineup's results")
    p.add_argument("--prompt", help="pick the lineup whose prompt contains this (default: the one with the most models)")
    p.add_argument("--skip", default="", help="model keys to leave out, comma-separated")
    p.add_argument("--max", type=int, default=12)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--tile", default="240x144", help="each clip's size (default 240x144)")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--secs", type=float, default=4)
    p.add_argument("--out", required=True, help=".gif or .webp")
    a = p.parse_args()
    root = Path(a.dir)
    vb.load_registry()
    names = {c["key"]: c["name"] for c in vb.CONTESTANTS}
    prompt, runs = lineup(root / "results.jsonl", a.prompt, {k.strip() for k in a.skip.split(",") if k.strip()})
    runs = runs[:a.max]
    w, h = (int(v) for v in a.tile.split("x"))
    band, cols = 22, a.cols
    rows = -(-len(runs) // cols)
    times = [i / a.fps for i in range(round(a.secs * a.fps))]
    f = font(12)
    frames = [Image.new("RGB", (cols * w, rows * (h + band)), (16, 18, 20)) for _ in times]
    for n, r in enumerate(runs):
        x, y = (n % cols) * w, (n // cols) * (h + band)
        label = f"{n + 1}. {names.get(r['lineup_model'], r['lineup_model'])}"
        took = vb.dur(r["total_seconds"])
        print(f"{label:<44} {took:>8}  {r['files']['clip']}")
        while f.getlength(label) > w - 20 - f.getlength(took) and len(label) > 4:  # never under the time
            label = label[:-2] + "…"
        for frame, tile in zip(frames, tiles(root / r["files"]["clip"], times, w, h)):
            frame.paste(tile, (x, y))
            d = ImageDraw.Draw(frame)
            d.text((x + 6, y + h + 4), label, font=f, fill=(233, 236, 239))
            d.text((x + w - 6, y + h + 4), took, font=f, fill=(124, 196, 255), anchor="ra")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ms = round(1000 / a.fps)
    if out.suffix.lower() == ".webp":
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0, quality=70, method=6)
    else:
        pal = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        q = [fr.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for fr in frames]
        q[0].save(out, save_all=True, append_images=q[1:], duration=ms, loop=0, optimize=True)
    print(f"\n{len(runs)} clips, {len(frames)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"prompt: {prompt}")


if __name__ == "__main__":
    main()
