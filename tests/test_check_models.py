"""The daily model check (tools/check_models.py), offline: a made-up template index and a
fake Hugging Face."""
import importlib.util
import json
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_models", ROOT / "tools" / "check_models.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
REG = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))


def tpl(name, models, tags=("Video", "Text to Video"), open_source=True, **kw):
    t = {"name": name, "models": list(models), "tags": list(tags), "date": "2026-09-01", **kw}
    if open_source is not None:
        t["openSource"] = open_source
    return t


INDEX = [
    {"category": "Video", "templates": [
        tpl("video_minimax_h3_t2v", ["MiniMax H3"]),
        tpl("video_ltx2_3_t2v", ["LTX-2.3", "Lightricks"]),
        tpl("video_wan2_2_14B_t2v", ["Wan2.2", "Wan"], open_source=None),  # older templates have no openSource
        tpl("video_wan3_t2v", ["Wan3.0", "Wan"], minComfyUIVersion="0.5.0", tutorialUrl="https://docs.example/wan3"),
        tpl("api_wan3_t2v", ["Wan3.0 Pro"], tags=("API", "Text to Video"), open_source=False),  # API only
        tpl("video_wan3_i2v", ["Wan3.0 Animate"], tags=("Video", "Image to Video")),  # not text-to-video
    ]},
    {"category": "Image", "templates": [tpl("image_zeta", ["Zeta Image"], tags=("Image", "Text to Image"))]},
]


class NewModels(unittest.TestCase):
    def test_only_new_open_text_to_video(self):
        found = cm.new_models(INDEX, REG)
        self.assertEqual(list(found), ["Wan3.0"])
        self.assertEqual([t["name"] for t in found["Wan3.0"]], ["video_wan3_t2v"])

    def test_watch_ignore_any_case(self):
        reg = dict(REG, watch_ignore=REG["watch_ignore"] + ["wan3.0"])
        self.assertEqual(cm.new_models(INDEX, reg), {})

    def test_issue_says_what_to_do(self):
        body = cm.new_model_issue("Wan3.0", cm.new_models(INDEX, REG)["Wan3.0"], REG)
        for s in ("video_wan3_t2v", "0.5.0", "https://docs.example/wan3", "watch_ignore", "--update-models", "e5a38e3"):
            self.assertIn(s, body)


class ChangedFiles(unittest.TestCase):
    FILES = {
        "a": {"url": "https://huggingface.co/o/r/resolve/main/sub/a.safetensors", "sha256": "a" * 64},
        "b": {"url": "https://huggingface.co/o/r/resolve/main/sub/b.safetensors", "sha256": "b" * 64},
        "c": {"url": "https://huggingface.co/o/r/resolve/main/c.gguf", "sha256": "c" * 64},
        "d": {"url": "https://huggingface.co/o/gated/resolve/main/d.safetensors"},  # nothing pinned: not checked
        "e": {"url": "https://huggingface.co/o/down/resolve/main/x/e.safetensors", "sha256": "e" * 64},
    }

    def fake(self, listing):
        calls = []

        def fetch(url):
            calls.append(url)
            for tail, entries in listing.items():
                if url.endswith(tail):
                    if isinstance(entries, Exception):
                        raise entries
                    return entries
            return []
        return fetch, calls

    def test_changed_gone_and_unreachable(self):
        fetch, calls = self.fake({
            "/o/r/tree/main/sub": [{"path": "sub/a.safetensors", "lfs": {"oid": "a" * 64}},
                                   {"path": "sub/b.safetensors", "lfs": {"oid": "f" * 64}}],
            "/o/r/tree/main/": [{"path": "other.gguf"}],
            "/o/down/tree/main/x": urllib.error.URLError("down")})
        changed, warnings = cm.changed_files({"files": self.FILES}, fetch)
        self.assertEqual([k for k, _ in changed], ["b", "c"])
        self.assertEqual(len(warnings), 1)  # a folder it couldn't list is a warning, not an issue
        self.assertEqual(sum(u.endswith("/o/r/tree/main/sub") for u in calls), 1)  # one listing per folder
        self.assertFalse(any("/gated/" in u for u in calls))

    def test_a_hidden_hash_is_not_a_change(self):
        fetch, _ = self.fake({"/o/r/tree/main/sub": [{"path": "sub/a.safetensors", "lfs": {"oid": "*" * 64}}]})
        self.assertEqual(cm.changed_files({"files": {"a": self.FILES["a"]}}, fetch), ([], []))


if __name__ == "__main__":
    unittest.main()
