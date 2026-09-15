"""videobench's tests: everything that runs without a GPU. Stdlib only.

    python -m unittest discover -s tests -v
"""
import contextlib
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import videobench as vb  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "golden_graphs.json").read_text(encoding="utf-8"))
BUNDLED = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))
TILES = {"normal": {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8},
         "lowmem": {"tile_size": 256, "overlap": 32, "temporal_size": 32, "temporal_overlap": 4}}
GPU = {"backend": "cuda", "gpu": "Test GPU", "vram_gb": 24.0, "unified": False, "tflops": 52.0}


def roundtrip(x):
    return json.loads(json.dumps(x))


def quiet():
    """videobench logs to stdout; tests don't need to see it."""
    return contextlib.redirect_stdout(io.StringIO())


def model(**over):
    """One valid models.json model entry, built on the benchmark's own LTX-Video files."""
    m = {"key": "test-model", "name": "Test Model", "released": "2026-01", "builder": "ltxv",
         "files": {"dit": "ltxv2b", "te": "t5", "vae": "ltxvae"}, "size": "768x512", "frames": 121, "fps": 24,
         "steps": 8, "dit_gb": 2.3, "te_gb": 3.4, "ref_s": 17, "license": {"name": "Test", "url": "https://example.com/l"}}
    m.update(over)
    return m


def hf_file(**over):
    f = {"folder": "diffusion_models", "name": "test-model.safetensors", "gb": 1.0,
         "url": "https://huggingface.co/org/repo/resolve/main/sub/test-model.safetensors"}
    f.update(over)
    return f


def registry(models=None, files=None, **over):
    data = {"schema": 1, "revision": 1, "files": files if files is not None else {},
            "models": models if models is not None else [model()]}
    data.update(over)
    return data


class BundledRegistry(unittest.TestCase):
    """The models.json that ships with videobench."""

    def setUp(self):
        vb.load_registry()

    def test_loads_with_nothing_left_out(self):
        self.assertEqual(vb.REGISTRY["problems"], [])
        self.assertEqual(vb.REGISTRY["newer"], [])
        self.assertEqual([c["key"] for c in vb.CONTESTANTS], [m["key"] for m in BUNDLED["models"]])

    def test_every_file_is_used_and_every_used_file_is_listed(self):
        used = {k for c in vb.CONTESTANTS for k in c["files"]}
        self.assertEqual(set(BUNDLED["files"]) - used, set(), "files no model uses")
        for k in used:
            self.assertIn(k, vb.FILES)

    def test_restricted_means_a_territory_limit(self):
        ex = {c["key"]: c["license"]["excluded"] for c in vb.CONTESTANTS if c["license"].get("excluded")}
        self.assertEqual(ex, {"minimax-h3": ["EU", "GB", "KR", "US"], "hunyuan-1.5": ["EU", "GB", "KR"],
                              "hunyuan-1.0": ["EU", "GB", "KR"]})

    def test_pinned_hashes_for_every_file_but_the_gated_ones(self):
        gated = {k for c in vb.CONTESTANTS if c.get("gated") for k in c["files"]}
        for k, f in BUNDLED["files"].items():
            self.assertEqual("sha256" in f, k not in gated, k)

    def test_readme_names_every_model(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for c in vb.CONTESTANTS:
            self.assertIn(f"`{c['key']}`", readme)


class SameWorkflowsAsBefore(unittest.TestCase):
    """The builders fed from models.json make exactly the ComfyUI workflows videobench 1.3.0
    made from code (tests/golden_graphs.json, frozen from 1.3.0)."""

    def setUp(self):
        vb.load_registry()

    def test_lineup(self):
        p = GOLDEN["prompt"]
        for c in vb.CONTESTANTS:
            for t, tiles in TILES.items():
                with self.subTest(model=c["key"], tiles=t):
                    self.assertEqual(roundtrip(c["build"](p, 24, tiles, f"videobench/K-{c['key']}")),
                                     GOLDEN["lineup"][f"{c['key']}/{t}"])
            with self.subTest(model=c["key"], resized=True):
                g = c["build"](p, 7, TILES["normal"], "videobench/R")
                vb.resize(g, 640, 480)
                vb.relength(g, 97)
                self.assertEqual(roundtrip(g), GOLDEN["lineup"][f"{c['key']}/resized"])
            if c.get("split"):
                with self.subTest(model=c["key"], split=True):
                    g = c["build"](p, 24, TILES["lowmem"], f"videobench/K-{c['key']}")
                    self.assertEqual(roundtrip(vb.split_phases(g, f"K-{c['key']}")), GOLDEN["split"][c["key"]])
        self.assertEqual(len(GOLDEN["lineup"]), 3 * len(vb.CONTESTANTS))

    def test_benchmark(self):
        p = GOLDEN["prompt"]
        for kind in vb.STILLS:
            self.assertEqual(roundtrip(vb.still_graph(kind, p, 1024, 768, "videobench/K-still", seed=24)),
                             GOLDEN["still"][kind])
        for prof, spec in vb.PROFILES.items():
            for low in (False, True):
                got = vb.clip_phases(spec["video"], "K-still.png", p, *spec["clip"], "K", "videobench/K-clip", low,
                                     seed=24, frames=121)
                self.assertEqual(roundtrip(got), GOLDEN["clip_phases"][f"{prof}/{'lowmem' if low else 'normal'}"])


class Checks(unittest.TestCase):
    """What check_registry lets through. The file names become paths and the URLs are
    downloaded, so those are held to a narrow shape."""

    def check(self, data):
        return vb.check_registry(data)

    def file_problem(self, **over):
        _, _, problems, _ = self.check(registry(files={"extra": hf_file(**over)}))
        return problems

    def model_problem(self, **over):
        _, models, problems, newer = self.check(registry(models=[model(**over)]))
        return problems, models, newer

    def test_a_good_one(self):
        files, models, problems, newer = self.check(registry(files={"extra": hf_file()}))
        self.assertEqual((problems, newer), ([], []))
        self.assertEqual(list(files), ["extra"])
        self.assertEqual([m["key"] for m in models], ["test-model"])

    def test_files_only_from_hugging_face(self):
        for url in ("http://huggingface.co/org/repo/resolve/main/test-model.safetensors",
                    "https://example.com/org/repo/resolve/main/test-model.safetensors",
                    "https://huggingface.co.evil.com/org/repo/resolve/main/test-model.safetensors",
                    "https://huggingface.co/org/repo/blob/main/test-model.safetensors",
                    "https://huggingface.co/org/repo/resolve/main/../../test-model.safetensors",
                    "https://huggingface.co/org/repo/resolve/main/other-name.safetensors"):
            with self.subTest(url=url):
                self.assertTrue(self.file_problem(url=url))

    def test_file_names_stay_inside_the_models_folder(self):
        for name in ("../escape.safetensors", "..\\escape.safetensors", "sub/dir.safetensors", "/abs.safetensors",
                     "C:\\x.safetensors", ".hidden.safetensors"):
            with self.subTest(name=name):
                self.assertTrue(self.file_problem(name=name, url="https://huggingface.co/o/r/resolve/main/" + name))
        for folder in ("..", "custom_nodes", "../models", ""):
            with self.subTest(folder=folder):
                self.assertTrue(self.file_problem(folder=folder))

    def test_only_weights_formats_that_cannot_run_code(self):
        for ext in (".ckpt", ".pt", ".pth", ".bin", ".py", ".zip"):
            with self.subTest(ext=ext):
                self.assertTrue(self.file_problem(name="m" + ext, url="https://huggingface.co/o/r/resolve/main/m" + ext))

    def test_hash_shape(self):
        self.assertTrue(self.file_problem(sha256="ABC"))
        self.assertTrue(self.file_problem(sha256="A" * 64))
        self.assertFalse(self.file_problem(sha256="a" * 64))

    def test_benchmark_files_cannot_be_replaced(self):
        _, _, problems, _ = self.check(registry(files={"t5": hf_file()}))
        self.assertTrue(problems)

    def test_model_entries(self):
        bad = [dict(builder="ltxv", files={"dit": "ltxv2b", "te": "t5"}),                     # a part missing
               dict(files={"dit": "ltxv2b", "te": "t5", "vae": "ltxvae", "extra": "t5"}),     # a part it doesn't have
               dict(files={"dit": "nope", "te": "t5", "vae": "ltxvae"}),                      # a file that isn't listed
               dict(params={"cfgg": 2}),                                                      # a setting it doesn't have
               dict(params={"cfg": "high"}), dict(params={"sampler": 3}), dict(params={"cfg": True}),
               dict(size="big"), dict(frames=0), dict(frames=12.5), dict(ref_s=-1), dict(released="August"),
               dict(key="Bad Key"), dict(name=""), dict(license={"name": "X"}),
               dict(license={"name": "X", "url": "http://x"}), dict(gated=True), dict(gated="https://example.com/x"),
               dict(license={"name": "X", "url": "https://x", "excluded": ["EU"]}),             # codes, no words
               dict(license={"name": "X", "url": "https://x", "restricted": "not in X"}),       # words, no codes
               dict(license={"name": "X", "url": "https://x", "restricted": "r", "excluded": ["Canada"]}),
               dict(license={"name": "X", "url": "https://x", "restricted": "r", "excluded": []}),
               dict(builder="minimax_h3", files={"dit": "ltxv2b", "te": "t5", "vae": "ltxvae", "avae": "ltxvae"}, split=True)]
        for over in bad:
            with self.subTest(**{k: str(v) for k, v in over.items()}):
                problems, models, _ = self.model_problem(**over)
                self.assertTrue(problems)
                self.assertEqual(models, [])

    def test_optional_parts_and_settings(self):
        problems, models, _ = self.model_problem(params={"cfg": 2.5, "sampler": "euler_ancestral"}, split=True)
        self.assertEqual(problems, [])
        problems, models, _ = self.model_problem(
            license={"name": "X", "url": "https://x", "restricted": "not licensed in Y", "excluded": ["EU", "KR"]})
        self.assertEqual(problems, [])
        problems, models, _ = self.model_problem(
            builder="minimax_h3", files={"dit": "ltxv2b", "te": "t5", "vae": "ltxvae", "avae": "ltxvae"})
        self.assertEqual(problems, [])  # the LoRA is optional

    def test_a_builder_from_a_newer_videobench(self):
        problems, models, newer = self.model_problem(builder="wan3")
        self.assertEqual((problems, models, newer), ([], [], ["test-model"]))

    def test_duplicate_keys(self):
        _, models, problems, _ = self.check(registry(models=[model(), model(name="Again")]))
        self.assertEqual(len(models), 1)
        self.assertTrue(problems)

    def test_whole_file(self):
        for data in ({}, [], {"schema": 1}, {"revision": 1}, registry(schema=2), registry(schema="1")):
            with self.subTest(data=str(data)[:40]):
                with self.assertRaises(vb.RegistryError):
                    self.check(data)


class WhichListWins(unittest.TestCase):
    """The bundled models.json and the one --update-models saved: the higher revision wins."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        vb.load_registry()
        self.tmp.cleanup()

    def save(self, data):
        (self.root / "models.json").write_text(json.dumps(data), encoding="utf-8")

    def test_newer_in_dir(self):
        self.save(registry(revision=BUNDLED["revision"] + 1))
        vb.load_registry(self.root)
        self.assertEqual(Path(vb.REGISTRY["path"]), self.root / "models.json")
        self.assertEqual([c["key"] for c in vb.CONTESTANTS], ["test-model"])

    def test_older_in_dir(self):
        self.save(registry(revision=BUNDLED["revision"] - 1))
        vb.load_registry(self.root)
        self.assertEqual(Path(vb.REGISTRY["path"]), ROOT / "models.json")

    def test_broken_in_dir(self):
        (self.root / "models.json").write_text("{not json", encoding="utf-8")
        vb.load_registry(self.root)
        self.assertEqual(Path(vb.REGISTRY["path"]), ROOT / "models.json")
        self.assertTrue(any("isn't usable" in p for p in vb.REGISTRY["problems"]))

    def test_none_anywhere(self):
        with mock.patch.object(vb, "registry_paths", return_value=[self.root / "models.json"]):
            vb.load_registry(self.root)
        self.assertEqual(vb.CONTESTANTS, [])
        self.assertNotIn("h3", vb.FILES)
        self.assertIn("ltxv2b", vb.FILES)  # the benchmark never depends on the list
        self.assertIn("--update-models", "\n".join(vb.registry_lines()))


class UpdateModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "vb"
        self.src = Path(self.tmp.name) / "online.json"
        vb.load_registry(self.root)

    def tearDown(self):
        vb.load_registry()
        self.tmp.cleanup()

    def update(self, data):
        self.src.write_text(json.dumps(data), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            vb.update_models(self.root, url=self.src.as_uri())
        return out.getvalue()

    def test_newer_list_is_saved_and_used(self):
        data = copy.deepcopy(BUNDLED)
        data["revision"] += 1
        data["models"].append(model(key="brand-new", name="Brand New"))
        data["models"] = [m for m in data["models"] if m["key"] != "mochi-1"]
        data["models"][0]["steps"] += 1
        out = self.update(data)
        self.assertTrue((self.root / "models.json").exists())
        self.assertIn("brand-new (Brand New)", out)
        self.assertIn("removed: mochi-1", out)
        self.assertIn(f"changed: {data['models'][0]['key']}", out)
        self.assertIn("brand-new", [c["key"] for c in vb.CONTESTANTS])

    def test_same_or_older_changes_nothing(self):
        for rev in (BUNDLED["revision"], BUNDLED["revision"] - 1):
            out = self.update(registry(revision=rev))
            self.assertIn("already up to date", out)
            self.assertFalse((self.root / "models.json").exists())

    def test_unusable_list_changes_nothing(self):
        for data in (registry(revision=99, schema=2), {"hello": 1}):
            with self.assertRaises(SystemExit), quiet():
                self.update(data)
            self.assertFalse((self.root / "models.json").exists())


class Restricted(unittest.TestCase):
    """A model whose license excludes countries is offered once the user says where they are,
    and only where it's licensed."""

    def setUp(self):
        vb.load_registry()
        by = {c["key"]: c for c in vb.CONTESTANTS}
        self.h3, self.hy, self.wan = by["minimax-h3"], by["hunyuan-1.5"], by["wan-2.2-5b"]

    def tearDown(self):
        vb.OPTS["country"] = None

    def offered(self, c):
        return vb.fit(c, GPU, 64)[0] != "restricted"

    def test_off_until_the_user_says_where(self):
        self.assertEqual(vb.fit(self.h3, GPU, 64), ("restricted", "needs --country"))
        self.assertTrue(self.offered(self.wan))

    def test_by_country(self):
        for cc, h3, hy in (("CA", True, True), ("JP", True, True), ("NO", True, True), ("US", False, True),
                           ("DE", False, False), ("GB", False, False), ("KR", False, False)):
            vb.OPTS["country"] = cc
            with self.subTest(country=cc):
                self.assertEqual((self.offered(self.h3), self.offered(self.hy), self.offered(self.wan)), (h3, hy, True))

    def test_country_codes(self):
        for text, cc in ((" ca ", "CA"), ("uk", "GB"), ("GB", "GB"), ("EU", None), ("Canada", None), ("", None), (None, None)):
            self.assertEqual(vb.country_code(text), cc, text)

    def test_picking(self):
        rows = [(n, c, vb.fit(c, GPU, 64)[0]) for n, c in enumerate(vb.CONTESTANTS, 1)]
        picks, _ = vb.parse_picks("all", rows)
        self.assertNotIn(self.h3, picks)
        n = next(n for n, c, _ in rows if c is self.h3)
        picks, skipped = vb.parse_picks(str(n), rows)
        self.assertEqual((picks, skipped), ([], [(self.h3, "restricted")]))
        self.assertIn("--country", vb.skip_reason(self.h3, "restricted"))
        vb.OPTS["country"] = "DE"
        self.assertIn("you're in DE", vb.skip_reason(self.h3, "restricted"))

    def menu(self):
        with mock.patch.object(vb, "measured_times", return_value={}), \
             mock.patch.object(vb, "missing_gb", return_value=0.0):
            return vb.lineup_menu(Path("."), GPU, 64)[1]

    def test_menu_says_why(self):
        text = self.menu()
        self.assertIn("minimax-h3 is not licensed in the EU, the UK, South Korea or the US", text)
        self.assertIn("--country", text)
        self.assertIn("MiniMax H3 Community", text)
        vb.OPTS["country"] = "DE"
        self.assertIn("Not licensed in DE, so not offered: minimax-h3, hunyuan-1.5, hunyuan-1.0", self.menu())

    def test_kept_on_this_computer(self):
        with tempfile.TemporaryDirectory() as d:
            vb.save_country(Path(d), "CA")
            self.assertEqual((vb._settings(Path(d)), vb.OPTS["country"]), ({"country": "CA"}, "CA"))
            vb.save_country(Path(d), "")  # a skipped question: remembered, so it isn't asked again
            self.assertEqual((vb._settings(Path(d)), vb.OPTS["country"]), ({"country": ""}, None))


class PinnedDownloads(unittest.TestCase):
    """A file models.json pins is checked against that sha256, not just against what
    Hugging Face says today."""
    URL = "https://huggingface.co/o/r/resolve/main/x.safetensors"

    def test_a_different_file_upstream_stops_before_downloading(self):
        with tempfile.TemporaryDirectory() as d, quiet(), \
             mock.patch.object(vb, "hf_meta", return_value=(10, "b" * 64)), \
             mock.patch.object(vb.urllib.request, "urlopen") as opened:
            with self.assertRaises(SystemExit):
                vb.fetch(self.URL, Path(d) / "x.safetensors", 1.0, "a" * 64)
            opened.assert_not_called()

    def test_the_pinned_hash_is_what_counts(self):
        data = b"weights"
        with tempfile.TemporaryDirectory() as d, quiet():
            dest = Path(d) / "x.safetensors"
            dest.with_name(dest.name + ".part").write_bytes(data)
            with mock.patch.object(vb, "hf_meta", return_value=(len(data), None)):  # Hugging Face gives no hash
                vb.fetch(self.URL, dest, 1.0, hashlib.sha256(data).hexdigest())
            self.assertEqual(dest.read_bytes(), data)


class Machine(unittest.TestCase):
    def test_results_carry_no_computer_name(self):
        m = vb.machine({"gpu": "G", "vram_gb": 8, "unified": False, "backend": "cuda", "torch": "t"})
        self.assertNotIn("host", m)
        self.assertNotIn(__import__("platform").node() or "\0", json.dumps(m))

    def test_measured_times_match_the_hardware(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{"lineup_model": "a", "status": "ok", "total_seconds": 60, "machine": {"gpu": "G", "cpu": "C"}},
                    {"lineup_model": "b", "status": "ok", "total_seconds": 70, "machine": {"gpu": "Other", "cpu": "C"}},
                    {"lineup_model": "c", "status": "failed", "total_seconds": 5, "machine": {"gpu": "G", "cpu": "C"}}]
            (Path(d) / "results.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
            with mock.patch.object(vb, "cpu_name", return_value="C"):
                self.assertEqual(vb.measured_times(Path(d), {"gpu": "G"}), {"a": 60})


class CheckRead(unittest.TestCase):
    def test_reads(self):
        self.assertTrue(vb.check_read("Buy the Mudsplat today", "buy the mud splat today", ["Mudsplat"])["ok"])
        self.assertTrue(vb.check_read("Walk up stairs now", "walk upstairs now")["ok"])
        self.assertTrue(vb.check_read("Only 3,000 left", "only three thousand left")["ok"])
        self.assertFalse(vb.check_read("A short line", "")["ok"])
        self.assertFalse(vb.check_read("A short line here", "a short line here and then five more words")["ok"])
        self.assertFalse(vb.check_read("one two quick brown fox jumps over lazy dogs", "quick fox")["ok"])

    def test_a_word_split_in_three_joins_back_up_when_the_whole_word_is_heard(self):
        # "Ca-ta-pult Lunch-box Bud-dy!": the pair joins rejoined lunch+box and bud+dy but never
        # ca+ta+pult, so a faithful read scored 88% and the job failed.
        script = ("CATAPULT LUNCHBOX BUDDY!\nCa-ta-pult Lunch-box Bud-dy!\nLaunch your sandwich over the moon!\n"
                  "Collect Soup Slinger and Juice Jumper! Sold separately. Batteries not included.")
        heard = ("Catapult Lunchbox Buddy! Catapult Lunchbox Buddy! Launch your sandwich over the moon! "
                 "Collect Soup Slinger and Juice Jumper! Sold separately. Batteries not included.")
        r = vb.check_read(script, heard, ["Catapult Lunchbox Buddy"])
        self.assertEqual((r["ok"], r["overlap"]), (True, 1.0))
        self.assertTrue(vb.check_read("A spring-loaded one-of-a-kind toy", "a springloaded oneofakind toy")["ok"])
        self.assertFalse(vb.check_read("Ca-ta-pult! Soak the sky!", "Soak the sky!", ["Catapult"])["ok"],
                         "a chant that was never sung must still fail")


class Small(unittest.TestCase):
    def test_frames_for(self):
        self.assertEqual(vb.frames_for(None, 24, 121), 121)
        self.assertEqual(vb.frames_for(5, 24, 121), 121)
        self.assertEqual(vb.frames_for(10, 24, 121), 241)

    def test_no_private_references(self):
        text = (ROOT / "videobench.py").read_text(encoding="utf-8").lower()
        for word in ("rigmatch", "slopmart", "davepc", "platform.node"):
            self.assertNotIn(word, text)


class CommandLine(unittest.TestCase):
    """The commands CI can run on any machine: no GPU, no downloads."""

    def run_vb(self, *args, d=None, code=0):
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, str(ROOT / "videobench.py"), "--dir", d or tmp, *args],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                               stdin=subprocess.DEVNULL)
        self.assertEqual(r.returncode, code, r.stdout + r.stderr)
        return r.stdout + r.stderr

    def test_help(self):
        out = self.run_vb("--help")
        for flag in ("--update-models", "--country", "--no-ask", "--list-models"):
            self.assertIn(flag, out)

    def test_list_models(self):
        out = self.run_vb("--list-models")
        self.assertIn("models.json revision", out)
        self.assertIn("license", out)
        self.assertIn("--country CA", out)

    def test_country(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("Not licensed in DE", self.run_vb("--list-models", "--country", "de", d=d))
            self.assertEqual(json.loads((Path(d) / "settings.json").read_text(encoding="utf-8")), {"country": "DE"})
            self.assertIn("Not licensed in DE", self.run_vb("--list-models", d=d))  # remembered
        self.run_vb("--list-models", "--country", "Canada", code=2)

    def test_list(self):
        self.assertIn("THE PLAN", self.run_vb("--list"))


if __name__ == "__main__":
    unittest.main()
