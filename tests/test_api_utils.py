import pytest
from pathlib import Path

from gui.api_utils import sanitize_filename_stem, resolve_safe_checkpoint_path


class TestSanitizeFilenameStem:
    def test_plain_name_unchanged(self):
        assert sanitize_filename_stem("fist") == "fist"

    def test_mixed_case_lowered(self):
        assert sanitize_filename_stem("OpenHand") == "openhand"

    def test_spaces_replaced(self):
        assert sanitize_filename_stem("thumbs up!") == "thumbs_up"

    def test_path_traversal_neutralized(self):
        stem = sanitize_filename_stem("../../evil")
        assert "/" not in stem
        assert "\\" not in stem
        assert ".." not in stem
        Path(stem)  # valid single component

    def test_backslash_traversal_neutralized(self):
        stem = sanitize_filename_stem("..\\..\\windows\\system32")
        assert "\\" not in stem and ".." not in stem

    def test_length_capped(self):
        assert len(sanitize_filename_stem("a" * 500)) <= 64

    @pytest.mark.parametrize("bad", ["", "   ", "...", "__", "-_-"])
    def test_empty_results_fall_back(self, bad):
        assert sanitize_filename_stem(bad) == "sample"

    def test_custom_fallback(self):
        assert sanitize_filename_stem("???", fallback="pose") == "pose"


class TestResolveSafeCheckpointPath:
    @pytest.fixture()
    def ckpt_tree(self, tmp_path: Path):
        (tmp_path / "sub").mkdir()
        good = tmp_path / "model.ckpt"
        good.write_bytes(b"x")
        nested = tmp_path / "sub" / "last.ckpt"
        nested.write_bytes(b"y")
        not_ckpt = tmp_path / "notes.txt"
        not_ckpt.write_text("hi")
        outside = tmp_path.parent / "outside.ckpt"
        outside.write_bytes(b"z")
        return {"base": tmp_path, "good": good, "nested": nested,
                "txt": not_ckpt, "outside": outside}

    def test_accepts_valid_file(self, ckpt_tree):
        out = resolve_safe_checkpoint_path(str(ckpt_tree["good"]), ckpt_tree["base"])
        assert out == ckpt_tree["good"].resolve()

    def test_accepts_nested_file(self, ckpt_tree):
        out = resolve_safe_checkpoint_path(str(ckpt_tree["nested"]), ckpt_tree["base"])
        assert out == ckpt_tree["nested"].resolve()

    def test_relative_path_resolved_against_base(self, tmp_path, monkeypatch):
        good = tmp_path / "rel.ckpt"
        good.write_bytes(b"x")
        monkeypatch.chdir(tmp_path)
        out = resolve_safe_checkpoint_path("rel.ckpt", tmp_path)
        assert out == good.resolve()

    def test_rejects_missing_file(self, tmp_path):
        assert resolve_safe_checkpoint_path(str(tmp_path / "nope.ckpt"), tmp_path) is None

    def test_rejects_non_ckpt_extension(self, ckpt_tree):
        assert resolve_safe_checkpoint_path(str(ckpt_tree["txt"]), ckpt_tree["base"]) is None

    def test_rejects_escape_outside_base(self, ckpt_tree):
        assert resolve_safe_checkpoint_path(str(ckpt_tree["outside"]), ckpt_tree["base"]) is None

    def test_rejects_traversal_string(self, tmp_path):
        evil = str(tmp_path / ".." / "escape.ckpt")
        assert resolve_safe_checkpoint_path(evil, tmp_path) is None

    @pytest.mark.parametrize("empty", [None, "", 12])
    def test_rejects_invalid_input(self, tmp_path, empty):
        assert resolve_safe_checkpoint_path(empty, tmp_path) is None
