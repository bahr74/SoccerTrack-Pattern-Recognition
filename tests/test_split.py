"""Member 1 M1 split guards and provenance tests; no real dataset access."""
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from soccertrack import config, split


@pytest.fixture
def cfg():
    # Read YAML directly in tests while Member 4's loader remains a placeholder.
    path = Path(__file__).resolve().parents[1] / "configs/project.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_01(cfg):
    assert split.matches_for("train", cfg) == ["118575", "118576", "118577", "118578", "128058"]
    assert split.matches_for("val", cfg) == ["117093"]
    assert split.matches_for("test", cfg) == ["128057", "132831"]
    cfg["split"]["use_train_optional"] = True
    cfg["split"]["use_val_optional"] = True
    assert split.matches_for("train", cfg)[-1] == "117092"
    assert split.matches_for("val", cfg) == ["117093", "132877"]
    result = split.matches_for("train", cfg)
    result.clear()
    assert len(cfg["split"]["train"]) == 5


def test_02(cfg):
    split.assert_disjoint(cfg)
    cfg["split"]["val"].append("118575")
    with pytest.raises(ValueError, match="118575"):
        split.assert_disjoint(cfg)


def test_03():
    with pytest.raises(PermissionError, match="--frozen-test"):
        split.require_test_access("test", False)
    split.require_test_access("test", True)
    for name in ("train", "val"):
        split.require_test_access(name, False)
        split.require_test_access(name, True)


def test_04(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load_config", Mock(return_value={"project": cfg}))
    monkeypatch.setattr(config, "repo_root", Mock(return_value=tmp_path))
    digest = Mock(return_value="a" * 64)
    monkeypatch.setattr(config, "config_sha256", digest)
    path = tmp_path / cfg["paths"]["test_log"]
    path.parent.mkdir(parents=True)
    original = b"# Existing log\r\n"
    path.write_bytes(original)
    before = datetime.now(timezone.utc).replace(microsecond=0)
    for method in ("scratch_fixture_a", "scratch_fixture_b"):
        split.log_test_access("Member 1", method, ["outputs/fixture.csv"], {"fixture": "b" * 64})
    content = path.read_text(encoding="utf-8")
    assert content.startswith("# Existing log\n")
    entries = [json.loads(block.split("```", 1)[0]) for block in content.split("```json\n")[1:]]
    assert [e["method"] for e in entries] == ["scratch_fixture_a", "scratch_fixture_b"]
    for entry in entries:
        assert entry["member"] == "Member 1"
        assert entry["config_sha256"] == "a" * 64
        assert entry["model_hashes"] == {"fixture": "b" * 64}
        assert entry["artefacts"] == ["outputs/fixture.csv"]
        assert entry["date"].endswith("Z")
        assert before <= datetime.fromisoformat(entry["date"]) <= datetime.now(timezone.utc)
    assert path.read_bytes().startswith(original)
    assert b"\r\n" not in path.read_bytes()[len(original):]
    assert digest.call_count == 2


def test_split_lookup_and_bundle(cfg):
    for name in ("train", "val", "test"):
        for match in cfg["split"][name]:
            assert split.split_of(match, cfg) == name
            assert split.split_of(match, {"project": cfg}) == name
    for match in ("999999", "117092", "132877"):
        with pytest.raises(ValueError, match="Unknown or disabled"):
            split.split_of(match, cfg)
    cfg["split"]["use_train_optional"] = True
    assert split.split_of("117092", cfg) == "train"


@pytest.mark.parametrize("location", ["val", "val_optional", "test"])
def test_disabled_optional_overlap(cfg, location):
    cfg["split"][location].append("117092")
    with pytest.raises(ValueError, match="117092"):
        split.assert_disjoint(cfg)
    with pytest.raises(ValueError, match="117092"):
        split.matches_for("train", cfg)


@pytest.mark.parametrize("name", ["validation", "TEST", "", None])
def test_unknown_split_is_rejected(cfg, name):
    with pytest.raises(ValueError, match="Unknown split"):
        split.matches_for(name, cfg)
    with pytest.raises(ValueError, match="Unknown split"):
        split.require_test_access(name, True)


@pytest.mark.parametrize("match", [117093, "../117093", "", "١١٧٠٩٣"])
def test_invalid_match_id(cfg, match):
    cfg["split"]["val"] = [match]
    with pytest.raises(ValueError, match="ASCII digits"):
        split.assert_disjoint(cfg)


@pytest.mark.parametrize("flag", [None, 1, "true", "False"])
def test_test_access_requires_explicit_boolean(flag):
    with pytest.raises(PermissionError):
        split.require_test_access("test", flag)


def test_log_creates_parent_and_missing_hash_writes_nothing(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "load_config", lambda: {"project": cfg})
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    unavailable = Mock(side_effect=NotImplementedError("Member 4 M0 hash"))
    monkeypatch.setattr(config, "config_sha256", unavailable)
    log = tmp_path / cfg["paths"]["test_log"]
    with pytest.raises(NotImplementedError, match="M0 hash"):
        split.log_test_access("Member 1", "fixture", [], {})
    assert not log.parent.exists()
    monkeypatch.setattr(config, "config_sha256", lambda: "c" * 64)
    split.log_test_access("Member 1", "fixture", [], {})
    assert log.is_file()


def test_log_rejects_path_escape(cfg, tmp_path, monkeypatch):
    cfg["paths"]["test_log"] = "../outside.md"
    monkeypatch.setattr(config, "load_config", lambda: {"project": cfg})
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="inside the repository"):
        split.log_test_access("Member 1", "fixture", [], {})


@pytest.mark.parametrize('value', [
    '/tmp/file.csv', r'\rooted\file.csv', 'C:/file.csv', r'C:\file.csv', 'C:file.csv',
    r'\\server\share\file.csv', '//server/share/file.csv', '../file.csv',
    'outputs/../../file.csv', r'outputs\..\..\file.csv', '', '.', 'outputs/..',
    'outputs/file.csv:stream', 'outputs/line\nbreak.csv',
])
def test_invalid_artifact_paths_do_not_write(cfg, tmp_path, monkeypatch, value):
    monkeypatch.setattr(config, 'load_config', lambda: {'project': cfg})
    monkeypatch.setattr(config, 'repo_root', lambda: tmp_path)
    digest = Mock(return_value='a' * 64)
    monkeypatch.setattr(config, 'config_sha256', digest)
    log = tmp_path / cfg['paths']['test_log']
    with pytest.raises(ValueError):
        split.log_test_access('Member 1', 'fixture', ['outputs/valid.csv', value], {})
    assert not log.parent.exists()
    digest.assert_not_called()
    log.parent.mkdir()
    log.write_bytes(b'original\n')
    with pytest.raises(ValueError):
        split.log_test_access('Member 1', 'fixture', [value], {})
    assert log.read_bytes() == b'original\n'


@pytest.mark.parametrize(('value', 'expected'), [
    (r'outputs\tables\fixture.csv', 'outputs/tables/fixture.csv'),
    ('./outputs/fixture.csv', 'outputs/fixture.csv'),
    ('outputs/temp/../fixture.csv', 'outputs/fixture.csv'),
])
def test_artifact_paths_normalized(cfg, tmp_path, monkeypatch, value, expected):
    monkeypatch.setattr(config, 'load_config', lambda: {'project': cfg})
    monkeypatch.setattr(config, 'repo_root', lambda: tmp_path)
    monkeypatch.setattr(config, 'config_sha256', lambda: 'a' * 64)
    split.log_test_access('Member 1', 'fixture', [value], {})
    text = (tmp_path / cfg['paths']['test_log']).read_text(encoding='utf-8')
    entry = json.loads(text.split('```json\n', 1)[1].split('```', 1)[0])
    assert entry['artefacts'] == [expected]
