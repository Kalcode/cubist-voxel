import json
import textwrap

import pytest

from cubist.build.runner import BuildError, build_script, find_style, load_palette
from cubist.io.vox import read_vox

STYLE = """
[project]
name = "test"

[palette]
wood = ["#3b2414", "#5e3a1f", "#8a5a32"]
eye = "#ffd23f"
skin = { goblin = "#6b8e23" }
"""


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(STYLE)
    (tmp_path / "assets").mkdir()
    return tmp_path


def write_script(project, name, body):
    path = project / "assets" / f"{name}.py"
    path.write_text(textwrap.dedent(body))
    return path


def test_find_style_walks_up_from_the_script(project):
    script = write_script(project, "a", "")
    assert find_style(script) == project / "style.toml"


def test_find_style_errors_when_missing(tmp_path):
    (tmp_path / "a.py").write_text("")
    with pytest.raises(BuildError, match="style.toml"):
        find_style(tmp_path / "a.py")


def test_load_palette_reads_lists_tables_and_single_colors(project):
    p = load_palette(project / "style.toml")
    assert p.wood.mid.index == 2
    assert p.eye.index == 4
    assert p.skin.goblin.index == 5


def test_build_writes_vox_and_summary_under_project_out(project):
    script = write_script(
        project,
        "crate",
        """
        from cubist import *

        @asset(name="crate", size=(4, 4, 4), tags=["prop"])
        def crate(p):
            return box((4, 4, 4), color=p.wood.dark)
    """,
    )
    [result] = build_script(script)
    out = project / "out" / "crate"
    assert result.vox_path == out / "crate.vox"
    assert read_vox(result.vox_path).parts[0].grid.count() == 64
    summary = json.loads((out / "build.json").read_text())
    assert summary["name"] == "crate"
    assert summary["voxels"] == 64
    assert summary["parts"][0] == {
        "name": "crate",
        "voxels": 64,
        "bounds": [[0, 0, 0], [4, 4, 4]],
        "pivot": [0, 0, 0],
        "parent": None,
    }
    assert summary["colors"] == {"wood.dark": 64}


def test_unknown_color_is_a_build_error_with_a_suggestion(project):
    script = write_script(
        project,
        "bad",
        """
        from cubist import *

        @asset(name="bad", size=(1, 1, 1))
        def bad(p):
            return box((1, 1, 1), color=p.wod.dark)
    """,
    )
    with pytest.raises(BuildError, match="wood"):
        build_script(script)


def test_script_exception_reports_the_script_line(project):
    script = write_script(
        project,
        "boom",
        """
        from cubist import *

        @asset(name="boom", size=(1, 1, 1))
        def boom(p):
            raise RuntimeError("kaboom")
    """,
    )
    with pytest.raises(BuildError, match=r"(?s)boom\.py\", line 6, in boom.*RuntimeError: kaboom"):
        build_script(script)


def test_script_without_assets_is_an_error(project):
    script = write_script(project, "empty", "x = 1\n")
    with pytest.raises(BuildError, match="@asset"):
        build_script(script)


def test_runaway_script_times_out(project):
    script = write_script(
        project,
        "loop",
        """
        from cubist import *

        @asset(name="loop", size=(1, 1, 1))
        def loop(p):
            while True:
                pass
    """,
    )
    with pytest.raises(BuildError, match="timed out"):
        build_script(script, timeout=2)


def test_builds_are_byte_identical(project):
    script = write_script(
        project,
        "orb",
        """
        from cubist import *

        @asset(name="orb", size=(12, 12, 12))
        def orb(p):
            return sphere(5.3, color=p.wood.mid).at(6, 6, 6).rotate(17, axis="z")
    """,
    )
    [a] = build_script(script)
    first = a.vox_path.read_bytes()
    [b] = build_script(script)
    assert b.vox_path.read_bytes() == first
