"""Project workflow: briefs and scaffolding, the example library, diffs, palette import."""

import json
import textwrap
import tomllib

import numpy as np
import pytest

from cubist.build.runner import build_script, load_project_style
from cubist.cli import main
from cubist.core.style import StyleError
from cubist.examples import ExampleError, promote, search_examples
from cubist.project import ProjectError, find_project, load_brief, new_asset
from cubist.render.diff import change_grid, diff_sheet

STYLE = """[project]
name = "test"

[palette]
# woods, dark to light
wood = ["#3b2414", "#5e3a1f", "#8a5a32"]
leaf = "#3f6b2a"
"""


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(STYLE)
    return tmp_path


def script(project, name, body, size="(4, 4, 4)", tags='["prop", "wood"]'):
    path = project / "assets" / f"{name}.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        textwrap.dedent(f"""
        from cubist import *

        @asset(name="{name}", size={size}, tags={tags}, grounded=True)
        def make(p):
            return {body}
    """)
    )
    return path


# --- briefs and scaffolding -------------------------------------------------


def test_find_project_walks_up_to_style_toml(project):
    (project / "assets" / "deep").mkdir(parents=True)
    assert find_project(project / "assets" / "deep") == project


def test_new_asset_writes_a_brief_and_a_buildable_stub(project):
    brief_path, script_path = new_asset(
        project, "lantern", category="prop", tags=["metal"], description="A hanging lantern."
    )
    brief = load_brief(brief_path)
    assert brief["name"] == "lantern" and brief["category"] == "prop"
    assert brief["size"] == [16, 16, 16]
    assert brief["tags"] == ["prop", "metal"]
    assert brief["description"] == "A hanging lantern."
    assert brief["must_haves"] == []
    [result] = build_script(script_path)
    assert result.summary["name"] == "lantern"


def test_new_asset_sizes_follow_the_category(project):
    brief_path, _ = new_asset(project, "orc", category="character")
    assert load_brief(brief_path)["size"] == [16, 24, 12]


def test_new_asset_refuses_to_overwrite(project):
    new_asset(project, "lantern")
    with pytest.raises(ProjectError, match="lantern"):
        new_asset(project, "lantern")


def test_new_asset_rejects_bad_names(project):
    with pytest.raises(ProjectError, match="name"):
        new_asset(project, "my lantern!")


# --- example library --------------------------------------------------------


def test_promote_copies_script_sheet_and_meta(project):
    path = script(project, "crate", "box((4, 4, 4), color=p.wood.mid)")
    build_script(path)
    example = promote(project, "crate", note="Clean silhouette, palette used as a ramp.")
    assert (example / "crate.py").read_text() == path.read_text()
    assert (example / "contact.png").is_file()
    meta = tomllib.loads((example / "meta.toml").read_text())
    assert meta["tags"] == ["prop", "wood"]
    assert meta["note"] == "Clean silhouette, palette used as a ramp."


def test_promote_refuses_assets_with_lint_errors(project):
    path = script(project, "big", "box((5, 4, 4), color=p.wood.mid)")
    build_script(path)
    with pytest.raises(ExampleError, match="lint"):
        promote(project, "big", note="no")


def test_search_ranks_by_category_and_tag_overlap(project):
    for name, tags in [
        ("crate", '["prop", "wood"]'),
        ("bush", '["prop", "nature"]'),
        ("hut", '["building", "wood"]'),
    ]:
        build_script(script(project, name, "box((4, 4, 4), color=p.wood.mid)", tags=tags))
        promote(project, name, note=f"{name} note")
    found = search_examples(project, tags=["wood"], category="prop")
    assert [e["name"] for e in found] == ["crate", "bush", "hut"]  # 3, 2, 1 points
    assert found[0]["script"].endswith("crate.py")
    assert found[0]["contact_sheet"].endswith("contact.png")
    assert search_examples(project, tags=["nature"], limit=1)[0]["name"] == "bush"


def test_search_with_no_library_is_empty(project):
    assert search_examples(project, tags=["wood"]) == []


# --- diffs ------------------------------------------------------------------


def test_rebuild_keeps_the_previous_vox(project):
    path = script(project, "crate", "box((4, 4, 4), color=p.wood.mid)")
    build_script(path)
    first = (project / "out" / "crate" / "crate.vox").read_bytes()
    script(project, "crate", "box((4, 3, 4), color=p.wood.mid)")
    build_script(path)
    assert (project / "out" / "crate" / "crate.prev.vox").read_bytes() == first


def test_change_grid_classifies_voxels():
    from cubist.core.grid import VoxelGrid

    before = VoxelGrid.from_voxels(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.array([1, 1, 1], dtype=np.uint8)
    )
    after = VoxelGrid.from_voxels(
        np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]]), np.array([1, 2, 1], dtype=np.uint8)
    )
    grid, counts = change_grid(before, after)
    assert counts == {"same": 1, "added": 1, "removed": 1, "recolored": 1}
    assert grid.count() == 4


def test_diff_sheet_stacks_before_after_and_changes(project):
    from cubist.core.grid import VoxelGrid

    p = load_project_style(project / "style.toml").palette
    a = VoxelGrid.from_voxels(np.array([[0, 0, 0]]), np.array([1], dtype=np.uint8))
    b = VoxelGrid.from_voxels(np.array([[0, 0, 0], [0, 1, 0]]), np.array([1, 2], dtype=np.uint8))
    img = diff_sheet(a, b, p.rgba_table(), name="x")
    assert img.mode == "RGB"
    assert img.height > 2 * img.width / 3


def test_cli_diff_against_the_previous_build(project, capsys):
    path = script(project, "crate", "box((4, 4, 4), color=p.wood.mid)")
    build_script(path)
    bump = "box((1, 1, 1), color=p.leaf).at(1, 4, 1)"
    script(project, "crate", f"box((4, 4, 4), color=p.wood.mid) | {bump}", size="(4, 5, 4)")
    build_script(path)
    capsys.readouterr()
    assert main(["diff", str(project / "out" / "crate")]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["added"] == 1 and info["removed"] == 0
    assert (project / "out" / "crate" / "diff.png").is_file()


# --- palette import ---------------------------------------------------------


def test_palette_import_sets_the_source_and_keeps_comments(project, capsys):
    pal = project.parent / "lospec.hex"
    pal.write_text("1a1c2c\n5d275d\nb13e53\n")
    assert main(["palette", "import", str(pal), "--style", str(project / "style.toml")]) == 0
    text = (project / "style.toml").read_text()
    assert 'source = "lospec.hex"' in text
    assert "# woods, dark to light" in text
    assert (project / "lospec.hex").is_file()
    style = load_project_style(project / "style.toml")
    assert style.palette.source[2].rgba[:3] == (0xB1, 0x3E, 0x53)
    info = json.loads(capsys.readouterr().out)
    assert info["colors"][0] == {"name": "source.0", "hex": "#1a1c2c"}


def test_palette_import_replaces_an_existing_source(project):
    (project / "a.hex").write_text("111111\n")
    (project / "b.hex").write_text("222222\n")
    style = project / "style.toml"
    main(["palette", "import", str(project / "a.hex"), "--style", str(style)])
    main(["palette", "import", str(project / "b.hex"), "--style", str(style)])
    text = style.read_text()
    assert text.count("source =") == 1 and 'source = "b.hex"' in text


def test_palette_import_rejects_unreadable_files(project):
    (project / "bad.txt").write_text("not a color\n")
    with pytest.raises(StyleError):
        from cubist.palette_import import import_palette

        import_palette(project / "style.toml", project / "bad.txt")


def test_promoted_examples_carry_no_machine_specific_paths(project):
    build_script(script(project, "crate", "box((4, 4, 4), color=p.wood.mid)"))
    example = promote(project, "crate", note="ok")
    summary = json.loads((example / "build.json").read_text())
    assert summary["script"] == "crate.py"
    assert "outputs" not in summary
    assert str(project) not in (example / "build.json").read_text()
