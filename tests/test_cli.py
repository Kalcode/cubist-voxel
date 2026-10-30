import json
import textwrap

from cubist.cli import main

STYLE = '[palette]\nwood = ["#3b2414", "#8a5a32"]\n'
SCRIPT = """
from cubist import *

@asset(name="post", size=(2, 6, 2))
def post(p):
    return Model(parts=[
        Part("base", box((2, 1, 2), color=p.wood.dark)),
        Part("pole", box((1, 5, 1), color=p.wood.light).at(0, 1, 0), parent="base"),
    ])
"""


def setup(tmp_path):
    (tmp_path / "style.toml").write_text(STYLE)
    script = tmp_path / "post.py"
    script.write_text(textwrap.dedent(SCRIPT))
    return script


def test_build_prints_json_summary(tmp_path, capsys):
    script = setup(tmp_path)
    assert main(["build", str(script)]) == 0
    [summary] = json.loads(capsys.readouterr().out)
    assert summary["name"] == "post"
    assert summary["voxels"] == 9
    assert (tmp_path / "out" / "post" / "post.vox").is_file()


def test_build_only_builds_the_named_asset(tmp_path, capsys):
    script = setup(tmp_path)
    with script.open("a") as f:
        f.write(
            "\n@asset(name='peg', size=(1, 2, 1))\ndef peg(p):\n"
            "    return box((1, 2, 1), color=p.wood.dark)\n"
        )
    assert main(["build", str(script), "--only", "peg"]) == 0
    [summary] = json.loads(capsys.readouterr().out)
    assert summary["name"] == "peg"
    assert not (tmp_path / "out" / "post").exists()


def test_build_error_exits_nonzero_with_message(tmp_path, capsys):
    (tmp_path / "style.toml").write_text(STYLE)
    bad = tmp_path / "bad.py"
    bad.write_text("from cubist import *\n\n@asset(name='bad')\ndef bad(p):\n    return p.nope\n")
    assert main(["build", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "unknown color nope" in err


def test_inspect_reports_parts_and_colors(tmp_path, capsys):
    script = setup(tmp_path)
    main(["build", str(script)])
    capsys.readouterr()
    assert main(["inspect", str(tmp_path / "out" / "post" / "post.vox")]) == 0
    info = json.loads(capsys.readouterr().out)
    assert [p["name"] for p in info["parts"]] == ["base", "pole"]
    assert info["parts"][1]["bounds"] == [[0, 1, 0], [1, 6, 1]]
    assert info["voxels"] == 9
    assert info["colors"] == {"wood.dark": 4, "wood.light": 5}


def test_inspect_missing_file_exits_nonzero(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "nope.vox")]) == 1
    assert "nope.vox" in capsys.readouterr().err


def test_render_writes_render_png_and_leaves_the_build_sheet_alone(tmp_path, capsys):
    script = setup(tmp_path)
    main(["build", str(script)])
    asset_dir = tmp_path / "out" / "post"
    (asset_dir / "contact.png").unlink()
    capsys.readouterr()
    assert main(["render", str(asset_dir / "post.vox"), "--views", "front,iso"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["contact_sheet"] == str(asset_dir / "render.png")
    assert (asset_dir / "render.png").is_file()
    assert not (asset_dir / "contact.png").exists()


def test_render_slices_use_palette_names_from_the_build(tmp_path, capsys):
    script = setup(tmp_path)
    main(["build", str(script)])
    capsys.readouterr()
    assert main(["render", str(tmp_path / "out" / "post"), "--slices", "y=0..1"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("legend: . = empty, a = wood.dark, b = wood.light")
    assert "y=1  (rows: z, back to front; columns: x)" in out


def test_render_bad_view_exits_nonzero(tmp_path, capsys):
    script = setup(tmp_path)
    main(["build", str(script)])
    capsys.readouterr()
    assert main(["render", str(tmp_path / "out" / "post"), "--views", "sideways"]) == 1
    assert "sideways" in capsys.readouterr().err


def test_build_seed_flag(tmp_path, capsys):
    script = setup(tmp_path)
    assert main(["build", str(script), "--seed", "3"]) == 0
    [summary] = json.loads(capsys.readouterr().out)
    assert summary["seed"] == 3
    assert "lint" in summary


def test_lint_command_reports_clean_asset(tmp_path, capsys):
    script = setup(tmp_path)
    main(["build", str(script)])
    capsys.readouterr()
    assert main(["lint", str(tmp_path / "out" / "post")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True


def test_lint_command_exits_2_on_errors(tmp_path, capsys):
    (tmp_path / "style.toml").write_text(STYLE)
    bad = tmp_path / "wide.py"
    bad.write_text(
        "from cubist import *\n\n@asset(name='wide', size=(1, 1, 1))\n"
        "def wide(p):\n    return box((2, 1, 1), color=p.wood.dark)\n"
    )
    main(["build", str(bad)])
    capsys.readouterr()
    assert main(["lint", str(tmp_path / "out" / "wide" / "wide.vox")]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["errors"] == 1


def test_style_command_prints_the_parsed_bible(tmp_path, capsys):
    (tmp_path / "style.toml").write_text(STYLE)
    assert main(["style", "--style", str(tmp_path / "style.toml")]) == 0
    d = json.loads(capsys.readouterr().out)
    assert [c["name"] for c in d["palette"]] == ["wood.dark", "wood.light"]


def test_inspect_accepts_the_asset_directory_and_names_colors(tmp_path, capsys):
    script = setup(tmp_path)
    main(["build", str(script)])
    capsys.readouterr()
    assert main(["inspect", str(tmp_path / "out" / "post")]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["colors"] == {"wood.dark": 4, "wood.light": 5}
