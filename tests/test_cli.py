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
    assert info["colors"] == {"#3b2414": 4, "#8a5a32": 5}


def test_inspect_missing_file_exits_nonzero(tmp_path, capsys):
    assert main(["inspect", str(tmp_path / "nope.vox")]) == 1
    assert "nope.vox" in capsys.readouterr().err


def test_render_writes_contact_sheet_next_to_the_asset(tmp_path, capsys):
    script = setup(tmp_path)
    main(["build", str(script)])
    asset_dir = tmp_path / "out" / "post"
    (asset_dir / "contact.png").unlink()
    capsys.readouterr()
    assert main(["render", str(asset_dir / "post.vox"), "--views", "front,iso"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["contact_sheet"] == str(asset_dir / "contact.png")
    assert (asset_dir / "contact.png").is_file()


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
