import json
import os
import stat
import textwrap

import pytest
from pygltflib import GLTF2

from cubist.build.runner import build_script
from cubist.cli import main
from cubist.export import ExportBlocked, export_asset
from cubist.io.vengi import VengiError

STYLE = '[project]\nvoxel_size_m = 0.25\n\n[palette]\nwood = ["#3b2414", "#8a5a32"]\n'


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(STYLE)
    return tmp_path


def built(project, name="crate", body="box((2, 2, 2), color=p.wood.dark)", size="(2, 2, 2)"):
    script = project / f"{name}.py"
    script.write_text(
        textwrap.dedent(f"""
        from cubist import *

        @asset(name="{name}", size={size})
        def make(p):
            return {body}
    """)
    )
    [result] = build_script(script)
    return result.vox_path.parent


def test_export_writes_glb_and_vox_to_the_project_export_dir(project):
    out = export_asset(built(project))
    assert out["glb"] == str(project / "export" / "crate.glb")
    assert out["vox"] == str(project / "export" / "crate.vox")
    g = GLTF2.load_from_bytes((project / "export" / "crate.glb").read_bytes())
    assert g.nodes[0].name == "crate"
    pos = g.accessors[g.meshes[0].primitives[0].attributes.POSITION]
    assert pos.max[1] == 0.5  # 2 voxels at 0.25 m


def test_lint_errors_block_export_unless_forced(project):
    asset_dir = built(project, name="wide", body="box((3, 2, 2), color=p.wood.dark)")
    with pytest.raises(ExportBlocked, match="bounds"):
        export_asset(asset_dir)
    assert not (project / "export" / "wide.glb").exists()
    out = export_asset(asset_dir, force=True)
    assert os.path.exists(out["glb"])


def test_formats_and_options(project, tmp_path):
    out = export_asset(
        built(project), formats=("glb",), out_dir=tmp_path / "game", vertex_colors=True
    )
    assert set(out) == {"glb"}
    g = GLTF2.load_from_bytes(open(out["glb"], "rb").read())
    assert g.meshes[0].primitives[0].attributes.COLOR_0 is not None


def test_vengi_missing_is_a_clear_error(project, monkeypatch):
    monkeypatch.setenv("PATH", str(project / "empty"))
    with pytest.raises(VengiError, match="vengi-voxconvert"):
        export_asset(built(project), formats=("obj",))


def test_vengi_bridge_converts_other_formats(project, monkeypatch):
    bin_dir = project / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "vengi-voxconvert"
    fake.write_text(
        textwrap.dedent("""\
        #!/bin/sh
        # fake converter: record the arguments and create the output file
        echo "$@" > "$(dirname "$0")/args.txt"
        while [ $# -gt 0 ]; do
          if [ "$1" = "--output" ]; then touch "$2"; fi
          shift
        done
    """)
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    out = export_asset(built(project), formats=("obj",))
    assert out["obj"] == str(project / "export" / "crate.obj")
    args = (bin_dir / "args.txt").read_text().split()
    assert args[args.index("--input") + 1].endswith("crate.vox")


def test_cli_export(project, capsys):
    asset_dir = built(project)
    capsys.readouterr()
    assert main(["export", str(asset_dir), "--glb", "--ao"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert set(info["exported"]) == {"glb"}


def test_cli_export_blocked_exits_2(project, capsys):
    asset_dir = built(project, name="wide", body="box((3, 2, 2), color=p.wood.dark)")
    capsys.readouterr()
    assert main(["export", str(asset_dir)]) == 2
    assert "blocked" in capsys.readouterr().err
