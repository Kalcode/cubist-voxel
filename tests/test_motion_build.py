"""Motion through the whole pipeline: build outputs, the CLI, and the MCP server."""

import json
import textwrap
import time

import anyio
import pytest
from mcp.client import Client
from PIL import Image

from cubist.build.runner import build_script
from cubist.cli import main
from cubist.mcp_server import create_server

STYLE = textwrap.dedent("""\
    [project]
    name = "t"
    voxel_size_m = 0.25
    [palette]
    wood = ["#3b2414", "#5e3a1f", "#8a5a32"]
    """)

CHEST = textwrap.dedent("""\
    from cubist import *

    @asset(name="chest", size=(8, 7, 6), grounded=True)
    def chest(p):
        body = box((8, 4, 6), color=p.wood.mid) - box((6, 3, 4)).at(1, 1, 1)
        lid = box((8, 2, 6), color=p.wood.light).at(0, 4, 0)
        return Model(
            parts=[
                Part("body", body, pivot=(4, 0, 3)),
                Part("lid", lid, parent="body", pivot=(4, 4, 0), joint=hinge("-x", (0, 120))),
            ],
            clips=CLIPS,
        )
    """)
TWO = '[clip("open", 0.6, {"lid": [0, 110, 100]}), clip("slam", 0.3, {"lid": [100, 0]})]'
ONE = '[clip("open", 0.6, {"lid": [0, 110, 100]})]'


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(STYLE)
    (tmp_path / "assets").mkdir()
    return tmp_path


def write(project, clips=TWO, name="chest"):
    script = project / "assets" / f"{name}.py"
    script.write_text(CHEST.replace("CLIPS", clips))
    return script


def test_build_writes_motion_outputs_for_animated_asset(project):
    [r] = build_script(write(project))
    d = r.vox_path.parent
    assert (d / "motion.png").exists() and (d / "open.gif").exists() and (d / "slam.gif").exists()
    manifest = json.loads((d / "manifest.json").read_text())
    assert [c["name"] for c in manifest["clips"]] == ["open", "slam"]
    outputs = r.summary["outputs"]
    assert outputs["motion"] == str(d / "motion.png")
    assert outputs["gifs"] == {"open": str(d / "open.gif"), "slam": str(d / "slam.gif")}
    assert outputs["manifest"] == str(d / "manifest.json")


def test_rebuild_keeps_the_previous_filmstrip(project):
    script = write(project)
    build_script(script)
    [r] = build_script(script)
    assert (r.vox_path.parent / "motion.prev.png").exists()


def test_build_of_static_asset_writes_manifest_but_no_motion(project):
    script = project / "assets" / "crate.py"
    script.write_text(
        "from cubist import *\n\n@asset(name='crate', size=(4, 4, 4))\n"
        "def crate(p):\n    return box((4, 4, 4), color=p.wood.mid)\n"
    )
    [r] = build_script(script)
    d = r.vox_path.parent
    assert (d / "manifest.json").exists()
    assert not (d / "motion.png").exists() and not list(d.glob("*.gif"))
    assert "motion" not in r.summary["outputs"]


def test_rebuild_removes_gifs_of_deleted_clips(project):
    build_script(write(project, TWO))
    [r] = build_script(write(project, ONE))
    d = r.vox_path.parent
    assert (d / "open.gif").exists() and not (d / "slam.gif").exists()


def test_cli_render_clip_and_at(project, capsys):
    [r] = build_script(write(project))
    d = r.vox_path.parent
    capsys.readouterr()
    assert main(["render", str(d), "--clip", "open", "--frames", "4", "--view", "front"]) == 0
    info = json.loads(capsys.readouterr().out)
    strip = Image.open(info["filmstrip"])
    assert main(["render", str(d), "--clip", "open", "--at", "0.3"]) == 0
    info = json.loads(capsys.readouterr().out)
    sheet = Image.open(info["contact_sheet"])
    assert strip.size != sheet.size
    assert info["contact_sheet"] == str(d / "render.png")


def test_cli_render_unknown_clip_lists_clips(project, capsys):
    [r] = build_script(write(project))
    capsys.readouterr()
    assert main(["render", str(r.vox_path.parent), "--clip", "wave"]) == 1
    err = capsys.readouterr().err
    assert "'wave'" in err and "open" in err and "slam" in err


def call(project, tool, **args):
    async def go():
        async with Client(create_server(project)) as client:
            return await client.call_tool(tool, args)

    return anyio.run(go)


def test_mcp_render_clip_returns_image(project):
    build_script(write(project))
    result = call(project, "render", asset="chest", clip="open", frames=4)
    assert [c.type for c in result.content] == ["image"]
    result = call(project, "render", asset="chest", clip="open", at=0.2)
    assert [c.type for c in result.content] == ["image"]


def test_mcp_build_returns_the_filmstrip_too(project):
    write(project)
    result = call(project, "build_asset", script_path="assets/chest.py")
    assert [c.type for c in result.content].count("image") == 2
    assert "motion.png" in result.content[0].text


def test_animated_build_under_15s(project):
    big = textwrap.dedent("""\
        from cubist import *

        @asset(name="big", size=(64, 64, 64))
        def big(p):
            base = box((64, 32, 64), color=p.wood.mid).hollow(1)
            door = box((24, 30, 2), color=p.wood.light).at(4, 1, 62)
            hatch = box((30, 2, 30), color=p.wood.dark).at(17, 32, 17)
            fan = sphere(10, color=p.wood.light).at(32, 50, 32)
            return Model(
                parts=[
                    Part("base", base - box((24, 30, 2)).at(4, 1, 62), pivot=(32, 0, 32)),
                    Part("door", door, parent="base", pivot=(4, 1, 63),
                         joint=hinge("y", (-110, 0))),
                    Part("hatch", hatch, parent="base", pivot=(17, 32, 17),
                         joint=hinge("-x", (0, 100))),
                    Part("fan", fan, parent="base", pivot=(32, 50, 32), joint=spin("y", rps=1)),
                ],
                clips=[
                    clip("door", 1.0, {"door": [0, -100]}),
                    clip("hatch", 1.0, {"hatch": [0, 95]}),
                    clip("fan", 1.0, {"fan": [0, 1]}, loop=True, ease="linear"),
                ],
            )
        """)
    script = project / "assets" / "big.py"
    script.write_text(big)
    start = time.perf_counter()
    build_script(script, timeout=60)
    assert time.perf_counter() - start < 15


def test_cli_diff_also_compares_filmstrips(project, capsys):
    script = write(project, ONE)
    build_script(script)
    script.write_text(script.read_text().replace("[0, 110, 100]", "[0, 80, 70]"))
    [r] = build_script(script)
    capsys.readouterr()
    assert main(["diff", str(r.vox_path.parent)]) == 0
    info = json.loads(capsys.readouterr().out)
    strip = Image.open(info["motion_diff"])
    one = Image.open(r.vox_path.parent / "motion.png")
    assert strip.height > 2 * one.height - 10


def test_mcp_diff_returns_the_filmstrip_diff(project):
    script = write(project, ONE)
    build_script(script)
    script.write_text(script.read_text().replace("[0, 110, 100]", "[0, 80, 70]"))
    build_script(script)
    result = call(project, "diff", asset="chest")
    assert [c.type for c in result.content].count("image") == 2


def test_promoted_examples_keep_their_filmstrip(project):
    from cubist.examples import promote, search_examples

    build_script(write(project, ONE))
    dest = promote(project, "chest", note="opens nicely")
    assert (dest / "motion.png").exists()
    [ex] = search_examples(project)
    assert ex["filmstrip"] == str(dest / "motion.png")
    result = call(project, "search_examples")
    assert [c.type for c in result.content].count("image") == 2
