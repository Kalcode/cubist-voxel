"""The MCP server, exercised over the real protocol with an in-memory client."""

import json

import anyio
import pytest
from mcp.client import Client

from cubist.build.runner import build_script
from cubist.examples import promote
from cubist.mcp_server import create_server

STYLE = '[project]\nvoxel_size_m = 0.25\n\n[palette]\nwood = ["#3b2414", "#5e3a1f", "#8a5a32"]\n'
CRATE = """from cubist import *

@asset(name="crate", size=(4, 4, 4), tags=["prop", "wood"], grounded=True)
def crate(p):
    return box((4, 4, 4), color=p.wood.mid)
"""


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(STYLE)
    return tmp_path


def call(project, tool, **args):
    async def go():
        async with Client(create_server(project)) as client:
            return await client.call_tool(tool, args)

    return anyio.run(go)


def texts(result):
    return [c.text for c in result.content if c.type == "text"]


def images(result):
    return [c for c in result.content if c.type == "image"]


def test_lists_the_design_tools(project):
    async def go():
        async with Client(create_server(project)) as client:
            return await client.list_tools()

    names = {t.name for t in anyio.run(go).tools}
    assert {
        "build_asset",
        "render",
        "lint",
        "inspect",
        "export",
        "get_style_bible",
        "search_examples",
    } <= names


def test_server_instructions_carry_the_workflow(project):
    async def go():
        async with Client(create_server(project)) as client:
            return client.instructions

    assert "5 iterations" in anyio.run(go)


def test_build_asset_from_code_returns_lint_and_the_contact_sheet(project):
    result = call(project, "build_asset", code=CRATE, name="crate")
    assert not result.is_error
    text = "\n".join(texts(result))
    assert "lint: 0 errors" in text and "crate.vox" in text
    [image] = images(result)
    assert image.mime_type == "image/png"
    assert (project / "assets" / "crate.py").read_text() == CRATE


def test_build_errors_come_back_as_tool_errors_with_the_reason(project):
    bad = CRATE.replace("p.wood.mid", "p.wod.mid")
    result = call(project, "build_asset", code=bad, name="crate")
    assert result.is_error
    assert "wood" in result.content[0].text


def test_render_slices_and_views(project):
    call(project, "build_asset", code=CRATE, name="crate")
    sliced = call(project, "render", asset="crate", slices="y=0")
    assert "y=0" in texts(sliced)[0] and not images(sliced)
    viewed = call(project, "render", asset="crate", views=["front", "iso"])
    assert len(images(viewed)) == 1


def test_lint_inspect_and_style_return_json(project):
    call(project, "build_asset", code=CRATE, name="crate")
    assert json.loads(texts(call(project, "lint", asset="crate"))[0])["ok"] is True
    info = json.loads(texts(call(project, "inspect", asset="crate"))[0])
    assert info["voxels"] == 64 and info["parts"][0]["name"] == "crate"
    style = json.loads(texts(call(project, "get_style_bible"))[0])
    assert style["palette"][0]["name"] == "wood.dark"


def test_export_writes_files(project):
    call(project, "build_asset", code=CRATE, name="crate")
    out = json.loads(texts(call(project, "export", asset="crate", formats=["glb"]))[0])
    assert out["glb"].endswith("crate.glb")


def test_search_examples_returns_scripts_and_sheets(project):
    script = project / "assets" / "crate.py"
    script.parent.mkdir()
    script.write_text(CRATE)
    build_script(script)
    promote(project, "crate", note="Solid and simple.")
    result = call(project, "search_examples", tags=["wood"])
    assert "Solid and simple." in texts(result)[0]
    assert "box((4, 4, 4)" in texts(result)[0]
    assert len(images(result)) == 1


def test_diff_after_a_rebuild(project):
    call(project, "build_asset", code=CRATE, name="crate")
    call(
        project,
        "build_asset",
        code=CRATE.replace("(4, 4, 4), color", "(4, 3, 4), color"),
        name="crate",
    )
    result = call(project, "diff", asset="crate")
    assert "16 removed" in texts(result)[0]
    assert len(images(result)) == 1


def test_new_asset_scaffolds_a_brief(project):
    result = call(
        project,
        "new_asset",
        name="lamp",
        category="prop",
        tags=["metal"],
        description="A street lamp.",
    )
    assert "briefs/lamp.toml" in texts(result)[0]
    assert (project / "assets" / "lamp.py").is_file()


def test_cli_entry_point_serves_over_stdio(project):
    import sys

    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "cubist.cli", "mcp", "--project", str(project)]
    )

    async def go():
        async with Client(params) as client:
            tools = await client.list_tools()
            style = await client.call_tool("get_style_bible", {})
            return {t.name for t in tools.tools}, style

    names, style = anyio.run(go)
    assert "build_asset" in names
    assert json.loads(style.content[0].text)["palette"][0]["name"] == "wood.dark"
