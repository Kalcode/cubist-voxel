import json
import textwrap

import pytest

from cubist.build.runner import build_script
from cubist.cli import main
from cubist.evaluation import evaluate, iterations
from cubist.project import new_asset

STYLE = '[palette]\nwood = ["#3b2414", "#5e3a1f", "#8a5a32"]\nleaf = "#3f6b2a"\n'


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(STYLE)
    return tmp_path


def write(project, name, body, size="(4, 4, 4)"):
    path = project / "assets" / f"{name}.py"
    path.write_text(
        textwrap.dedent(f"""
        from cubist import *

        @asset(name="{name}", size={size}, tags=["prop"], grounded=True)
        def make(p):
            return {body}
    """)
    )
    return path


def test_builds_append_to_the_history_log(project):
    new_asset(project, "crate")
    path = write(project, "crate", "box((4, 4, 4), color=p.wood.mid)")
    build_script(path)
    build_script(path)  # same script again: not a new iteration
    write(project, "crate", "box((4, 3, 4), color=p.wood.mid)")
    build_script(path)
    log = (project / "out" / "crate" / "history.jsonl").read_text().splitlines()
    assert len(log) == 3
    entry = json.loads(log[-1])
    assert {"time", "script_sha256", "voxels", "errors", "warnings", "seed"} <= set(entry)
    assert iterations(project, "crate") == 2


def test_evaluate_summarizes_briefs(project):
    for name in ("crate", "bush", "hut"):
        new_asset(project, name, category="building" if name == "hut" else "prop")
    build_script(write(project, "crate", "box((4, 4, 4), color=p.wood.mid)"))
    build_script(write(project, "bush", "box((5, 4, 4), color=p.leaf)"))  # out of bounds
    (project / "eval").mkdir()
    (project / "eval" / "scores.toml").write_text("crate = 4\nbush = 2\n")
    report = evaluate(project)
    rows = {r["name"]: r for r in report["assets"]}
    assert rows["crate"]["lint_ok"] is True and rows["crate"]["iterations"] == 1
    assert rows["crate"]["accepted"] is True
    assert rows["bush"]["lint_ok"] is False and rows["bush"]["accepted"] is False
    assert rows["hut"]["built"] is False
    s = report["summary"]
    assert s["briefs"] == 3 and s["built"] == 2
    assert s["lint_pass_rate"] == pytest.approx(1 / 3)
    assert s["by_category"]["prop"]["accepted"] == 1
    assert s["mean_score"] == pytest.approx(3.0)


def test_cli_eval_writes_report_gallery_and_score_template(project, capsys):
    new_asset(project, "crate")
    build_script(write(project, "crate", "box((4, 4, 4), color=p.wood.mid)"))
    capsys.readouterr()
    assert main(["eval", "--project", str(project)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["built"] == 1
    assert (project / "eval" / "report.md").is_file()
    assert (project / "eval" / "gallery.png").is_file()
    scores = (project / "eval" / "scores.toml").read_text()
    assert "crate = 0" in scores
    assert "| crate |" in (project / "eval" / "report.md").read_text()
