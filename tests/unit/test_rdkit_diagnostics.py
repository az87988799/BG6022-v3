from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Event

from tool_context import make_tool_context

from bg6022.config import load_config
from bg6022.diagnostics import configure_rdkit_logging
from bg6022.models import InputReference, Plan, Request, Run, Step
from bg6022.session import create_run, register_bytes_artifact, run_directory, utc_now
from bg6022.tools.molecule import _run_embedding_helper, execute_generate_geometry


def _config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        f"""[orca]
executable = '{(tmp_path / "missing-orca.exe").as_posix()}'

[runtime]
data_root = 'data'
semantic_planner_v1 = false
""",
        encoding="utf-8",
    )
    return load_config(path)


def test_rdkit_logging_filters_known_console_warning_but_keeps_file_diagnostics(
    tmp_path: Path, capsys
) -> None:
    logger = logging.getLogger("rdkit")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    for handler in original_handlers:
        logger.removeHandler(handler)
    try:
        configure_rdkit_logging(tmp_path / "data")
        logger.warning("WARNING: not removing hydrogen atom without neighbors")
        logger.error("visible-rdkit-error")
        for handler in logger.handlers:
            handler.flush()
        captured = capsys.readouterr()
        log_text = (tmp_path / "data" / "logs" / "rdkit.log").read_text(encoding="utf-8")
        assert "not removing hydrogen atom without neighbors" not in captured.err
        assert "visible-rdkit-error" in captured.err
        assert "not removing hydrogen atom without neighbors" in log_text
        assert "visible-rdkit-error" in log_text

        handler_count = len(logger.handlers)
        configure_rdkit_logging(tmp_path / "data")
        assert len(logger.handlers) == handler_count
    finally:
        for handler in tuple(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            logger.addHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_rdkit_helper_preserves_stderr_and_geometry_tool_writes_raw_diagnostic(
    tmp_path: Path,
) -> None:
    embedded = _run_embedding_helper(
        "[HH]",
        [61453],
        timeout_seconds=30,
        cancel=Event(),
    )
    assert embedded["symbols"] == ["H", "H"]
    assert "not removing hydrogen atom without neighbors" in embedded["rdkit_stderr"]

    config = _config(tmp_path)
    request = Request(id="request_rdkit", description="diagnostics", source="chat")
    step = Step(
        id="geometry",
        tool="generate_geometry",
        inputs={"molecule": InputReference(artifact_id="molecule_artifact")},
    )
    run = Run(
        id="run_rdkit",
        request=request,
        plan=Plan(id="plan_rdkit", request_id=request.id, steps=[step]),
        resources=config.resources,
        status="running",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    create_run(config.data_root_path, run)
    payload = {
        "facts": {
            "canonical_smiles": "[HH]",
            "isomeric_smiles": "[HH]",
            "element_counts": {"H": 2},
        },
        "source": "test:rdkit",
    }
    molecule = register_bytes_artifact(
        config.data_root_path,
        run,
        json.dumps(payload).encode("utf-8"),
        artifact_type="molecule",
        role="resolved_molecule",
        source="test:rdkit",
        extension=".json",
    )
    step.inputs["molecule"] = InputReference(artifact_id=molecule.id)

    result = execute_generate_geometry(
        config, step=step, context=make_tool_context(config, run, step)
    )

    assert result.status == "succeeded"
    raw_path = Path(result.diagnostics["raw_paths"]["rdkit_stderr"])
    assert (
        raw_path
        == run_directory(config.data_root_path, run.id) / "geometry/attempt-01/rdkit.stderr.log"
    )
    assert "not removing hydrogen atom without neighbors" in raw_path.read_text(encoding="utf-8")
