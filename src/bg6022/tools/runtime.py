"""Transient context for one Tool invocation; never persisted as a domain object."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

from bg6022.models import Artifact, Result, Run, Step
from bg6022.session import (
    artifact_path,
    register_bytes_artifact,
    register_file_artifact,
    save_run,
)


@dataclass(slots=True)
class ToolCallContext:
    data_root: Path
    run: Run
    step: Step
    attempt: int
    cancel: Event
    workdir: Path
    relative_attempt_path: str
    frozen_inputs: dict[str, Artifact] = field(default_factory=dict)
    attempt_record: dict[str, Any] = field(default_factory=dict)

    def read_input(self, name: str) -> bytes:
        artifact = self.frozen_inputs.get(name)
        if artifact is None:
            raise ValueError(f"Tool input {name!r} was not frozen for this invocation")
        path = artifact_path(self.data_root, self.run, artifact)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError(f"Tool input {name!r} changed after it was frozen")
        return payload

    def register_bytes(
        self,
        content: bytes,
        *,
        artifact_type: str,
        role: str,
        source: str,
        extension: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        return register_bytes_artifact(
            self.data_root,
            self.run,
            content,
            artifact_type=artifact_type,
            role=role,
            source=source,
            extension=extension or ".bin",
            step_id=self.step.id,
            attempt=self.attempt,
            metadata=metadata,
        )

    def register_file(
        self,
        source_path: str | Path,
        *,
        artifact_type: str,
        role: str,
        source: str,
        extension: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        return register_file_artifact(
            self.data_root,
            self.run,
            source_path,
            artifact_type=artifact_type,
            role=role,
            source=source,
            step_id=self.step.id,
            attempt=self.attempt,
            extension=extension,
            metadata=metadata,
        )

    def make_result(self, status: str, **values: Any) -> Result:
        values.setdefault("run_id", self.run.id)
        values.setdefault("step_id", self.step.id)
        values.setdefault("attempt", self.attempt)
        values.setdefault("attempt_relative_path", self.relative_attempt_path)
        values.setdefault(
            "input_bindings", {name: artifact.id for name, artifact in self.frozen_inputs.items()}
        )
        values.setdefault(
            "input_artifact_ids",
            list(dict.fromkeys(artifact.id for artifact in self.frozen_inputs.values())),
        )
        values["status"] = status
        return Result(**values)

    def update_attempt(self, *, checkpoint: bool = True, **updates: Any) -> None:
        self.attempt_record.update(updates)
        if checkpoint:
            self.checkpoint()

    def checkpoint(self) -> None:
        save_run(self.data_root, self.run)


__all__ = ["ToolCallContext"]
