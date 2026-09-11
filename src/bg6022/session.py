"""Small JSON/file persistence layer for Runs and their source artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Artifact, Result, Run


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execution_guard_path(data_root: str | Path) -> Path:
    return Path(data_root).resolve() / "execution_guard.json"


def read_execution_guard(data_root: str | Path) -> dict[str, Any] | None:
    path = execution_guard_path(data_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"execution guard exists but cannot be read: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"execution guard is not a JSON object: {path}")
    return payload


def write_execution_guard(data_root: str | Path, payload: dict[str, Any]) -> None:
    """Atomically persist the marker that means execution is not confirmed ended."""

    atomic_write_json(execution_guard_path(data_root), payload)


def clear_execution_guard(data_root: str | Path, execution_id: str) -> bool:
    """Remove only a guard owned by ``execution_id``; return whether it was removed."""

    path = execution_guard_path(data_root)
    if not path.exists():
        return True
    payload = read_execution_guard(data_root)
    if payload is None:
        return True
    owner = payload.get("execution_id")
    if owner != execution_id:
        raise RuntimeError(f"execution guard at {path} belongs to {owner!r}, not {execution_id!r}")
    path.unlink()
    return True


def execution_fingerprint(
    plan: Any,
    resources: dict[str, Any],
    initial_artifacts: list[Artifact],
    *,
    snapshot: dict[str, Any] | None = None,
) -> str:
    """Hash stable execution content, excluding mutable run state and clocks."""

    referenced_ids = {
        reference.artifact_id
        for step in plan.steps
        for reference in step.inputs.values()
        if reference.artifact_id is not None
    }
    direct = [
        {
            "id": artifact.id,
            "artifact_type": artifact.artifact_type,
            "sha256": artifact.sha256,
        }
        for artifact in initial_artifacts
        if artifact.id in referenced_ids
    ]
    if snapshot is not None:
        bound_ids = {
            item.get("id") for item in snapshot.get("bound_inputs", []) if isinstance(item, dict)
        }
        direct.extend(
            {
                "id": artifact.id,
                "artifact_type": artifact.artifact_type,
                "sha256": artifact.sha256,
            }
            for artifact in initial_artifacts
            if artifact.id in bound_ids and artifact.id not in {item["id"] for item in direct}
        )
    payload = {
        "plan": plan.model_dump(mode="json"),
        "resources": {key: resources[key] for key in sorted(resources)},
        "initial_artifacts": sorted(direct, key=lambda item: item["id"]),
    }
    if snapshot is not None:
        payload["accepted_snapshot"] = snapshot
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


def _guard_message(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"execution guard exists at {path}; process cleanup must be explicitly verified"
    if isinstance(payload, dict):
        run_id = payload.get("run_id", "unknown")
        phase = payload.get("phase", "unknown")
        reason = payload.get("reason")
        suffix = f", reason={reason}" if reason else ""
        return (
            f"execution guard exists at {path} (run_id={run_id}, phase={phase}{suffix}); "
            "process cleanup must be explicitly verified"
        )
    return f"execution guard exists at {path}; process cleanup must be explicitly verified"


class RuntimeLock:
    """One-byte cross-process lock for a data root."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).resolve()
        self.path = self.data_root / ".runtime.lock"
        self._handle = None

    def __enter__(self) -> RuntimeLock:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0)
        self._handle.write(b"0")
        self._handle.flush()
        self._handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                f"another real calculation is active for {self.data_root}"
            ) from error
        guard = execution_guard_path(self.data_root)
        if guard.exists():
            message = _guard_message(guard)
            self.__exit__(None, None, None)
            raise RuntimeError(message)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


class ChatSessionLock:
    """Separate entry lock for the single local chat instance."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).resolve()
        self.path = self.data_root / ".chat.lock"
        self._handle = None

    def __enter__(self) -> ChatSessionLock:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0)
        self._handle.write(b"0")
        self._handle.flush()
        self._handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"another chat instance is active for {self.data_root}") from error
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def run_directory(data_root: str | Path, run_id: str) -> Path:
    return Path(data_root).resolve() / "runs" / run_id


def session_directory(data_root: str | Path) -> Path:
    return Path(data_root).resolve() / "sessions"


def session_path(data_root: str | Path, session_id: str) -> Path:
    if not session_id or "/" in session_id or "\\" in session_id or session_id in {".", ".."}:
        raise ValueError("invalid session id")
    return session_directory(data_root) / f"{session_id}.json"


def save_session(data_root: str | Path, session_id: str, payload: dict[str, Any]) -> Path:
    """Persist only bounded conversational context, never full ORCA output."""

    bounded = dict(payload)
    messages = bounded.get("recent_messages")
    if isinstance(messages, list):
        bounded["recent_messages"] = messages[-12:]
    results = bounded.get("recent_results")
    if isinstance(results, list):
        bounded["recent_results"] = results[-3:]
    path = session_path(data_root, session_id)
    atomic_write_json(path, bounded)
    return path


def load_session(data_root: str | Path, session_id: str) -> dict[str, Any]:
    path = session_path(data_root, session_id)
    if not path.exists():
        return {
            "session_id": session_id,
            "recent_messages": [],
            "recent_results": [],
            "active_run_id": None,
            "pending_prompt": None,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"session file is invalid: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"session file is not a JSON object: {path}")
    return payload


def attempt_directory(data_root: str | Path, run_id: str, step_id: str, attempt: int) -> Path:
    if attempt < 1:
        raise ValueError("attempt number must be positive")
    return run_directory(data_root, run_id) / step_id / f"attempt-{attempt:02d}"


def create_run(data_root: str | Path, run: Run) -> Path:
    directory = run_directory(data_root, run.id)
    (directory / "artifacts").mkdir(parents=True, exist_ok=False)
    save_run(data_root, run)
    return directory


def save_run(data_root: str | Path, run: Run) -> None:
    directory = run_directory(data_root, run.id)
    directory.mkdir(parents=True, exist_ok=True)
    # Every durable checkpoint carries the elapsed active interval.  This
    # matters when a Tool saves a prepared/started/finished attempt before the
    # Agent reaches its next explicit checkpoint.
    run.checkpoint_active()
    run.updated_at = utc_now()
    atomic_write_json(directory / "run.json", run.model_dump(mode="json"))


def load_run(data_root: str | Path, run_id: str) -> Run:
    path = run_directory(data_root, run_id) / "run.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Run.model_validate(payload, strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"Run does not exist: {run_id}") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Run file is invalid: {path}: {error}") from error


def save_result(data_root: str | Path, run: Run, result: Result) -> Path:
    directory = run_directory(data_root, run.id) / result.attempt_relative_path
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    atomic_write_json(path, result.model_dump(mode="json"))
    return path


def register_file_artifact(
    data_root: str | Path,
    run: Run,
    source_path: str | Path,
    *,
    artifact_type: str,
    role: str,
    source: str,
    step_id: str | None = None,
    attempt: int | None = None,
    extension: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    source = str(source)
    source_path = Path(source_path)
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError(f"artifact source is not a regular file: {source_path}")
    artifact_id = new_id("artifact")
    suffix = extension if extension is not None else source_path.suffix
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    destination = run_directory(data_root, run.id) / "artifacts" / f"{artifact_id}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination.resolve():
        shutil.copyfile(source_path, destination)
    artifact = Artifact(
        id=artifact_id,
        artifact_type=artifact_type,
        role=role,
        relative_path=destination.relative_to(run_directory(data_root, run.id)).as_posix(),
        size_bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        source=source,
        run_id=run.id,
        step_id=step_id,
        attempt=attempt,
        metadata=metadata or {},
    )
    run.artifact_index.append(artifact)
    return artifact


def register_bytes_artifact(
    data_root: str | Path,
    run: Run,
    content: bytes,
    *,
    artifact_type: str,
    role: str,
    source: str,
    extension: str = ".bin",
    step_id: str | None = None,
    attempt: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    artifact_id = new_id("artifact")
    if not extension.startswith("."):
        extension = "." + extension
    destination = run_directory(data_root, run.id) / "artifacts" / f"{artifact_id}{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, content)
    artifact = Artifact(
        id=artifact_id,
        artifact_type=artifact_type,
        role=role,
        relative_path=destination.relative_to(run_directory(data_root, run.id)).as_posix(),
        size_bytes=len(content),
        sha256=sha256_bytes(content),
        source=source,
        run_id=run.id,
        step_id=step_id,
        attempt=attempt,
        metadata=metadata or {},
    )
    run.artifact_index.append(artifact)
    return artifact


def artifact_path(data_root: str | Path, run: Run, artifact: Artifact) -> Path:
    root = run_directory(data_root, run.id).resolve()
    path = (root / artifact.relative_path).resolve()
    if root not in path.parents:
        raise ValueError("artifact path escapes the Run directory")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"artifact file is missing: {path}")
    if path.stat().st_size != artifact.size_bytes or sha256_file(path) != artifact.sha256:
        raise ValueError(f"artifact hash or size mismatch: {artifact.id}")
    return path


def find_artifact(run: Run, artifact_id: str) -> Artifact:
    for artifact in run.artifact_index:
        if artifact.id == artifact_id:
            return artifact
    raise ValueError(f"Run artifact is not registered: {artifact_id}")


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


__all__ = [
    "ChatSessionLock",
    "RuntimeLock",
    "artifact_path",
    "attempt_directory",
    "atomic_write_bytes",
    "atomic_write_json",
    "clear_execution_guard",
    "create_run",
    "execution_fingerprint",
    "execution_guard_path",
    "find_artifact",
    "load_run",
    "new_id",
    "read_execution_guard",
    "register_bytes_artifact",
    "register_file_artifact",
    "run_directory",
    "load_session",
    "save_session",
    "session_directory",
    "session_path",
    "save_result",
    "save_run",
    "sha256_bytes",
    "sha256_file",
    "utc_now",
    "write_execution_guard",
]
