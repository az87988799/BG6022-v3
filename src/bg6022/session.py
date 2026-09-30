"""Small JSON/file persistence layer for Runs and their source artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from .models import Artifact, Result, Run, Step, Tool
from .output_contracts import is_compatible_value

# Retain the active Run plus up to five distinct earlier Runs.
MAX_RECENT_RUNS = 6


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


MAX_METADATA_FILE_BYTES = 4 * 1024 * 1024
MAX_METADATA_TURN_BYTES = 16 * 1024 * 1024
MAX_HISTORY_INDEX_BYTES = 8 * 1024 * 1024
MAX_PENDING_QUERY_BYTES = 8192


def _valid_pending_query(value, session_id):
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    question, evidence = value.get("origin_question"), value.get("origin_evidence")
    sources = value.get("source_bindings")
    return (
        value.get("schema") == "bg6022.pending_query.v1"
        and value.get("session_id") == session_id
        and isinstance(value.get("pending_id"), str)
        and 1 <= len(value["pending_id"]) <= 80
        and isinstance(question, str)
        and 1 <= len(question) <= 240
        and isinstance(evidence, str)
        and evidence.strip()
        and question.count(evidence) == 1
        and isinstance(sources, list)
        and len(sources) <= 3
        and all(isinstance(s, dict) and s.get("session_id") == session_id for s in sources)
        and isinstance(value.get("missing_slots"), list)
        and len(value["missing_slots"]) <= 3
        and all(
            isinstance(s, str) and s in {"source", "property", "mode"}
            for s in value["missing_slots"]
        )
        and isinstance(value.get("resolved_slots", {}), dict)
        and set(value.get("resolved_slots", {})) <= {"mode"}
        and value.get("resolved_slots", {}).get("mode", "read_existing") == "read_existing"
        and all(
            isinstance(value.get(key, []), list)
            and len(value.get(key, [])) <= bound
            and all(isinstance(v, str) and 1 <= len(v) <= 80 for v in value.get(key, []))
            for key, bound in (("property_candidates", 6), ("search_terms", 4))
        )
        and (
            value.get("property_hint") is None
            or (isinstance(value["property_hint"], str) and 1 <= len(value["property_hint"]) <= 80)
        )
        and len(json.dumps(value, ensure_ascii=False).encode()) <= MAX_PENDING_QUERY_BYTES
    )


def new_metadata_budget(cancel=None):
    return {
        "cancel": cancel or Event(),
        "deadline": time.monotonic() + 2.0,
        "bytes": 0,
        "candidates": 0,
        "cache": {},
    }


def read_metadata_json(path, budget, *, max_file_bytes=MAX_METADATA_FILE_BYTES):
    try:
        return _read_metadata_json(path, budget, max_file_bytes=max_file_bytes)
    except ArtifactReadError as error:
        budget["diagnostic"] = error.category
        raise


def _read_metadata_json(path, budget, *, max_file_bytes):
    """Bound every metadata read before decoding, with a shared turn allowance."""
    check_read_deadline(budget["cancel"], budget["deadline"])
    path = Path(path)
    key = str(path.resolve())
    if key in budget["cache"]:
        return budget["cache"][key]
    if budget["candidates"] >= 64:
        raise ArtifactReadError("metadata_candidates")
    budget["candidates"] += 1
    size = path.stat().st_size
    if size > max_file_bytes:
        raise ArtifactReadError("metadata_file_bytes")
    if size + budget["bytes"] > MAX_METADATA_TURN_BYTES:
        raise ArtifactReadError("metadata_turn_bytes")
    chunks = []
    read = 0
    with path.open("rb") as handle:
        while True:
            check_read_deadline(budget["cancel"], budget["deadline"])
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            read += len(chunk)
            budget["bytes"] += len(chunk)
            if read > max_file_bytes or budget["bytes"] > MAX_METADATA_TURN_BYTES:
                raise ArtifactReadError("metadata_turn_bytes")
            chunks.append(chunk)
    check_read_deadline(budget["cancel"], budget["deadline"])
    value = json.loads(b"".join(chunks))
    check_read_deadline(budget["cancel"], budget["deadline"])
    budget["cache"][key] = value
    return value


def index_session_run(session, run):
    if run.session_id not in {None, session.get("session_id")}:
        return
    entries = session.setdefault("run_history", [])
    item = {
        "run_id": run.id,
        "created_at": str(run.created_at),
        "description": run.request.description[:240],
    }
    previous = next((i for i, entry in enumerate(entries) if entry.get("run_id") == run.id), None)
    candidate = list(entries)
    if previous is None:
        candidate.append(item)
    else:
        candidate[previous] = item
    if len(json.dumps(candidate, ensure_ascii=False).encode("utf-8")) > MAX_HISTORY_INDEX_BYTES:
        session["history_diagnostic"] = "history_index_bytes"
        return
    session["run_history"] = candidate


def list_session_run_summaries(data_root, session, *, cursor=None, limit=24, cancel=None):
    """Browse a session directory, independently of its conversational window."""
    if type(limit) is not int or not 1 <= limit <= 24:
        raise ValueError("directory page limit must be between 1 and 24")
    cursors = session.setdefault("history_cursors", {})
    if cursor is not None and cursor not in cursors:
        raise ValueError("directory cursor was not issued by this session")
    state = dict(cursors[cursor]) if cursor else {"offset": 0, "scanned": []}
    budget = new_metadata_budget(cancel)
    diagnostic = None
    unavailable = []
    overflow = list(state.get("pending", []))
    history = session.setdefault("run_history", [])
    indexed = {item.get("run_id") for item in history}
    indexed.update(item.get("run_id") for item in session.get("recent_results", []))
    if session.get("active_run_id"):
        indexed.add(session["active_run_id"])
    scanned = set(state.get("scanned", []))
    try:
        check_read_deadline(budget["cancel"], budget["deadline"])
        root = Path(data_root).resolve() / "runs"
        if root.exists():
            with os.scandir(root) as directories:
                for entry in directories:
                    check_read_deadline(budget["cancel"], budget["deadline"])
                    if entry.name in scanned:
                        continue
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    try:
                        run = load_run(data_root, entry.name, metadata_budget=budget)
                        if run.session_id == session.get("session_id") or (
                            run.session_id is None and run.id in indexed
                        ):
                            index_session_run(session, run)
                            if not any(item["run_id"] == run.id for item in session["run_history"]):
                                overflow.append(
                                    {
                                        "run_id": run.id,
                                        "created_at": str(run.created_at),
                                        "description": run.request.description[:240],
                                    }
                                )
                    except ArtifactReadError as error:
                        if error.category == "metadata_file_bytes":
                            unavailable.append(
                                "metadata_file_bytes: controlled large-record reading required"
                            )
                        else:
                            raise
                    except (OSError, ValueError):
                        pass
                    scanned.add(entry.name)
    except ArtifactReadError as error:
        diagnostic = error.category
    history = session.get("run_history", []) + overflow
    offset = state["offset"]
    items = history[offset : offset + limit] if diagnostic != "cancelled" else []
    next_offset = offset + len(items)
    more = next_offset < len(history) or diagnostic is not None
    next_cursor = None
    if more:
        next_cursor = uuid.uuid4().hex
        cursors[next_cursor] = {
            "offset": next_offset,
            "scanned": sorted(scanned),
            "pending": overflow,
        }
        # Cursors are capabilities issued for this directory, never paths.
        while len(cursors) > 64:
            del cursors[next(iter(cursors))]
    return {
        "items": items,
        "cursor": next_cursor,
        "complete": not more,
        "status": diagnostic or ("limited" if unavailable else "ok"),
        "unavailable": unavailable,
        "metadata_bytes": budget["bytes"],
        "metadata_candidates": budget["candidates"],
        "indexed_count": len(history),
    }


def save_session(data_root: str | Path, session_id: str, payload: dict[str, Any]) -> Path:
    """Persist only bounded conversational context, never full ORCA output."""

    bounded = dict(payload)
    if not _valid_pending_query(bounded.get("pending_query"), session_id):
        raise ArtifactReadError("pending_query_invalid")
    messages = bounded.get("recent_messages")
    if isinstance(messages, list):
        bounded["recent_messages"] = messages[-12:]
    results = bounded.get("recent_results")
    if isinstance(results, list):
        recent_by_run: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            run_id = item.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                continue
            recent_by_run = [entry for entry in recent_by_run if entry.get("run_id") != run_id]
            recent_by_run.append(item)
        bounded["recent_results"] = recent_by_run[-MAX_RECENT_RUNS:]
    delivery = bounded.get("last_delivery")
    if isinstance(delivery, list):
        bounded["last_delivery"] = [dict(item) for item in delivery[-8:] if isinstance(item, dict)]
    if len(json.dumps(bounded, ensure_ascii=False).encode("utf-8")) > MAX_HISTORY_INDEX_BYTES:
        raise ArtifactReadError("session_index_bytes")
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
            "run_history": [],
            "last_delivery": [],
            "active_run_id": None,
            "pending_prompt": None,
        }
    try:
        payload = read_metadata_json(
            path, new_metadata_budget(), max_file_bytes=MAX_HISTORY_INDEX_BYTES
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"session file is invalid: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"session file is not a JSON object: {path}")
    payload.setdefault("run_history", [])
    if not _valid_pending_query(payload.get("pending_query"), session_id):
        payload.pop("pending_query", None)
        payload["history_diagnostic"] = "pending_query_invalid"
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


def load_run(data_root: str | Path, run_id: str, *, metadata_budget=None) -> Run:
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise ValueError("invalid run id")
    path = run_directory(data_root, run_id) / "run.json"
    try:
        payload = (
            read_metadata_json(path, metadata_budget)
            if metadata_budget is not None
            else json.loads(path.read_text(encoding="utf-8"))
        )
        return Run.model_validate(payload, strict=True)
    except ArtifactReadError:
        raise
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


def publish_step_result(
    data_root: str | Path,
    run: Run,
    step: Step,
    tool: Tool,
    result: Result,
    *,
    expected_input_bindings: dict[str, str],
    expected_input_hashes: dict[str, str],
    parameter_sources: dict[str, str] | None = None,
) -> str:
    """Validate and publish one candidate Result before a downstream Step can run."""

    if result.run_id != run.id or result.step_id != step.id:
        raise ValueError("Tool Result is not bound to the active Run and Step")
    if result.status not in {"succeeded", "failed", "cancelled", "interrupted", "needs_input"}:
        raise ValueError("Tool Result has an unsupported status")
    if result.input_bindings != expected_input_bindings:
        raise ValueError("Tool-reported input bindings differ from the frozen Step inputs")
    if result.input_artifact_ids != list(expected_input_bindings.values()):
        raise ValueError("Tool-reported input Artifact ids differ from its input bindings")
    if set(expected_input_hashes) != set(expected_input_bindings.values()):
        raise ValueError("frozen input hashes do not cover the bound Step inputs")

    run_root = run_directory(data_root, run.id).resolve()
    attempt_root = (run_root / result.attempt_relative_path).resolve()
    if run_root not in attempt_root.parents:
        raise ValueError("Tool Result attempt path escapes the Run directory")
    if attempt_root.name != f"attempt-{result.attempt:02d}":
        raise ValueError("Tool Result attempt directory does not match its attempt number")

    if set(result.values) - set(tool.results):
        raise ValueError("Tool Result contains undeclared values")
    for name, value in result.values.items():
        if not is_compatible_value(value, tool.results[name]):
            raise ValueError(f"Tool Result value {name!r} violates its declared type")
    if set(result.output_ports) - set(tool.output_ports):
        raise ValueError("Tool Result contains undeclared output ports")
    if set(result.scientific_checks) - set(tool.scientific_checks):
        raise ValueError("Tool Result contains undeclared scientific checks")
    known_artifacts = {item.id: item for item in run.artifact_index}
    for artifact_id, expected_hash in expected_input_hashes.items():
        artifact = known_artifacts.get(artifact_id)
        if artifact is None or artifact.sha256 != expected_hash:
            raise ValueError("a frozen input Artifact changed ownership during execution")
        source_path = artifact_path(data_root, run, artifact)
        if sha256_file(source_path) != expected_hash:
            raise ValueError("a frozen input Artifact changed while the Tool was running")
    if set(result.artifact_ids) - set(known_artifacts):
        raise ValueError("Tool Result refers to an Artifact outside the active Run")
    for port, artifact_id in result.output_ports.items():
        artifact = known_artifacts.get(artifact_id)
        if (
            artifact is None
            or artifact_id not in result.artifact_ids
            or artifact.run_id != run.id
            or artifact.step_id != step.id
            or artifact.attempt != result.attempt
            or artifact.artifact_type != tool.output_ports[port]
            or artifact.role == "restart_candidate"
        ):
            raise ValueError(f"Tool output port {port!r} is not bound to a verified Artifact")
        artifact_path(data_root, run, artifact)
    if result.status == "succeeded" and not tool.validate_result(run, step, result):
        raise ValueError("Tool Result does not satisfy its declared verified-result contract")

    result.step_fingerprint = _step_fingerprint_for_publish(step)
    result.parameter_sources = dict(parameter_sources or result.parameter_sources)
    relative = f"{result.attempt_relative_path}/result.json"
    snapshot = run.model_copy(deep=True)
    try:
        save_result(data_root, run, result)
        if relative not in run.result_index:
            run.result_index.append(relative)
        run.step_status[step.id] = result.status
        if result.status == "succeeded":
            run.current_results[step.id] = relative
        save_run(data_root, run)
    except Exception:
        for name in Run.model_fields:
            setattr(run, name, getattr(snapshot, name))
        raise
    return relative


def _step_fingerprint_for_publish(step: Step) -> str:
    payload = json.dumps(
        step.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


class ArtifactReadError(ValueError):
    """A bounded read failed before verified bytes could be delivered."""

    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


def check_read_deadline(cancel: Event, deadline: float) -> None:
    if cancel.is_set():
        raise ArtifactReadError("cancelled")
    if time.monotonic() >= deadline:
        raise ArtifactReadError("deadline")


def registered_read_path(data_root: str | Path, run: Run, relative: str) -> Path:
    """Check unresolved components, including Windows junctions, before opening."""
    root = Path(data_root).absolute()
    rel = Path(relative)
    if rel.is_absolute() or rel.drive or any(p in {"..", "."} or ":" in p for p in rel.parts):
        raise ArtifactReadError("unsafe_path")
    if not run.id or run.id in {".", ".."} or Path(run.id).name != run.id or ":" in run.id:
        raise ArtifactReadError("unsafe_path")
    path = root / "runs" / run.id / rel
    try:
        for component in [*reversed(path.parents), path]:
            info = component.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ArtifactReadError("unsafe_path")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ArtifactReadError("unsafe_path")
    except OSError as error:
        raise ArtifactReadError("missing_file") from error
    return path


def read_registered_artifact_bytes(
    data_root: str | Path,
    run: Run,
    artifact: Artifact,
    *,
    max_bytes: int,
    deadline: float,
    cancel: Event,
) -> bytes:
    """One bounded read/hash pass; consumers use only these verified bytes."""
    check_read_deadline(cancel, deadline)
    try:
        registered = find_artifact(run, artifact.id)
    except ValueError as error:
        raise ArtifactReadError("invalid_source") from error
    if registered != artifact or artifact.run_id != run.id:
        raise ArtifactReadError("invalid_source")
    if artifact.size_bytes > max_bytes:
        raise ArtifactReadError("byte_limit")
    path = registered_read_path(data_root, run, artifact.relative_path)
    before = path.stat()
    if before.st_size > max_bytes:
        raise ArtifactReadError("byte_limit")
    if before.st_size != artifact.size_bytes:
        raise ArtifactReadError("size_mismatch")
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
                raise ArtifactReadError("changed_file")
            if opened.st_size != artifact.size_bytes:
                raise ArtifactReadError("size_mismatch")
            while len(payload) < artifact.size_bytes:
                check_read_deadline(cancel, deadline)
                chunk = handle.read(min(65536, artifact.size_bytes - len(payload)))
                check_read_deadline(cancel, deadline)
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > artifact.size_bytes or len(payload) > max_bytes:
                    raise ArtifactReadError("byte_limit")
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ArtifactReadError("changed_file")
            checked = registered_read_path(data_root, run, artifact.relative_path)
            if not os.path.samestat(after, checked.stat()):
                raise ArtifactReadError("changed_file")
    except OSError as error:
        raise ArtifactReadError("missing_file") from error
    if len(payload) != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
        raise ArtifactReadError("hash_mismatch")
    check_read_deadline(cancel, deadline)
    return bytes(payload)


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
    "MAX_RECENT_RUNS",
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
