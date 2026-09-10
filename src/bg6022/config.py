"""Strict TOML configuration and local resource checks."""

from __future__ import annotations

import ctypes
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OrcaSettings(StrictConfig):
    executable: StrictStr


class RuntimeSettings(StrictConfig):
    data_root: StrictStr
    cores: StrictInt = Field(default=4, gt=0)
    memory_mb: StrictInt = Field(default=2048, gt=0)
    maxcore_mb: StrictInt = Field(default=384, gt=0)
    max_concurrent_jobs: StrictInt = Field(default=1, ge=1, le=1)
    confirm_before_compute: StrictBool = True
    attempt_timeout_seconds: StrictInt = Field(default=1200, gt=0)
    run_active_timeout_seconds: StrictInt = Field(default=3600, gt=0)
    output_limit_mb: StrictInt = Field(default=64, gt=0)
    workdir_limit_mb: StrictInt = Field(default=512, gt=0)


class DefaultSettings(StrictConfig):
    method_profile: StrictStr = "r2scan3c"
    environment: StrictStr = "gas"


class AppConfig(StrictConfig):
    orca: OrcaSettings
    runtime: RuntimeSettings
    defaults: DefaultSettings
    config_path: str = Field(exclude=True)
    executable_path: str = Field(exclude=True)
    data_root_path: str = Field(exclude=True)

    @property
    def output_limit_bytes(self) -> int:
        return self.runtime.output_limit_mb * 1024 * 1024

    @property
    def workdir_limit_bytes(self) -> int:
        return self.runtime.workdir_limit_mb * 1024 * 1024

    @property
    def resources(self) -> dict[str, Any]:
        return {
            "cores": self.runtime.cores,
            "memory_mb": self.runtime.memory_mb,
            "maxcore_mb": self.runtime.maxcore_mb,
            "max_concurrent_jobs": self.runtime.max_concurrent_jobs,
            "attempt_timeout_seconds": self.runtime.attempt_timeout_seconds,
            "run_active_timeout_seconds": self.runtime.run_active_timeout_seconds,
            "output_limit_bytes": self.output_limit_bytes,
            "workdir_limit_bytes": self.workdir_limit_bytes,
        }


def _resolve_path(raw: str, base: Path) -> Path:
    value = Path(raw).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def _available_physical_memory_bytes() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullAvailPhys)
    return None


def validate_runtime_capacity(config: AppConfig) -> None:
    runtime = config.runtime
    if runtime.cores * runtime.maxcore_mb > (runtime.memory_mb * 75) // 100:
        raise ValueError("cores * maxcore_mb must not exceed 75% of memory_mb")
    available = _available_physical_memory_bytes()
    if available is not None and available < runtime.memory_mb * 1024 * 1024:
        raise ValueError(
            f"available physical memory ({available} bytes) is below the configured "
            f"budget ({runtime.memory_mb * 1024 * 1024} bytes)"
        )


def load_config(path: str | Path, *, create_data_root: bool = True) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML in {config_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a TOML table")
    unknown_sections = set(raw) - {"orca", "runtime", "defaults"}
    if unknown_sections:
        raise ValueError(f"unknown configuration section(s): {sorted(unknown_sections)}")
    base = config_path.parent
    try:
        orca_raw = dict(raw.get("orca", {}))
        runtime_raw = dict(raw.get("runtime", {}))
        orca_raw["executable"] = str(_resolve_path(orca_raw["executable"], base))
        runtime_raw["data_root"] = str(_resolve_path(runtime_raw["data_root"], base))
        model = AppConfig.model_validate(
            {
                "orca": orca_raw,
                "runtime": runtime_raw,
                "defaults": raw.get("defaults", {}),
                "config_path": str(config_path),
                "executable_path": orca_raw["executable"],
                "data_root_path": runtime_raw["data_root"],
            },
            strict=True,
        )
    except KeyError as error:
        raise ValueError(f"configuration is missing {error.args[0]!r}") from error
    except Exception as error:
        raise ValueError(f"invalid configuration: {error}") from error
    executable = Path(model.executable_path)
    if executable.suffix.casefold() != ".exe" or not executable.is_file():
        raise ValueError(f"ORCA executable is not an existing .exe file: {executable}")
    data_root = Path(model.data_root_path)
    if create_data_root:
        data_root.mkdir(parents=True, exist_ok=True)
    if not data_root.is_dir():
        raise ValueError(f"data_root is not a directory: {data_root}")
    try:
        probe = data_root / ".bg6022-write-probe"
        probe.write_bytes(b"probe")
        probe.unlink()
    except OSError as error:
        raise ValueError(f"data_root is not writable: {data_root}: {error}") from error
    if model.defaults.method_profile != "r2scan3c":
        raise ValueError(f"unsupported default method profile: {model.defaults.method_profile}")
    if model.defaults.environment != "gas":
        raise ValueError(f"unsupported default environment: {model.defaults.environment}")
    validate_runtime_capacity(model)
    return model


def environment_for_child() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    return env


def runtime_summary(config: AppConfig) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "executable": config.executable_path,
        "data_root": config.data_root_path,
        **config.resources,
    }


__all__ = [
    "AppConfig",
    "DefaultSettings",
    "OrcaSettings",
    "RuntimeSettings",
    "environment_for_child",
    "load_config",
    "runtime_summary",
    "validate_runtime_capacity",
]
