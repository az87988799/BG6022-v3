"""Small Windows Job Object wrapper used by the ORCA runner."""

from __future__ import annotations

import ctypes
import os
import platform
from ctypes import wintypes

CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
THREAD_SUSPEND_RESUME = 0x0002
TH32CS_SNAPTHREAD = 0x00000004


class WindowsJobObject:
    """Own a process tree and enforce a total job memory limit."""

    def __init__(self, *, memory_limit_bytes: int) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Objects are only available on Windows")
        if type(memory_limit_bytes) is not int or memory_limit_bytes <= 0:
            raise ValueError("job memory limit must be positive")
        self.memory_limit_bytes = memory_limit_bytes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = self._kernel32.CreateJobObjectW
        create.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        create.restype = wintypes.HANDLE
        self._handle = create(None, None)
        if not self._handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        try:
            self._configure_limits()
        except Exception:
            self.close()
            raise

    def _configure_limits(self) -> None:
        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_JOB_MEMORY
        )
        info.JobMemoryLimit = self.memory_limit_bytes
        set_information = self._kernel32.SetInformationJobObject
        set_information.argtypes = [wintypes.HANDLE, wintypes.INT, ctypes.c_void_p, wintypes.DWORD]
        set_information.restype = wintypes.BOOL
        if not set_information(
            self._handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    def assign_pid(self, pid: int) -> None:
        if type(pid) is not int or pid <= 0:
            raise ValueError("process PID must be positive")
        open_process = self._kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        process = open_process(
            PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not process:
            raise OSError(ctypes.get_last_error(), "OpenProcess failed")
        try:
            assign = self._kernel32.AssignProcessToJobObject
            assign.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            assign.restype = wintypes.BOOL
            if not assign(self._handle, process):
                raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        finally:
            self._close_handle(process)

    def resume_pid(self, pid: int) -> None:
        """Resume the suspended child only after it has joined this Job Object."""

        class ThreadEntry(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG),
                ("tpDeltaPri", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
            ]

        api = self._kernel32
        snapshot = api.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        invalid = ctypes.c_void_p(-1).value
        if snapshot == invalid:
            raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
        api.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
        api.Thread32First.restype = wintypes.BOOL
        api.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
        api.Thread32Next.restype = wintypes.BOOL
        api.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenThread.restype = wintypes.HANDLE
        api.ResumeThread.argtypes = [wintypes.HANDLE]
        api.ResumeThread.restype = wintypes.DWORD
        resumed = 0
        try:
            entry = ThreadEntry()
            entry.dwSize = ctypes.sizeof(entry)
            present = api.Thread32First(snapshot, ctypes.byref(entry))
            while present:
                if entry.th32OwnerProcessID == pid:
                    thread = api.OpenThread(THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                    if not thread:
                        raise OSError(ctypes.get_last_error(), "OpenThread failed")
                    try:
                        if api.ResumeThread(thread) != 1:
                            raise OSError("child initial thread was not suspended exactly once")
                        resumed += 1
                    finally:
                        self._close_handle(thread)
                present = api.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            self._close_handle(snapshot)
        if resumed != 1:
            raise OSError(f"expected one suspended initial thread, resumed {resumed}")

    def terminate(self, exit_code: int = 1) -> None:
        if not self._handle:
            raise OSError("Job Object is closed")
        terminate = self._kernel32.TerminateJobObject
        terminate.argtypes = [wintypes.HANDLE, wintypes.UINT]
        terminate.restype = wintypes.BOOL
        if not terminate(self._handle, exit_code):
            raise OSError(ctypes.get_last_error(), "TerminateJobObject failed")

    def active_process_count(self) -> int:
        class Accounting(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        info = Accounting()
        query = self._kernel32.QueryInformationJobObject
        query.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        query.restype = wintypes.BOOL
        if not query(
            self._handle,
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        ):
            raise OSError(ctypes.get_last_error(), "QueryInformationJobObject failed")
        return int(info.ActiveProcesses)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._close_handle(self._handle)
            self._handle = None

    def _close_handle(self, handle: wintypes.HANDLE) -> None:
        close_handle = self._kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        close_handle(handle)

    def __enter__(self) -> WindowsJobObject:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def process_start_marker(pid: int) -> float | None:
    """Return a best-effort creation marker so a stale PID is never trusted."""

    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    process = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not process:
        return None
    try:
        get_exit = kernel32.GetExitCodeProcess
        get_exit.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        get_exit.restype = wintypes.BOOL
        code = wintypes.DWORD()
        if not get_exit(process, ctypes.byref(code)) or code.value != 259:
            return None
        get_times = kernel32.GetProcessTimes
        get_times.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        get_times.restype = wintypes.BOOL
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if not get_times(
            process,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return ticks / 10_000_000.0
    finally:
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        close_handle(process)


def host_identity() -> str:
    return f"{platform.node()}:{os.name}"


__all__ = [
    "CREATE_NEW_PROCESS_GROUP",
    "CREATE_SUSPENDED",
    "WindowsJobObject",
    "host_identity",
    "process_start_marker",
]
