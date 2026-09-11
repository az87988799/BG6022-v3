@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo BG6022-v3 Python environment is missing. Run: uv sync --group dev
  exit /b 2
)
".venv\Scripts\python.exe" start_chat.py
exit /b %errorlevel%
