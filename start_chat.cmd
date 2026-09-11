@echo off
setlocal
cd /d "%~dp0"
if not exist ".env" (
  echo .env is missing. Copy .env.example to .env and set DEEPSEEK_API_KEY.
  exit /b 2
)
if not exist ".venv\Scripts\uv.exe" (
  echo uv is missing. Run: uv sync --group dev
  exit /b 2
)
".venv\Scripts\uv.exe" run --frozen --env-file ".env" python start_chat.py
exit /b %errorlevel%
