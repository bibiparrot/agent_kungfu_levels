@echo off
setlocal
pushd "%~dp0"
set "PYTHONPATH=%CD%;%CD%\src;%PYTHONPATH%"
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" serve_demos.py %*
set "DEMO_EXIT=%ERRORLEVEL%"
if not "%DEMO_EXIT%"=="0" (
  echo.
  echo Marimo failed to start. Install the project with:
  echo   python -m pip install -e ".[harness,eval,dev]"
)
popd
endlocal & exit /b %DEMO_EXIT%
