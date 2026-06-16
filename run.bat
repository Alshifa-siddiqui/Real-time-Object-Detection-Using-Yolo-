@echo off
REM Convenience launcher for YOLOv8 detection.
REM Usage examples (run from anywhere):
REM   run                 -> webcam (camera 0)
REM   run bus.jpg --save  -> detect on an image and save
REM   run video.mp4 --save
REM   run 0 --conf 0.4
REM
REM Any arguments are passed straight through to main.py. If the first
REM argument doesn't start with "--", it's treated as the --source value.

setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [run.bat] Could not find %PY%
  echo [run.bat] Create the venv first:  py -3.12 -m venv .venv
  exit /b 1
)

REM If a source was given as the first arg (not a flag), prepend --source.
set "FIRST=%~1"
if "%FIRST%"=="" (
  "%PY%" yolov8_test\main.py
) else (
  set "LEAD=%FIRST:~0,2%"
  if "%FIRST:~0,2%"=="--" (
    "%PY%" yolov8_test\main.py %*
  ) else (
    "%PY%" yolov8_test\main.py --source %*
  )
)

endlocal
