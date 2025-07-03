@echo off

echo [oichi2] FramePack-oichi2 Starting (Traditional Chinese)...

call environment.bat

cd /d %~dp0webui

REM HF_HOME setting
set HF_HOME=%~dp0webui\hf_download
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo [oichi2] Starting oneframe_ichi2.py (Traditional Chinese)...
"%DIR%\python\python.exe" oneframe_ichi2.py --server 127.0.0.1 --lang zh-tw --inbrowser

:done
pause