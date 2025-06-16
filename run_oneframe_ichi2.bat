@echo off

call environment.bat

cd %~dp0webui

"%DIR%\python\python.exe" oneframe_ichi2.py --server 127.0.0.1 --inbrowser

:done
pause