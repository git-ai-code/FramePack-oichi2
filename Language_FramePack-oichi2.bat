@echo off
:SEL
cls
echo.
echo ============================================================
echo          FramePack-oichi2 Language Selection
echo ============================================================
echo.
echo  1  Japan (ja)----------------------FramePack-oichi2
echo  2  English (en)--------------------FramePack-oichi2
echo  3  Traditional Chinese (zh-tw)-----FramePack-oichi2
echo  4  Russian (ru)--------------------FramePack-oichi2
echo.
echo  99 Go to Official FramePack
echo  00 Go to Official FramePack-oichi2
echo.
set /p Type=Please select language (number):
if "%Type%"=="1" goto JP-1
if "%Type%"=="2" goto EN-2
if "%Type%"=="3" goto TW-3
if "%Type%"=="4" goto RU-4
if "%Type%"=="99" goto FPO
if "%Type%"=="00" goto FPOO
if "%Type%"=="" goto PP

:JP-1
cls
@echo FramePack-oichi2 Japan Language...
call environment.bat
cd /d %~dp0webui
set HF_HOME=%~dp0webui\hf_download
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"%DIR%\python\python.exe" oneframe_ichi2.py --server 127.0.0.1 --inbrowser
goto PP

:EN-2
cls
@echo FramePack-oichi2 English Language...
call environment.bat
cd /d %~dp0webui
set HF_HOME=%~dp0webui\hf_download
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"%DIR%\python\python.exe" oneframe_ichi2.py --server 127.0.0.1 --lang en --inbrowser
goto PP

:TW-3
cls
@echo FramePack-oichi2 Traditional Chinese Language...
call environment.bat
cd /d %~dp0webui
set HF_HOME=%~dp0webui\hf_download
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"%DIR%\python\python.exe" oneframe_ichi2.py --server 127.0.0.1 --lang zh-tw --inbrowser
goto PP

:RU-4
cls
@echo FramePack-oichi2 Russian Language...
call environment.bat
cd /d %~dp0webui
set HF_HOME=%~dp0webui\hf_download
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
"%DIR%\python\python.exe" oneframe_ichi2.py --server 127.0.0.1 --lang ru --inbrowser
goto PP

:FPO
cls
@echo Go to FramePack Official...
Start https://github.com/lllyasviel/FramePack
goto PP

:FPOO
cls
@echo Go to FramePack-oichi2 Official...
Start https://github.com/git-ai-code/FramePack-oichi2
goto PP

:PP
pause