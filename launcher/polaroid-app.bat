@echo off
setlocal EnableExtensions
rem ---------------------------------------------------------------------------
rem  Polaroid processor - local web app for Windows
rem
rem  Double-click to start the app: it opens in your browser. Scan folders
rem  dragged onto this file become projects in the app (the first one opens);
rem  each is processed into a sibling folder "<folder>_processed". The scans
rem  themselves are never changed. If the app is already running, the browser
rem  opens on that one. Close this window to stop the app.
rem  A folder name with an ampersand must reach this file as one quoted name;
rem  if Explorer does not quote it, the name is cut there. Start such a
rem  folder from a terminal with polaroid app and the quoted path.
rem
rem  Settings: edit the lines below once, or set the same names as Windows
rem  environment variables.
rem    POLAROID_EXE        full path to polaroid.exe (optional)
rem    POLAROID_CONDA_ENV  conda environment that has polaroid-processor installed
rem    POLAROID_APP_ARGS   extra options for "polaroid app", e.g. --port 8780
rem ---------------------------------------------------------------------------
if not defined POLAROID_EXE set "POLAROID_EXE="
if not defined POLAROID_CONDA_ENV set "POLAROID_CONDA_ENV="
if not defined POLAROID_APP_ARGS set "POLAROID_APP_ARGS="

call :find_polaroid || goto :fail

set "FOLDERS="
:next
if "%~1"=="" goto :run
set "ONE=%~f1"
rem a drive root such as E:\ ends in a backslash that would escape the closing quote
if "%ONE:~-1%"=="\" set "ONE=%ONE%."
set FOLDERS=%FOLDERS% "%ONE%"
shift
goto :next

:run
echo Starting the Polaroid app. Close this window to stop it.
"%POLAROID_EXE%" app %FOLDERS% %POLAROID_APP_ARGS%
if errorlevel 1 goto :fail
goto :end

rem ---------------------------------------------------------------------------
:find_polaroid
if defined POLAROID_EXE if exist "%POLAROID_EXE%" exit /b 0
if defined POLAROID_CONDA_ENV (
    for %%R in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%LOCALAPPDATA%\miniconda3") do (
        if exist "%%~R\envs\%POLAROID_CONDA_ENV%\Scripts\polaroid.exe" (
            call :use_env "%%~R\envs\%POLAROID_CONDA_ENV%"
            exit /b 0
        )
    )
    echo Conda environment "%POLAROID_CONDA_ENV%" with polaroid.exe not found.
    exit /b 1
)
for %%X in (polaroid.exe) do set "POLAROID_EXE=%%~$PATH:X"
if defined POLAROID_EXE exit /b 0
rem started from Explorer, no conda environment is active: look in every one
for %%R in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%LOCALAPPDATA%\miniconda3") do (
    for /d %%E in ("%%~R\envs\*") do (
        if exist "%%~E\Scripts\polaroid.exe" (
            call :use_env "%%~E"
            echo Using %%~E\Scripts\polaroid.exe
            exit /b 0
        )
    )
)
echo polaroid.exe not found. Install polaroid-processor, or set POLAROID_EXE or
echo POLAROID_CONDA_ENV at the top of this file.
exit /b 1

:use_env
set "POLAROID_EXE=%~1\Scripts\polaroid.exe"
set "PATH=%~1;%~1\Library\bin;%~1\Scripts;%PATH%"
exit /b 0

rem ---------------------------------------------------------------------------
:fail
set "FAILED=1"
:end
if defined FAILED (
    echo.
    echo The app did not start or stopped with a problem. See the messages above.
    if not defined POLAROID_NO_PAUSE pause
    exit /b 1
)
exit /b 0
