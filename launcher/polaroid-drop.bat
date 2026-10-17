@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem ---------------------------------------------------------------------------
rem  Polaroid processor - drag and drop launcher for Windows
rem
rem  Drag one or more scan folders onto this file (or double-click it and type
rem  a folder path). Each folder is processed into a sibling folder named
rem  "<folder>_processed". The scans themselves are never changed.
rem
rem  Settings: edit the lines below once, or set the same names as Windows
rem  environment variables.
rem    POLAROID_EXE        full path to polaroid.exe (optional)
rem    POLAROID_CONDA_ENV  conda environment that has polaroid-processor installed
rem    POLAROID_ARGS       extra options for "polaroid process", e.g. --overlay
rem ---------------------------------------------------------------------------
if not defined POLAROID_EXE set "POLAROID_EXE="
if not defined POLAROID_CONDA_ENV set "POLAROID_CONDA_ENV="
if not defined POLAROID_ARGS set "POLAROID_ARGS="

call :find_polaroid || goto :fail

if "%~1"=="" (
    set "ONE="
    set /p "ONE=Scan folder: "
    if not defined ONE goto :end
    call :process_one "!ONE:"=!"
    goto :end
)

:next
if "%~1"=="" goto :end
call :process_one "%~1"
shift
goto :next

rem ---------------------------------------------------------------------------
:process_one
set "SRC=%~f1"
if exist "%SRC%\" (
    set "OUT=%SRC%_processed"
) else if exist "%SRC%" (
    rem a single scan file: output next to its folder
    for %%D in ("%~dp1.") do set "OUT=%%~fD_processed"
) else (
    echo Not found: %SRC%
    set "FAILED=1"
    exit /b 1
)
echo.
echo === %SRC%
echo     output: %OUT%
"%POLAROID_EXE%" process "%SRC%" -o "%OUT%" -v %POLAROID_ARGS%
if errorlevel 2 (
    echo Could not process %SRC%.
    set "FAILED=1"
) else (
    if errorlevel 1 set "FAILED=1"
    set "LAST_OUT=%OUT%"
)
exit /b 0

rem ---------------------------------------------------------------------------
:find_polaroid
if defined POLAROID_EXE if exist "%POLAROID_EXE%" exit /b 0
if defined POLAROID_CONDA_ENV (
    for %%R in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%LOCALAPPDATA%\miniconda3") do (
        if exist "%%~R\envs\%POLAROID_CONDA_ENV%\Scripts\polaroid.exe" (
            set "POLAROID_EXE=%%~R\envs\%POLAROID_CONDA_ENV%\Scripts\polaroid.exe"
            set "PATH=%%~R\envs\%POLAROID_CONDA_ENV%;%%~R\envs\%POLAROID_CONDA_ENV%\Library\bin;%%~R\envs\%POLAROID_CONDA_ENV%\Scripts;!PATH!"
            exit /b 0
        )
    )
    echo Conda environment "%POLAROID_CONDA_ENV%" with polaroid.exe not found.
    exit /b 1
)
for %%X in (polaroid.exe) do set "POLAROID_EXE=%%~$PATH:X"
if defined POLAROID_EXE exit /b 0
echo polaroid.exe not found. Install polaroid-processor, or set POLAROID_EXE or
echo POLAROID_CONDA_ENV at the top of this file.
exit /b 1

rem ---------------------------------------------------------------------------
:fail
set "FAILED=1"
:end
echo.
if defined FAILED (
    echo Finished with problems. See the messages above.
) else (
    echo Done. Scans marked "needs review" are exported but worth a look.
)
if defined LAST_OUT if not defined POLAROID_NO_EXPLORER start "" explorer "%LAST_OUT%"
if not defined POLAROID_NO_PAUSE pause
if defined FAILED exit /b 1
exit /b 0
