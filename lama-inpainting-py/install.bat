@echo off
REM install.bat - Install the LaMa sidecar plug-in for GIMP 3.2.
REM
REM Usage:
REM   install.bat
REM   install.bat "C:\Path\To\python.exe"
REM
REM The installer creates a unique user-level GIMP interpreter alias for the
REM plug-in shebang. It never changes GIMP's installation interpreter files and
REM never uses vendor\.

setlocal EnableExtensions

set "SRC=%~dp0"
set "DEST=%APPDATA%\GIMP\3.2\plug-ins\lama-inpaint"
set "INTERPRETER_DIR=%APPDATA%\GIMP\3.2\interpreters"
set "WORKER_PYTHON="
set "GIMP_ROOT="
set "GIMP_PYTHON="

if not defined APPDATA (
    echo ERROR: APPDATA is not defined.
    exit /b 1
)

for %%F in (lama-inpaint.py lama_worker.py lama_inpaint.py) do (
    if not exist "%SRC%%%F" (
        echo ERROR: Required file not found: %SRC%%%F
        exit /b 1
    )
)

REM The LaMa model is fetched from HuggingFace if not present in the
REM source dir. This keeps the repo small (model is ~200 MB) while
REM still giving the user a one-command install.
set "LAMA_MODEL_URL=https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx?download=true"
set "LAMA_MODEL_PATH=%SRC%lama_fp32.onnx"
if not exist "%LAMA_MODEL_PATH%" (
    echo.
    echo LaMa model not found locally. Downloading from:
    echo   %LAMA_MODEL_URL%
    echo This is a ~200 MB file and may take a moment.
    echo.
    powershell -NoProfile -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%LAMA_MODEL_URL%' -OutFile '%LAMA_MODEL_PATH%' -UseBasicParsing } catch { exit 1 }"
    if errorlevel 1 (
        echo.
        echo WARNING: Could not download the LaMa model automatically.
        echo   Please download it manually from:
        echo     %LAMA_MODEL_URL%
        echo   and place it at:
        echo     %LAMA_MODEL_PATH%
        echo The plug-in will not work without it.
        echo.
        set "LAMA_MODEL_DOWNLOAD_FAILED=1"
    ) else (
        echo Download complete: %LAMA_MODEL_PATH%
    )
)

call :find_gimp
if not defined GIMP_ROOT goto :no_gimp
set "GIMP_PYTHON=%GIMP_ROOT%\bin\python.exe"
if not exist "%GIMP_PYTHON%" goto :no_gimp

echo GIMP install: %GIMP_ROOT%
echo GIMP Python:  %GIMP_PYTHON%
echo.

if not "%~1"=="" goto :use_argument
call :detect_python
if not defined WORKER_PYTHON goto :no_python
goto :python_ready

:use_argument
if not exist "%~1" (
    echo ERROR: Python was not found at: %~1
    exit /b 1
)
call :try_candidate "%~1"
if not defined WORKER_PYTHON (
    echo ERROR: The requested interpreter is not Python 3.10 or newer:
    echo   %~1
    exit /b 1
)

:python_ready
echo Worker Python: %WORKER_PYTHON%
"%WORKER_PYTHON%" --version
if errorlevel 1 (
    echo ERROR: Could not run the selected worker Python interpreter.
    exit /b 1
)

echo Checking Pillow, NumPy, and ONNX Runtime...
"%WORKER_PYTHON%" -c "import PIL, numpy, onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ML dependencies missing. Installing them now...
    echo.
    "%WORKER_PYTHON%" -m pip install pillow numpy onnxruntime
    if errorlevel 1 goto :pip_failed

    "%WORKER_PYTHON%" -c "import PIL, numpy, onnxruntime" >nul 2>&1
    if errorlevel 1 goto :dependency_failed
)

set "LAMA_CONFIG_PATH=%SRC%lama_config.json"
set "LAMA_CONFIG_PYTHON=%WORKER_PYTHON%"
"%WORKER_PYTHON%" -c "import json, os; from pathlib import Path; Path(os.environ['LAMA_CONFIG_PATH']).write_text(json.dumps({'worker_python': os.path.abspath(os.environ['LAMA_CONFIG_PYTHON'])}, indent=2) + '\n', encoding='utf-8')"
if errorlevel 1 (
    echo ERROR: Could not write %SRC%lama_config.json
    exit /b 1
)

set "LAMA_INTERPRETER_DIR=%INTERPRETER_DIR%"
set "LAMA_GIMP_PYTHON=%GIMP_PYTHON%"
set "LAMA_GIMP_PYTHONW=%GIMP_ROOT%\bin\pythonw.exe"
if not exist "%LAMA_GIMP_PYTHONW%" set "LAMA_GIMP_PYTHONW=%GIMP_PYTHON%"
set "LAMA_GIMP_PYTHON_GUI=%LAMA_GIMP_PYTHONW%"
"%WORKER_PYTHON%" -c "import os; from pathlib import Path; directory=Path(os.environ['LAMA_INTERPRETER_DIR']); directory.mkdir(parents=True, exist_ok=True); console='lama-gimp-python=' + Path(os.environ['LAMA_GIMP_PYTHON']).resolve().as_posix() + chr(10); gui='lama-gimp-python=' + Path(os.environ['LAMA_GIMP_PYTHON_GUI']).resolve().as_posix() + chr(10); (directory / 'lama-gimp-python.interp').write_text(console, encoding='utf-8'); (directory / 'lama-gimp-python_win.interp').write_text(gui, encoding='utf-8')"
if errorlevel 1 goto :interpreter_failed

if not exist "%INTERPRETER_DIR%\lama-gimp-python.interp" goto :interpreter_failed
if not exist "%INTERPRETER_DIR%\lama-gimp-python_win.interp" goto :interpreter_failed

echo Creating plug-in directory: %DEST%
if not exist "%DEST%" mkdir "%DEST%"
if errorlevel 1 (
    echo ERROR: Could not create the plug-in directory.
    exit /b 1
)

copy /Y "%SRC%lama-inpaint.py" "%DEST%\lama-inpaint.py" >nul || goto :copy_failed
copy /Y "%SRC%lama_worker.py" "%DEST%\lama_worker.py" >nul || goto :copy_failed
copy /Y "%SRC%lama_inpaint.py" "%DEST%\lama_inpaint.py" >nul || goto :copy_failed
copy /Y "%SRC%lama_config.json" "%DEST%\lama_config.json" >nul || goto :copy_failed
if exist "%SRC%lama_fp32.onnx" (
    copy /Y "%SRC%lama_fp32.onnx" "%DEST%\lama_fp32.onnx" >nul || goto :copy_failed
)

echo.
echo Attempting optional Rust worker build...

REM The Rust worker is an opt-in drop-in replacement for the Python
REM worker. It is fully optional: the Python worker is the default,
REM and any failure here is a warning, not a fatal error.
set "RUST_DEST=%DEST%\lama_worker_rust.exe"
set "RUST_BUILD_DIR=%DEST%\rust-target"
set "RUST_BUILT=%RUST_BUILD_DIR%\bin\lama-worker.exe"
set "RUST_SRC_DIR=%SRC%..\lama-worker-rs"
set "RUST_STATUS=skipped"

if exist "%RUST_DEST%" (
    set "RUST_STATUS=already present"
    goto :rust_done
)

where cargo >nul 2>&1
if errorlevel 1 (
    echo NOTE: cargo is not on PATH; skipping the optional Rust worker build.
    echo       The Python worker is the default and is fully working.
    echo       To build the Rust worker manually, run:
    echo         cargo build --release --manifest-path "%RUST_SRC_DIR%\Cargo.toml"
    echo       then copy "%RUST_SRC_DIR%\target\release\lama-worker.exe" to "%RUST_DEST%".
    set "RUST_STATUS=not built (no cargo on PATH)"
    goto :rust_done
)

if not exist "%RUST_SRC_DIR%\Cargo.toml" (
    echo NOTE: Rust source not found at %RUST_SRC_DIR%; skipping the optional Rust build.
    set "RUST_STATUS=not built (source not present)"
    goto :rust_done
)

echo Running: cargo build --release --manifest-path "%RUST_SRC_DIR%\Cargo.toml"
cargo build --release --manifest-path "%RUST_SRC_DIR%\Cargo.toml"
if errorlevel 1 (
    echo WARNING: cargo build failed; the Rust worker will not be installed.
    echo          The Python worker is the default and is fully working.
    set "RUST_STATUS=not built (cargo build failed)"
    goto :rust_done
)

if not exist "%RUST_SRC_DIR%\target\release\lama-worker.exe" (
    echo WARNING: cargo build reported success but the binary is missing:
    echo          %RUST_SRC_DIR%\target\release\lama-worker.exe
    set "RUST_STATUS=not built (binary missing)"
    goto :rust_done
)

copy /Y "%RUST_SRC_DIR%\target\release\lama-worker.exe" "%RUST_DEST%" >nul
if errorlevel 1 (
    echo WARNING: Could not copy the Rust worker to %RUST_DEST%.
    set "RUST_STATUS=not built (copy failed)"
) else (
    echo Rust worker installed: %RUST_DEST%
    set "RUST_STATUS=installed"
)

:rust_done
REM When the Rust binary is present (pre-existing or freshly built),
REM set worker_kind to rust so the plug-in prefers it by default.
set "RUST_PRESENT=0"
if exist "%DEST%\lama_worker_rust.exe" set "RUST_PRESENT=1"
if "%RUST_PRESENT%"=="1" (
    "%WORKER_PYTHON%" -c "import json; p=r'%DEST%\lama_config.json'; d=json.loads(open(p).read()); d['worker_kind']='rust'; open(p,'w').write(json.dumps(d,indent=2)+'\n')" >nul 2>&1
)
echo.
echo Installed LaMa Inpaint to:
echo   %DEST%
echo User-level GIMP interpreter mappings:
echo   %INTERPRETER_DIR%
echo Rust worker status: %RUST_STATUS%
echo.
echo Restart GIMP, then use Filters ^> Enhance ^> LaMa Inpaint...
echo The Rust worker is the default when available (CPU, no GPU
echo dependencies). Set LAMA_USE_RUST_WORKER=0 or
echo "worker_kind": "python" to force the Python worker.
echo.
echo The plug-in needs one ONNX model:
echo   lama_fp32.onnx      (LaMa inpainter, ~198 MB, fast ~2 s)
echo.
echo This was downloaded from HuggingFace during install. If the
echo download failed, get it from:
echo   %LAMA_MODEL_URL%
echo and place it at:
echo   %DEST%\lama_fp32.onnx
endlocal
exit /b 0

:no_gimp
echo ERROR: Could not find GIMP 3 bundled Python.
echo Looked for:
echo   %LOCALAPPDATA%\Programs\GIMP 3\bin\python.exe
echo   %ProgramFiles%\GIMP 3\bin\python.exe
echo   %ProgramFiles(x86)%\GIMP 3\bin\python.exe
exit /b 1

:no_python
echo ERROR: No usable Python 3.10 or newer worker interpreter was found.
echo Install Python from https://www.python.org/ or run:
echo   install.bat "C:\Path\To\python.exe"
exit /b 1

:pip_failed
echo.
echo ERROR: Package installation failed. Run this command manually:
echo   "%WORKER_PYTHON%" -m pip install pillow numpy onnxruntime
exit /b 1

:dependency_failed
echo ERROR: Pillow, NumPy, or ONNX Runtime still cannot be imported.
echo Re-run the installer with a different Python 3.10+ interpreter.
exit /b 1

:interpreter_failed
echo ERROR: Failed to write the GIMP interpreter mappings to:
echo   %INTERPRETER_DIR%
exit /b 1

:copy_failed
echo ERROR: Failed to copy a plug-in file to:
echo   %DEST%
echo Existing unrelated files in that directory were left in place.
exit /b 1

:find_gimp
if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\Programs\GIMP 3\bin\python.exe" set "GIMP_ROOT=%LOCALAPPDATA%\Programs\GIMP 3"
if not defined GIMP_ROOT if defined ProgramFiles if exist "%ProgramFiles%\GIMP 3\bin\python.exe" set "GIMP_ROOT=%ProgramFiles%\GIMP 3"
if not defined GIMP_ROOT if defined ProgramFiles(x86) if exist "%ProgramFiles(x86)%\GIMP 3\bin\python.exe" set "GIMP_ROOT=%ProgramFiles(x86)%\GIMP 3"
exit /b 0

:detect_python
for /f "delims=" %%P in ('where python.exe 2^>nul') do (
    if not defined WORKER_PYTHON call :try_candidate "%%P"
)
for /f "delims=" %%P in ('where python3.exe 2^>nul') do (
    if not defined WORKER_PYTHON call :try_candidate "%%P"
)
if defined LOCALAPPDATA (
    for %%V in (314 313 312 311 310) do (
        if not defined WORKER_PYTHON call :try_candidate "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
    )
)
if not defined WORKER_PYTHON (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        if not defined WORKER_PYTHON call :try_candidate "%%P"
    )
)
exit /b 0

:try_candidate
if not exist "%~1" exit /b 1
"%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 exit /b 1
set "WORKER_PYTHON=%~f1"
exit /b 0
