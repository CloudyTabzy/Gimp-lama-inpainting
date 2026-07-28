@echo off
REM Launch GIMP 3 with a visible console window.
REM After GIMP exits, shows the plug-in log.

echo Starting GIMP 3.2 with console messages...
echo.
echo The plug-in writes status to:
echo   %%APPDATA%%\GIMP\3.2\plug-ins\lama-inpaint\lama.log
echo.
echo Close this window or quit GIMP to finish.

"C:\Users\User\AppData\Local\Programs\GIMP 3\bin\gimp-3.2.exe" --new-instance --console-messages --verbose

echo.
echo ===== Plug-in log =====
if exist "%APPDATA%\GIMP\3.2\plug-ins\lama-inpaint\lama.log" (
    type "%APPDATA%\GIMP\3.2\plug-ins\lama-inpaint\lama.log"
) else (
    echo (no log entries yet - run the plug-in first)
)
echo =======================
echo.
pause
