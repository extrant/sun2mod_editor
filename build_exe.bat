@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..") do set "SOURCE_DIR=%%~fI"
for %%I in ("%~dp0.") do set "TARGET_DIR=%%~fI"
set "BUILD_DIR=%TEMP%\SUN_MOD_Editor_Build"
set "ICON_FILE=%~dp0favicon.ico"
set "BUILD_ICON=%BUILD_DIR%\favicon.ico"

if not exist "%SOURCE_DIR%\mod_editor.pyw" goto :missing_source
if not exist "%ICON_FILE%" goto :missing_icon

if /I "%~1"=="--check" (
    echo SOURCE_DIR=%SOURCE_DIR%
    echo TARGET_DIR=%TARGET_DIR%
    echo BUILD_DIR=%BUILD_DIR%
    echo ICON_FILE=%ICON_FILE%
    echo Build script check passed.
    exit /b 0
)

echo [1/2] Checking Nuitka build tools...
py -3 -c "import nuitka, ordered_set, zstandard" >nul 2>&1
if errorlevel 1 (
    py -3 -m pip install "Nuitka[app]" ordered-set zstandard
    if errorlevel 1 goto :failed
)

echo [2/2] Building SUN_MOD_Editor.exe...
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if errorlevel 1 goto :failed
copy /Y "%ICON_FILE%" "%BUILD_ICON%" >nul
if errorlevel 1 goto :failed

pushd "%SOURCE_DIR%"
py -3 -m nuitka ^
    --mode=onefile ^
    --onefile-cache-mode=cached ^
    --enable-plugins=tk-inter ^
    --windows-console-mode=disable ^
    --assume-yes-for-downloads ^
    --remove-output ^
    --output-dir="%BUILD_DIR%" ^
    --output-filename=SUN_MOD_Editor.exe ^
    --company-name=SUNModTools_QQGroup_221860548 ^
    --product-name=SUN2_MOD_Editor_QQGroup_221860548 ^
    --file-version=1.0.0.0 ^
    --product-version=1.0.0.0 ^
    --file-description=SUN_MOD_Resource_Editor_QQGroup_221860548 ^
    --windows-icon-from-ico="%BUILD_ICON%" ^
    --include-data-files="%BUILD_ICON%=favicon.ico" ^
    --include-module=wzm_viewer ^
    --include-module=ewz_inspect ^
    --nofollow-import-to=matplotlib ^
    mod_editor.pyw
set "BUILD_RESULT=%ERRORLEVEL%"
popd
if not "%BUILD_RESULT%"=="0" goto :failed

copy /Y "%BUILD_DIR%\SUN_MOD_Editor.exe" "%TARGET_DIR%\SUN_MOD_Editor.exe" >nul
if errorlevel 1 goto :failed

echo.
echo Build complete: "%TARGET_DIR%\SUN_MOD_Editor.exe"
pause
exit /b 0

:missing_source
echo ERROR: Cannot find "%SOURCE_DIR%\mod_editor.pyw".
goto :failed

:missing_icon
echo ERROR: Cannot find "%ICON_FILE%".
goto :failed

:failed
echo.
echo Build failed. Review the error message above.
pause
exit /b 1
