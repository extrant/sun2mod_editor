@echo off
call "%~dp0build_exe.bat" %*
exit /b %errorlevel%
