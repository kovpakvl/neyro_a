@echo off
setlocal

rem Run from this script's directory
cd /d "%~dp0"

rem Start API in a separate window
start "API Server" python api.py

rem Give the server a moment to start
ping 127.0.0.1 -n 3 >nul

rem Start client in a separate window
start "API Client" python app_api_client.py

endlocal

