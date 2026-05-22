@echo off
cd /d "%~dp0"
start "" "C:\Users\Vulli\anaconda3\envs\sign_env\python.exe" app.py
timeout /t 5 /nobreak >nul
start "" "http://127.0.0.1:5000/"
