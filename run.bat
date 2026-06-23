@echo off
title NeuralRAG Production Launcher
color 0B

echo =========================================
echo  Starting NeuralRAG (HTML UI + Flask)
echo =========================================
echo.

echo [1/2] Starting Flask Backend...
start "NeuralRAG Server" cmd /k "call venv\Scripts\activate.bat && python server.py"

echo Waiting 5 seconds for server to boot...
timeout /t 5 /nobreak >nul

echo [2/2] Starting Ngrok Tunnel (Port 5000)...
start "Ngrok Tunnel" cmd /k "ngrok http 5000"

echo.
echo =========================================
echo  DONE! Look at the Ngrok window for your
echo  public URL (e.g., https://something.ngrok.app)
echo  and open it on your phone or PC!
echo =========================================
pause
