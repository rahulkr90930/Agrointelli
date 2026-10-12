@echo off
title AgroIntelli Pinggy Public Tunnel
echo ========================================================
echo   🌿 AgroIntelli - Instant Public Tunnel (Pinggy)
echo ========================================================
echo.
echo Make sure AgroIntelli is running on port 5000 (python run.py)
echo Starting SSH tunnel via Pinggy...
echo.
ssh -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new -p 443 -R0:localhost:5000 a.pinggy.io
pause
