@echo off
title AgroIntelli Public Tunnel (Pinggy)
echo ===================================================
echo   AgroIntelli Public Shareable Tunnel (Pinggy)
echo ===================================================
echo Forwarding port 5000 to the public web...
echo.
ssh -p 443 -R0:localhost:5000 a.pinggy.io
pause
