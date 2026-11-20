@echo off
echo ===================================================
echo Starting FreshTrack Static Frontend Client
echo Portal URL: http://127.0.0.1:5555/index.html
echo ===================================================
cd /d "%~dp0frontend"
python -m http.server 5555
pause
