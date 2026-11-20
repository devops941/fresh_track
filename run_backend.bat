@echo off
echo ===================================================
echo Starting FreshTrack FastAPI Backend Server & UI
echo Web App: http://127.0.0.1:8000
echo API Docs: http://127.0.0.1:8000/docs
echo ===================================================
cd /d "%~dp0backend"
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
