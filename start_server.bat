@echo off
echo Starting Mining Prediction API Server...
echo.
echo Make sure you're in the backend directory and virtual environment is activated
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause
