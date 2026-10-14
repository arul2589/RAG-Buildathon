@echo off
cd /d "%~dp0"

start "FastAPI" cmd /k "call .venv\Scripts\activate.bat && uvicorn app:api --reload --port 8000"
timeout /t 3 /nobreak >nul

call .venv\Scripts\activate.bat
python -m streamlit run app.py --server.port 8503